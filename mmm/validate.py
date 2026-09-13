"""Validation: does the fit reproduce the data, recover the truth, and predict weeks it never saw?

Three questions, in increasing order of how much they hurt.

**Posterior predictive checks** ask whether data simulated from the fitted model looks like the data that
was actually observed. The Bayesian p-value is the fraction of replicated datasets whose test statistic is
at least as extreme as the observed one; values near 0 or 1 mean the model cannot produce what happened.
The statistics chosen here are the ones an additive-normal MMM tends to fail: week-over-week volatility
(fails when real sales are heteroscedastic), residual autocorrelation (fails when a promotion effect is
missing), and the maximum (fails when a single launch week dominates the year).

**Parameter recovery** is only possible on synthetic data, and it is the only place where "is the posterior
right?" has a checkable answer. Coverage is the metric: a well-calibrated 90% interval should contain the
truth about 90% of the time across repeated simulations -- not always, which would mean the intervals are
too wide, and certainly not half the time.

**Holdout forecasting** is the honest business test. Fit on the first weeks, predict the rest, and report
both error and interval coverage. It is where an over-flexible seasonal basis is exposed: in-sample fit
improves monotonically with harmonics, and out-of-sample error does not.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .diagnostics import quantile
from .model import MMM, Dataset, Parameters


def flatten_natural(model: MMM, params: Parameters) -> list[float]:
    """Natural-scale parameters in the same order as ``model.names``."""
    flat = [params.base, params.trend]
    flat.extend(params.season)
    flat.extend(params.beta)
    flat.extend(params.decay)
    flat.extend(params.half)
    flat.extend(params.shape)
    flat.extend(params.control)
    flat.append(params.sigma)
    return flat


def natural_columns(model: MMM, posterior) -> list[list[list[float]]]:
    """Posterior draws on the natural scale, as ``[coordinate][chain][draw]``.

    Summaries are reported on this scale because nobody negotiates a media budget in units of log kappa.
    """
    columns: list[list[list[float]]] = [[] for _ in model.names]
    for chain in posterior.chains:
        transformed = [flatten_natural(model, model.unpack(draw)) for draw in chain.draws]
        for index in range(len(model.names)):
            columns[index].append([row[index] for row in transformed])
    return columns


# ---------------------------------------------------------------------------------------------
# posterior predictive checks
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PPCResult:
    statistic: str
    observed: float
    replicated_mean: float
    bayes_p: float

    @property
    def suspicious(self) -> bool:
        return self.bayes_p < 0.05 or self.bayes_p > 0.95

    def row(self) -> str:
        flag = "  <-- the model cannot reproduce this" if self.suspicious else ""
        return (
            f"{self.statistic:<28}{self.observed:>14.2f}{self.replicated_mean:>16.2f}"
            f"{self.bayes_p:>10.3f}{flag}"
        )


def _standard_deviation(values: list[float]) -> float:
    count = len(values)
    average = sum(values) / count
    return math.sqrt(sum((value - average) ** 2 for value in values) / max(count - 1, 1))


def _lag_one_autocorrelation(values: list[float]) -> float:
    count = len(values)
    average = sum(values) / count
    numerator = sum((values[t] - average) * (values[t + 1] - average) for t in range(count - 1))
    denominator = sum((value - average) ** 2 for value in values)
    return numerator / denominator if denominator > 0 else 0.0


def _mean_absolute_change(values: list[float]) -> float:
    return sum(abs(values[t + 1] - values[t]) for t in range(len(values) - 1)) / max(
        len(values) - 1, 1
    )


STATISTICS = {
    "standard deviation": _standard_deviation,
    "maximum": max,
    "minimum": min,
    "lag-1 autocorrelation": _lag_one_autocorrelation,
    "mean week-over-week change": _mean_absolute_change,
}


def posterior_predictive_check(
    model: MMM, posterior, seed: int = 0, draws: int = 200
) -> list[PPCResult]:
    """Simulate replicated datasets from the posterior and compare summary statistics with the data."""
    rng = random.Random(seed)
    samples = posterior.draws()
    if not samples:
        raise ValueError("the posterior contains no draws")
    stride = max(len(samples) // max(draws, 1), 1)
    selected = samples[::stride][:draws]

    replicated = [model.posterior_predictive(theta, rng) for theta in selected]
    results: list[PPCResult] = []
    for name, statistic in STATISTICS.items():
        observed = statistic(model.data.y)
        values = [statistic(series) for series in replicated]
        at_least = sum(1 for value in values if value >= observed)
        results.append(
            PPCResult(
                statistic=name,
                observed=observed,
                replicated_mean=sum(values) / len(values),
                bayes_p=at_least / len(values),
            )
        )
    return results


def ppc_table(results: list[PPCResult]) -> str:
    header = f"{'statistic':<28}{'observed':>14}{'replicated':>16}{'p':>10}"
    lines = [header, "-" * len(header)]
    lines.extend(result.row() for result in results)
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# parameter recovery
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryResult:
    name: str
    truth: float
    posterior_mean: float
    q05: float
    q95: float

    @property
    def covered(self) -> bool:
        return self.q05 <= self.truth <= self.q95

    @property
    def relative_error(self) -> float:
        scale = max(abs(self.truth), 1e-9)
        return (self.posterior_mean - self.truth) / scale

    def row(self) -> str:
        mark = "yes" if self.covered else "NO"
        return (
            f"{self.name:<22}{self.truth:>14.3f}{self.posterior_mean:>14.3f}"
            f"{self.q05:>13.3f}{self.q95:>13.3f}{mark:>7}"
        )


def parameter_recovery(model: MMM, posterior, truth: Parameters) -> list[RecoveryResult]:
    """Compare the 90% credible interval for every parameter with the value that generated the data."""
    true_values = flatten_natural(model, truth)
    columns = natural_columns(model, posterior)
    results: list[RecoveryResult] = []
    for name, value, chains in zip(model.names, true_values, columns):
        pooled = [draw for chain in chains for draw in chain]
        results.append(
            RecoveryResult(
                name=name,
                truth=value,
                posterior_mean=sum(pooled) / len(pooled),
                q05=quantile(pooled, 0.05),
                q95=quantile(pooled, 0.95),
            )
        )
    return results


def recovery_table(results: list[RecoveryResult]) -> str:
    header = f"{'parameter':<22}{'truth':>14}{'posterior':>14}{'5%':>13}{'95%':>13}{'in':>7}"
    lines = [header, "-" * len(header)]
    lines.extend(result.row() for result in results)
    covered = sum(1 for result in results if result.covered)
    lines.append("")
    lines.append(
        f"coverage: {covered}/{len(results)} parameters inside their 90% interval "
        f"({covered / len(results):.0%}; nominal 90%)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# holdout forecasting
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    weeks: int
    rmse: float
    mape: float
    coverage: float
    naive_rmse: float

    @property
    def skill(self) -> float:
        """Fraction of the naive seasonal forecast's error removed. Negative means it is worse."""
        if self.naive_rmse <= 0.0:
            return 0.0
        return 1.0 - self.rmse / self.naive_rmse

    def summary(self) -> str:
        return (
            f"{self.weeks}-week holdout: RMSE {self.rmse:,.0f} vs naive {self.naive_rmse:,.0f} "
            f"(skill {self.skill:+.0%}), MAPE {self.mape:.1%}, "
            f"90% interval coverage {self.coverage:.0%}"
        )


def forecast_evaluation(
    model_full: MMM,
    posterior,
    holdout: int,
    seed: int = 0,
    draws: int = 200,
) -> Forecast:
    """Score a posterior fitted on the training weeks against the held-out tail.

    ``model_full`` must be built on the **whole** dataset while ``posterior`` was fitted on the training
    weeks alone. Evaluating the mean function over the full horizon is what makes carryover from the
    training weeks flow correctly into the holdout instead of being reset to zero at the boundary.
    """
    if holdout < 1:
        raise ValueError("holdout must be at least one week")
    weeks = model_full.weeks
    if holdout >= weeks:
        raise ValueError("holdout must be shorter than the series")

    rng = random.Random(seed)
    samples = posterior.draws()
    stride = max(len(samples) // max(draws, 1), 1)
    selected = samples[::stride][:draws]

    predictions: list[list[float]] = []
    for theta in selected:
        params = model_full.unpack(theta)
        mean = model_full.mean(params)
        predictions.append(
            [value + rng.gauss(0.0, params.sigma) for value in mean[weeks - holdout :]]
        )

    actual = model_full.data.y[weeks - holdout :]
    errors: list[float] = []
    percentage: list[float] = []
    inside = 0
    for index, observed in enumerate(actual):
        column = [prediction[index] for prediction in predictions]
        centre = sum(column) / len(column)
        errors.append((observed - centre) ** 2)
        if observed != 0.0:
            percentage.append(abs(observed - centre) / abs(observed))
        if quantile(column, 0.05) <= observed <= quantile(column, 0.95):
            inside += 1

    # The benchmark is last year's same week where possible, otherwise the training mean: a forecast that
    # cannot beat that has no business informing a budget.
    period = int(model_full.data.period)
    training = model_full.data.y[: weeks - holdout]
    training_mean = sum(training) / len(training)
    naive: list[float] = []
    for offset in range(holdout):
        index = weeks - holdout + offset - period
        naive.append(model_full.data.y[index] if index >= 0 else training_mean)
    naive_errors = [(observed - reference) ** 2 for observed, reference in zip(actual, naive)]

    return Forecast(
        weeks=holdout,
        rmse=math.sqrt(sum(errors) / len(errors)),
        mape=sum(percentage) / len(percentage) if percentage else float("nan"),
        coverage=inside / len(actual),
        naive_rmse=math.sqrt(sum(naive_errors) / len(naive_errors)),
    )


def train_test_split(data: Dataset, holdout: int) -> tuple[Dataset, Dataset]:
    """Chronological split. Never random: shuffling a time series leaks the future into the past."""
    if holdout < 1 or holdout >= data.weeks:
        raise ValueError("holdout must be between 1 and weeks - 1")
    return data.slice(0, data.weeks - holdout), data.slice(data.weeks - holdout, data.weeks)
