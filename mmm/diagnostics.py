"""Convergence diagnostics: rank-normalised split R-hat, effective sample size, Monte Carlo error.

A sampler that has not converged produces confident nonsense, and MMM posteriors are exactly the shape
that hides it -- long ridges where a chain can wander for hundreds of draws without ever visiting the
other end. So no number from this repository is reported without its diagnostics attached.

**Split R-hat** halves every chain before comparing between-chain and within-chain variance, so a single
chain that drifts is caught even when all chains drift the same way. **Rank normalisation** (Vehtari,
Gelman, Simpson, Carpenter & Burkner, 2021) replaces each draw by its normal-score rank before computing
the statistic, which makes it well defined for heavy-tailed posteriors where the variance-based version
can look fine while the tails disagree completely.

**Effective sample size** uses Geyer's initial positive sequence on the multi-chain autocorrelation
estimate: pair the autocorrelations, sum while the pairs stay positive, and stop. The truncation is what
makes the estimator stable; without it, the noise in the long-lag autocorrelations dominates the sum.

The thresholds used in the summaries are the current recommendations: R-hat below 1.01, and at least 400
effective draws per parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

NORMAL = NormalDist()
MAX_LAG = 500  # autocorrelations past this are noise, and Geyer's rule stops long before it


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def _split(chains: list[list[float]]) -> list[list[float]]:
    """Halve each chain: a chain that has not mixed with itself has not converged."""
    halves: list[list[float]] = []
    for chain in chains:
        middle = len(chain) // 2
        if middle < 2:
            raise ValueError("chains are too short to split")
        halves.append(chain[:middle])
        halves.append(chain[middle : 2 * middle])
    return halves


def _rank_normalise(chains: list[list[float]]) -> list[list[float]]:
    """Replace each draw by ``Phi^-1((rank - 3/8)/(N - 1/4))``, pooling all chains for the ranking."""
    pooled = [(value, index, position) for index, chain in enumerate(chains) for position, value in enumerate(chain)]
    pooled.sort(key=lambda item: item[0])
    total = len(pooled)
    out = [[0.0] * len(chain) for chain in chains]
    for rank, (_, index, position) in enumerate(pooled, start=1):
        quantile = (rank - 0.375) / (total + 0.25)
        out[index][position] = NORMAL.inv_cdf(quantile)
    return out


def rhat(chains: list[list[float]]) -> float:
    """Plain (unsplit, un-normalised) potential scale reduction factor.

    ``sqrt(var_plus / W)`` with ``var_plus = ((n-1)W + B)/n``. Kept public because it is the version in
    the textbooks; ``split_rhat`` is the version to trust.
    """
    if len(chains) < 2:
        raise ValueError("R-hat needs at least two chains")
    lengths = {len(chain) for chain in chains}
    if len(lengths) != 1:
        raise ValueError("R-hat needs chains of equal length")
    n = lengths.pop()
    if n < 4:
        raise ValueError("chains are too short for R-hat")

    within = _mean([_variance(chain) for chain in chains])
    if within <= 0.0:
        return 1.0  # every chain is a constant: degenerate but not divergent
    means = [_mean(chain) for chain in chains]
    between = n * _variance(means)
    var_plus = ((n - 1) * within + between) / n
    return math.sqrt(var_plus / within)


def split_rhat(chains: list[list[float]], rank_normalise: bool = True) -> float:
    """The recommended diagnostic: rank-normalised split R-hat."""
    prepared = _rank_normalise(chains) if rank_normalise else chains
    return rhat(_split(prepared))


def _autocovariance(chain: list[float], max_lag: int) -> list[float]:
    """Biased autocovariance about the chain's own mean, lags 0..max_lag."""
    n = len(chain)
    average = _mean(chain)
    centred = [value - average for value in chain]
    out: list[float] = []
    for lag in range(min(max_lag, n - 1) + 1):
        total = 0.0
        for index in range(n - lag):
            total += centred[index] * centred[index + lag]
        out.append(total / n)
    return out


def effective_sample_size(chains: list[list[float]]) -> float:
    """ESS via Geyer's initial positive sequence over the pooled autocorrelation estimate.

    ``tau = -1 + 2 * sum of positive autocorrelation pairs`` and ``ESS = m * n / tau``. The pairing is
    not cosmetic: individual autocorrelation estimates can be negative by noise alone, and summing them
    until the first negative value badly overestimates ESS for slowly mixing chains.
    """
    lengths = {len(chain) for chain in chains}
    if len(lengths) != 1:
        raise ValueError("ESS needs chains of equal length")
    n = lengths.pop()
    m = len(chains)
    if n < 8:
        raise ValueError("chains are too short for an ESS estimate")

    within = _mean([_variance(chain) for chain in chains])
    if within <= 0.0:
        return float(m * n)  # constant chains: no information, but no autocorrelation either

    if m > 1:
        var_plus = within * (n - 1) / n + _variance([_mean(chain) for chain in chains])
    else:
        var_plus = within * (n - 1) / n

    max_lag = min(MAX_LAG, n - 2)
    covariances = [_autocovariance(chain, max_lag) for chain in chains]
    pooled = [
        _mean([covariance[lag] for covariance in covariances]) for lag in range(max_lag + 1)
    ]
    rho = [1.0 - (within - value) / var_plus for value in pooled]

    total = 0.0
    lag = 1
    while lag + 1 <= max_lag:
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0.0:
            break
        total += pair
        lag += 2

    tau = max(-1.0 + 2.0 * total, 1.0 / math.log10(max(m * n, 11)))
    return m * n / tau


def mcse_mean(chains: list[list[float]]) -> float:
    """Monte Carlo standard error of the posterior mean: ``sd / sqrt(ESS)``.

    This is the number that says whether a reported posterior mean is worth three decimal places.
    """
    pooled = [value for chain in chains for value in chain]
    ess = effective_sample_size(chains)
    return math.sqrt(_variance(pooled) / max(ess, 1.0))


def quantile(values: list[float], probability: float) -> float:
    """Linear-interpolation quantile; no dependency and no surprises about the convention used."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class ParameterSummary:
    name: str
    mean: float
    sd: float
    q05: float
    q50: float
    q95: float
    ess: float
    rhat: float
    mcse: float

    @property
    def healthy(self) -> bool:
        return self.rhat < 1.01 and self.ess >= 400.0

    def row(self) -> str:
        flag = "" if self.healthy else "  <-- check"
        return (
            f"{self.name:<22}{self.mean:>12.3f}{self.sd:>11.3f}{self.q05:>12.3f}"
            f"{self.q95:>12.3f}{self.ess:>9.0f}{self.rhat:>8.3f}{flag}"
        )


def summarise_columns(
    names: list[str], columns_by_chain: list[list[list[float]]]
) -> list[ParameterSummary]:
    """One summary row per coordinate. ``columns_by_chain[i][c]`` is coordinate ``i`` in chain ``c``."""
    if len(names) != len(columns_by_chain):
        raise ValueError("names and columns disagree in length")
    summaries: list[ParameterSummary] = []
    for name, chains in zip(names, columns_by_chain):
        pooled = [value for chain in chains for value in chain]
        summaries.append(
            ParameterSummary(
                name=name,
                mean=_mean(pooled),
                sd=math.sqrt(_variance(pooled)),
                q05=quantile(pooled, 0.05),
                q50=quantile(pooled, 0.50),
                q95=quantile(pooled, 0.95),
                ess=effective_sample_size(chains) if len(chains[0]) >= 8 else float("nan"),
                rhat=split_rhat(chains) if len(chains) > 1 else float("nan"),
                mcse=mcse_mean(chains) if len(chains[0]) >= 8 else float("nan"),
            )
        )
    return summaries


def summary_table(summaries: list[ParameterSummary]) -> str:
    header = (
        f"{'parameter':<22}{'mean':>12}{'sd':>11}{'5%':>12}{'95%':>12}{'ESS':>9}{'R-hat':>8}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(summary.row() for summary in summaries)
    unhealthy = [summary.name for summary in summaries if not summary.healthy]
    if unhealthy:
        lines.append("")
        lines.append(
            f"{len(unhealthy)} parameter(s) below the R-hat < 1.01 / ESS >= 400 bar: "
            + ", ".join(unhealthy[:6])
            + ("..." if len(unhealthy) > 6 else "")
        )
        lines.append("Treat those posterior summaries as unreliable, not merely imprecise.")
    return "\n".join(lines)
