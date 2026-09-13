"""Validation: predictive checks, recovery, and forecasting -- plus one end-to-end fit.

Most of this file avoids sampling on purpose. Feeding hand-made draws through the validation code proves
the arithmetic without waiting for a posterior, and the single ``slow`` test at the end exercises the real
pipeline: simulate, sample, recover, forecast.
"""

import math
import random
from dataclasses import replace

import pytest

from mmm.data import ChannelTruth, simulate
from mmm.model import MMM
from mmm.nuts import NUTSConfig, sample
from mmm.validate import (
    Forecast,
    flatten_natural,
    forecast_evaluation,
    natural_columns,
    parameter_recovery,
    posterior_predictive_check,
    ppc_table,
    recovery_table,
    train_test_split,
)


class FakeChain:
    def __init__(self, draws):
        self.draws = draws


class FakePosterior:
    """Hand-made draws in the shape the validation code expects."""

    def __init__(self, chains):
        self.chains = [FakeChain([list(theta) for theta in chain]) for chain in chains]

    def draws(self):
        return [draw for chain in self.chains for draw in chain.draws]


@pytest.fixture(scope="module")
def fitted():
    dataset, truth = simulate(weeks=104, seed=23)
    return MMM(dataset), truth


def jittered_posterior(model, truth, spread: float = 0.02, count: int = 40, chains: int = 2):
    """Draws scattered tightly around the truth: what a well-behaved posterior would look like."""
    rng = random.Random(0)
    theta = model.pack(truth.parameters())
    return FakePosterior(
        [
            [
                [value + rng.gauss(0.0, spread * max(abs(value), 1e-3)) for value in theta]
                for _ in range(count)
            ]
            for _ in range(chains)
        ]
    )


# -- ordering and shapes ----------------------------------------------------------------------


def test_flattening_follows_the_declared_parameter_order(fitted):
    model, truth = fitted
    flat = flatten_natural(model, truth.parameters())
    assert len(flat) == model.dim == len(model.names)
    assert flat[model.names.index("base")] == pytest.approx(truth.base)
    assert flat[model.names.index("sigma")] == pytest.approx(truth.sigma)
    for name in model.channels:
        channel = truth.channel(name)
        assert flat[model.names.index(f"beta[{name}]")] == pytest.approx(channel.beta)
        assert flat[model.names.index(f"decay[{name}]")] == pytest.approx(channel.decay)
        assert flat[model.names.index(f"half[{name}]")] == pytest.approx(channel.half)
        assert flat[model.names.index(f"shape[{name}]")] == pytest.approx(channel.shape)


def test_natural_columns_are_indexed_by_coordinate_then_chain(fitted):
    model, truth = fitted
    posterior = jittered_posterior(model, truth, count=15, chains=3)
    columns = natural_columns(model, posterior)
    assert len(columns) == model.dim
    assert len(columns[0]) == 3
    assert len(columns[0][0]) == 15
    # Constrained coordinates must arrive already transformed to their natural scale.
    decay_index = model.names.index(f"decay[{model.channels[0]}]")
    assert all(0.0 < value < 1.0 for chain in columns[decay_index] for value in chain)


# -- posterior predictive checks --------------------------------------------------------------


def test_a_well_specified_model_reproduces_its_own_data(fitted):
    """Replicated datasets from the truth should look like the observed data on every statistic."""
    model, truth = fitted
    results = posterior_predictive_check(model, jittered_posterior(model, truth), draws=150)
    assert len(results) == 5
    suspicious = [result.statistic for result in results if result.suspicious]
    assert len(suspicious) <= 1, f"unexpected misfit: {suspicious}"
    assert all(math.isfinite(result.bayes_p) for result in results)


def test_the_checks_catch_a_variance_the_model_cannot_produce(fitted):
    """Shrink the noise and the observed volatility becomes impossible to replicate.

    This is the check working as intended: it is not measuring fit, it is asking whether the generative
    model can produce what was seen.
    """
    model, truth = fitted
    understated = replace(truth, sigma=truth.sigma / 6.0)
    posterior = jittered_posterior(model, understated, spread=0.001, count=60)
    results = posterior_predictive_check(model, posterior, draws=60)
    by_name = {result.statistic: result for result in results}
    assert by_name["standard deviation"].suspicious
    assert by_name["mean week-over-week change"].suspicious
    assert "cannot reproduce" in ppc_table(results)


# -- recovery ---------------------------------------------------------------------------------


def test_recovery_reports_coverage_against_the_answer_key(fitted):
    model, truth = fitted
    posterior = jittered_posterior(model, truth, spread=0.05)
    results = parameter_recovery(model, posterior, truth.parameters())
    assert len(results) == model.dim
    covered = sum(1 for result in results if result.covered)
    assert covered >= model.dim - 1, "draws centred on the truth must cover the truth"
    table = recovery_table(results)
    assert "coverage" in table and "nominal 90%" in table


def test_recovery_notices_when_the_truth_is_outside_the_interval(fitted):
    model, truth = fitted
    displaced = replace(truth, base=truth.base * 1.5)
    posterior = jittered_posterior(model, truth, spread=0.001)
    results = parameter_recovery(model, posterior, displaced.parameters())
    by_name = {result.name: result for result in results}
    assert not by_name["base"].covered
    assert by_name["base"].relative_error < 0.0, "the posterior sits below the displaced truth"


# -- forecasting ------------------------------------------------------------------------------


def test_the_split_is_chronological(fitted):
    model, _ = fitted
    train, test = train_test_split(model.data, 13)
    assert train.weeks == model.weeks - 13
    assert test.weeks == 13
    assert train.y == model.data.y[: model.weeks - 13]
    assert test.y == model.data.y[model.weeks - 13 :]
    for name in model.data.spend:
        assert train.spend[name] + test.spend[name] == model.data.spend[name]


def test_an_impossible_split_is_refused(fitted):
    model, _ = fitted
    with pytest.raises(ValueError, match="holdout"):
        train_test_split(model.data, 0)
    with pytest.raises(ValueError, match="holdout"):
        train_test_split(model.data, model.weeks)


def test_forecast_skill_is_measured_against_the_naive_benchmark():
    assert Forecast(13, 50.0, 0.05, 0.9, 100.0).skill == pytest.approx(0.5)
    assert Forecast(13, 200.0, 0.2, 0.9, 100.0).skill == pytest.approx(-1.0)
    assert "coverage" in Forecast(13, 50.0, 0.05, 0.9, 100.0).summary()


def test_forecasting_from_the_truth_is_accurate_and_calibrated(fitted):
    """With the true parameters the holdout error should be about the noise level, and the 90% intervals
    should contain most of the held-out weeks. Anything else means the forecast path is wrong."""
    model, truth = fitted
    posterior = jittered_posterior(model, truth, spread=0.001, count=100)
    result = forecast_evaluation(model, posterior, holdout=13, draws=100)
    assert result.rmse < 2.0 * truth.sigma
    assert result.coverage >= 0.6
    assert math.isfinite(result.naive_rmse)


def test_the_holdout_must_fit_inside_the_series(fitted):
    model, truth = fitted
    posterior = jittered_posterior(model, truth, count=10)
    with pytest.raises(ValueError, match="holdout"):
        forecast_evaluation(model, posterior, holdout=0)
    with pytest.raises(ValueError, match="holdout"):
        forecast_evaluation(model, posterior, holdout=model.weeks)


# -- one real fit -----------------------------------------------------------------------------


@pytest.mark.slow
def test_the_whole_pipeline_runs_and_recovers_something():
    """Simulate, sample, recover, check -- for real, on a deliberately small problem.

    The thresholds are loose because this is a smoke test of the pipeline, not a claim about statistical
    efficiency: one channel, a bit over a year of weeks, a few hundred draws. The strong recovery claims
    live in the demonstrations, where the sampler is given a realistic budget.
    """
    channels = [ChannelTruth("tv", beta=8000.0, decay=0.4, half=15000.0, shape=1.0, level=18000.0)]
    dataset, truth = simulate(weeks=60, seed=31, channels=channels, harmonics=1)
    model = MMM(dataset, harmonics=1)
    posterior = sample(model, NUTSConfig(draws=200, warmup=200, chains=2, seed=5))

    assert posterior.total_draws == 400
    assert posterior.divergences < 0.25 * posterior.total_draws

    results = parameter_recovery(model, posterior, truth.parameters())
    covered = sum(1 for result in results if result.covered)
    assert covered >= 0.4 * len(results), recovery_table(results)

    checks = posterior_predictive_check(model, posterior, draws=100)
    assert all(math.isfinite(result.bayes_p) for result in checks)
