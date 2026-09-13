"""The sampler and the diagnostics, on targets whose answers are known in closed form.

A sampler cannot be tested against the model it was written for -- if both are wrong in the same way, the
fit looks perfect. So NUTS is tested against a correlated Gaussian whose mean, scales and correlation are
known by algebra, and the diagnostics are tested against sequences constructed to have a specific
autocorrelation or a specific disagreement between chains.
"""

import math
import random

import pytest

from mmm.diagnostics import (
    effective_sample_size,
    mcse_mean,
    quantile,
    rhat,
    split_rhat,
    summarise_columns,
    summary_table,
)
from mmm.nuts import NUTSConfig, sample


class Gaussian:
    """A correlated Gaussian wearing the model interface, so the sampler cannot tell the difference."""

    def __init__(self, correlation: float = 0.9, scales=(1.0, 5.0)) -> None:
        self.correlation = correlation
        self.scales = scales
        self.dim = 2
        self.names = ["x", "y"]

    def log_posterior_and_gradient(self, theta):
        rho = self.correlation
        sx, sy = self.scales
        x, y = theta[0] / sx, theta[1] / sy
        factor = 1.0 / (1.0 - rho * rho)
        value = -0.5 * factor * (x * x - 2.0 * rho * x * y + y * y)
        return value, [-factor * (x - rho * y) / sx, -factor * (y - rho * x) / sy]

    def log_posterior(self, theta):
        return self.log_posterior_and_gradient(theta)[0]

    def initial_point(self, rng, jitter: float = 1.0):
        return [rng.gauss(0.0, jitter * self.scales[0]), rng.gauss(0.0, jitter * self.scales[1])]


class Impossible:
    """A target with no density anywhere. The sampler must say so rather than loop forever."""

    dim = 1
    names = ["x"]

    def log_posterior_and_gradient(self, theta):
        return -math.inf, [0.0]

    def log_posterior(self, theta):
        return -math.inf

    def initial_point(self, rng, jitter: float = 1.0):
        return [0.0]


@pytest.fixture(scope="module")
def gaussian_posterior():
    target = Gaussian(correlation=0.9, scales=(1.0, 5.0))
    return target, sample(target, NUTSConfig(draws=600, warmup=600, chains=2, seed=3))


# -- the sampler ------------------------------------------------------------------------------


def test_nuts_recovers_the_moments_of_a_known_gaussian(gaussian_posterior):
    target, posterior = gaussian_posterior
    x = posterior.column(0)
    y = posterior.column(1)
    count = len(x)
    mean_x, mean_y = sum(x) / count, sum(y) / count
    sd_x = math.sqrt(sum((v - mean_x) ** 2 for v in x) / (count - 1))
    sd_y = math.sqrt(sum((v - mean_y) ** 2 for v in y) / (count - 1))
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / (count - 1)

    assert abs(mean_x) < 0.15, f"mean of x drifted to {mean_x:.3f}"
    assert abs(mean_y) < 0.75, f"mean of y drifted to {mean_y:.3f}"
    assert sd_x == pytest.approx(target.scales[0], rel=0.15)
    assert sd_y == pytest.approx(target.scales[1], rel=0.15)
    assert covariance / (sd_x * sd_y) == pytest.approx(target.correlation, abs=0.08)


def test_a_smooth_target_produces_no_divergences(gaussian_posterior):
    """Divergences on a Gaussian would mean the integrator or the metric is broken."""
    _, posterior = gaussian_posterior
    assert posterior.divergences == 0


def test_the_metric_adapts_to_wildly_different_scales():
    """The diagonal metric should learn that one coordinate is ten times wider than the other.

    Without this, a single step size has to satisfy the tightest direction and the sampler crawls along
    the wide one -- the exact failure that makes plain HMC useless on an MMM posterior.
    """
    posterior = sample(
        Gaussian(correlation=0.0, scales=(1.0, 10.0)),
        NUTSConfig(draws=400, warmup=600, chains=1, seed=5),
    )
    metric = posterior.chains[0].inverse_metric
    assert metric[1] / metric[0] > 10.0, f"metric did not adapt: {metric}"


def test_the_same_seed_gives_the_same_draws():
    target = Gaussian()
    first = sample(target, NUTSConfig(draws=100, warmup=100, chains=1, seed=9))
    second = sample(target, NUTSConfig(draws=100, warmup=100, chains=1, seed=9))
    assert first.chains[0].draws == second.chains[0].draws


def test_different_seeds_give_different_draws():
    target = Gaussian()
    first = sample(target, NUTSConfig(draws=100, warmup=100, chains=1, seed=1))
    second = sample(target, NUTSConfig(draws=100, warmup=100, chains=1, seed=2))
    assert first.chains[0].draws != second.chains[0].draws


def test_hamiltonian_sampling_beats_independence_at_worst_slightly(gaussian_posterior):
    """ESS should be a substantial fraction of the draws, not a handful."""
    _, posterior = gaussian_posterior
    ess = effective_sample_size(posterior.columns_by_chain(0))
    assert ess > 0.2 * posterior.total_draws, f"ESS collapsed to {ess:.0f}"


def test_the_chain_reports_its_own_tuning(gaussian_posterior):
    _, posterior = gaussian_posterior
    for chain in posterior.chains:
        assert chain.step_size > 0.0
        assert len(chain.inverse_metric) == 2
        assert 0.0 < sum(chain.accept_stat) / chain.length <= 1.0
        assert "divergences" in chain.summary()


def test_a_target_with_no_density_is_reported_not_hidden():
    with pytest.raises(ValueError, match="finite posterior density"):
        sample(Impossible(), NUTSConfig(draws=10, warmup=50, chains=1, seed=0))


def test_impossible_configurations_are_refused():
    with pytest.raises(ValueError, match="draws"):
        NUTSConfig(draws=0)
    with pytest.raises(ValueError, match="warmup"):
        NUTSConfig(warmup=10)
    with pytest.raises(ValueError, match="target_accept"):
        NUTSConfig(target_accept=1.5)
    with pytest.raises(ValueError, match="max_treedepth"):
        NUTSConfig(max_treedepth=0)


# -- diagnostics ------------------------------------------------------------------------------


def iid_chains(count: int, length: int, seed: int = 0, mean: float = 0.0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(mean, 1.0) for _ in range(length)] for _ in range(count)]


def ar1_chain(length: int, phi: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    value = 0.0
    out = []
    for _ in range(length):
        value = phi * value + rng.gauss(0.0, math.sqrt(1.0 - phi * phi))
        out.append(value)
    return out


def test_rhat_is_one_for_chains_from_the_same_distribution():
    assert rhat(iid_chains(4, 500, seed=1)) < 1.05
    assert split_rhat(iid_chains(4, 500, seed=2)) < 1.05


def test_rhat_catches_chains_that_disagree():
    chains = iid_chains(1, 300, seed=3) + iid_chains(1, 300, seed=4, mean=4.0)
    assert rhat(chains) > 1.5
    assert split_rhat(chains) > 1.5


def test_splitting_catches_a_chain_that_drifts_even_when_the_means_agree():
    """Two identical drifting chains: unsplit R-hat sees nothing, split R-hat sees everything.

    This is the failure mode that matters in practice -- a sampler slowly sliding down a ridge looks
    perfectly converged to any between-chain comparison.
    """
    rng = random.Random(5)
    drift = [[index * 0.02 + rng.gauss(0.0, 0.05) for index in range(400)] for _ in range(2)]
    assert rhat(drift) < 1.05, "the chains agree with each other, so the plain statistic is blind"
    assert split_rhat(drift) > 1.2, "but neither chain agrees with itself"


def test_rhat_needs_at_least_two_chains_of_equal_length():
    with pytest.raises(ValueError, match="two chains"):
        rhat([[1.0] * 100])
    with pytest.raises(ValueError, match="equal length"):
        rhat([[1.0] * 100, [1.0] * 50])


def test_ess_is_about_the_sample_size_for_independent_draws():
    chains = iid_chains(2, 500, seed=6)
    ess = effective_sample_size(chains)
    assert 500 < ess < 2000, f"ESS of independent draws came out at {ess:.0f} of 1000"


def test_ess_collapses_for_a_strongly_autocorrelated_chain():
    """AR(1) with phi = 0.9 has an integrated autocorrelation time of (1+phi)/(1-phi) = 19."""
    chains = [ar1_chain(2000, 0.9, seed=7), ar1_chain(2000, 0.9, seed=8)]
    ess = effective_sample_size(chains)
    assert 50 < ess < 700, f"ESS came out at {ess:.0f} of 4000 draws"
    assert ess < 0.25 * 4000


def test_ess_ordering_follows_the_autocorrelation():
    low = effective_sample_size([ar1_chain(1500, 0.2, seed=9), ar1_chain(1500, 0.2, seed=10)])
    high = effective_sample_size([ar1_chain(1500, 0.95, seed=9), ar1_chain(1500, 0.95, seed=10)])
    assert low > 5.0 * high


def test_mcse_is_the_standard_error_of_the_mean():
    chains = iid_chains(2, 1000, seed=11)
    assert mcse_mean(chains) == pytest.approx(1.0 / math.sqrt(2000), rel=0.5)


def test_quantiles_are_computed_by_interpolation():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert quantile(values, 0.0) == 1.0
    assert quantile(values, 1.0) == 5.0
    assert quantile(values, 0.5) == 3.0
    assert quantile(values, 0.25) == 2.0
    assert quantile([10.0, 20.0], 0.5) == 15.0
    with pytest.raises(ValueError, match="probability"):
        quantile(values, 1.5)


def test_the_summary_flags_what_is_not_converged():
    good = summarise_columns(["fine"], [iid_chains(4, 500, seed=12)])
    assert good[0].healthy
    assert "fine" in summary_table(good)

    bad_chains = iid_chains(1, 300, seed=13) + iid_chains(1, 300, seed=14, mean=6.0)
    bad = summarise_columns(["broken"], [bad_chains])
    assert not bad[0].healthy
    table = summary_table(bad)
    assert "broken" in table and "unreliable" in table


def test_the_summary_reports_the_right_moments():
    chains = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
    summary = summarise_columns(["x"], [chains])[0]
    pooled = chains[0] + chains[1]
    assert summary.mean == pytest.approx(sum(pooled) / len(pooled))
    assert summary.q05 == pytest.approx(quantile(pooled, 0.05))
    assert summary.q95 == pytest.approx(quantile(pooled, 0.95))
