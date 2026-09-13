"""Transformations: hand-computed values, and every derivative against finite differences.

The derivative tests are the point of this file. An analytic gradient that is wrong in one term does not
raise anything -- it produces a sampler that explores a slightly different distribution than the one
written down, and no amount of staring at trace plots will reveal it.
"""

import math

import pytest

from mmm.transforms import (
    carryover_summary,
    delayed_adstock,
    geometric_adstock,
    geometric_adstock_gradient,
    hill,
    hill_partials,
    inflection_point,
    is_convex_region,
    logistic_saturation,
    marginal_response,
)


def central_difference(function, value, step=1e-6):
    scale = max(abs(value), 1.0)
    h = step * scale
    return (function(value + h) - function(value - h)) / (2.0 * h)


# -- carryover --------------------------------------------------------------------------------


def test_geometric_adstock_matches_the_recursion_by_hand():
    assert geometric_adstock([100.0, 0.0, 0.0], 0.5, normalize=False) == [100.0, 50.0, 25.0]
    assert geometric_adstock([100.0, 0.0, 0.0], 0.5, normalize=True) == [50.0, 25.0, 12.5]


def test_zero_decay_is_no_carryover_at_all():
    spend = [10.0, 0.0, 5.0]
    assert geometric_adstock(spend, 0.0, normalize=False) == spend
    assert geometric_adstock(spend, 0.0, normalize=True) == spend


def test_normalisation_conserves_total_volume():
    """The reason normalisation is the default: total adstocked spend equals total spend."""
    spend = [1000.0] + [0.0] * 400
    for decay in (0.1, 0.5, 0.9):
        assert sum(geometric_adstock(spend, decay, normalize=True)) == pytest.approx(1000.0, rel=1e-6)
        # Without it, volume inflates by exactly 1/(1 - decay), which is what forces beta to shrink.
        assert sum(geometric_adstock(spend, decay, normalize=False)) == pytest.approx(
            1000.0 / (1.0 - decay), rel=1e-6
        )


def test_carryover_is_a_convolution_so_it_is_linear():
    a = [10.0, 0.0, 3.0, 7.0]
    b = [2.0, 5.0, 0.0, 1.0]
    combined = geometric_adstock([x + y for x, y in zip(a, b)], 0.4)
    separate = [
        x + y for x, y in zip(geometric_adstock(a, 0.4), geometric_adstock(b, 0.4))
    ]
    for left, right in zip(combined, separate):
        assert left == pytest.approx(right)


@pytest.mark.parametrize("decay", [0.05, 0.3, 0.6, 0.85, 0.95])
@pytest.mark.parametrize("normalize", [True, False])
def test_the_adstock_derivative_matches_finite_differences(decay, normalize):
    spend = [120.0, 0.0, 340.0, 80.0, 0.0, 0.0, 500.0, 45.0]
    values, derivatives = geometric_adstock_gradient(spend, decay, normalize)
    assert values == pytest.approx(geometric_adstock(spend, decay, normalize))
    for index in range(len(spend)):
        numeric = central_difference(
            lambda d: geometric_adstock(spend, d, normalize)[index], decay, step=1e-6
        )
        assert derivatives[index] == pytest.approx(numeric, rel=1e-5, abs=1e-8)


def test_a_decay_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="decay"):
        geometric_adstock([1.0], 1.0)
    with pytest.raises(ValueError, match="decay"):
        geometric_adstock_gradient([1.0], -0.1)


def test_carryover_reads_in_weeks():
    summary = carryover_summary(0.5)
    assert summary.half_life == pytest.approx(1.0)
    assert summary.weeks_to_90_percent == pytest.approx(math.log(0.1) / math.log(0.5))
    assert summary.mean_lag == pytest.approx(1.0)
    assert "half-life" in summary.summary()


def test_longer_carryover_means_a_longer_half_life():
    assert carryover_summary(0.8).half_life > carryover_summary(0.4).half_life
    assert carryover_summary(0.0).half_life == 0.0


def test_delayed_adstock_peaks_after_the_exposure():
    spike = [1.0] + [0.0] * 10
    delayed = delayed_adstock(spike, decay=0.6, peak=2.0, length=8)
    assert delayed.index(max(delayed)) == 2, "a kernel peaking at lag 2 must peak at lag 2"
    assert sum(delayed) == pytest.approx(1.0), "the weights are normalised, so volume is conserved"
    immediate = delayed_adstock(spike, decay=0.6, peak=0.0, length=8)
    assert immediate.index(max(immediate)) == 0


# -- saturation -------------------------------------------------------------------------------


def test_the_half_saturation_point_is_exactly_half():
    for half in (1.0, 250.0, 98000.0):
        for shape in (0.5, 1.0, 2.5):
            assert hill(half, half, shape) == pytest.approx(0.5)


def test_hill_is_zero_at_zero_and_approaches_one():
    assert hill(0.0, 100.0, 1.5) == 0.0
    assert hill(1e9, 100.0, 1.5) == pytest.approx(1.0, abs=1e-6)


def test_hill_is_increasing():
    previous = -1.0
    for spend in (0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        current = hill(spend, 500.0, 1.3)
        assert current > previous
        previous = current


def test_hill_values_by_hand():
    assert hill(20.0, 10.0, 1.0) == pytest.approx(20.0 / 30.0)
    assert hill(10.0, 20.0, 1.0) == pytest.approx(10.0 / 30.0)
    assert hill(20.0, 10.0, 2.0) == pytest.approx(400.0 / 500.0)


def test_the_partials_at_the_half_saturation_point_by_hand():
    """At ``u = kappa = 10`` with ``alpha = 1``: ds/du = 0.025, ds/dkappa = -0.025, ds/dalpha = 0."""
    value, d_u, d_half, d_shape = hill_partials(10.0, 10.0, 1.0)
    assert value == pytest.approx(0.5)
    assert d_u == pytest.approx(0.025)
    assert d_half == pytest.approx(-0.025)
    assert d_shape == pytest.approx(0.0)


@pytest.mark.parametrize("spend", [1.0, 50.0, 300.0, 1200.0, 9000.0])
@pytest.mark.parametrize("half", [100.0, 700.0])
@pytest.mark.parametrize("shape", [0.6, 1.0, 1.7, 3.0])
def test_every_hill_partial_matches_finite_differences(spend, half, shape):
    _, d_u, d_half, d_shape = hill_partials(spend, half, shape)
    assert d_u == pytest.approx(
        central_difference(lambda u: hill(u, half, shape), spend), rel=1e-4, abs=1e-12
    )
    assert d_half == pytest.approx(
        central_difference(lambda k: hill(spend, k, shape), half), rel=1e-4, abs=1e-12
    )
    assert d_shape == pytest.approx(
        central_difference(lambda a: hill(spend, half, a), shape), rel=1e-4, abs=1e-12
    )


def test_zero_spend_has_zero_partials_by_convention():
    assert hill_partials(0.0, 100.0, 0.8) == (0.0, 0.0, 0.0, 0.0)


def test_invalid_saturation_parameters_are_refused():
    with pytest.raises(ValueError, match="half-saturation"):
        hill(10.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="shape"):
        hill_partials(10.0, 5.0, 0.0)


def test_the_inflection_point_exists_only_for_s_curves():
    assert inflection_point(100.0, 0.9) is None
    assert inflection_point(100.0, 1.0) is None
    assert inflection_point(10.0, 2.0) == pytest.approx(10.0 * math.sqrt(1.0 / 3.0))


def test_the_inflection_point_is_where_the_marginal_return_peaks():
    """Below it the curve accelerates; above it, diminishing returns. This is what breaks concavity."""
    half, shape = 500.0, 2.4
    peak = inflection_point(half, shape)
    assert peak is not None
    at_peak = hill_partials(peak, half, shape)[1]
    for offset in (0.5, 0.8, 1.25, 2.0):
        assert hill_partials(peak * offset, half, shape)[1] <= at_peak + 1e-12
    assert is_convex_region(peak * 0.5, half, shape)
    assert not is_convex_region(peak * 1.5, half, shape)


def test_a_concave_curve_has_no_convex_region():
    assert not is_convex_region(1.0, 100.0, 1.0)
    assert not is_convex_region(1e-6, 100.0, 0.5)


def test_marginal_response_scales_with_beta():
    single = marginal_response(400.0, 500.0, 1.2, beta=1.0)
    assert marginal_response(400.0, 500.0, 1.2, beta=7.0) == pytest.approx(7.0 * single)


def test_logistic_saturation_partials_match_finite_differences():
    for spend, rate in ((0.0, 0.5), (2.0, 0.5), (10.0, 0.1)):
        value, d_u, d_rate = logistic_saturation(spend, rate)
        assert value == pytest.approx(1.0 - math.exp(-rate * spend))
        assert d_u == pytest.approx(
            central_difference(lambda u: logistic_saturation(u, rate)[0], spend), rel=1e-5, abs=1e-9
        )
        assert d_rate == pytest.approx(
            central_difference(lambda r: logistic_saturation(spend, r)[0], rate), rel=1e-5, abs=1e-9
        )


def test_logistic_saturation_is_always_concave():
    """No inflection point anywhere: the conservative alternative to Hill."""
    previous = math.inf
    for spend in (0.1, 1.0, 5.0, 20.0):
        slope = logistic_saturation(spend, 0.3)[1]
        assert slope < previous
        previous = slope
