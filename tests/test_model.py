"""The model: the gradient, the transformations, and whether the generator and the model agree.

The gradient check is the load-bearing test in this repository. Everything downstream -- the sampler, the
diagnostics, the ROAS numbers, the budget -- rests on the analytic gradient being the gradient of the stated
log-posterior. It is checked at the true parameters and at random points, coordinate by coordinate, against
central finite differences.

The tolerance is 1e-4 relative, and that is a considered number rather than a loose one. The log-posterior
is of order 1e3, so cancellation in ``f(x+h) - f(x-h)`` leaves roughly 1e-13 of absolute noise, and dividing
by ``2h`` with ``h`` around 1e-5 puts the floor on the finite-difference estimate near 1e-8. Coordinates
whose gradient is itself small -- the seasonal coefficients -- therefore cannot be verified to better than
about 1e-5 relative no matter how correct the analysis is. Every error this test is designed to catch (a
missing log-Jacobian, a dropped chain-rule factor, a wrong sign) shows up at order 0.1 or larger.
"""

import math
import random

import pytest

from mmm.data import DEFAULT_CHANNELS, simulate, simulate_flat_spend
from mmm.model import MMM, Dataset, Priors, gradient_error, numerical_gradient

TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def fitted():
    dataset, truth = simulate(weeks=78, seed=3)
    return MMM(dataset), truth


# -- the gradient -----------------------------------------------------------------------------


def test_the_gradient_matches_finite_differences_at_the_truth(fitted):
    model, truth = fitted
    error, where = gradient_error(model, model.pack(truth.parameters()))
    assert error < TOLERANCE, f"worst coordinate: {model.names[where]} ({error:.2e})"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_gradient_matches_finite_differences_at_random_points(fitted, seed):
    model, _ = fitted
    rng = random.Random(seed)
    theta = model.initial_point(rng, jitter=1.0)
    error, where = gradient_error(model, theta)
    assert error < TOLERANCE, f"worst coordinate: {model.names[where]} ({error:.2e})"


def test_the_gradient_is_checked_in_every_parameter_block(fitted):
    """Coordinate by coordinate, so one wrong block cannot hide behind the others."""
    model, truth = fitted
    theta = model.pack(truth.parameters())
    analytic = model.gradient(theta)
    numeric = numerical_gradient(model.log_posterior, theta)
    for index, name in enumerate(model.names):
        scale = max(abs(analytic[index]), abs(numeric[index]), 1e-8)
        assert abs(analytic[index] - numeric[index]) / scale < TOLERANCE, name


def test_the_value_returned_with_the_gradient_is_the_log_posterior(fitted):
    model, truth = fitted
    theta = model.pack(truth.parameters())
    value, _ = model.log_posterior_and_gradient(theta)
    assert value == pytest.approx(model.log_posterior(theta))
    assert value == pytest.approx(model.log_likelihood(theta) + model.log_prior(theta))


def test_the_gradient_has_one_entry_per_coordinate(fitted):
    model, truth = fitted
    assert len(model.gradient(model.pack(truth.parameters()))) == model.dim
    assert len(model.names) == model.dim


# -- transformations --------------------------------------------------------------------------


def test_pack_and_unpack_are_inverses(fitted):
    model, truth = fitted
    params = truth.parameters()
    recovered = model.unpack(model.pack(params))
    assert recovered.base == pytest.approx(params.base)
    assert recovered.sigma == pytest.approx(params.sigma)
    for index in range(model.n_channels):
        assert recovered.beta[index] == pytest.approx(params.beta[index])
        assert recovered.decay[index] == pytest.approx(params.decay[index])
        assert recovered.half[index] == pytest.approx(params.half[index])
        assert recovered.shape[index] == pytest.approx(params.shape[index])


def test_constrained_parameters_stay_inside_their_support(fitted):
    """Whatever the sampler proposes, decay is a probability and sigma is positive."""
    model, _ = fitted
    rng = random.Random(7)
    for _ in range(50):
        theta = [rng.gauss(0.0, 25.0) for _ in range(model.dim)]
        try:
            params = model.unpack(theta)
        except OverflowError:
            continue
        assert all(0.0 < value < 1.0 for value in params.decay)
        assert all(value > 0.0 for value in params.beta)
        assert all(value > 0.0 for value in params.half)
        assert all(value > 0.0 for value in params.shape)
        assert params.sigma > 0.0


def test_an_absurd_proposal_is_rejected_rather_than_raising(fitted):
    model, truth = fitted
    theta = model.pack(truth.parameters())
    theta[model.names.index("sigma")] = 5000.0  # exp(5000) overflows
    assert model.log_posterior(theta) == -math.inf


def test_the_wrong_number_of_parameters_is_refused(fitted):
    model, _ = fitted
    with pytest.raises(ValueError, match="expected"):
        model.unpack([0.0] * (model.dim - 1))


# -- the mean function ------------------------------------------------------------------------


def test_the_generator_and_the_model_agree(fitted):
    """Residuals at the true parameters must look like the noise that was added, and nothing more.

    If the model's mean function differed from the generator's by any term -- a shifted Fourier phase, an
    un-normalised adstock, a trend divided by T instead of T-1 -- this residual standard deviation would
    come out inflated. It is the cheapest possible check that two independently written pieces of
    arithmetic describe the same model.
    """
    model, truth = fitted
    residuals = model.residuals(truth.parameters())
    count = len(residuals)
    mean = sum(residuals) / count
    sd = math.sqrt(sum((value - mean) ** 2 for value in residuals) / (count - 1))
    assert sd == pytest.approx(truth.sigma, rel=0.25)
    assert abs(mean) < 0.5 * truth.sigma


def test_the_mean_is_the_baseline_plus_the_channels(fitted):
    model, truth = fitted
    params = truth.parameters()
    baseline = model.baseline(params)
    contributions = model.channel_contributions(params)
    total = model.mean(params)
    for t in range(model.weeks):
        assert total[t] == pytest.approx(baseline[t] + sum(series[t] for series in contributions))


def test_media_contributions_are_non_negative(fitted):
    model, truth = fitted
    for series in model.channel_contributions(truth.parameters()):
        assert all(value >= 0.0 for value in series)


def test_the_log_likelihood_is_the_normal_density(fitted):
    model, truth = fitted
    params = truth.parameters()
    residuals = model.residuals(params)
    expected = sum(
        -0.5 * (residual / params.sigma) ** 2
        - math.log(params.sigma)
        - 0.5 * math.log(2.0 * math.pi)
        for residual in residuals
    )
    assert model.log_likelihood(model.pack(params)) == pytest.approx(expected)


# -- data and priors --------------------------------------------------------------------------


def test_series_of_the_wrong_length_are_refused():
    with pytest.raises(ValueError, match="has 3 points"):
        Dataset(y=[1.0] * 10, spend={"tv": [1.0, 2.0, 3.0]})


def test_negative_spend_is_refused():
    with pytest.raises(ValueError, match="negative"):
        Dataset(y=[1.0] * 10, spend={"tv": [-1.0] + [0.0] * 9})


def test_too_few_observations_are_refused():
    with pytest.raises(ValueError, match="at least 8"):
        Dataset(y=[1.0] * 4, spend={"tv": [1.0] * 4})


def test_priors_are_scaled_to_the_data(fitted):
    model, _ = fitted
    priors = Priors.from_data(model.data)
    assert priors.base_mean == pytest.approx(sum(model.data.y) / model.weeks)
    assert priors.sigma_scale > 0.0
    # The half-saturation prior is centred on observed median spend: the data cannot identify a
    # saturation point far outside the range of spend it contains.
    assert math.exp(priors.half_log_mean) > 0.0


def test_the_layout_covers_every_parameter():
    dataset, _ = simulate(weeks=60, seed=1)
    for harmonics in (0, 1, 3):
        model = MMM(dataset, harmonics=harmonics)
        expected = 2 + 2 * harmonics + 4 * model.n_channels + model.n_controls + 1
        assert model.dim == expected
        assert len(model.names) == expected
        assert len(set(model.names)) == expected, "names must be unique to be readable"


def test_starting_points_are_dispersed_and_valid(fitted):
    """R-hat compares chains, so chains must start apart -- and every start must be samplable."""
    model, _ = fitted
    rng = random.Random(11)
    points = [model.initial_point(rng) for _ in range(6)]
    for point in points:
        assert math.isfinite(model.log_posterior(point))
    assert max(point[0] for point in points) > min(point[0] for point in points)


def test_the_flat_spend_regime_really_is_flat():
    dataset, _ = simulate_flat_spend(weeks=60, seed=2)
    for series in dataset.spend.values():
        assert (max(series) - min(series)) / (sum(series) / len(series)) < 0.35


def test_the_default_channels_span_the_interesting_cases():
    """A saturated concave channel, an S-curve, and one with headroom: they behave differently."""
    shapes = [channel.shape for channel in DEFAULT_CHANNELS]
    assert any(shape <= 1.0 for shape in shapes)
    assert any(shape > 1.5 for shape in shapes)
