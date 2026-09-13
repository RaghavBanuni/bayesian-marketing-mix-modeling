"""Decisions: the ROAS arithmetic, and whether the optimiser really finds the optimum.

The allocation test is the one that matters. An optimiser on a non-concave objective can return a
stationary point, report success, and be wrong by a wide margin -- so the analytic result is compared
against brute force over a simplex grid, on the S-curve parameters where concavity actually fails.
"""

import math
from dataclasses import replace

import pytest

from mmm.data import simulate
from mmm.decision import (
    allocate,
    channel_results,
    decomposition,
    grid_search_allocation,
    incremental_contribution,
    marginal_roas,
    posterior_allocation,
    results_table,
)
from mmm.model import MMM
from mmm.transforms import inflection_point


@pytest.fixture(scope="module")
def fitted():
    dataset, truth = simulate(weeks=104, seed=17)
    return MMM(dataset), truth.parameters()


class FakePosterior:
    """A posterior with hand-made draws, so the plumbing can be tested without sampling."""

    def __init__(self, thetas):
        self._draws = [list(theta) for theta in thetas]
        self.chains = [type("Chain", (), {"draws": self._draws})()]
        self.names: list[str] = []

    def draws(self):
        return self._draws


# -- contributions and ROAS -------------------------------------------------------------------


def test_the_decomposition_accounts_for_all_of_sales(fitted):
    model, params = fitted
    shares = decomposition(model, params)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert 0.0 < shares["baseline"] < 1.0
    assert all(shares[name] > 0.0 for name in model.channels)


def test_carryover_pushes_response_past_the_end_of_the_window(fitted):
    model, params = fitted
    results = channel_results(model, params)
    for result in results:
        assert result.contribution >= result.in_window_contribution
        assert 0.0 <= result.tail_share < 0.5
    assert any(result.tail_share > 0.0 for result in results), "carryover must leak somewhere"


def test_without_carryover_nothing_leaks(fitted):
    """Zero decay means this week's spend sells this week only, so the tail must be empty."""
    model, params = fitted
    instant = replace(params, decay=[0.0] * model.n_channels)
    for index in range(model.n_channels):
        total, in_window = incremental_contribution(model, instant, index, tail=26)
        assert total == pytest.approx(in_window)


def test_a_channel_with_no_spend_earns_no_credit(fitted):
    model, params = fitted
    dataset = model.data
    silent = MMM(
        type(dataset)(
            y=dataset.y,
            spend={name: [0.0] * dataset.weeks for name in dataset.spend},
            controls=dataset.controls,
            period=dataset.period,
        )
    )
    for index in range(silent.n_channels):
        total, _ = incremental_contribution(silent, params, index)
        assert total == pytest.approx(0.0)
        assert marginal_roas(silent, params, index) == 0.0


def test_marginal_roas_is_below_average_roas_on_a_concave_channel(fitted):
    """Diminishing returns, stated as an inequality: ``f(a)/a >= f'(a)`` for concave ``f`` through zero.

    This is why ranking channels by average ROAS misallocates budget -- the average is a report on money
    already spent, and the decision depends on the slope where you are standing.
    """
    model, params = fitted
    concave = replace(params, shape=[0.8] * model.n_channels)
    for result in channel_results(model, concave):
        assert result.marginal_roas < result.roas, result.name


def test_marginal_roas_is_above_average_roas_inside_the_s_curve_toe(fitted):
    """Below the inflection point the curve is convex and the inequality reverses.

    A planner using average ROAS here would under-fund a channel that is about to start working.
    """
    model, params = fitted
    ceiling = max(max(series) for series in model.data.spend.values())
    toe = replace(
        params,
        shape=[3.0] * model.n_channels,
        half=[ceiling * 20.0] * model.n_channels,
    )
    for index in range(model.n_channels):
        peak = inflection_point(toe.half[index], toe.shape[index])
        assert peak is not None and peak > ceiling, "the spend range must sit in the convex toe"
    for result in channel_results(model, toe):
        assert result.marginal_roas > result.roas, result.name


def test_the_results_table_names_every_channel(fitted):
    model, params = fitted
    table = results_table(channel_results(model, params))
    for name in model.channels:
        assert name in table
    assert "mROAS" in table


def test_an_uplift_must_be_positive(fitted):
    model, params = fitted
    with pytest.raises(ValueError, match="uplift"):
        marginal_roas(model, params, 0, uplift=0.0)


# -- allocation -------------------------------------------------------------------------------


def test_the_allocation_spends_the_whole_budget(fitted):
    model, params = fitted
    budget = 40000.0
    allocation = allocate(model, params, budget)
    assert sum(allocation.spend.values()) == pytest.approx(budget, rel=1e-6)
    assert all(value >= 0.0 for value in allocation.spend.values())
    assert sum(allocation.shares().values()) == pytest.approx(1.0)


def test_concave_channels_end_with_equal_marginal_returns(fitted):
    """The KKT condition for an interior optimum: every funded channel returns the same on the next pound."""
    model, params = fitted
    concave = replace(params, shape=[0.8] * model.n_channels)
    allocation = allocate(model, concave, 50000.0)
    assert allocation.concave
    funded = [allocation.marginal[name] for name in model.channels if allocation.spend[name] > 1e-6]
    assert len(funded) == model.n_channels, "a concave channel always deserves some budget"
    assert max(funded) - min(funded) < 0.02 * max(funded)


@pytest.mark.parametrize("budget", [20000.0, 60000.0, 150000.0])
def test_the_optimiser_matches_brute_force(fitted, budget):
    """Against a simplex grid on the default S-curve parameters, where the problem is not concave."""
    model, params = fitted
    allocation = allocate(model, params, budget)
    _, grid_response = grid_search_allocation(model, params, budget, steps=30)
    assert allocation.response >= grid_response - 1e-6, (
        f"the optimiser is below a coarse grid at budget {budget:,.0f}: "
        f"{allocation.response:,.2f} < {grid_response:,.2f}"
    )
    assert allocation.response <= grid_response * 1.10


def test_an_s_curve_channel_too_small_to_fund_gets_nothing(fitted):
    """Concavity would fund every channel a little. An S-curve channel is worth all or nothing."""
    model, params = fitted
    hopeless = replace(
        params,
        beta=[params.beta[0]] + [1.0] * (model.n_channels - 1),
        shape=[1.0] + [4.0] * (model.n_channels - 1),
        half=[params.half[0]] + [1e7] * (model.n_channels - 1),
    )
    allocation = allocate(model, hopeless, 30000.0)
    assert allocation.spend[model.channels[0]] == pytest.approx(30000.0, rel=1e-6)
    for name in model.channels[1:]:
        assert allocation.spend[name] == pytest.approx(0.0, abs=1e-6)


def test_more_budget_never_buys_less_response(fitted):
    model, params = fitted
    previous = -math.inf
    for budget in (10000.0, 30000.0, 60000.0, 120000.0):
        response = allocate(model, params, budget).response
        assert response >= previous
        previous = response


def test_a_non_positive_budget_is_refused(fitted):
    model, params = fitted
    with pytest.raises(ValueError, match="budget"):
        allocate(model, params, 0.0)


def test_the_allocation_summary_is_readable(fitted):
    model, params = fitted
    text = allocate(model, params, 45000.0).summary()
    assert "budget" in text and "marginal" in text


def test_posterior_allocation_returns_intervals_per_channel(fitted):
    """The deliverable is a distribution over splits, not a single split."""
    model, params = fitted
    theta = model.pack(params)
    jittered = []
    for index in range(6):
        shifted = list(theta)
        shifted[model.names.index(f"beta[{model.channels[0]}]")] += 0.1 * (index - 3)
        jittered.append(shifted)
    intervals = posterior_allocation(model, FakePosterior(jittered), 40000.0, draws=6)
    assert set(intervals) == set(model.channels)
    for low, median, high in intervals.values():
        assert 0.0 <= low <= median <= high <= 1.0
    assert sum(median for _, median, _ in intervals.values()) == pytest.approx(1.0, abs=0.2)
