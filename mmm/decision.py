"""From posterior to decision: contributions, ROAS, marginal ROAS, and budget allocation.

This is where MMMs are used and misused. Three distinctions decide whether the output is worth acting on.

**Average ROAS answers a question nobody asked.** Total contribution divided by total spend tells you what
the channel did on average, over a range of spend you have already committed. The decision -- move the next
pound -- depends on the *marginal* return at the spend level you are actually at. On a saturated channel the
two differ by a factor of several, and the ranking by average ROAS can be the reverse of the ranking by
marginal ROAS. Both are computed here, side by side, because the gap is the point.

**Carryover leaks past the window.** Spend in the last week of the horizon is still selling after the data
ends. Attributing only the in-window response systematically understates recent, and therefore usually
growing, channels. ``incremental_contribution`` extends the horizon with zero-spend weeks so the tail is
counted, and reports what fraction of the credit falls outside the window.

**Optimisation must be done at a defined operating point.** Optimising a per-week, per-channel spend matrix
over two years is a large non-convex problem whose answer nobody can execute. What a planner can execute is
a steady weekly budget, and there the normalised adstock earns its keep: for constant spend ``s`` the
adstocked series converges to exactly ``s``, so the steady-state weekly response is ``beta * Hill(s)`` and
the allocation problem becomes

    maximise   sum_c beta_c * Hill(s_c; kappa_c, alpha_c)      subject to   sum_c s_c = B,   s_c >= 0.

For ``alpha <= 1`` every channel is concave, the problem is concave, and the optimum equalises marginal
returns across funded channels -- water-filling by bisection on the shadow price. For ``alpha > 1`` the
S-curve toe is convex and the problem is **not** concave: equalising marginal returns can land on a local
optimum, and the true answer may fund a channel at scale or not at all. That case is handled by enumerating
which channels are on, solving the concave problem within each subset, and keeping the best; with a handful
of channels the enumeration is free. ``tests/test_decision.py`` checks the result against a brute-force
simplex grid search, because an optimiser that quietly returns a local optimum is worse than none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from .model import MMM, Parameters
from .transforms import geometric_adstock, hill, hill_partials, inflection_point


# ---------------------------------------------------------------------------------------------
# contributions and ROAS
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelResult:
    name: str
    spend: float
    contribution: float
    in_window_contribution: float
    roas: float
    marginal_roas: float
    saturation: float  # how far up its own curve the channel sits at mean active spend

    @property
    def tail_share(self) -> float:
        """How much of the credited response arrives after the data window ends."""
        if self.contribution <= 0.0:
            return 0.0
        return (self.contribution - self.in_window_contribution) / self.contribution

    def row(self) -> str:
        return (
            f"{self.name:<12}{self.spend:>14,.0f}{self.contribution:>16,.0f}"
            f"{self.roas:>10.2f}{self.marginal_roas:>12.2f}{self.saturation:>13.0%}"
            f"{self.tail_share:>10.0%}"
        )


def incremental_contribution(
    model: MMM, params: Parameters, channel: int, tail: int = 26
) -> tuple[float, float]:
    """Total and in-window incremental response for one channel.

    Incremental means the difference between the fitted response and the counterfactual with that
    channel's spend set to zero. Because the model is additive in channels, that difference is the
    channel's own term -- but only if the carryover tail is included, which is what ``tail`` extra
    zero-spend weeks provide.
    """
    spend = list(model.data.spend[model.channels[channel]])
    weeks = len(spend)
    padded = spend + [0.0] * max(tail, 0)
    adstocked = geometric_adstock(padded, params.decay[channel], model.normalize_adstock)
    responses = [
        params.beta[channel] * hill(value, params.half[channel], params.shape[channel])
        for value in adstocked
    ]
    return sum(responses), sum(responses[:weeks])


def marginal_roas(
    model: MMM, params: Parameters, channel: int, uplift: float = 0.05, tail: int = 26
) -> float:
    """Response to a proportional spend increase, divided by its cost.

    A finite difference rather than a derivative, on purpose: this is the question a planner asks -- "if I
    add five percent to this channel, what comes back?" -- and at 5% the answer already differs from the
    derivative when the channel sits near its inflection point.
    """
    if uplift <= 0.0:
        raise ValueError("uplift must be positive")
    name = model.channels[channel]
    spend = list(model.data.spend[name])
    total_spend = sum(spend)
    if total_spend <= 0.0:
        return 0.0

    base_total, _ = incremental_contribution(model, params, channel, tail)
    scaled = [value * (1.0 + uplift) for value in spend]
    padded = scaled + [0.0] * max(tail, 0)
    adstocked = geometric_adstock(padded, params.decay[channel], model.normalize_adstock)
    lifted = sum(
        params.beta[channel] * hill(value, params.half[channel], params.shape[channel])
        for value in adstocked
    )
    return (lifted - base_total) / (uplift * total_spend)


def channel_results(
    model: MMM, params: Parameters, uplift: float = 0.05, tail: int = 26
) -> list[ChannelResult]:
    results: list[ChannelResult] = []
    for index, name in enumerate(model.channels):
        spend = model.data.spend[name]
        total_spend = sum(spend)
        total, in_window = incremental_contribution(model, params, index, tail)
        active = [value for value in spend if value > 0.0]
        mean_spend = sum(active) / len(active) if active else 0.0
        results.append(
            ChannelResult(
                name=name,
                spend=total_spend,
                contribution=total,
                in_window_contribution=in_window,
                roas=total / total_spend if total_spend > 0 else 0.0,
                marginal_roas=marginal_roas(model, params, index, uplift, tail),
                saturation=hill(mean_spend, params.half[index], params.shape[index]),
            )
        )
    return results


def results_table(results: list[ChannelResult]) -> str:
    header = (
        f"{'channel':<12}{'spend':>14}{'contribution':>16}{'ROAS':>10}"
        f"{'mROAS':>12}{'saturation':>13}{'tail':>10}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(result.row() for result in results)
    return "\n".join(lines)


def decomposition(model: MMM, params: Parameters) -> dict[str, float]:
    """Sales split into baseline and media, as shares of observed sales.

    The baseline is not "what we would sell with no marketing ever" -- it absorbs brand equity built by
    years of past spend that this window cannot see. It is "what the model cannot attribute to media
    within this window", and calling it organic demand is the most common overclaim in the field.
    """
    total = sum(model.data.y)
    if total <= 0.0:
        raise ValueError("total sales must be positive to take shares")
    contributions = model.channel_contributions(params)
    shares = {"baseline": sum(model.baseline(params)) / total}
    for index, name in enumerate(model.channels):
        shares[name] = sum(contributions[index]) / total
    shares["unexplained"] = 1.0 - sum(shares.values())
    return shares


# ---------------------------------------------------------------------------------------------
# budget allocation at steady state
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Allocation:
    """A recommended weekly budget split, with the evidence that it is actually optimal."""

    spend: dict[str, float]
    response: float
    marginal: dict[str, float]
    budget: float
    active: tuple[str, ...]
    concave: bool

    def summary(self) -> str:
        parts = ", ".join(f"{name} {value:,.0f}" for name, value in sorted(self.spend.items()))
        marginals = ", ".join(
            f"{name} {value:.2f}"
            for name, value in sorted(self.marginal.items())
            if self.spend[name] > 0
        )
        return (
            f"budget {self.budget:,.0f}/week -> response {self.response:,.0f}/week\n"
            f"  split:     {parts}\n"
            f"  marginal:  {marginals}"
            + ("" if self.concave else "\n  (S-curve present: funded set chosen by enumeration)")
        )

    def shares(self) -> dict[str, float]:
        return {name: value / self.budget for name, value in self.spend.items()}


def _steady_state_response(params: Parameters, channel: int, spend: float) -> float:
    """Weekly response to a constant weekly spend.

    Normalised geometric adstock of a constant series converges to the constant itself: the recursion
    ``A = s + lambda A`` has fixed point ``s/(1-lambda)``, and the ``(1-lambda)`` normaliser cancels it.
    """
    return params.beta[channel] * hill(spend, params.half[channel], params.shape[channel])


def _steady_state_marginal(params: Parameters, channel: int, spend: float) -> float:
    _, d_u, _, _ = hill_partials(spend, params.half[channel], params.shape[channel])
    return params.beta[channel] * d_u


def _spend_for_marginal(params: Parameters, channel: int, price: float, ceiling: float) -> float:
    """Invert the marginal-return curve on its decreasing branch: find ``s`` with ``beta * ds/du = price``.

    On a concave channel that branch is the whole positive axis. On an S-curve it begins at the inflection
    point; below that point the marginal return is *rising*, so a root there is a minimum rather than a
    maximum, and taking it is the classic optimiser error on S-curves.
    """
    if price <= 0.0:
        return ceiling
    lower = inflection_point(params.half[channel], params.shape[channel]) or 1e-9
    if _steady_state_marginal(params, channel, lower) < price:
        return 0.0  # even at its most productive point the channel cannot pay this price
    high = max(ceiling, lower * 2.0)
    if _steady_state_marginal(params, channel, high) > price:
        return high  # still worth more than the price at the ceiling
    for _ in range(200):
        middle = 0.5 * (lower + high)
        if _steady_state_marginal(params, channel, middle) > price:
            lower = middle
        else:
            high = middle
        if high - lower < 1e-9 * max(1.0, high):
            break
    return 0.5 * (lower + high)


def _total_at_price(
    params: Parameters, active: tuple[int, ...], price: float, budget: float
) -> dict[int, float]:
    return {channel: _spend_for_marginal(params, channel, price, budget) for channel in active}


def _water_fill(params: Parameters, active: tuple[int, ...], budget: float) -> dict[int, float]:
    """Equalise marginal returns across the active channels, by bisection on the shadow price.

    The upper bracket is found by doubling rather than guessed from a formula: on an S-curve the largest
    marginal return sits at the inflection point, not at zero spend, so any closed-form guess based on
    small-spend behaviour brackets the wrong interval and the bisection converges to a price that
    overspends.
    """
    if not active:
        return {}
    low_price = 0.0
    high_price = 1.0
    for _ in range(200):
        if sum(_total_at_price(params, active, high_price, budget).values()) <= budget:
            break
        high_price *= 2.0
    else:
        high_price = float("inf")

    if math.isinf(high_price):
        # No finite price rations this budget (numerically degenerate parameters); split it evenly.
        return {channel: budget / len(active) for channel in active}

    for _ in range(200):
        price = 0.5 * (low_price + high_price)
        total = sum(_total_at_price(params, active, price, budget).values())
        if total > budget:
            low_price = price  # too much spend demanded: the shadow price must rise
        else:
            high_price = price
        if abs(total - budget) < 1e-9 * max(1.0, budget):
            break

    allocation = _total_at_price(params, active, 0.5 * (low_price + high_price), budget)
    total = sum(allocation.values())
    if total > 0.0:
        # Exhaust the budget exactly: the bisection lands within a rounding error, and a plan that does
        # not add up to the budget invites a spreadsheet to "fix" it.
        factor = budget / total
        allocation = {channel: value * factor for channel, value in allocation.items()}
    return allocation


def allocate(model: MMM, params: Parameters, budget: float) -> Allocation:
    """Optimal steady-state weekly split of ``budget`` across channels.

    Concave everywhere: one water-filling pass. With any S-curve: every subset of channels is tried, the
    concave problem is solved within each, and the best total response wins. That enumeration is what makes
    the answer global rather than merely stationary.
    """
    if budget <= 0.0:
        raise ValueError("budget must be positive")
    count = model.n_channels
    if count == 0:
        raise ValueError("no channels to allocate across")

    concave = all(shape <= 1.0 for shape in params.shape)
    if concave:
        subsets: list[tuple[int, ...]] = [tuple(range(count))]
    else:
        subsets = []
        for size in range(1, count + 1):
            subsets.extend(combinations(range(count), size))

    best_spend: dict[int, float] = {}
    best_response = -math.inf
    best_active: tuple[int, ...] = ()
    for active in subsets:
        allocation = _water_fill(params, active, budget)
        if not allocation:
            continue
        response = sum(
            _steady_state_response(params, channel, spend)
            for channel, spend in allocation.items()
        )
        if response > best_response:
            best_response = response
            best_spend = allocation
            best_active = active

    spend = {name: 0.0 for name in model.channels}
    for channel, value in best_spend.items():
        spend[model.channels[channel]] = value
    marginal = {
        model.channels[channel]: _steady_state_marginal(
            params, channel, spend[model.channels[channel]]
        )
        for channel in range(count)
    }
    return Allocation(
        spend=spend,
        response=best_response,
        marginal=marginal,
        budget=budget,
        active=tuple(model.channels[channel] for channel in best_active),
        concave=concave,
    )


def grid_search_allocation(
    model: MMM, params: Parameters, budget: float, steps: int = 40
) -> tuple[dict[str, float], float]:
    """Brute force over a simplex grid. Slow, obviously correct, and used to check ``allocate``."""
    count = model.n_channels
    if count > 4:
        raise ValueError("the grid search is only intended for small channel counts")
    unit = budget / steps
    best_response = -math.inf
    best: tuple[int, ...] = ()

    def walk(prefix: tuple[int, ...], remaining: int, depth: int) -> None:
        nonlocal best_response, best
        if depth == count - 1:
            candidate = prefix + (remaining,)
            response = sum(
                _steady_state_response(params, channel, units * unit)
                for channel, units in enumerate(candidate)
            )
            if response > best_response:
                best_response = response
                best = candidate
            return
        for units in range(remaining + 1):
            walk(prefix + (units,), remaining - units, depth + 1)

    walk((), steps, 0)
    spend = {model.channels[channel]: units * unit for channel, units in enumerate(best)}
    return spend, best_response


def posterior_allocation(
    model: MMM, posterior, budget: float, draws: int = 200
) -> dict[str, tuple[float, float, float]]:
    """The recommended split under each posterior draw, reported as (5%, 50%, 95%) shares.

    A single split computed from posterior means is a point estimate of a decision, and it hides the only
    thing a planner needs to know: whether the recommendation is robust. If the credible interval for a
    channel's share runs from 5% to 40%, the honest recommendation is "we cannot tell yet, and here is the
    experiment that would settle it".
    """
    from .diagnostics import quantile

    samples = posterior.draws()
    if not samples:
        raise ValueError("the posterior contains no draws")
    stride = max(len(samples) // max(draws, 1), 1)
    selected = samples[::stride][:draws]

    shares: dict[str, list[float]] = {name: [] for name in model.channels}
    for theta in selected:
        params = model.unpack(theta)
        allocation = allocate(model, params, budget)
        for name, value in allocation.spend.items():
            shares[name].append(value / budget)
    return {
        name: (quantile(values, 0.05), quantile(values, 0.50), quantile(values, 0.95))
        for name, values in shares.items()
    }
