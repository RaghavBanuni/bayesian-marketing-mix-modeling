"""No-U-Turn Sampler, written out in full: leapfrog, slice-based tree doubling, dual averaging.

Why not Metropolis? An MMM posterior is a dozen correlated parameters in a curved geometry -- beta trades
off against kappa, kappa against alpha, lambda against trend. A random-walk proposal in twelve dimensions
either takes microscopic steps or is rejected; effective sample sizes come out in single digits and the
chains look converged because they are barely moving. Hamiltonian dynamics uses the gradient to travel
along the ridge instead of across it, and NUTS removes the one parameter nobody can tune by hand -- the
trajectory length -- by doubling the trajectory until it turns back on itself (Hoffman & Gelman, 2014).

What is implemented here:

* **Leapfrog integration** with a diagonal mass matrix. Momentum ``r ~ Normal(0, M)`` with ``M = diag(1/v)``,
  kinetic energy ``K = 0.5 * sum(v_i r_i^2)``. ``v`` is the estimated posterior variance per coordinate, which
  is what lets one integrator step suit both ``base`` (revenue units, in the thousands) and ``log alpha``
  (dimensionless, order one). Skip this and the step size is dictated by the tightest coordinate.
* **Slice-sampled tree doubling** with the no-U-turn stopping rule, checked on every subtree so the
  detailed-balance argument holds, plus a hard ``max_treedepth`` cap.
* **Dual averaging** on the step size during warmup, targeting a nominal acceptance statistic (Nesterov's
  scheme as adapted in the NUTS paper, with ``gamma = 0.05``, ``t0 = 10``, ``kappa = 0.75``).
* **Divergence counting.** A divergence means the integrator fell off a cliff in the log density, and the
  region it was trying to enter is therefore under-sampled. Divergences are reported, never swallowed:
  a fit with divergences is not a fit.

Three-phase warmup: a short step-size search, a variance-estimation window that yields the mass matrix, and
a final step-size adaptation at the new metric. This is a simplification of Stan's expanding-window schedule;
it is named as such rather than presented as equivalent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

MAX_ENERGY_ERROR = 1000.0  # beyond this the trajectory has diverged


@dataclass(frozen=True)
class NUTSConfig:
    draws: int = 500
    warmup: int = 500
    chains: int = 2
    target_accept: float = 0.8
    max_treedepth: int = 10
    seed: int = 0
    jitter: float = 0.5

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError("draws must be positive")
        if self.warmup < 50:
            raise ValueError("warmup below 50 draws cannot adapt anything")
        if self.chains < 1:
            raise ValueError("chains must be positive")
        if not 0.0 < self.target_accept < 1.0:
            raise ValueError("target_accept must lie in (0, 1)")
        if self.max_treedepth < 1:
            raise ValueError("max_treedepth must be positive")


@dataclass
class Chain:
    """One chain: post-warmup draws plus the sampler's own account of how it behaved."""

    draws: list[list[float]] = field(default_factory=list)
    log_posterior: list[float] = field(default_factory=list)
    tree_depth: list[int] = field(default_factory=list)
    accept_stat: list[float] = field(default_factory=list)
    divergences: int = 0
    step_size: float = 0.0
    inverse_metric: list[float] = field(default_factory=list)
    max_depth_hits: int = 0

    @property
    def length(self) -> int:
        return len(self.draws)

    def column(self, index: int) -> list[float]:
        return [draw[index] for draw in self.draws]

    def summary(self) -> str:
        depth = sum(self.tree_depth) / max(len(self.tree_depth), 1)
        accept = sum(self.accept_stat) / max(len(self.accept_stat), 1)
        return (
            f"{self.length} draws, step size {self.step_size:.4f}, mean tree depth {depth:.1f}, "
            f"mean accept {accept:.2f}, {self.divergences} divergences"
        )


@dataclass
class Posterior:
    """All chains, with the names of the coordinates so results can be read without a decoder ring."""

    chains: list[Chain]
    names: list[str]

    @property
    def divergences(self) -> int:
        return sum(chain.divergences for chain in self.chains)

    @property
    def total_draws(self) -> int:
        return sum(chain.length for chain in self.chains)

    def column(self, index: int) -> list[float]:
        """One coordinate, all chains concatenated."""
        values: list[float] = []
        for chain in self.chains:
            values.extend(chain.column(index))
        return values

    def columns_by_chain(self, index: int) -> list[list[float]]:
        return [chain.column(index) for chain in self.chains]

    def draws(self) -> list[list[float]]:
        pooled: list[list[float]] = []
        for chain in self.chains:
            pooled.extend(chain.draws)
        return pooled

    def index_of(self, name: str) -> int:
        return self.names.index(name)


class _Target:
    """The log-posterior with the arithmetic accidents caught.

    A sampler must be able to propose an absurd point and be told it is absurd. An exception thrown from
    deep inside a leapfrog step would abort the run; ``-inf`` merely rejects the proposal.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.evaluations = 0

    def __call__(self, theta: list[float]) -> tuple[float, list[float]]:
        self.evaluations += 1
        try:
            value, grad = self.model.log_posterior_and_gradient(theta)
        except (ValueError, OverflowError, ZeroDivisionError):
            return -math.inf, [0.0] * len(theta)
        if not math.isfinite(value):
            return -math.inf, [0.0] * len(theta)
        for element in grad:
            if not math.isfinite(element):
                return -math.inf, [0.0] * len(theta)
        return value, grad


def _kinetic(momentum: list[float], inverse_metric: list[float]) -> float:
    return 0.5 * sum(v * r * r for v, r in zip(inverse_metric, momentum))


def _draw_momentum(rng: random.Random, inverse_metric: list[float]) -> list[float]:
    # r ~ Normal(0, M) with M = diag(1/v), so the standard deviation is 1/sqrt(v).
    return [rng.gauss(0.0, 1.0 / math.sqrt(v)) for v in inverse_metric]


def _leapfrog(
    target: _Target,
    theta: list[float],
    momentum: list[float],
    grad: list[float],
    step: float,
    inverse_metric: list[float],
) -> tuple[list[float], list[float], float, list[float]]:
    """One step of the standard half-kick / drift / half-kick integrator."""
    half = [r + 0.5 * step * g for r, g in zip(momentum, grad)]
    moved = [q + step * v * r for q, v, r in zip(theta, inverse_metric, half)]
    value, new_grad = target(moved)
    momentum_out = [r + 0.5 * step * g for r, g in zip(half, new_grad)]
    return moved, momentum_out, value, new_grad


def _turning(
    theta_minus: list[float],
    theta_plus: list[float],
    momentum_minus: list[float],
    momentum_plus: list[float],
    inverse_metric: list[float],
) -> bool:
    """The no-U-turn criterion, in position space: has the trajectory started to come back?"""
    difference = [plus - minus for plus, minus in zip(theta_plus, theta_minus)]
    forward = sum(
        d * v * r for d, v, r in zip(difference, inverse_metric, momentum_plus)
    )
    backward = sum(
        d * v * r for d, v, r in zip(difference, inverse_metric, momentum_minus)
    )
    return forward <= 0.0 or backward <= 0.0


@dataclass
class _Subtree:
    theta_minus: list[float]
    momentum_minus: list[float]
    grad_minus: list[float]
    theta_plus: list[float]
    momentum_plus: list[float]
    grad_plus: list[float]
    proposal: list[float]
    proposal_value: float
    valid: int
    keep_going: bool
    accept_sum: float
    steps: int
    diverged: bool


def _build_tree(
    target: _Target,
    theta: list[float],
    momentum: list[float],
    grad: list[float],
    log_slice: float,
    direction: int,
    depth: int,
    step: float,
    joint_start: float,
    inverse_metric: list[float],
    rng: random.Random,
) -> _Subtree:
    if depth == 0:
        moved, moved_momentum, value, new_grad = _leapfrog(
            target, theta, momentum, grad, direction * step, inverse_metric
        )
        joint = value - _kinetic(moved_momentum, inverse_metric)
        valid = 1 if joint >= log_slice else 0
        energy_error = joint_start - joint
        diverged = not math.isfinite(joint) or energy_error > MAX_ENERGY_ERROR
        accept = math.exp(min(0.0, joint - joint_start)) if math.isfinite(joint) else 0.0
        return _Subtree(
            theta_minus=moved,
            momentum_minus=moved_momentum,
            grad_minus=new_grad,
            theta_plus=moved,
            momentum_plus=moved_momentum,
            grad_plus=new_grad,
            proposal=moved,
            proposal_value=value,
            valid=valid,
            keep_going=not diverged,
            accept_sum=accept,
            steps=1,
            diverged=diverged,
        )

    first = _build_tree(
        target,
        theta,
        momentum,
        grad,
        log_slice,
        direction,
        depth - 1,
        step,
        joint_start,
        inverse_metric,
        rng,
    )
    if not first.keep_going:
        return first

    if direction == -1:
        second = _build_tree(
            target,
            first.theta_minus,
            first.momentum_minus,
            first.grad_minus,
            log_slice,
            direction,
            depth - 1,
            step,
            joint_start,
            inverse_metric,
            rng,
        )
        theta_minus, momentum_minus, grad_minus = (
            second.theta_minus,
            second.momentum_minus,
            second.grad_minus,
        )
        theta_plus, momentum_plus, grad_plus = (
            first.theta_plus,
            first.momentum_plus,
            first.grad_plus,
        )
    else:
        second = _build_tree(
            target,
            first.theta_plus,
            first.momentum_plus,
            first.grad_plus,
            log_slice,
            direction,
            depth - 1,
            step,
            joint_start,
            inverse_metric,
            rng,
        )
        theta_minus, momentum_minus, grad_minus = (
            first.theta_minus,
            first.momentum_minus,
            first.grad_minus,
        )
        theta_plus, momentum_plus, grad_plus = (
            second.theta_plus,
            second.momentum_plus,
            second.grad_plus,
        )

    total_valid = first.valid + second.valid
    proposal, proposal_value = first.proposal, first.proposal_value
    # Progressive sampling: accept the new subtree's proposal with probability n2/(n1+n2), which makes
    # the final draw uniform over all valid points in the trajectory.
    if second.valid > 0 and rng.random() < second.valid / max(total_valid, 1):
        proposal, proposal_value = second.proposal, second.proposal_value

    keep_going = (
        first.keep_going
        and second.keep_going
        and not _turning(
            theta_minus, theta_plus, momentum_minus, momentum_plus, inverse_metric
        )
    )
    return _Subtree(
        theta_minus=theta_minus,
        momentum_minus=momentum_minus,
        grad_minus=grad_minus,
        theta_plus=theta_plus,
        momentum_plus=momentum_plus,
        grad_plus=grad_plus,
        proposal=proposal,
        proposal_value=proposal_value,
        valid=total_valid,
        keep_going=keep_going,
        accept_sum=first.accept_sum + second.accept_sum,
        steps=first.steps + second.steps,
        diverged=first.diverged or second.diverged,
    )


def _find_initial_step(
    target: _Target, theta: list[float], inverse_metric: list[float], rng: random.Random
) -> float:
    """Heuristic step-size search: double or halve until the acceptance crosses 0.5."""
    step = 1.0
    value, grad = target(theta)
    if not math.isfinite(value):
        raise ValueError("the starting point has zero posterior density")
    momentum = _draw_momentum(rng, inverse_metric)
    joint = value - _kinetic(momentum, inverse_metric)

    _, moved_momentum, moved_value, _ = _leapfrog(
        target, theta, momentum, grad, step, inverse_metric
    )
    moved_joint = moved_value - _kinetic(moved_momentum, inverse_metric)
    direction = 1 if moved_joint - joint > math.log(0.5) else -1

    for _ in range(60):
        step = step * (2.0 if direction == 1 else 0.5)
        _, moved_momentum, moved_value, _ = _leapfrog(
            target, theta, momentum, grad, step, inverse_metric
        )
        moved_joint = moved_value - _kinetic(moved_momentum, inverse_metric)
        difference = moved_joint - joint
        if direction == 1 and not difference > math.log(0.5):
            break
        if direction == -1 and not difference < math.log(0.5):
            break
    return step


class _DualAveraging:
    """Nesterov dual averaging on ``log step``, as used for step-size adaptation in NUTS."""

    def __init__(self, initial_step: float, target: float) -> None:
        self.mu = math.log(10.0 * initial_step)
        self.target = target
        self.log_step = math.log(initial_step)
        self.log_step_bar = 0.0
        self.h_bar = 0.0
        self.iteration = 0
        self.gamma = 0.05
        self.t0 = 10.0
        self.kappa = 0.75

    def update(self, accept: float) -> float:
        self.iteration += 1
        weight = 1.0 / (self.iteration + self.t0)
        self.h_bar = (1.0 - weight) * self.h_bar + weight * (self.target - accept)
        self.log_step = self.mu - math.sqrt(self.iteration) / self.gamma * self.h_bar
        decay = self.iteration ** (-self.kappa)
        self.log_step_bar = decay * self.log_step + (1.0 - decay) * self.log_step_bar
        return math.exp(self.log_step)

    @property
    def averaged(self) -> float:
        return math.exp(self.log_step_bar)


def _one_chain(model, config: NUTSConfig, seed: int) -> Chain:
    rng = random.Random(seed)
    target = _Target(model)
    dimension = model.dim
    theta = model.initial_point(rng, config.jitter)
    inverse_metric = [1.0] * dimension

    value, grad = target(theta)
    if not math.isfinite(value):
        for _ in range(50):
            theta = model.initial_point(rng, config.jitter * 0.5)
            value, grad = target(theta)
            if math.isfinite(value):
                break
        else:
            raise ValueError("could not find a starting point with finite posterior density")

    step = _find_initial_step(target, theta, inverse_metric, rng)
    adapter = _DualAveraging(step, config.target_accept)

    chain = Chain()
    # Warmup is split: search, then estimate the metric, then re-adapt the step at that metric.
    metric_start = config.warmup // 4
    metric_end = (3 * config.warmup) // 4
    collected: list[list[float]] = []

    for iteration in range(config.warmup + config.draws):
        warming = iteration < config.warmup
        momentum = _draw_momentum(rng, inverse_metric)
        joint = value - _kinetic(momentum, inverse_metric)
        # log u ~ Uniform(0, exp(joint))  <=>  log u = joint - Exponential(1)
        log_slice = joint - rng.expovariate(1.0)

        theta_minus = theta_plus = theta
        momentum_minus = momentum_plus = momentum
        grad_minus = grad_plus = grad
        proposal, proposal_value, proposal_grad = theta, value, grad
        valid = 1
        depth = 0
        keep_going = True
        accept_sum = 0.0
        steps = 0
        diverged = False

        while keep_going and depth < config.max_treedepth:
            direction = 1 if rng.random() < 0.5 else -1
            if direction == -1:
                subtree = _build_tree(
                    target,
                    theta_minus,
                    momentum_minus,
                    grad_minus,
                    log_slice,
                    direction,
                    depth,
                    step,
                    joint,
                    inverse_metric,
                    rng,
                )
                theta_minus, momentum_minus, grad_minus = (
                    subtree.theta_minus,
                    subtree.momentum_minus,
                    subtree.grad_minus,
                )
            else:
                subtree = _build_tree(
                    target,
                    theta_plus,
                    momentum_plus,
                    grad_plus,
                    log_slice,
                    direction,
                    depth,
                    step,
                    joint,
                    inverse_metric,
                    rng,
                )
                theta_plus, momentum_plus, grad_plus = (
                    subtree.theta_plus,
                    subtree.momentum_plus,
                    subtree.grad_plus,
                )

            accept_sum += subtree.accept_sum
            steps += subtree.steps
            diverged = diverged or subtree.diverged

            if subtree.keep_going and subtree.valid > 0:
                if rng.random() < subtree.valid / max(valid, 1):
                    proposal = subtree.proposal
                    proposal_value = subtree.proposal_value
                    proposal_grad = None  # recomputed below only if the proposal is taken
            valid += subtree.valid
            keep_going = subtree.keep_going and not _turning(
                theta_minus, theta_plus, momentum_minus, momentum_plus, inverse_metric
            )
            depth += 1

        if depth >= config.max_treedepth and keep_going:
            chain.max_depth_hits += 1

        if proposal is not theta:
            theta = proposal
            value = proposal_value
            if proposal_grad is None:
                value, grad = target(theta)

        accept_stat = accept_sum / max(steps, 1)
        if diverged:
            chain.divergences += 1

        if warming:
            step = adapter.update(accept_stat)
            if metric_start <= iteration < metric_end:
                collected.append(list(theta))
            if iteration == metric_end - 1 and len(collected) > 10:
                inverse_metric = _estimate_variance(collected)
                step = _find_initial_step(target, theta, inverse_metric, rng)
                adapter = _DualAveraging(step, config.target_accept)
            if iteration == config.warmup - 1:
                step = adapter.averaged
        else:
            chain.draws.append(list(theta))
            chain.log_posterior.append(value)
            chain.tree_depth.append(depth)
            chain.accept_stat.append(accept_stat)

    chain.step_size = step
    chain.inverse_metric = inverse_metric
    return chain


def _estimate_variance(samples: list[list[float]]) -> list[float]:
    """Per-coordinate variance with a shrinkage floor, which becomes the diagonal metric."""
    count = len(samples)
    dimension = len(samples[0])
    variances: list[float] = []
    for index in range(dimension):
        column = [row[index] for row in samples]
        mean = sum(column) / count
        variance = sum((value - mean) ** 2 for value in column) / max(count - 1, 1)
        # Stan's regularisation: pull toward 1e-3 so a coordinate that happened not to move during the
        # window cannot produce a zero (and therefore a frozen) metric entry.
        shrunk = (count / (count + 5.0)) * variance + 1e-3 * (5.0 / (count + 5.0))
        variances.append(max(shrunk, 1e-10))
    return variances


def sample(model, config: NUTSConfig | None = None) -> Posterior:
    """Run every chain and return the posterior.

    Chains start from dispersed points by design: R-hat compares between-chain and within-chain
    variance, so chains that start together and stay together prove nothing at all.
    """
    config = config or NUTSConfig()
    chains = [
        _one_chain(model, config, seed=config.seed + 1000 * index)
        for index in range(config.chains)
    ]
    return Posterior(chains=chains, names=model.names)
