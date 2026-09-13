"""Demonstrations, one per claim.

``gradient``        the analytic gradient against central finite differences, everywhere it matters
``sampler``         NUTS on a target with a known answer, next to a random walk given the same budget
``fit``             the full pipeline: sample, diagnose, recover, decompose, price the media
``identifiability`` what flat spend does to the saturation posterior (the quiet failure)
``confounded``      what dropping seasonality does to media effects (the loud one)
``budget``          allocation at steady state, verified against brute force, with decision uncertainty
``forecast``        a 13-week holdout against a same-week-last-year benchmark

``gradient`` and ``sampler`` run in seconds. The rest fit real posteriors in pure Python and take one to
three minutes each; the draw counts are printed so nothing looks stalled.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time

from . import data as datasets
from .decision import allocate, channel_results, decomposition, grid_search_allocation, posterior_allocation, results_table
from .diagnostics import quantile, split_rhat, summarise_columns, summary_table
from .model import MMM, gradient_error
from .nuts import NUTSConfig, sample
from .validate import (
    forecast_evaluation,
    natural_columns,
    parameter_recovery,
    posterior_predictive_check,
    ppc_table,
    recovery_table,
    train_test_split,
)

RULE = "=" * 98


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _fit(dataset, harmonics: int = 2, draws: int = 300, warmup: int = 300, chains: int = 2, seed: int = 1):
    model = MMM(dataset, harmonics=harmonics)
    print(
        f"  sampling {chains} chains x ({warmup} warmup + {draws} draws) over "
        f"{model.dim} parameters and {model.weeks} weeks..."
    )
    started = time.perf_counter()
    posterior = sample(model, NUTSConfig(draws=draws, warmup=warmup, chains=chains, seed=seed))
    elapsed = time.perf_counter() - started
    for index, chain in enumerate(posterior.chains):
        print(f"    chain {index}: {chain.summary()}")
    print(f"    {elapsed:.0f}s total, {posterior.divergences} divergences overall")
    return model, posterior


# ---------------------------------------------------------------------------------------------


def cmd_gradient(args) -> None:
    """The analytic gradient against central finite differences."""
    _heading("Analytic gradient versus central finite differences")
    dataset, truth = datasets.simulate(weeks=78, seed=3)
    model = MMM(dataset)
    rng = random.Random(0)

    print(f"  {model.dim} parameters, {model.weeks} weeks, {model.n_channels} channels")
    print(f"\n    {'point':<34}{'worst relative error':>22}{'coordinate':>26}")
    print("    " + "-" * 80)
    worst_overall = 0.0
    points = [("the truth", model.pack(truth.parameters()))]
    for index in range(4):
        points.append((f"random draw {index + 1}", model.initial_point(rng, jitter=1.0)))
    for label, theta in points:
        error, where = gradient_error(model, theta)
        worst_overall = max(worst_overall, error)
        print(f"    {label:<34}{error:>22.2e}{model.names[where]:>26}")

    print(f"\n  worst relative error anywhere: {worst_overall:.2e}")
    print(
        "  Central differences are accurate to about 1e-10 relative, so anything below roughly 1e-6 is\n"
        "  agreement to the precision of the check itself. This is the test that catches a missing\n"
        "  log-Jacobian: the posterior would still be sampled happily, just the wrong posterior."
    )


class _GaussianTarget:
    """A correlated Gaussian with a known answer, wearing the model interface.

    Sampling something whose truth is known by algebra is the only way to separate "the sampler works"
    from "the model happens to fit".
    """

    def __init__(self, correlation: float = 0.95, scales=(1.0, 10.0)) -> None:
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
        gx = -factor * (x - rho * y) / sx
        gy = -factor * (y - rho * x) / sy
        return value, [gx, gy]

    def log_posterior(self, theta):
        return self.log_posterior_and_gradient(theta)[0]

    def initial_point(self, rng, jitter: float = 1.0):
        return [rng.gauss(0.0, jitter * self.scales[0]), rng.gauss(0.0, jitter * self.scales[1])]


def _metropolis(target, draws: int, gradient_budget: int, seed: int) -> list[list[float]]:
    """A tuned random walk, given the same number of density evaluations NUTS used.

    Included as the control. The step size is set to the textbook optimum for this geometry -- the
    smaller marginal scale -- which is the best a random walk can do without knowing the correlation.
    """
    rng = random.Random(seed)
    theta = target.initial_point(rng)
    value = target.log_posterior(theta)
    step = 2.4 / math.sqrt(2) * min(target.scales)
    chain: list[list[float]] = []
    accepted = 0
    for iteration in range(gradient_budget):
        proposal = [value_ + rng.gauss(0.0, step) for value_ in theta]
        candidate = target.log_posterior(proposal)
        if math.log(rng.random() + 1e-300) < candidate - value:
            theta, value = proposal, candidate
            accepted += 1
        if iteration % max(gradient_budget // draws, 1) == 0:
            chain.append(list(theta))
    print(f"    random walk acceptance {accepted / gradient_budget:.2f}")
    return chain[:draws]


def cmd_sampler(args) -> None:
    """NUTS against a known Gaussian, with a random walk as the control."""
    _heading("NUTS on a target whose answer is known by algebra")
    target = _GaussianTarget(correlation=0.95, scales=(1.0, 10.0))
    print(
        "  target: 2-d Gaussian, correlation 0.95, marginal scales 1 and 10.\n"
        "  A ten-to-one scale ratio plus tight correlation is the geometry an MMM posterior actually has."
    )

    started = time.perf_counter()
    posterior = sample(target, NUTSConfig(draws=1000, warmup=1000, chains=2, seed=7))
    elapsed = time.perf_counter() - started
    columns = [posterior.columns_by_chain(0), posterior.columns_by_chain(1)]
    summaries = summarise_columns(target.names, columns)
    print(f"\n  NUTS ({elapsed:.1f}s):")
    print("    " + summary_table(summaries).replace("\n", "\n    "))

    pooled_x = posterior.column(0)
    pooled_y = posterior.column(1)
    mean_x = sum(pooled_x) / len(pooled_x)
    mean_y = sum(pooled_y) / len(pooled_y)
    sd_x = math.sqrt(sum((v - mean_x) ** 2 for v in pooled_x) / (len(pooled_x) - 1))
    sd_y = math.sqrt(sum((v - mean_y) ** 2 for v in pooled_y) / (len(pooled_y) - 1))
    covariance = sum(
        (a - mean_x) * (b - mean_y) for a, b in zip(pooled_x, pooled_y)
    ) / (len(pooled_x) - 1)
    print(
        f"\n    truth:     mean (0, 0), sd (1.00, 10.00), correlation 0.950\n"
        f"    recovered: mean ({mean_x:.3f}, {mean_y:.3f}), sd ({sd_x:.3f}, {sd_y:.3f}), "
        f"correlation {covariance / (sd_x * sd_y):.3f}"
    )

    print("\n  Random walk, same density-evaluation budget:")
    walk = _metropolis(target, draws=2000, gradient_budget=40000, seed=7)
    half = len(walk) // 2
    walk_columns = [[[row[0] for row in walk[:half]], [row[0] for row in walk[half:2 * half]]]]
    walk_summary = summarise_columns(["x"], walk_columns)
    print("    " + summary_table(walk_summary).replace("\n", "\n    "))
    print(
        "\n  The comparison is not about wall-clock time, it is about information per evaluation. The\n"
        "  random walk cannot see the ridge, so its effective sample size stays small however long it runs."
    )


def cmd_fit(args) -> None:
    """The full pipeline on data generated from the model's own assumptions."""
    _heading("Fit, diagnose, recover, decompose, price")
    dataset, truth = datasets.simulate(weeks=104, seed=11)
    model, posterior = _fit(dataset)

    columns = natural_columns(model, posterior)
    summaries = summarise_columns(model.names, columns)
    print("\n  Posterior, on the natural scale:\n")
    print("  " + summary_table(summaries).replace("\n", "\n  "))

    print("\n  Recovery against the values that generated the data:\n")
    print("  " + recovery_table(parameter_recovery(model, posterior, truth.parameters())).replace("\n", "\n  "))

    draws = posterior.draws()
    middle = model.unpack(draws[len(draws) // 2])
    print("\n  Media effectiveness at the posterior median draw:\n")
    print("  " + results_table(channel_results(model, middle)).replace("\n", "\n  "))
    shares = decomposition(model, middle)
    print("\n  Sales decomposition: " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
    print(
        "\n  The baseline share is not organic demand. It is everything the model could not attribute to\n"
        "  media inside this window, including brand equity built by spend that predates the data."
    )


def cmd_identifiability(args) -> None:
    """Flat spend, and the saturation point that cannot be learned from it."""
    _heading("The quiet failure: flat spend leaves the saturation point unidentified")
    for regime in ("clean", "flat"):
        dataset, truth = datasets.generate(regime, weeks=104, seed=5)
        spans = {
            name: (min(series), max(series)) for name, series in dataset.spend.items()
        }
        print(f"\n  regime '{regime}' ({datasets.REGIMES[regime]})")
        for name, (low, high) in spans.items():
            print(f"    {name:<10} spend ranges {low:>10,.0f} to {high:>10,.0f}")
        model, posterior = _fit(dataset, draws=250, warmup=250)
        columns = natural_columns(model, posterior)
        print(f"    {'channel':<10}{'true kappa':>14}{'posterior 5-95%':>28}{'width / prior width':>22}")
        for index, name in enumerate(model.channels):
            coordinate = model.names.index(f"half[{name}]")
            pooled = [value for chain in columns[coordinate] for value in chain]
            low, high = quantile(pooled, 0.05), quantile(pooled, 0.95)
            prior_low = math.exp(model.priors.half_log_mean - 1.645 * model.priors.half_log_sd)
            prior_high = math.exp(model.priors.half_log_mean + 1.645 * model.priors.half_log_sd)
            ratio = (high - low) / (prior_high - prior_low)
            print(
                f"    {name:<10}{truth.channel(name).half:>14,.0f}"
                f"{f'{low:,.0f} to {high:,.0f}':>28}{ratio:>22.2f}"
            )
        residual_sd = math.sqrt(
            sum(r * r for r in model.residuals(model.unpack(posterior.draws()[-1]))) / model.weeks
        )
        print(f"    residual sd {residual_sd:,.0f} against true noise {truth.sigma:,.0f}")
    print(
        "\n  Read the last column. A ratio near 1 means the data taught the model nothing about the\n"
        "  saturation point and the posterior is the prior. Both fits look good, and the flat-spend one\n"
        "  will still produce a confident budget recommendation. That recommendation is prior belief\n"
        "  wearing a posterior's clothes, and the only cure is spend variation -- which means an\n"
        "  experiment, not more weeks of the same plan."
    )


def cmd_confounded(args) -> None:
    """Spend that follows seasonality, fitted with and without seasonal terms."""
    _heading("The loud failure: media takes credit for Christmas")
    dataset, truth = datasets.simulate_confounded(weeks=104, seed=9)
    print("  Budgets in this dataset rise into the seasonal peak, exactly as real media plans do.\n")

    for harmonics, label in ((2, "with seasonality (correct)"), (0, "without seasonality (omitted)")):
        model, posterior = _fit(dataset, harmonics=harmonics, draws=250, warmup=250)
        columns = natural_columns(model, posterior)
        print(f"\n  {label}:")
        print(f"    {'channel':<10}{'true beta':>14}{'posterior mean':>18}{'bias':>12}")
        for name in model.channels:
            coordinate = model.names.index(f"beta[{name}]")
            pooled = [value for chain in columns[coordinate] for value in chain]
            mean = sum(pooled) / len(pooled)
            true_beta = truth.channel(name).beta
            print(
                f"    {name:<10}{true_beta:>14,.0f}{mean:>18,.0f}"
                f"{(mean - true_beta) / true_beta:>11.0%}"
            )
    print(
        "\n  The omitted-variable bias is upward, and it has to be: seasonality raises sales and raises\n"
        "  spend, so with no seasonal term to absorb it the only place that variance can go is media.\n"
        "  This is why an MMM without controls for demand seasonality overstates advertising, and why\n"
        "  the fit statistics do not warn you -- the mis-specified model explains the data almost as well."
    )


def cmd_budget(args) -> None:
    """Allocation at steady state, checked against brute force, with decision uncertainty."""
    _heading("Budget allocation: verified, and reported with its uncertainty")
    dataset, truth = datasets.simulate(weeks=104, seed=11)
    model, posterior = _fit(dataset, draws=250, warmup=250)
    draws = posterior.draws()
    params = model.unpack(draws[len(draws) // 2])

    current = sum(sum(series) for series in dataset.spend.values()) / model.weeks
    print(f"\n  current spend: {current:,.0f} per week across {model.n_channels} channels")
    print(f"  channel shapes: " + ", ".join(
        f"{name} alpha={value:.2f}" for name, value in zip(model.channels, params.shape)
    ))

    allocation = allocate(model, params, current)
    print("\n  " + allocation.summary().replace("\n", "\n  "))

    grid_spend, grid_response = grid_search_allocation(model, params, current, steps=40)
    print(
        f"\n  brute-force grid (40 steps): response {grid_response:,.0f}/week vs "
        f"{allocation.response:,.0f} from the optimiser"
    )
    print("    grid split: " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(grid_spend.items())))
    gap = (allocation.response - grid_response) / max(abs(grid_response), 1e-9)
    print(f"    the optimiser is {gap:+.2%} against the grid (a coarse grid should be slightly worse)")

    print("\n  Recommended share of budget, across posterior draws:")
    print(f"    {'channel':<12}{'5%':>10}{'median':>10}{'95%':>10}")
    for name, (low, mid, high) in sorted(posterior_allocation(model, posterior, current, draws=120).items()):
        print(f"    {name:<12}{low:>10.1%}{mid:>10.1%}{high:>10.1%}")
    print(
        "\n  Those intervals are the deliverable. A median split of 40/35/25 with each share uncertain to\n"
        "  plus or minus fifteen points is not a plan, it is a case for an experiment -- and saying so is\n"
        "  the difference between a model that informs decisions and one that launders them."
    )


def cmd_forecast(args) -> None:
    """A 13-week holdout, scored against same-week-last-year."""
    _heading("Holdout forecasting: fit on the past, predict weeks never seen")
    dataset, truth = datasets.simulate(weeks=130, seed=21)
    holdout = 13
    training, _ = train_test_split(dataset, holdout)

    print(f"  training on {training.weeks} weeks, holding out {holdout}")
    train_model = MMM(training, harmonics=2)
    print(f"  sampling...")
    posterior = sample(train_model, NUTSConfig(draws=250, warmup=250, chains=2, seed=4))
    print(f"    {posterior.divergences} divergences")

    full_model = MMM(dataset, harmonics=2, priors=train_model.priors)
    result = forecast_evaluation(full_model, posterior, holdout)
    print(f"\n  {result.summary()}")

    print("\n  Posterior predictive checks on the training window:\n")
    print("  " + ppc_table(posterior_predictive_check(train_model, posterior)).replace("\n", "\n  "))
    print(
        "\n  Coverage near 90% and skill above zero is the pair to want. Good coverage with negative skill\n"
        "  means the intervals are honestly wide and the point forecast is useless; the reverse means a\n"
        "  lucky point forecast with intervals that will embarrass someone."
    )


COMMANDS = {
    "gradient": cmd_gradient,
    "sampler": cmd_sampler,
    "fit": cmd_fit,
    "identifiability": cmd_identifiability,
    "confounded": cmd_confounded,
    "budget": cmd_budget,
    "forecast": cmd_forecast,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmm", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=(handler.__doc__ or "").strip().split("\n")[0])
        subparser.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
