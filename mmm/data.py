"""Synthetic data with known ground truth, including data the model is wrong about.

A model tested only against data drawn from itself has been tested for typos, not for validity. So this
module generates four regimes:

``simulate``            the model's own assumptions, held exactly. Parameter recovery must work here, and
                        if it does not, something is broken rather than merely hard.
``simulate_flat_spend`` spend held nearly constant. Nothing about the model changes, and the saturation
                        point becomes unidentifiable: the posterior for kappa reverts to its prior because
                        the data never observes the curve bending. This is the single most common way a
                        real MMM lies, and it does not announce itself -- the fit is excellent.
``simulate_delayed``    carryover that peaks a week or two after exposure, which geometric adstock cannot
                        represent. The model is misspecified and the recovered decay compensates.
``simulate_confounded`` media spend driven by the same seasonality that drives sales. Fit the model without
                        the seasonal terms and media takes credit for Christmas.

Every generator returns the truth alongside the data, so a test can ask the only question that matters:
does the posterior cover the value that produced the data?
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .model import Dataset, Parameters
from .transforms import delayed_adstock, geometric_adstock, hill


@dataclass(frozen=True)
class ChannelTruth:
    """The truth for one channel, in the units the model uses."""

    name: str
    beta: float  # maximum incremental sales per week at full saturation
    decay: float  # geometric carryover
    half: float  # half-saturation point, in adstocked spend units
    shape: float  # Hill shape; > 1 is an S-curve
    level: float  # typical weekly spend when active
    duty_cycle: float = 0.75  # fraction of weeks with any spend at all


@dataclass(frozen=True)
class Truth:
    """Everything used to generate a dataset. The answer key."""

    base: float
    trend: float
    season: list[float]
    channels: list[ChannelTruth]
    control: list[float]
    sigma: float
    weeks: int
    harmonics: int
    period: float = 52.0
    notes: str = ""

    def parameters(self) -> Parameters:
        """The truth in the model's own parameter object, so it can be packed and evaluated directly."""
        return Parameters(
            base=self.base,
            trend=self.trend,
            season=list(self.season),
            beta=[channel.beta for channel in self.channels],
            decay=[channel.decay for channel in self.channels],
            half=[channel.half for channel in self.channels],
            shape=[channel.shape for channel in self.channels],
            control=list(self.control),
            sigma=self.sigma,
        )

    def channel(self, name: str) -> ChannelTruth:
        for entry in self.channels:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def true_contribution(self, spend: list[float], channel: ChannelTruth) -> list[float]:
        adstocked = geometric_adstock(spend, channel.decay, normalize=True)
        return [channel.beta * hill(value, channel.half, channel.shape) for value in adstocked]


DEFAULT_CHANNELS = [
    # A large always-on channel that is already saturated: high spend, low marginal return. The one a
    # naive last-click report over-credits.
    ChannelTruth("search", beta=9000.0, decay=0.25, half=12000.0, shape=1.0, level=22000.0, duty_cycle=0.98),
    # A bursty brand channel with long carryover and an S-curve: needs scale to work at all.
    ChannelTruth("tv", beta=14000.0, decay=0.65, half=30000.0, shape=1.8, level=38000.0, duty_cycle=0.45),
    # A small efficient channel operating below saturation: the one with headroom.
    ChannelTruth("social", beta=6000.0, decay=0.35, half=9000.0, shape=1.2, level=6000.0, duty_cycle=0.85),
]


def _spend_series(
    rng: random.Random, channel: ChannelTruth, weeks: int, variation: float = 0.55
) -> list[float]:
    """Flighted spend: bursts, dark weeks, and a slow drift, spanning the saturation point.

    The range matters more than the mean. Spend that never crosses ``half`` leaves the saturation point
    unidentified no matter how many weeks are collected, so the generator deliberately sweeps from well
    below to well above it.
    """
    series: list[float] = []
    burst_remaining = 0
    for t in range(weeks):
        if burst_remaining > 0:
            burst_remaining -= 1
            active = True
        else:
            active = rng.random() < channel.duty_cycle
            if active and channel.duty_cycle < 0.6:
                burst_remaining = rng.randint(1, 3)  # campaigns run in flights, not single weeks
        if not active:
            series.append(0.0)
            continue
        # A slow sinusoidal drift in budget across the horizon plus lognormal week-to-week noise.
        drift = 1.0 + 0.45 * math.sin(2.0 * math.pi * t / max(weeks, 1) + rng.random() * 0.1)
        noise = math.exp(rng.gauss(0.0, variation))
        series.append(max(channel.level * drift * noise, 0.0))
    return series


def _seasonality(rng: random.Random, harmonics: int, scale: float) -> list[float]:
    coefficients: list[float] = []
    for j in range(1, harmonics + 1):
        damping = 1.0 / j
        coefficients.append(rng.gauss(0.0, scale * damping))
        coefficients.append(rng.gauss(0.0, scale * damping))
    return coefficients


def _assemble(
    truth: Truth,
    spend: dict[str, list[float]],
    controls: dict[str, list[float]],
    rng: random.Random,
    contributions: dict[str, list[float]] | None = None,
) -> Dataset:
    weeks = truth.weeks
    y = []
    for t in range(weeks):
        value = truth.base + truth.trend * (t / max(weeks - 1, 1))
        for j in range(1, truth.harmonics + 1):
            angle = 2.0 * math.pi * j * t / truth.period
            value += truth.season[2 * (j - 1)] * math.sin(angle)
            value += truth.season[2 * (j - 1) + 1] * math.cos(angle)
        for index, name in enumerate(controls):
            value += truth.control[index] * controls[name][t]
        y.append(value)

    for channel in truth.channels:
        series = (
            contributions[channel.name]
            if contributions is not None
            else truth.true_contribution(spend[channel.name], channel)
        )
        for t in range(weeks):
            y[t] += series[t]

    y = [value + rng.gauss(0.0, truth.sigma) for value in y]
    return Dataset(y=y, spend=spend, controls=controls, period=truth.period)


def _controls(rng: random.Random, weeks: int, period: float) -> dict[str, list[float]]:
    """A price index (mean-centred, autocorrelated) and a holiday flag."""
    price: list[float] = []
    level = 0.0
    for _ in range(weeks):
        level = 0.85 * level + rng.gauss(0.0, 0.06)
        price.append(level)
    holiday = [1.0 if (t % int(period)) in (46, 47, 48, 49) else 0.0 for t in range(weeks)]
    return {"price_index": price, "holiday": holiday}


def simulate(
    weeks: int = 104,
    seed: int = 0,
    channels: list[ChannelTruth] | None = None,
    harmonics: int = 2,
    noise: float = 4000.0,
    base: float = 60000.0,
    trend: float = 9000.0,
) -> tuple[Dataset, Truth]:
    """Data from exactly the model that will be fitted to it. The recovery test lives here."""
    rng = random.Random(seed)
    channels = list(channels or DEFAULT_CHANNELS)
    controls = _controls(rng, weeks, 52.0)
    truth = Truth(
        base=base,
        trend=trend,
        season=_seasonality(rng, harmonics, 0.12 * base),
        channels=channels,
        control=[-0.9 * base, 0.18 * base],  # price elasticity is negative; holidays lift sales
        sigma=noise,
        weeks=weeks,
        harmonics=harmonics,
        notes="the model's own assumptions, exactly",
    )
    spend = {channel.name: _spend_series(rng, channel, weeks) for channel in channels}
    return _assemble(truth, spend, controls, rng), truth


def simulate_flat_spend(
    weeks: int = 104, seed: int = 0, harmonics: int = 2
) -> tuple[Dataset, Truth]:
    """Spend that barely moves, so the saturation curve is never traced out.

    The likelihood is nearly flat in ``kappa`` over a wide range: many (beta, kappa) pairs reproduce the
    same fitted contribution, because within a narrow band of spend the curve is locally linear. Expect a
    posterior for ``kappa`` that looks like its prior and a ``beta`` posterior with a matching ridge. The
    model fit will nonetheless be excellent, which is the trap.
    """
    rng = random.Random(seed)
    channels = list(DEFAULT_CHANNELS)
    controls = _controls(rng, weeks, 52.0)
    truth = Truth(
        base=60000.0,
        trend=6000.0,
        season=_seasonality(rng, harmonics, 7000.0),
        channels=channels,
        control=[-54000.0, 11000.0],
        sigma=4000.0,
        weeks=weeks,
        harmonics=harmonics,
        notes="near-constant spend: kappa is not identified",
    )
    spend = {
        channel.name: [
            max(channel.level * (1.0 + rng.gauss(0.0, 0.03)), 0.0) for _ in range(weeks)
        ]
        for channel in channels
    }
    return _assemble(truth, spend, controls, rng), truth


def simulate_delayed(
    weeks: int = 104, seed: int = 0, harmonics: int = 2, peak: float = 2.0
) -> tuple[Dataset, Truth]:
    """Carryover that peaks after exposure. Geometric adstock cannot express it.

    The recovered ``decay`` will not match anything in the answer key, because there is no geometric decay
    that equals a delayed kernel. What the fit does instead is interesting and worth showing: it lengthens
    carryover to cover the delay, and the sales decomposition stays roughly right while the media timing
    is wrong -- so ROAS survives and any "when should the campaign start" conclusion does not.
    """
    rng = random.Random(seed)
    channels = list(DEFAULT_CHANNELS)
    controls = _controls(rng, weeks, 52.0)
    truth = Truth(
        base=60000.0,
        trend=7000.0,
        season=_seasonality(rng, harmonics, 7000.0),
        channels=channels,
        control=[-54000.0, 11000.0],
        sigma=4000.0,
        weeks=weeks,
        harmonics=harmonics,
        notes=f"delayed carryover peaking at week {peak:g}: the model is misspecified",
    )
    spend = {channel.name: _spend_series(rng, channel, weeks) for channel in channels}
    contributions = {}
    for channel in channels:
        adstocked = delayed_adstock(spend[channel.name], channel.decay, peak)
        contributions[channel.name] = [
            channel.beta * hill(value, channel.half, channel.shape) for value in adstocked
        ]
    return _assemble(truth, spend, controls, rng, contributions), truth


def simulate_confounded(
    weeks: int = 104, seed: int = 0, harmonics: int = 2, strength: float = 1.4
) -> tuple[Dataset, Truth]:
    """Budgets that follow the seasonal demand they are supposed to be causing.

    Every real media plan does this -- spend rises into the season -- and it is the reason an MMM without
    seasonal control terms overstates media. The demonstration fits the same data twice, with and without
    the Fourier terms, and the difference in recovered ``beta`` is the confounding bias, made visible.
    """
    rng = random.Random(seed)
    channels = list(DEFAULT_CHANNELS)
    controls = _controls(rng, weeks, 52.0)
    season = _seasonality(rng, harmonics, 9000.0)
    truth = Truth(
        base=60000.0,
        trend=5000.0,
        season=season,
        channels=channels,
        control=[-54000.0, 11000.0],
        sigma=4000.0,
        weeks=weeks,
        harmonics=harmonics,
        notes="spend tracks seasonality: media and season are confounded",
    )

    seasonal_shape = []
    for t in range(weeks):
        value = 0.0
        for j in range(1, harmonics + 1):
            angle = 2.0 * math.pi * j * t / 52.0
            value += season[2 * (j - 1)] * math.sin(angle) + season[2 * (j - 1) + 1] * math.cos(angle)
        seasonal_shape.append(value)
    largest = max(abs(value) for value in seasonal_shape) or 1.0

    spend = {}
    for channel in channels:
        baseline = _spend_series(rng, channel, weeks)
        spend[channel.name] = [
            max(value * (1.0 + strength * shape / largest), 0.0)
            for value, shape in zip(baseline, seasonal_shape)
        ]
    return _assemble(truth, spend, controls, rng), truth


REGIMES: dict[str, str] = {
    "clean": "the model's own assumptions",
    "flat": "near-constant spend, kappa unidentified",
    "delayed": "delayed carryover, model misspecified",
    "confounded": "spend follows seasonality",
}


def generate(regime: str, weeks: int = 104, seed: int = 0) -> tuple[Dataset, Truth]:
    if regime == "clean":
        return simulate(weeks=weeks, seed=seed)
    if regime == "flat":
        return simulate_flat_spend(weeks=weeks, seed=seed)
    if regime == "delayed":
        return simulate_delayed(weeks=weeks, seed=seed)
    if regime == "confounded":
        return simulate_confounded(weeks=weeks, seed=seed)
    raise ValueError(f"unknown regime '{regime}'; choose from {sorted(REGIMES)}")
