"""The model: mean function, priors, log-posterior, and its analytic gradient.

    y_t = base + trend * (t/T) + seasonality_t + sum_c beta_c * Hill(adstock(x_c)_t) + sum_k eta_k z_kt + e_t
    e_t ~ Normal(0, sigma)

Everything the sampler sees lives in an unconstrained space, because Hamiltonian dynamics cannot respect
a boundary: a decay of 1.01 or a negative sigma is not a bad sample, it is a crash. So

    beta = exp(b)          positive: media does not reduce sales
    kappa = exp(h)         positive: a half-saturation point is a spend level
    alpha = exp(a)         positive: a curve shape
    lambda = sigmoid(l)    in (0, 1): carryover that neither vanishes nor explodes
    sigma = exp(s)         positive

Each transformation contributes a log-Jacobian, and leaving one out is the classic silent MMM bug -- the
chains still converge, to the wrong posterior. Two of them simplify pleasantly and it is worth knowing why:

* A LogNormal(mu, tau) prior on beta plus the Jacobian ``|dbeta/db| = beta`` is exactly a Normal(mu, tau)
  prior on ``b``. The ``-log beta`` in the LogNormal density and the ``+log beta`` from the Jacobian cancel.
* A Beta(a, b) prior on lambda plus ``|dlambda/dl| = lambda(1 - lambda)`` gives
  ``a * log lambda + b * log(1 - lambda) - log B(a, b)`` -- the exponents shift by one, and the gradient
  collapses to ``a(1 - lambda) - b * lambda``.

The gradient is hand-derived (see ``gradient``), not automatic, and ``tests/test_model.py`` holds it against
central finite differences at random points in every parameter block. An analytic gradient that is wrong in
one coordinate produces a sampler that is subtly biased in that coordinate and looks fine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from .transforms import geometric_adstock, geometric_adstock_gradient, hill_partials

LOG_TWO_PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Weekly observations. ``spend`` and ``controls`` are ordered dictionaries of equal-length series."""

    y: list[float]
    spend: dict[str, list[float]]
    controls: dict[str, list[float]] = field(default_factory=dict)
    period: float = 52.0

    def __post_init__(self) -> None:
        if len(self.y) < 8:
            raise ValueError(f"need at least 8 observations, got {len(self.y)}")
        for name, series in list(self.spend.items()) + list(self.controls.items()):
            if len(series) != len(self.y):
                raise ValueError(
                    f"series '{name}' has {len(series)} points, y has {len(self.y)}"
                )
        for name, series in self.spend.items():
            if any(value < 0.0 for value in series):
                raise ValueError(f"spend '{name}' contains negative values")
        if self.period <= 1.0:
            raise ValueError("period must exceed one observation")

    @property
    def weeks(self) -> int:
        return len(self.y)

    @property
    def channels(self) -> list[str]:
        return list(self.spend)

    @property
    def control_names(self) -> list[str]:
        return list(self.controls)

    def total_spend(self, channel: str) -> float:
        return sum(self.spend[channel])

    def slice(self, start: int, stop: int) -> "Dataset":
        """A contiguous window, for rolling-origin validation."""
        return Dataset(
            y=self.y[start:stop],
            spend={name: series[start:stop] for name, series in self.spend.items()},
            controls={name: series[start:stop] for name, series in self.controls.items()},
            period=self.period,
        )


# ---------------------------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Priors:
    """Weakly informative and deliberately data-scaled.

    MMM data is short -- two or three years of weekly points against a dozen parameters -- so the priors
    are doing real work and pretending otherwise would be dishonest. What they encode:

    * ``beta``: LogNormal, median a modest fraction of average sales. Positive by construction, because
      the sign of a media effect is not something to learn from 104 noisy weeks.
    * ``lambda``: Beta(2, 4), mean 1/3, i.e. a half-life of about half a week to a couple of weeks. It
      pulls away from the 0.9+ region where carryover becomes indistinguishable from trend.
    * ``kappa``: LogNormal centred on observed median spend. This is the one that matters most: the data
      only identifies the saturation point if spend actually varies around it.
    * ``alpha``: LogNormal centred just under 1, so concavity is the default and an S-curve has to earn
      its keep against the prior.
    * ``sigma``: HalfNormal at the scale of the observed standard deviation.
    """

    base_mean: float
    base_sd: float
    trend_sd: float
    season_sd: float
    control_sd: float
    beta_log_mean: float
    beta_log_sd: float
    decay_a: float
    decay_b: float
    half_log_mean: float
    half_log_sd: float
    shape_log_mean: float
    shape_log_sd: float
    sigma_scale: float

    @classmethod
    def from_data(cls, data: Dataset) -> "Priors":
        weeks = data.weeks
        mean_y = sum(data.y) / weeks
        variance = sum((value - mean_y) ** 2 for value in data.y) / max(weeks - 1, 1)
        sd_y = math.sqrt(max(variance, 1e-12))

        positive: list[float] = []
        for series in data.spend.values():
            positive.extend(value for value in series if value > 0.0)
        median_spend = _median(positive) if positive else 1.0
        channels = max(len(data.spend), 1)

        return cls(
            base_mean=mean_y,
            base_sd=2.0 * sd_y + 1e-9,
            trend_sd=2.0 * sd_y + 1e-9,
            season_sd=sd_y + 1e-9,
            control_sd=sd_y + 1e-9,
            beta_log_mean=math.log(max(0.25 * mean_y / channels, 1e-9)),
            beta_log_sd=1.0,
            decay_a=2.0,
            decay_b=4.0,
            half_log_mean=math.log(max(median_spend, 1e-9)),
            half_log_sd=0.7,
            shape_log_mean=math.log(0.9),
            shape_log_sd=0.4,
            sigma_scale=sd_y + 1e-9,
        )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _log_beta_function(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ---------------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameters:
    """Parameters on their natural scale, which is the only scale worth reporting."""

    base: float
    trend: float
    season: list[float]
    beta: list[float]
    decay: list[float]
    half: list[float]
    shape: list[float]
    control: list[float]
    sigma: float

    def with_spend_effect(self, channel: int, factor: float) -> "Parameters":
        scaled = list(self.beta)
        scaled[channel] *= factor
        return replace(self, beta=scaled)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponentiated = math.exp(value)
    return exponentiated / (1.0 + exponentiated)


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"cannot take the logit of {value}")
    return math.log(value / (1.0 - value))


# ---------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------


class MMM:
    """A marketing mix model over one dataset, exposing a log-posterior and its gradient."""

    def __init__(
        self,
        data: Dataset,
        harmonics: int = 2,
        priors: Priors | None = None,
        normalize_adstock: bool = True,
    ) -> None:
        if harmonics < 0:
            raise ValueError("harmonics must be non-negative")
        self.data = data
        self.harmonics = harmonics
        self.priors = priors or Priors.from_data(data)
        self.normalize_adstock = normalize_adstock

        self.channels = data.channels
        self.control_names = data.control_names
        self.weeks = data.weeks
        self.n_channels = len(self.channels)
        self.n_controls = len(self.control_names)

        # Fixed design pieces, built once.
        self._time = [t / max(self.weeks - 1, 1) for t in range(self.weeks)]
        self._fourier: list[list[float]] = []
        for j in range(1, harmonics + 1):
            angle = [2.0 * math.pi * j * t / data.period for t in range(self.weeks)]
            self._fourier.append([math.sin(value) for value in angle])
            self._fourier.append([math.cos(value) for value in angle])
        self._spend = [data.spend[name] for name in self.channels]
        self._controls = [data.controls[name] for name in self.control_names]

        self._offsets = self._layout()

    # -- layout ------------------------------------------------------------------------------

    def _layout(self) -> dict[str, int]:
        cursor = 0
        offsets: dict[str, int] = {}
        for name, size in (
            ("base", 1),
            ("trend", 1),
            ("season", len(self._fourier)),
            ("beta", self.n_channels),
            ("decay", self.n_channels),
            ("half", self.n_channels),
            ("shape", self.n_channels),
            ("control", self.n_controls),
            ("sigma", 1),
        ):
            offsets[name] = cursor
            cursor += size
        self._dim = cursor
        return offsets

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def names(self) -> list[str]:
        """Human-readable names in the order the unconstrained vector uses."""
        labels = ["base", "trend"]
        for j in range(1, self.harmonics + 1):
            labels.extend([f"season_sin{j}", f"season_cos{j}"])
        for prefix in ("beta", "decay", "half", "shape"):
            labels.extend(f"{prefix}[{name}]" for name in self.channels)
        labels.extend(f"control[{name}]" for name in self.control_names)
        labels.append("sigma")
        return labels

    def unpack(self, theta: list[float]) -> Parameters:
        """Unconstrained vector -> natural parameters."""
        if len(theta) != self._dim:
            raise ValueError(f"expected {self._dim} parameters, got {len(theta)}")
        at = self._offsets
        channels = self.n_channels
        return Parameters(
            base=theta[at["base"]],
            trend=theta[at["trend"]],
            season=list(theta[at["season"] : at["season"] + len(self._fourier)]),
            beta=[math.exp(theta[at["beta"] + c]) for c in range(channels)],
            decay=[_sigmoid(theta[at["decay"] + c]) for c in range(channels)],
            half=[math.exp(theta[at["half"] + c]) for c in range(channels)],
            shape=[math.exp(theta[at["shape"] + c]) for c in range(channels)],
            control=list(theta[at["control"] : at["control"] + self.n_controls]),
            sigma=math.exp(theta[at["sigma"]]),
        )

    def pack(self, params: Parameters) -> list[float]:
        """Natural parameters -> unconstrained vector. The exact inverse of ``unpack``."""
        theta = [0.0] * self._dim
        at = self._offsets
        theta[at["base"]] = params.base
        theta[at["trend"]] = params.trend
        for index, value in enumerate(params.season):
            theta[at["season"] + index] = value
        for c in range(self.n_channels):
            theta[at["beta"] + c] = math.log(params.beta[c])
            theta[at["decay"] + c] = _logit(params.decay[c])
            theta[at["half"] + c] = math.log(params.half[c])
            theta[at["shape"] + c] = math.log(params.shape[c])
        for index, value in enumerate(params.control):
            theta[at["control"] + index] = value
        theta[at["sigma"]] = math.log(params.sigma)
        return theta

    # -- mean function ------------------------------------------------------------------------

    def baseline(self, params: Parameters) -> list[float]:
        """Everything that is not media: intercept, trend, seasonality, controls."""
        out = []
        for t in range(self.weeks):
            value = params.base + params.trend * self._time[t]
            for index, series in enumerate(self._fourier):
                value += params.season[index] * series[t]
            for index, series in enumerate(self._controls):
                value += params.control[index] * series[t]
            out.append(value)
        return out

    def channel_contributions(self, params: Parameters) -> list[list[float]]:
        """``beta_c * Hill(adstock(x_c))`` per channel: the incremental sales attributed to media."""
        contributions = []
        for c in range(self.n_channels):
            adstocked = geometric_adstock(
                self._spend[c], params.decay[c], self.normalize_adstock
            )
            saturated = [
                hill_partials(value, params.half[c], params.shape[c])[0] for value in adstocked
            ]
            contributions.append([params.beta[c] * value for value in saturated])
        return contributions

    def mean(self, params: Parameters) -> list[float]:
        total = self.baseline(params)
        for series in self.channel_contributions(params):
            total = [a + b for a, b in zip(total, series)]
        return total

    def residuals(self, params: Parameters) -> list[float]:
        return [observed - fitted for observed, fitted in zip(self.data.y, self.mean(params))]

    # -- log posterior ------------------------------------------------------------------------

    def log_prior(self, theta: list[float]) -> float:
        params = self.unpack(theta)
        priors = self.priors
        at = self._offsets
        total = 0.0

        total += _normal_logpdf(params.base, priors.base_mean, priors.base_sd)
        total += _normal_logpdf(params.trend, 0.0, priors.trend_sd)
        for value in params.season:
            total += _normal_logpdf(value, 0.0, priors.season_sd)
        for value in params.control:
            total += _normal_logpdf(value, 0.0, priors.control_sd)

        for c in range(self.n_channels):
            # LogNormal prior + Jacobian == Normal prior on the unconstrained coordinate.
            total += _normal_logpdf(theta[at["beta"] + c], priors.beta_log_mean, priors.beta_log_sd)
            total += _normal_logpdf(theta[at["half"] + c], priors.half_log_mean, priors.half_log_sd)
            total += _normal_logpdf(
                theta[at["shape"] + c], priors.shape_log_mean, priors.shape_log_sd
            )
            # Beta prior on lambda, with the logit Jacobian folded into the exponents.
            decay = params.decay[c]
            total += (
                priors.decay_a * math.log(decay)
                + priors.decay_b * math.log1p(-decay)
                - _log_beta_function(priors.decay_a, priors.decay_b)
            )

        # HalfNormal prior on sigma, plus log|dsigma/ds| = s.
        log_sigma = theta[at["sigma"]]
        total += (
            -0.5 * (params.sigma / priors.sigma_scale) ** 2
            + log_sigma
            + 0.5 * math.log(2.0 / math.pi)
            - math.log(priors.sigma_scale)
        )
        return total

    def log_likelihood(self, theta: list[float]) -> float:
        params = self.unpack(theta)
        sigma = params.sigma
        residuals = self.residuals(params)
        total = -self.weeks * (math.log(sigma) + 0.5 * LOG_TWO_PI)
        total -= 0.5 * sum(residual * residual for residual in residuals) / (sigma * sigma)
        return total

    def log_posterior(self, theta: list[float]) -> float:
        try:
            return self.log_likelihood(theta) + self.log_prior(theta)
        except (ValueError, OverflowError):
            return -math.inf

    # -- the gradient -------------------------------------------------------------------------

    def gradient(self, theta: list[float]) -> list[float]:
        """Analytic gradient of the log-posterior in unconstrained coordinates.

        With ``r_t = y_t - mu_t`` the likelihood contributes ``dLL/dmu_t = r_t / sigma^2``, and each
        parameter picks up its own ``dmu_t/dparam``:

            base            1
            trend           t/(T-1)
            season_j        sin or cos term
            control_k       z_kt
            log beta_c      beta_c * s_ct                                   (chain rule on exp)
            logit lambda_c  beta_c * ds/du * da_t/dlambda * lambda(1-lambda)
            log kappa_c     beta_c * ds/dkappa * kappa_c
            log alpha_c     beta_c * ds/dalpha * alpha_c

        and the scale coordinate is ``dLL/dlog sigma = sum_t (r_t^2/sigma^2 - 1)``.
        """
        value, grad = self.log_posterior_and_gradient(theta)
        del value
        return grad

    def log_posterior_and_gradient(self, theta: list[float]) -> tuple[float, list[float]]:
        """Both at once, because the sampler always wants both and the work is shared."""
        try:
            params = self.unpack(theta)
        except ValueError:
            raise
        at = self._offsets
        priors = self.priors
        grad = [0.0] * self._dim

        # -- forward pass: mean, and the per-channel pieces needed for the backward pass
        fitted = self.baseline(params)
        adstocked: list[list[float]] = []
        adstock_derivative: list[list[float]] = []
        saturation: list[list[float]] = []
        d_saturation_du: list[list[float]] = []
        d_saturation_dhalf: list[list[float]] = []
        d_saturation_dshape: list[list[float]] = []

        for c in range(self.n_channels):
            values, derivative = geometric_adstock_gradient(
                self._spend[c], params.decay[c], self.normalize_adstock
            )
            adstocked.append(values)
            adstock_derivative.append(derivative)
            column = [hill_partials(value, params.half[c], params.shape[c]) for value in values]
            saturation.append([entry[0] for entry in column])
            d_saturation_du.append([entry[1] for entry in column])
            d_saturation_dhalf.append([entry[2] for entry in column])
            d_saturation_dshape.append([entry[3] for entry in column])
            beta = params.beta[c]
            for t in range(self.weeks):
                fitted[t] += beta * saturation[c][t]

        sigma = params.sigma
        precision = 1.0 / (sigma * sigma)
        residuals = [observed - value for observed, value in zip(self.data.y, fitted)]

        log_likelihood = -self.weeks * (math.log(sigma) + 0.5 * LOG_TWO_PI)
        log_likelihood -= 0.5 * precision * sum(r * r for r in residuals)

        # -- backward pass
        weighted = [residual * precision for residual in residuals]  # dLL/dmu_t

        grad[at["base"]] = sum(weighted)
        grad[at["trend"]] = sum(w * time for w, time in zip(weighted, self._time))
        for index, series in enumerate(self._fourier):
            grad[at["season"] + index] = sum(w * value for w, value in zip(weighted, series))
        for index, series in enumerate(self._controls):
            grad[at["control"] + index] = sum(w * value for w, value in zip(weighted, series))

        for c in range(self.n_channels):
            beta = params.beta[c]
            decay = params.decay[c]
            half = params.half[c]
            shape = params.shape[c]
            logistic_jacobian = decay * (1.0 - decay)

            beta_grad = 0.0
            decay_grad = 0.0
            half_grad = 0.0
            shape_grad = 0.0
            for t in range(self.weeks):
                w = weighted[t]
                beta_grad += w * saturation[c][t]
                decay_grad += w * d_saturation_du[c][t] * adstock_derivative[c][t]
                half_grad += w * d_saturation_dhalf[c][t]
                shape_grad += w * d_saturation_dshape[c][t]

            grad[at["beta"] + c] = beta_grad * beta  # dbeta/dlog beta = beta
            grad[at["decay"] + c] = decay_grad * beta * logistic_jacobian
            grad[at["half"] + c] = half_grad * beta * half
            grad[at["shape"] + c] = shape_grad * beta * shape

        grad[at["sigma"]] = sum(r * r for r in residuals) * precision - self.weeks

        # -- priors
        log_prior = 0.0

        log_prior += _normal_logpdf(params.base, priors.base_mean, priors.base_sd)
        grad[at["base"]] += -(params.base - priors.base_mean) / priors.base_sd**2

        log_prior += _normal_logpdf(params.trend, 0.0, priors.trend_sd)
        grad[at["trend"]] += -params.trend / priors.trend_sd**2

        for index, value in enumerate(params.season):
            log_prior += _normal_logpdf(value, 0.0, priors.season_sd)
            grad[at["season"] + index] += -value / priors.season_sd**2

        for index, value in enumerate(params.control):
            log_prior += _normal_logpdf(value, 0.0, priors.control_sd)
            grad[at["control"] + index] += -value / priors.control_sd**2

        for c in range(self.n_channels):
            for key, mean, sd in (
                ("beta", priors.beta_log_mean, priors.beta_log_sd),
                ("half", priors.half_log_mean, priors.half_log_sd),
                ("shape", priors.shape_log_mean, priors.shape_log_sd),
            ):
                coordinate = theta[at[key] + c]
                log_prior += _normal_logpdf(coordinate, mean, sd)
                grad[at[key] + c] += -(coordinate - mean) / sd**2

            decay = params.decay[c]
            log_prior += (
                priors.decay_a * math.log(decay)
                + priors.decay_b * math.log1p(-decay)
                - _log_beta_function(priors.decay_a, priors.decay_b)
            )
            # d/dl [a log lambda + b log(1-lambda)] with dlambda/dl = lambda(1-lambda)
            grad[at["decay"] + c] += priors.decay_a * (1.0 - decay) - priors.decay_b * decay

        log_prior += (
            -0.5 * (sigma / priors.sigma_scale) ** 2
            + theta[at["sigma"]]
            + 0.5 * math.log(2.0 / math.pi)
            - math.log(priors.sigma_scale)
        )
        grad[at["sigma"]] += 1.0 - (sigma / priors.sigma_scale) ** 2

        return log_likelihood + log_prior, grad

    # -- starting points ----------------------------------------------------------------------

    def initial_point(self, rng: random.Random, jitter: float = 0.5) -> list[float]:
        """A dispersed but sane starting point: prior medians with noise.

        Chains started from the same place cannot diagnose anything -- R-hat compares chains, so they
        have to begin apart.
        """
        priors = self.priors
        at = self._offsets
        theta = [0.0] * self._dim
        theta[at["base"]] = priors.base_mean + rng.gauss(0.0, jitter * priors.base_sd * 0.25)
        theta[at["trend"]] = rng.gauss(0.0, jitter * priors.trend_sd * 0.1)
        for index in range(len(self._fourier)):
            theta[at["season"] + index] = rng.gauss(0.0, jitter * priors.season_sd * 0.1)
        for index in range(self.n_controls):
            theta[at["control"] + index] = rng.gauss(0.0, jitter * priors.control_sd * 0.1)
        for c in range(self.n_channels):
            theta[at["beta"] + c] = priors.beta_log_mean + rng.gauss(0.0, jitter)
            theta[at["half"] + c] = priors.half_log_mean + rng.gauss(0.0, jitter * 0.5)
            theta[at["shape"] + c] = priors.shape_log_mean + rng.gauss(0.0, jitter * 0.3)
            mean_decay = priors.decay_a / (priors.decay_a + priors.decay_b)
            theta[at["decay"] + c] = _logit(mean_decay) + rng.gauss(0.0, jitter)
        theta[at["sigma"]] = math.log(priors.sigma_scale) + rng.gauss(0.0, jitter * 0.3)
        return theta

    def posterior_predictive(
        self, theta: list[float], rng: random.Random
    ) -> list[float]:
        """One replicated dataset from the fitted model: mean plus fresh noise."""
        params = self.unpack(theta)
        return [value + rng.gauss(0.0, params.sigma) for value in self.mean(params)]


def _normal_logpdf(value: float, mean: float, sd: float) -> float:
    if sd <= 0.0:
        raise ValueError("standard deviation must be positive")
    z = (value - mean) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * LOG_TWO_PI


def numerical_gradient(
    function, theta: list[float], step: float = 1e-5
) -> list[float]:
    """Central differences, for testing the analytic gradient. Never used in the sampler."""
    grad = []
    for index in range(len(theta)):
        forward = list(theta)
        backward = list(theta)
        scale = max(abs(theta[index]), 1.0)
        h = step * scale
        forward[index] += h
        backward[index] -= h
        grad.append((function(forward) - function(backward)) / (2.0 * h))
    return grad


def gradient_error(model: MMM, theta: list[float], step: float = 1e-5) -> tuple[float, int]:
    """Largest relative discrepancy between the analytic and numerical gradients, and where.

    Reported as a relative error so that coordinates with wildly different scales -- ``base`` is in
    revenue units, ``log alpha`` is dimensionless -- are judged on the same footing.
    """
    analytic = model.gradient(theta)
    numeric = numerical_gradient(model.log_posterior, theta, step)
    worst = 0.0
    where = 0
    for index, (a, n) in enumerate(zip(analytic, numeric)):
        scale = max(abs(a), abs(n), 1e-8)
        error = abs(a - n) / scale
        if error > worst:
            worst = error
            where = index
    return worst, where
