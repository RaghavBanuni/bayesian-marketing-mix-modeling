"""Media transformations, with the derivatives the sampler needs.

Two effects turn spend into response, and both are non-linear:

**Carryover.** Advertising seen this week still sells next week. Geometric adstock is the recursion

    A_t = x_t + lambda * A_{t-1},        A_{-1} = 0,        0 <= lambda < 1

which is an IIR filter with weights lambda^k. Left like that, total adstocked volume is
sum(x) / (1 - lambda), so a larger lambda inflates the input to saturation and the coefficient beta
shrinks to compensate: the posterior develops a banana-shaped beta-lambda ridge and the sampler has to
work for its living. Normalising by (1 - lambda) keeps total volume equal to total spend and removes
most of that correlation, which is why it is the default here.

**Diminishing returns.** The tenth impression is worth less than the first. The Hill function

    s(u) = u^alpha / (u^alpha + kappa^alpha)

is the usual choice because its parameters mean something: kappa is the half-saturation point in the
same units as adstocked spend (s(kappa) = 0.5 exactly), and alpha controls the shape. alpha <= 1 gives
a concave curve saturating from the origin; alpha > 1 gives an S-curve with a convex toe, which is a
real claim about the market -- small budgets do nothing -- and one that budget optimisers must respect,
because a convex region means the optimal allocation can be a corner rather than an interior point.

Every function that the log-posterior touches also returns its partial derivatives. They are derived
below and checked against central finite differences in ``tests/test_transforms.py``; nothing here is
assumed to be right because it looks right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EPS = 1e-12


# ---------------------------------------------------------------------------------------------
# carryover
# ---------------------------------------------------------------------------------------------


def geometric_adstock(spend: list[float], decay: float, normalize: bool = True) -> list[float]:
    """Geometric carryover: ``A_t = x_t + decay * A_{t-1}``.

    With ``normalize`` the series is multiplied by ``(1 - decay)`` so that total adstocked volume
    equals total spend, which keeps ``beta`` on a fixed scale as ``decay`` moves.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must lie in [0, 1), got {decay}")
    out: list[float] = []
    carried = 0.0
    for value in spend:
        carried = value + decay * carried
        out.append(carried)
    if normalize:
        scale = 1.0 - decay
        out = [value * scale for value in out]
    return out


def geometric_adstock_gradient(
    spend: list[float], decay: float, normalize: bool = True
) -> tuple[list[float], list[float]]:
    """The adstocked series and its derivative with respect to ``decay``.

    Differentiating the recursion gives another recursion:

        dA_t/dlambda = A_{t-1} + lambda * dA_{t-1}/dlambda

    and with normalisation ``a_t = (1 - lambda) A_t`` the product rule adds the ``-A_t`` term:

        da_t/dlambda = -A_t + (1 - lambda) * dA_t/dlambda
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must lie in [0, 1), got {decay}")
    values: list[float] = []
    derivatives: list[float] = []
    carried = 0.0
    carried_derivative = 0.0
    previous = 0.0
    for value in spend:
        carried = value + decay * previous
        carried_derivative = previous + decay * carried_derivative
        values.append(carried)
        derivatives.append(carried_derivative)
        previous = carried

    if normalize:
        scale = 1.0 - decay
        derivatives = [
            -raw + scale * derivative for raw, derivative in zip(values, derivatives)
        ]
        values = [raw * scale for raw in values]
    return values, derivatives


def delayed_adstock(
    spend: list[float], decay: float, peak: float, length: int = 12
) -> list[float]:
    """Carryover that peaks after the exposure rather than at it.

    Weights ``w_k = decay ** ((k - peak) ** 2)``, normalised to sum to one. Television and brand
    campaigns behave like this; the effect builds for a week or two before it fades.

    This is deliberately **not** differentiated and **not** used by the model. It exists so the data
    generator can produce data the model is wrong about, and ``python -m mmm misspecified`` can show
    what that costs. A model that can only be tested against data it generated itself has not been
    tested.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay must lie in (0, 1), got {decay}")
    if peak < 0.0:
        raise ValueError(f"peak must be non-negative, got {peak}")
    if length < 1:
        raise ValueError("length must be at least 1")
    weights = [decay ** ((k - peak) ** 2) for k in range(length)]
    total = sum(weights)
    weights = [weight / total for weight in weights]

    out: list[float] = []
    for t in range(len(spend)):
        accumulated = 0.0
        for k, weight in enumerate(weights):
            if t - k >= 0:
                accumulated += weight * spend[t - k]
        out.append(accumulated)
    return out


@dataclass(frozen=True)
class Carryover:
    """How a decay rate reads in weeks, which is how a marketer will ask about it."""

    decay: float
    half_life: float
    weeks_to_90_percent: float
    mean_lag: float

    def summary(self) -> str:
        return (
            f"decay {self.decay:.3f} -> half-life {self.half_life:.2f} weeks, "
            f"90% of the effect within {self.weeks_to_90_percent:.1f} weeks, "
            f"mean lag {self.mean_lag:.2f} weeks"
        )


def carryover_summary(decay: float) -> Carryover:
    """Half-life ``ln(0.5)/ln(lambda)``, the 90% mass point, and the mean lag ``lambda/(1-lambda)``."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must lie in [0, 1), got {decay}")
    if decay <= EPS:
        return Carryover(decay, 0.0, 0.0, 0.0)
    half_life = math.log(0.5) / math.log(decay)
    weeks_90 = math.log(0.1) / math.log(decay)
    return Carryover(decay, half_life, weeks_90, decay / (1.0 - decay))


# ---------------------------------------------------------------------------------------------
# diminishing returns
# ---------------------------------------------------------------------------------------------


def hill(u: float, half: float, shape: float) -> float:
    """``u^alpha / (u^alpha + kappa^alpha)``, equal to 0.5 at ``u = kappa`` by construction."""
    if half <= 0.0:
        raise ValueError(f"half-saturation must be positive, got {half}")
    if shape <= 0.0:
        raise ValueError(f"shape must be positive, got {shape}")
    if u <= EPS:
        return 0.0
    powered = u**shape
    reference = half**shape
    return powered / (powered + reference)


def hill_partials(u: float, half: float, shape: float) -> tuple[float, float, float, float]:
    """Return ``(s, ds/du, ds/dkappa, ds/dalpha)``.

    With ``U = u^alpha``, ``K = kappa^alpha`` and ``D = U + K``:

        s          = U / D
        ds/du      = alpha * u^(alpha-1) * K / D^2
        ds/dkappa  = -alpha * kappa^(alpha-1) * U / D^2
        ds/dalpha  = U * K * (ln u - ln kappa) / D^2

    At ``u = 0`` the function is zero and every partial is taken as zero. That is the correct limit for
    ``alpha > 1``; for ``alpha < 1`` the true ``ds/du`` diverges, and zero is a deliberate choice that
    keeps the sampler finite on weeks with no spend. Weeks with zero spend carry no information about
    the slope at zero, so nothing is lost, but it is an approximation and it is stated rather than
    hidden.
    """
    if half <= 0.0:
        raise ValueError(f"half-saturation must be positive, got {half}")
    if shape <= 0.0:
        raise ValueError(f"shape must be positive, got {shape}")
    if u <= EPS:
        return 0.0, 0.0, 0.0, 0.0

    powered = u**shape
    reference = half**shape
    denominator = powered + reference
    squared = denominator * denominator

    value = powered / denominator
    d_u = shape * (u ** (shape - 1.0)) * reference / squared
    d_half = -shape * (half ** (shape - 1.0)) * powered / squared
    d_shape = powered * reference * (math.log(u) - math.log(half)) / squared
    return value, d_u, d_half, d_shape


def hill_curve(spend: list[float], half: float, shape: float) -> list[float]:
    return [hill(value, half, shape) for value in spend]


def logistic_saturation(u: float, rate: float) -> tuple[float, float, float]:
    """``(1 - exp(-rate * u))``, and its partials in ``u`` and ``rate``.

    Always concave, so it cannot express an S-curve. Offered as the conservative alternative: if a
    budget recommendation survives both this and Hill, the recommendation is about the data rather
    than about the functional form.
    """
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate}")
    if u < 0.0:
        raise ValueError(f"spend must be non-negative, got {u}")
    decayed = math.exp(-rate * u)
    return 1.0 - decayed, rate * decayed, u * decayed


def marginal_response(u: float, half: float, shape: float, beta: float) -> float:
    """``beta * ds/du``: the extra response from one more unit of adstocked spend at ``u``."""
    _, d_u, _, _ = hill_partials(u, half, shape)
    return beta * d_u


def is_convex_region(u: float, half: float, shape: float) -> bool:
    """True where the response curve is still convex (the S-curve toe).

    The inflection point of the Hill function is at ``u* = kappa * ((alpha - 1)/(alpha + 1))^(1/alpha)``
    for ``alpha > 1``, and does not exist for ``alpha <= 1``. Below it, spending more has increasing
    returns -- which is exactly where a solver that assumes concavity gives the wrong answer.
    """
    if shape <= 1.0:
        return False
    inflection = half * ((shape - 1.0) / (shape + 1.0)) ** (1.0 / shape)
    return u < inflection


def inflection_point(half: float, shape: float) -> float | None:
    """Where the S-curve stops accelerating, or ``None`` when the curve is concave everywhere."""
    if shape <= 1.0:
        return None
    return half * ((shape - 1.0) / (shape + 1.0)) ** (1.0 / shape)
