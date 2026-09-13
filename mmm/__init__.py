"""Bayesian marketing mix modelling in pure Python.

The pipeline, in the order it runs:

    data      -> synthetic weekly sales and spend, with ground truth and three pathological regimes
    model     -> adstock, saturation, priors, log-posterior, analytic gradient
    nuts      -> No-U-Turn sampler with dual averaging and a diagonal metric
    diagnostics -> rank-normalised split R-hat, effective sample size, Monte Carlo error
    validate  -> posterior predictive checks, parameter recovery, holdout forecasting
    decision  -> contributions, ROAS, marginal ROAS, budget allocation with its uncertainty

No third-party packages are used anywhere, including in the tests.
"""

from .decision import (
    Allocation,
    ChannelResult,
    allocate,
    channel_results,
    decomposition,
    incremental_contribution,
    marginal_roas,
    posterior_allocation,
)
from .diagnostics import (
    ParameterSummary,
    effective_sample_size,
    mcse_mean,
    quantile,
    rhat,
    split_rhat,
    summarise_columns,
    summary_table,
)
from .model import (
    MMM,
    Dataset,
    Parameters,
    Priors,
    gradient_error,
    numerical_gradient,
)
from .nuts import Chain, NUTSConfig, Posterior, sample
from .transforms import (
    carryover_summary,
    delayed_adstock,
    geometric_adstock,
    geometric_adstock_gradient,
    hill,
    hill_partials,
    inflection_point,
)
from .validate import (
    Forecast,
    forecast_evaluation,
    natural_columns,
    parameter_recovery,
    posterior_predictive_check,
    train_test_split,
)

__version__ = "1.0.0"

__all__ = [
    "MMM",
    "Allocation",
    "Chain",
    "ChannelResult",
    "Dataset",
    "Forecast",
    "NUTSConfig",
    "ParameterSummary",
    "Parameters",
    "Posterior",
    "Priors",
    "allocate",
    "carryover_summary",
    "channel_results",
    "decomposition",
    "delayed_adstock",
    "effective_sample_size",
    "forecast_evaluation",
    "geometric_adstock",
    "geometric_adstock_gradient",
    "gradient_error",
    "hill",
    "hill_partials",
    "incremental_contribution",
    "inflection_point",
    "marginal_roas",
    "mcse_mean",
    "natural_columns",
    "numerical_gradient",
    "parameter_recovery",
    "posterior_allocation",
    "posterior_predictive_check",
    "quantile",
    "rhat",
    "sample",
    "split_rhat",
    "summarise_columns",
    "summary_table",
    "train_test_split",
]
