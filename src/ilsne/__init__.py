"""ilSNe MST inference utilities."""

from .diagnostics import diagnose_lambda_bias
from .magnification import LensDataset, LensedSN, infer_lambda_for_lens
from .microlens import (
    load_sigma_micro_from_csv,
    sigma_micro_constant,
    sigma_micro_parametric,
)
from .pivot import (
    FlatLCDM,
    m_sn_from_pivot,
    predict_unlensed_mag,
    profile_m_p_gaussian_prior,
)
from .sensitivity import sweep_sigma_micro_params, sweep_sigma_micro_scale
from .stacking import (
    exact_convolution_stack,
    gaussian_precision_stack,
    gaussian_stack_lambda,
    product_stack_common_lambda,
)
from .viz import plot_lambda_posterior
from .los_posterior import posterior_lambda_intrinsic_from_total
from .los import sample_kappa_ext

__all__ = [
    "FlatLCDM",
    "m_sn_from_pivot",
    "predict_unlensed_mag",
    "profile_m_p_gaussian_prior",
    "LensedSN",
    "LensDataset",
    "infer_lambda_for_lens",
    "sigma_micro_constant",
    "load_sigma_micro_from_csv",
    "sigma_micro_parametric",
    "gaussian_stack_lambda",
    "gaussian_precision_stack",
    "product_stack_common_lambda",
    "exact_convolution_stack",
    "plot_lambda_posterior",
    "posterior_lambda_intrinsic_from_total",
    "sample_kappa_ext",
    "diagnose_lambda_bias",
    "sweep_sigma_micro_params",
    "sweep_sigma_micro_scale",
]

try:  # Stage 4 placeholders; optional hierArc dependency
    from .hierarc_glue import PantheonPivotBlock, TDMagBlock  # noqa: F401

    __all__.extend(["PantheonPivotBlock", "TDMagBlock"])
except ImportError:  # pragma: no cover - optional dependency
    pass
