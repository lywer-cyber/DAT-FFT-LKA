from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .hologram_metrics import (
    calculate_hologram_hf_mae,
    calculate_hologram_spectral_corr,
    calculate_mae,
    calculate_rmse,
)
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = [
    'calculate_psnr', 'calculate_ssim', 'calculate_mae', 'calculate_rmse',
    'calculate_hologram_spectral_corr', 'calculate_hologram_hf_mae'
]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
