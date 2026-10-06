"""Single-channel hologram-intensity metrics for the unified SR comparison."""

import numpy as np

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


def _prepare(img, img2, crop_border):
    if img.shape != img2.shape:
        raise ValueError('prediction and target shapes do not match')
    img = reorder_image(img, input_order='HWC').astype(np.float64)
    img2 = reorder_image(img2, input_order='HWC').astype(np.float64)
    if crop_border:
        if min(img.shape[:2]) <= 2 * int(crop_border):
            raise ValueError('crop_border is too large for the hologram image')
        img = img[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    return img, img2


@METRIC_REGISTRY.register()
def calculate_mae(img, img2, crop_border=0, **kwargs):
    prediction, target = _prepare(img, img2, crop_border)
    return float(np.abs(prediction - target).mean())


@METRIC_REGISTRY.register()
def calculate_rmse(img, img2, crop_border=0, **kwargs):
    prediction, target = _prepare(img, img2, crop_border)
    return float(np.sqrt(np.square(prediction - target).mean()))


def _spectral_terms(img, img2, crop_border, min_radius, max_radius):
    prediction, target = _prepare(img, img2, crop_border)
    prediction = prediction.mean(axis=2)
    target = target.mean(axis=2)
    height, width = prediction.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    pred_log = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(prediction * window))))
    target_log = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(target * window))))
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    band = (radius >= float(min_radius)) & (radius <= float(max_radius))
    if not np.any(band):
        raise ValueError('configured Fourier band is empty')
    return pred_log[band], target_log[band]


@METRIC_REGISTRY.register()
def calculate_hologram_spectral_corr(img, img2, crop_border=0,
                                     min_radius=0.15, max_radius=0.5, **kwargs):
    prediction, target = _spectral_terms(img, img2, crop_border, min_radius, max_radius)
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = np.sqrt(np.square(prediction).sum() * np.square(target).sum())
    return float(np.dot(prediction, target) / max(float(denominator), 1e-12))


@METRIC_REGISTRY.register()
def calculate_hologram_hf_mae(img, img2, crop_border=0,
                              min_radius=0.15, max_radius=0.5, **kwargs):
    prediction, target = _spectral_terms(img, img2, crop_border, min_radius, max_radius)
    return float(np.abs(prediction - target).mean())
