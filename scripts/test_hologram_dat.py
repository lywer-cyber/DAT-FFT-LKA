"""Smoke test for DAT with the prepared hologram-intensity dataset contract."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.archs.dat_arch import DAT
from basicsr.data.hologram_intensity_dataset import HologramIntensityDataset
from basicsr.losses.losses import HologramSpectrumLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--scale', type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    dataset = HologramIntensityDataset({
        'phase': 'val',
        'dataroot': args.data_root,
        'scale': args.scale,
        'dataset_protocol': 'lab_like',
        'target_dir': 'hologram_clean',
    })
    sample = dataset[0]
    crop = 16
    lq = sample['lq'][..., :crop, :crop].unsqueeze(0)
    gt = sample['gt'][..., :crop * args.scale, :crop * args.scale].unsqueeze(0)
    baseline = F.interpolate(lq, scale_factor=args.scale, mode='bicubic', align_corners=False)
    expected = (1, 1, lq.shape[-2] * args.scale, lq.shape[-1] * args.scale)
    model = DAT(
        upscale=args.scale,
        in_chans=1,
        img_size=lq.shape[-1],
        img_range=1.0,
        split_size=[4, 8],
        depth=[1, 1],
        embed_dim=24,
        num_heads=[2, 2],
        expansion_factor=2,
        use_chk=False,
        upsampler='pixelshuffle',
        use_bicubic_residual=True,
        zero_init_residual_tail=True,
    ).eval()
    output = model(lq)
    assert tuple(output.shape) == expected, (tuple(output.shape), expected)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, baseline, rtol=0, atol=1e-6)
    spectrum_loss = HologramSpectrumLoss(
        loss_weight=0.01, min_radius=0.15, max_radius=0.5,
        correlation_weight=0.05,
    )(output, gt)
    assert torch.isfinite(spectrum_loss)
    spectrum_loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    print('pairs:', len(dataset))
    print('LR:', tuple(lq.shape), 'HR:', tuple(sample['gt'].shape))
    print('DAT output:', tuple(output.shape))
    print('DAT hologram smoke test passed')


if __name__ == '__main__':
    main()
