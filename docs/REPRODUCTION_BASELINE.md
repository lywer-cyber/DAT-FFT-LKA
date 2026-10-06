# DAT + FFT + LKA reproduction baseline

This repository contains the code and configuration used for the matched
DAT+FFT+LKA experiment. Dataset files, checkpoints, TensorBoard event files,
and generated images are intentionally excluded from Git; they remain on the
experiment server and are referenced by path in the experiment handoff.

## Matched training configuration

- model: DAT with the LKA replacement in residual groups `[4, 5]`
- data root: `/root/autodl-tmp/real_hologram_x4`
- protocol: `lab_like`, target directory `hologram_clean`
- input/target: 64x64 to 256x256, scale 4
- batch size: 2; seed: 2026
- optimizer: Adam, learning rate `1e-4`, betas `(0.9, 0.99)`
- AMP: enabled; TF32: enabled; checkpointing: disabled
- objective: L1 plus `HologramSpectrumLoss` with weight `0.01`
- EMA: `0.999`; total iterations: `25,500`
- validation/checkpoint interval: every `850` iterations
- scheduler milestones: `17000, 21250, 23800`, gamma `0.5`

Run:

```bash
PYTHONPATH=. python basicsr/train.py \
  -opt options/Train/train_DAT_holo_x4_fft_lka_matched_25500.yml
```

The full YAML is also copied under `configs/` for a stable, easy-to-find
reference. The `options/` copy is the file consumed by the training command.

## Baseline validation reference

The matched baseline log reported these values on the shared validation split:

| Iter | PSNR | SSIM | MAE | RMSE | Spectral corr. | HF spectral MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 850 | 33.8265 | 0.9471 | 2.0790 | 5.2022 | 0.4415 | 4.4037 |
| 1700 | 34.3726 | 0.9513 | 1.9259 | 4.8841 | 0.4601 | 4.0785 |

The complete original log is preserved on the server at:

`/root/autodl-tmp/strict_DAT_reference/DAT-main/experiments/matched_DAT_FFT_LKA_x4_25500/train_matched_DAT_FFT_LKA_x4_25500_20261006_194202.log`

These are validation results, not a claim of independent test-set performance.
