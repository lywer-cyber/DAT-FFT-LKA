# Experiment: shallow-guided multi-scale LKA

Branch: `experiment/fringe-guided-ms-shallow`

This branch is an unmerged hypothesis test against `main`. It computes a
shared structure-tensor guide from `DAT.conv_first` output and uses the guide
to gate horizontal, vertical, and dilation-rate `[1, 2, 3]` depthwise paths in
the LKA replacement. The original 7x7, dilation-3 path remains the base path;
the new correction starts at residual scale `0.01`. Only residual groups `[4,
5]` are replaced.

The guide's frequency channel is a local gradient-energy proxy. It is not an
explicit local FFT estimate; this distinction is intentional and recorded for
the ablation.

The run uses the same data/split/seed/AMP/TF32/loss/optimizer/scheduler/EMA and
25,500-iteration budget as the main baseline, with no resume checkpoint.

At iter 850, the running server result was:

| PSNR | SSIM | MAE | RMSE | Spectral corr. | HF spectral MAE |
|---:|---:|---:|---:|---:|---:|
| 33.7861 | 0.9471 | 2.0838 | 5.2266 | 0.4479 | 4.4616 |

Main baseline at the same iter: `33.8265 / 0.9471 / 2.0790 / 5.2022 /
0.4415 / 4.4037`. This is not a final conclusion; the experiment must finish
and be evaluated at matched checkpoints before considering a merge.

Server run directory:

`/root/autodl-tmp/strict_DAT_reference/DAT-main-fringe-guided-multiscale-shallow-v2/`
