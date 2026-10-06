# Experiment branch policy

`main` is the reproducible DAT+FFT+LKA baseline. Every architecture or loss
change is developed on a separate branch named `experiment/<short-name>` and
uses a unique experiment output directory.

Before merging a branch, compare it against the baseline at the same split,
seed, and iteration checkpoints. Report PSNR/SSIM, MAE/RMSE, spectral
correlation, and high-frequency spectral MAE. A branch is merged only when the
change is supported by the predeclared metrics and the run is complete. An
unsupported or harmful branch is kept as an auditable remote branch (or
closed) and is not merged into `main`.

Never commit dataset contents, passwords, checkpoints, or generated logs with
credentials. Put reproducibility information and server paths in `docs/`.
