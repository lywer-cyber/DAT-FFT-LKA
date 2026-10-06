import torch

from basicsr.archs.dat_arch import (
    Adaptive_Channel_Attention,
    DAT,
    LargeKernelSpatialMixer,
)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    common = dict(
        upscale=4,
        in_chans=1,
        img_size=64,
        img_range=1.,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=4,
        resi_connection='1conv',
        upsampler='pixelshuffle',
        use_bicubic_residual=True,
        zero_init_residual_tail=True,
    )

    baseline = DAT(**common)
    model = DAT(
        **common,
        spatial_mixer='large_kernel',
        spatial_mixer_groups=[4, 5],
        lka_kernel_size=7,
        lka_dilation=3,
    )
    replaced = sum(isinstance(module, LargeKernelSpatialMixer)
                   for module in model.modules())
    channel_blocks = sum(isinstance(module, Adaptive_Channel_Attention)
                         for module in model.modules())
    if replaced != 6:
        raise RuntimeError('expected 6 replaced DSTBs, found {}'.format(replaced))
    if channel_blocks != 18:
        raise RuntimeError('expected all 18 DCTBs to remain, found {}'.format(channel_blocks))

    # A compact runtime forward/backward check. The full model above verifies
    # the requested six replacements without requiring a full-size training step.
    miniature = DAT(
        upscale=4,
        in_chans=1,
        img_size=8,
        img_range=1.,
        split_size=[4, 4],
        depth=[2, 2],
        embed_dim=24,
        num_heads=[4, 4],
        expansion_factor=2,
        resi_connection='1conv',
        upsampler='pixelshuffle',
        spatial_mixer='large_kernel',
        spatial_mixer_groups=[0, 1],
        lka_kernel_size=7,
        lka_dilation=3,
    )
    sample = torch.randn(1, 1, 8, 8, requires_grad=True)
    output = miniature(sample)
    if output.shape != (1, 1, 32, 32):
        raise RuntimeError('unexpected output shape: {}'.format(output.shape))
    output.mean().backward()
    if sample.grad is None or not torch.isfinite(sample.grad).all():
        raise RuntimeError('LKA backward pass produced invalid gradients')

    print('smoke test passed')
    print('replaced DSTBs: {}'.format(replaced))
    print('preserved DCTBs: {}'.format(channel_blocks))
    print('baseline parameters: {:,}'.format(count_parameters(baseline)))
    print('LKA parameters: {:,}'.format(count_parameters(model)))
    print('miniature output shape: {}'.format(tuple(output.shape)))


if __name__ == '__main__':
    main()
