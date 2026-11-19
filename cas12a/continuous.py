"""Differentiable extension of the training features for Cas12a IG (#195).

ACGT one-hot inputs reproduce features.extract_seq_features. For soft inputs
this defines a polynomial, not a DNA sequence or a probability model. Keep the
tensor graph intact: argmax/count/NumPy would discard the feature chain rule.
"""

from cas12a.features import PROTOSPACER_SLICE, SEED_LENGTH, SEQUENCE_LENGTH


def derive_features(x):
    """Torch tensor [..., 34, 4] -> [..., 354], preserving dtype and gradients.

    The layout is GC total, GC distal tail, then 22 adjacent pairs in position /
    first ACGT / second ACGT order. D(0)=0. GC is linear; pairs are quadratic.
    Import torch only when called so reading this module cannot load a runtime
    in the UI process.
    """
    import torch

    if tuple(x.shape[-2:]) != (SEQUENCE_LENGTH, 4):
        raise ValueError(f"Cas12a requires [..., 34, 4], got {tuple(x.shape)}")
    p = x[..., PROTOSPACER_SLICE, :]
    gc = p[..., 1:3].sum(dim=(-2, -1)) / p.shape[-2]
    tail_gc = p[..., -SEED_LENGTH:, 1:3].sum(dim=(-2, -1)) / SEED_LENGTH
    pairs = p[..., :-1, :, None] * p[..., 1:, None, :]
    return torch.cat((gc[..., None], tail_gc[..., None], pairs.flatten(-3)), dim=-1)
