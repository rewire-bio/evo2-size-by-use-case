"""Run Arc's evo2 0.6.0 / vtx 1.1.0 code on a CPU, with the smallest changes that make it start.

Protocol G1 allows exactly these changes, and nothing in the model arithmetic beyond them:

1. `torch.cuda.device(...)` contexts, which Vortex opens around block construction and the input
   projections, become no-ops for non-CUDA devices. Without this the loader fails with
   "Expected a cuda device, but got: cpu" (ArcInstitute/evo2 issue 67).
2. The model config is loaded with `use_flash_attn=False`, so attention uses Vortex's own PyTorch
   `scaled_dot_product_attention` path, and `use_fp8_input_projections=False`, the bfloat16 path
   Arc documents for 7B without Transformer Engine.
3. Rotary position embedding uses Vortex's own PyTorch reference function
   (`vortex.model.rotary.apply_rotary_emb_torch`) instead of its Triton kernel, for the inference
   case only (fixed-length batch, integer offset). Any other case raises, so the run fails rather
   than computing something different.
4. After loading, every parameter and buffer is cast to float32.

Arc's `score_sequences` defaults to device "cuda:0"; workloads.py passes the device explicitly.
"""
from __future__ import annotations

import contextlib

import torch

_original_cuda_device = torch.cuda.device
CONFIG_OVERRIDES = {"use_flash_attn": False, "use_fp8_input_projections": False}


def _device_aware_cuda_device(device):
    """torch.cuda.device for CUDA devices; a no-op context for anything else."""
    kind = device.type if isinstance(device, torch.device) else str(device)
    if not kind.startswith("cuda"):
        return contextlib.nullcontext()
    return _original_cuda_device(device)


def _apply_rotary_torch(x, cos, sin, seqlen_offsets=0, cu_seqlens=None, max_seqlen=None, interleaved=False,
                       inplace=False, conjugate=False):
    """Drop-in for vortex.ops.embedding.rotary.apply_rotary using Vortex's PyTorch reference."""
    from vortex.model.rotary import apply_rotary_emb_torch

    if cu_seqlens is not None or not isinstance(seqlen_offsets, int) or conjugate:
        raise NotImplementedError("CPU rotary path supports fixed-length inference only")
    seqlen = x.shape[1]
    cos = cos[seqlen_offsets:seqlen_offsets + seqlen]
    sin = sin[seqlen_offsets:seqlen_offsets + seqlen]
    out = apply_rotary_emb_torch(x, cos, sin, interleaved)
    if inplace:
        x.copy_(out)
        return x
    return out


def apply() -> None:
    """Install the CPU changes. Call before constructing an Evo2 model."""
    torch.cuda.device = _device_aware_cuda_device

    from vortex.model import model as vortex_model
    from vortex.model import rotary as vortex_rotary

    vortex_rotary.apply_rotary = _apply_rotary_torch

    original_init = vortex_model.StripedHyena.__init__
    if getattr(original_init, "_evo2size_cpu", False):
        return

    def init_with_overrides(self, config, *args, **kwargs):
        for key, value in CONFIG_OVERRIDES.items():
            config[key] = value
        original_init(self, config, *args, **kwargs)

    init_with_overrides._evo2size_cpu = True
    vortex_model.StripedHyena.__init__ = init_with_overrides


def to_float32(evo2_model) -> None:
    """Cast the loaded Vortex model to float32 for CPU arithmetic."""
    evo2_model.model.to(torch.float32)
