"""Local wiring probe for G1: Arc's 7B architecture, cut to four layers with random weights, on CPU.

Checks that src/evo2size/cpu_patch.py lets Vortex build and run a forward pass without CUDA,
through one Hyena block of each kind and one attention block. It says nothing about accuracy,
which T1a tests with the real weights. No cloud spend.

The flash-attention CUDA extension and Triton exist in the pod image but not on macOS, so they are
stubbed here only so Vortex can be imported; any call into them raises.

  uv venv -p 3.12 /tmp/probe && uv pip install -p /tmp/probe/bin/python torch==2.7.0 pyyaml tqdm \
      einops huggingface_hub numpy rich biopython pandas && \
  uv pip install -p /tmp/probe/bin/python --no-deps vtx==1.1.0 evo2==0.6.0
  /tmp/probe/bin/python scripts/cpu_wiring_probe.py src
"""
import sys, types, pkgutil, yaml, time
sys.path.insert(0, sys.argv[1])  # study src/

def stub(name, **attrs):
    module = types.ModuleType(name); module.__dict__.update(attrs); sys.modules[name] = module; return module
def unavailable(*a, **k): raise RuntimeError("CUDA-only kernel called on CPU")
def lazy(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return unavailable
stub("flash_attn_2_cuda", __getattr__=lazy)
tl = stub("triton.language", constexpr=int, __getattr__=lazy)
stub("triton", jit=lambda f=None, **k: (f if f else (lambda g: g)), language=tl, cdiv=lambda a, b: -(-a // b),
     autotune=lambda *a, **k: (lambda g: g), Config=lambda *a, **k: None, heuristics=lambda *a, **k: (lambda g: g),
     __getattr__=lazy)

import torch
from evo2size import cpu_patch
cpu_patch.apply()
from vortex.model.model import StripedHyena

config = yaml.safe_load(pkgutil.get_data("evo2", "configs/evo2-7b-1m.yml"))
attn = config["attn_layer_idxs"][0]
keep = sorted({0, 1, 2, attn})
remap = {old: new for new, old in enumerate(keep)}
config["num_layers"] = len(keep)
for key in [k for k in config if k.endswith("layer_idxs")]:
    config[key] = [remap[i] for i in config[key] if i in remap]
print("layers kept", keep, "attention at", config["attn_layer_idxs"])
from evo2.models import dotdict as DD
model = StripedHyena(DD(config, Loader=yaml.FullLoader))
model.to(torch.float32).eval()
torch.manual_seed(0)
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.02)
        elif "scale" in name or "norm" in name:
            p.fill_(1.0)
        else:
            p.normal_(0, 0.02)
    for name, b in model.named_buffers():
        if b.is_floating_point() and not torch.isfinite(b).all():
            b.zero_()
assert not model.config.use_flash_attn
x = torch.randint(0, 512, (1, 256))
start = time.perf_counter()
with torch.inference_mode():
    logits, _ = model.forward(x)
print("logits", tuple(logits.shape), logits.dtype, "finite", bool(torch.isfinite(logits).all()),
      f"{time.perf_counter() - start:.1f}s")
