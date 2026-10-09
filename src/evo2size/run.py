"""Entry point on the pod.

    python -m evo2size.run --device cuda --tasks forward:evo2_7b,brca1:evo2_7b --out /work/out

Task kinds: forward (Arc's forward-pass test), brca1 (3,893 BRCA1 SNVs), brca1_subsample (the
60-SNV CPU subsample) and indels (ClinVar noncoding non-SNVs). Tasks are grouped by model so each
checkpoint is loaded once, in the order the models first appear.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import platform
import resource
import subprocess
import traceback
from pathlib import Path

import torch

from . import workloads as W
from .telemetry import GpuSampler, Timings

KINDS = ("forward", "brca1", "brca1_subsample", "indels")


def parse_tasks(text: str) -> list[tuple[str, str]]:
    tasks = []
    for item in text.split(","):
        kind, model = item.split(":")
        if kind not in KINDS or model not in W.pins.WEIGHTS:
            raise ValueError(f"Unknown task {item}")
        tasks.append((kind, model))
    return tasks


def environment(device: str) -> dict:
    env = {
        "platform_id": os.environ.get("PLATFORM_ID"),
        "image": os.environ.get("IMAGE_REF"),
        "code_sha256": os.environ.get("CODE_SHA256"),
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None,
        "torch_threads": torch.get_num_threads(),
    }
    if device.startswith("cuda"):
        query = "name,driver_version,memory.total,compute_cap"
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip()
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["torch_cuda"] = torch.version.cuda
    try:
        import transformer_engine  # noqa: F401
        env["transformer_engine_installed"] = True
    except ImportError:
        env["transformer_engine_installed"] = False
    return env


def run_task(kind: str, name: str, model, args, timings: Timings) -> dict:
    data, out = args.data, args.out
    with timings.measure(f"{kind}_{name}_s"):
        if kind == "forward":
            return W.forward_test(model, name, data / "prompts.csv", args.device)
        if kind == "brca1":
            pairs = W.brca1_pairs(data / "GRCh37.p13_chr17.fna.gz", data / "brca1_variants.tsv")
        elif kind == "brca1_subsample":
            pairs = W.brca1_pairs(data / "GRCh37.p13_chr17.fna.gz", data / "brca1_cpu_subsample.tsv")
        else:
            pairs = W.prepared_pairs(data / "clinvar_noncoding_nonsnv_windows.tsv.gz")
        return W.score_pairs(model, pairs, out / f"{kind}_{name}_scores.tsv", args.device, f"{kind} {name}",
                             args.time_limit_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-limit-s", type=float, default=None,
                        help="Stop variant scoring after this many seconds (CPU tier)")
    parser.add_argument("--data", type=Path, default=Path("/work/data"))
    parser.add_argument("--out", type=Path, default=Path("/work/out"))
    parser.add_argument("--cache", type=Path, default=Path("/work/hf"))
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)
    args.out.mkdir(parents=True, exist_ok=True)
    timings = Timings()
    receipt = {"started_utc": dt.datetime.now(dt.UTC).isoformat(), "tasks": args.tasks,
               "environment": environment(args.device), "timings_s": timings, "results": {}, "status": "running"}
    sampler = GpuSampler(args.out / "gpu-samples.csv") if args.device.startswith("cuda") else contextlib.nullcontext()
    models = list(dict.fromkeys(name for _, name in tasks))
    with sampler:
        try:
            for name in models:
                model = W.load_evo2(name, args.cache, timings, args.device)
                for kind, task_model in tasks:
                    if task_model == name:
                        receipt["results"][f"{kind}:{name}"] = run_task(kind, name, model, args, timings)
                W.unload(model, args.device)
            receipt["status"] = "completed"
        except Exception:
            receipt["status"] = "failed"
            receipt["error"] = traceback.format_exc()
            raise
        finally:
            # ru_maxrss is in kilobytes on Linux.
            receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            receipt["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
            W.write_json(args.out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
