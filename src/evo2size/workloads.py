"""Workloads run on the pod: Arc's forward-pass test and paired-window variant scoring."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import pins
from .genome import read_first_fasta_record, read_variants, read_window_pairs, snv_windows


from .weights import fetch_weights


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device: str) -> int | None:
    return torch.cuda.max_memory_allocated() if device.startswith("cuda") else None


def load_evo2(name: str, cache_dir: Path, timings: dict, device: str):
    from evo2 import Evo2

    start = time.perf_counter()
    path = fetch_weights(name, cache_dir)
    timings[f"{name}_download_and_verify_s"] = round(time.perf_counter() - start, 3)
    start = time.perf_counter()
    if device == "cpu":
        from . import cpu_patch
        cpu_patch.apply()
    model = Evo2(name, local_path=str(path))
    if device == "cpu":
        cpu_patch.to_float32(model)
    synchronize(device)
    timings[f"{name}_load_s"] = round(time.perf_counter() - start, 3)
    return model


def unload(model, device: str) -> None:
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def forward_test(model, name: str, prompts_csv: Path, device: str) -> dict:
    """Arc's forward-pass check (evo2/test/test_evo2.py), reimplemented without its CLI."""
    with prompts_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        sequences = [row[0] for row in reader]
    torch.manual_seed(1)
    losses, accuracies = [], []
    for seq in sequences:
        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=int).to(device)
        with torch.inference_mode():
            logits, _ = model.model.forward(input_ids.unsqueeze(0))
        target, pred = input_ids[1:], logits[0, :-1, :].float()
        losses.append(F.cross_entropy(pred, target.long()).item())
        accuracies.append((target == torch.argmax(pred, dim=-1)).float().mean().item())
    mean_loss = sum(losses) / len(losses)
    expected = pins.FORWARD_TEST_LOSS[name]
    return {
        "model": name,
        "per_sequence_loss": losses,
        "per_sequence_accuracy": accuracies,
        "mean_loss": mean_loss,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "expected_loss": expected,
        "passed": abs(mean_loss - expected) < pins.FORWARD_TEST_TOLERANCE,
    }


def _score(model, seq: str, device: str) -> float:
    """Mean log-likelihood of one sequence (batch size 1, as in Arc's notebook).

    On a GPU this is the same call as evo2-cloud-operations. Arc's wrapper hard-codes "cuda:0", so
    on a CPU the underlying scoring function is called with the device given explicitly.
    """
    if device.startswith("cuda"):
        return float(model.score_sequences([seq], batch_size=1, reduce_method="mean")[0])
    from evo2.scoring import score_sequences

    with torch.no_grad():
        return float(score_sequences([seq], model=model.model, tokenizer=model.tokenizer, batch_size=1,
                                     reduce_method="mean", device=device)[0])


def score_pairs(model, pairs: list[tuple[str, str, str]], out_tsv: Path, device: str, label: str,
                time_limit_s: float | None = None) -> dict:
    """Score (id, reference window, variant window) triples; delta = variant minus reference.

    Identical reference windows are scored once. Rows are written as they finish, so a time limit
    leaves a usable partial table. The limit is checked between variants.
    """
    reset_peak_memory(device)
    ref_cache: dict[str, float] = {}
    windows = 0
    completed = 0
    start = time.perf_counter()
    stopped_by_limit = False
    with out_tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "ref_score", "var_score", "delta"])
        for variant_id, ref_seq, var_seq in pairs:
            if time_limit_s is not None and time.perf_counter() - start > time_limit_s:
                stopped_by_limit = True
                break
            if ref_seq not in ref_cache:
                ref_cache[ref_seq] = _score(model, ref_seq, device)
                windows += 1
            var_score = _score(model, var_seq, device)
            windows += 1
            ref_score = ref_cache[ref_seq]
            writer.writerow([variant_id, repr(ref_score), repr(var_score), repr(var_score - ref_score)])
            handle.flush()
            completed += 1
            if completed % 250 == 0:
                rate = windows / (time.perf_counter() - start)
                print(f"[{label}] {completed}/{len(pairs)} variants, {rate:.3f} windows/s", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - start
    return {
        "variants_requested": len(pairs),
        "variants_scored": completed,
        "windows_scored": windows,
        "scoring_seconds": round(seconds, 3),
        "seconds_per_window": round(seconds / windows, 4) if windows else None,
        "stopped_by_time_limit": stopped_by_limit,
        "peak_memory_allocated_bytes": peak_memory_bytes(device),
    }


def brca1_pairs(chr17_fasta: Path, variants_tsv: Path, window: int = pins.WINDOW) -> list[tuple[str, str, str]]:
    """BRCA1 SNV windows built exactly as evo2-cloud-operations W1."""
    chromosome = read_first_fasta_record(chr17_fasta)
    pairs = []
    for row in read_variants(variants_tsv):
        ref_seq, var_seq = snv_windows(chromosome, row["pos"], row["ref"], row["alt"], window)
        pairs.append((row["id"], ref_seq, var_seq))
    return pairs


def prepared_pairs(path: Path) -> list[tuple[str, str, str]]:
    return [(row["id"], row["ref_window"], row["var_window"]) for row in read_window_pairs(path)]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
