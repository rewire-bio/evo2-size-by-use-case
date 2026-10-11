"""Assemble results.json (scientific outputs) and operations.json (timings, memory, cost) from the
files each tier copied back. Offline and deterministic."""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from study.analysis import agreement, grouped_bootstrap  # noqa: E402

# Runpod container disk list price, USD per GB per month (https://www.runpod.io/pricing, 9 October 2026).
RUNPOD_CONTAINER_DISK_GB_MONTH = 0.10
HOURS_PER_MONTH = 730
MIN_STRATUM_POSITIVES = 20
REFERENCE_SCORES = "data/reference/evo2ops_p3_w1_brca1_scores.tsv"


def load_config(path: Path, root: Path = ROOT) -> dict:
    """A run configuration with its tier definitions resolved from `tiers_file`."""
    config = json.loads(Path(path).read_text())
    if "tiers_file" in config:
        config["tiers"] = json.loads((root / config["tiers_file"]).read_text())["tiers"]
    return config


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def attempts(tier_dir: Path) -> list[Path]:
    return sorted(tier_dir.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]))


def chosen_attempt(tier_dir: Path) -> Path | None:
    """The last attempt with a completed receipt, else the last attempt with any receipt."""
    with_receipt = [a for a in attempts(tier_dir) if (a / "receipt.json").exists()]
    completed = [a for a in with_receipt if read_json(a / "receipt.json").get("status") == "completed"]
    return (completed or with_receipt or [None])[-1]


def forward_tests_passed(tier_dir: Path) -> bool:
    attempt = chosen_attempt(tier_dir)
    if attempt is None:
        return False
    results = read_json(attempt / "receipt.json")["results"]
    tests = [v for k, v in results.items() if k.startswith("forward:")]
    return bool(tests) and all(t["passed"] for t in tests)


def utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def events(attempt: Path) -> dict:
    path = attempt / "events.jsonl"
    if not path.exists():
        return {}
    return {row["event"]: row["utc"] for row in map(json.loads, path.read_text().splitlines())}


def peak_smi_memory_mib(attempt: Path) -> float | None:
    path = attempt / "gpu-samples.csv"
    if not path.exists():
        return None
    with path.open() as handle:
        rows = [r for r in csv.reader(handle) if len(r) > 1 and not r[0].startswith("timestamp")]
    values = [float(r[1]) for r in rows if r[1].strip().replace(".", "").isdigit()]
    return max(values) if values else None


def scores(attempt: Path, task: str) -> pd.Series | None:
    path = attempt / f"{task}_scores.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t").set_index("id")["delta"]


def tier_cost(attempt: Path) -> dict:
    """Runpod bills from pod creation (including the image pull) until deletion."""
    ev, instance = events(attempt), read_json(attempt / "instance.json") or {}
    hourly = instance.get("cost_per_hr")
    storage = round(instance.get("disk_gb", 0) * RUNPOD_CONTAINER_DISK_GB_MONTH / HOURS_PER_MONTH, 4)
    if "pod_created" not in ev or "terminated" not in ev or hourly is None:
        return {"billed_seconds": None, "usd": None, "instance_usd_per_hour": hourly}
    seconds = (utc(ev["terminated"]) - utc(ev["pod_created"])).total_seconds()
    return {"billed_seconds": round(seconds), "instance_usd_per_hour": hourly, "storage_usd_per_hour": storage,
            "usd": round(seconds / 3600 * (hourly + storage), 2),
            "price_source": "Runpod pod costPerHr returned by the REST API at launch"}


def hardware_comparison(other: pd.Series, reference: pd.Series, variants: pd.DataFrame, boot: dict) -> dict:
    """Agreement and AUROC difference against the H100 reference on the variants both scored."""
    ids = other.index.intersection(reference.index)
    table = variants.loc[ids]
    # More negative delta means predicted more damaging, so the negated delta predicts LOF.
    auc = grouped_bootstrap(table["lof"].to_numpy(),
                            {"reference": -reference.loc[ids].to_numpy(), "tier": -other.loc[ids].to_numpy()},
                            table["pos"].to_numpy(), n=boot["n"], seed=boot["seed"])
    agree = agreement(reference.loc[ids].to_numpy(), other.loc[ids].to_numpy())
    return {"agreement": agree, "auroc": auc,
            "auroc_abs_difference": abs(auc["reference"]["auroc"] - auc["tier"]["auroc"])}


def equivalent(forward_ok: bool, comparison: dict | None, criteria: dict) -> bool | None:
    if comparison is None:
        return None
    return bool(forward_ok and comparison["agreement"]["spearman"] >= criteria["min_spearman"]
                and comparison["auroc_abs_difference"] <= criteria["max_auroc_abs_difference"])


def model_size_comparison(score_sets: dict[str, pd.Series], variants: pd.DataFrame, boot: dict) -> dict:
    """AUROC and AUPRC of each model on the shared variants, paired against 40B, overall and by stratum."""
    ids = variants.index
    for series in score_sets.values():
        ids = ids.intersection(series.index)
    table = variants.loc[ids]
    out = {"n": int(len(ids)), "n_pathogenic": int(table["ClinSigSimple"].sum())}
    strata = {"all": np.ones(len(table), bool)}
    for element in ("splice_site", "intergenic"):
        strata[element] = (table["genomic_element"] == element).to_numpy()
    for name, mask in strata.items():
        sub = table[mask]
        n_pos = int(sub["ClinSigSimple"].sum())
        entry = {"n": int(mask.sum()), "n_pathogenic": n_pos}
        if n_pos >= MIN_STRATUM_POSITIVES and n_pos < len(sub):
            sets = {k: -v.loc[sub.index].to_numpy() for k, v in score_sets.items()}
            for metric in ("auroc", "auprc"):
                entry[metric] = grouped_bootstrap(sub["ClinSigSimple"].to_numpy(), sets, sub["group"].to_numpy(),
                                                  n=boot["n"], seed=boot["seed"], metric=metric)
        out[name] = entry
    return out


def assemble(config: dict, output: Path, root: Path) -> None:
    boot, criteria = config["bootstrap"], config["criteria"]
    brca1 = pd.read_csv(root / "data/derived/brca1_variants.tsv", sep="\t").set_index("id")
    clinvar = pd.read_csv(root / "data/derived/clinvar_noncoding_nonsnv.tsv", sep="\t").set_index("id")
    reference_path = root / REFERENCE_SCORES
    reference = pd.read_csv(reference_path, sep="\t").set_index("id")["delta"] if reference_path.exists() else None

    chosen = {t["id"]: chosen_attempt(output / "tiers" / t["id"]) for t in config["tiers"]}
    receipts = {k: read_json(a / "receipt.json") if a else None for k, a in chosen.items()}
    forward = {}
    for tier_id, receipt in receipts.items():
        if receipt:
            forward[tier_id] = {k.split(":")[1]: {"mean_loss": v["mean_loss"], "expected_loss": v["expected_loss"],
                                                  "passed": v["passed"]}
                                for k, v in receipt["results"].items() if k.startswith("forward:")}

    def forward_ok(tier_id: str) -> bool:
        tests = forward.get(tier_id, {})
        return bool(tests) and all(t["passed"] for t in tests.values())

    results: dict = {"forward_tests": forward, "reference_available": reference is not None}

    # G1 and G2: hardware equivalence of 7B against the H100 reference.
    for gap, tier_id, task in (("g1_cpu", "T1b", "brca1_subsample_evo2_7b"), ("g2_l4", "T2", "brca1_evo2_7b")):
        attempt = chosen.get(tier_id)
        tier_scores = scores(attempt, task) if attempt else None
        comparison = None
        if tier_scores is not None and reference is not None and len(tier_scores):
            comparison = hardware_comparison(tier_scores, reference, brca1, boot)
        gate_tier = "T1a" if gap == "g1_cpu" else tier_id
        results[gap] = {"tier": tier_id, "forward_test_tier": gate_tier, "forward_test_passed": forward_ok(gate_tier),
                        "variants_scored": None if tier_scores is None else int(len(tier_scores)),
                        "comparison": comparison,
                        "hardware_equivalent": equivalent(forward_ok(gate_tier), comparison, criteria)}

    # G3: model size on ClinVar noncoding non-SNVs.
    sets = {}
    for name, tier_id in (("evo2_40b", "T4"), ("evo2_20b", "T3"), ("evo2_7b", "T3")):
        attempt = chosen.get(tier_id)
        series = scores(attempt, f"indels_{name}") if attempt else None
        if series is not None and len(series):
            sets[name] = series
    g3: dict = {"models_scored": sorted(sets)}
    if "evo2_40b" in sets and len(sets) > 1:
        g3["comparison"] = model_size_comparison(sets, clinvar, boot)
        overall = g3["comparison"]["all"].get("auroc", {})
        g3["model_adequate"] = {}
        for name in sets:
            if name == "evo2_40b":
                continue
            # difference_vs_evo2_40b is 40B minus the smaller model; adequate if 40B's lead stays below 0.02.
            interval = overall.get(name, {}).get("difference_vs_evo2_40b_ci95")
            g3["model_adequate"][name] = None if interval is None else bool(
                -interval[1] > -criteria["max_auroc_deficit_vs_40b"])
    results["g3_indels"] = g3

    operations = {}
    for tier_id, attempt in chosen.items():
        receipt = receipts[tier_id]
        if attempt is None or receipt is None:
            operations[tier_id] = {"status": "no_receipt"}
            continue
        cost = tier_cost(attempt)
        tasks = {}
        for key, value in receipt["results"].items():
            if key.startswith("forward:"):
                continue
            entry = dict(value)
            hourly = cost.get("instance_usd_per_hour")
            if hourly and value.get("variants_scored"):
                entry["usd_per_1000_variants"] = round(
                    value["scoring_seconds"] / value["variants_scored"] * 1000 * hourly / 3600, 4)
            tasks[key] = entry
        operations[tier_id] = {"status": receipt["status"], "attempt": attempt.name, "environment": receipt["environment"],
                               "timings_s": receipt["timings_s"], "tasks": tasks, "cost": cost,
                               "peak_rss_bytes": receipt.get("peak_rss_bytes"),
                               "nvidia_smi_peak_memory_mib": peak_smi_memory_mib(attempt),
                               "events_utc": events(attempt), "instance": read_json(attempt / "instance.json")}

    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (output / "operations.json").write_text(json.dumps(operations, indent=2, sort_keys=True) + "\n")
