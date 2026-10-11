"""Run every configured tier on Runpod, then assemble results. Adapted from evo2-cloud-operations.

This spends money: each tier launches a pod. RUNPOD_API_KEY comes from the environment, never
from the repository.

  uv run --frozen python scripts/experiment.py --config configs/full.json --output results/full
  uv run --frozen python scripts/experiment.py --config configs/full.json --output results/full --collect-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect import assemble, forward_tests_passed, load_config, read_json  # noqa: E402
from data import verify_data  # noqa: E402

BUNDLE_DATA = {
    "prompts.csv": "data/raw/prompts.csv",
    "GRCh37.p13_chr17.fna.gz": "data/raw/GRCh37.p13_chr17.fna.gz",
    "brca1_variants.tsv": "data/derived/brca1_variants.tsv",
    "brca1_cpu_subsample.tsv": "data/derived/brca1_cpu_subsample.tsv",
    "clinvar_noncoding_nonsnv_windows.tsv.gz": "data/derived/clinvar_noncoding_nonsnv_windows.tsv.gz",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_bundle(bundle: Path, allow_dirty: bool) -> dict:
    """Code from the committed revision only, plus the verified inputs."""
    if git("status", "--porcelain", "--untracked-files=no") and not allow_dirty:
        raise SystemExit("Commit your changes first: runs use the committed revision only.")
    bundle.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(bundle / "code.tar.gz"), "HEAD", "src"],
                   cwd=ROOT, check=True)
    shutil.copy2(ROOT / "cloud/remote-run.sh", bundle / "remote-run.sh")
    (bundle / "data").mkdir()
    for name, source in BUNDLE_DATA.items():
        shutil.copy2(ROOT / source, bundle / "data" / name)
    return {
        "git_revision": git("rev-parse", "HEAD"),
        # The tarball embeds the commit time, so compare runs by the git tree of src/ instead.
        "src_tree": git("rev-parse", "HEAD:src"),
        "code_sha256": sha256(bundle / "code.tar.gz"),
        "inputs_sha256": {name: sha256(bundle / "data" / name) for name in BUNDLE_DATA},
    }


def launch(tier: dict, image: str, bundle: Path, output: Path, code_sha: str, attempts: int) -> dict:
    """Run one tier, retrying on failure up to the attempt limit. Every attempt's directory is kept."""
    script = ROOT / "cloud/runpod/launch.sh"
    for attempt in range(1, attempts + 1):
        existing = len(list((output / "tiers" / tier["id"]).glob("attempt-*")))
        out = output / "tiers" / tier["id"] / f"attempt-{existing + 1}"
        out.mkdir(parents=True, exist_ok=False)
        env = {**os.environ, "PLATFORM_ID": tier["id"], "IMAGE_REF": image, "BUNDLE": str(bundle),
               "OUT": str(out), "CODE_SHA256": code_sha, "TASKS": tier["tasks"], "DEVICE": tier["device"],
               "MAX_HOURS": str(tier["max_hours"]), "DISK_GB": str(tier["disk_gb"])}
        if tier.get("time_limit_s"):
            env["TIME_LIMIT_S"] = str(tier["time_limit_s"])
        env.update({k: str(v) for k, v in tier["env"].items()})
        with (out / "launcher.log").open("w") as log:
            status = subprocess.run(["bash", str(script)], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (out / "launcher-exit.json").write_text(json.dumps({"exit_code": status}) + "\n")
        print(f"{tier['id']} attempt {attempt}: exit {status}", flush=True)
        if status == 0:
            return {"id": tier["id"], "attempts": attempt, "status": "completed"}
    return {"id": tier["id"], "attempts": attempts, "status": "failed"}


def run_chain(chain: list[dict], image: str, bundle: Path, output: Path, code_sha: str, attempts: int) -> list[dict]:
    """Run tiers in order. A tier with `requires_forward_test_pass` runs only if that tier passed."""
    outcomes = []
    for tier in chain:
        gate = tier.get("requires_forward_test_pass")
        if gate and not forward_tests_passed(output / "tiers" / gate):
            outcomes.append({"id": tier["id"], "attempts": 0, "status": "not_run",
                             "reason": f"forward-pass test did not pass on {gate}"})
            print(f"{tier['id']}: not run, forward-pass test did not pass on {gate}", flush=True)
            continue
        outcomes.append(launch(tier, image, bundle, output, code_sha, attempts))
    return outcomes


def reuse_tiers(reuse: dict, output: Path, src_tree: str) -> dict:
    """Copy tier outputs from the latest completed recorded run of an earlier phase.

    The source run must belong to the current approval and have used the same study code (git tree
    of src/), so the copied tiers are what this run would have produced. Otherwise refuse.
    """
    approval = read_json(ROOT / ".research/approval.json")
    if approval is None:
        raise SystemExit("No protocol approval found; tiers can only be reused inside recorded runs.")
    candidates = []
    for manifest_path in (ROOT / ".research/runs").glob("*/manifest.json"):
        manifest = read_json(manifest_path)
        if (manifest.get("status") == "completed" and manifest.get("config", {}).get("phase") == reuse["from_phase"]
                and manifest.get("protocol_hash") == approval["protocol_hash"]):
            candidates.append(manifest)
    if not candidates:
        raise SystemExit(f"No completed phase {reuse['from_phase']} run under the current approval to reuse.")
    source = max(candidates, key=lambda m: m["finished_at"])
    source_output = ROOT / ".research/runs" / source["id"] / "output"
    source_provenance = read_json(source_output / "provenance.json")
    if source_provenance["src_tree"] != src_tree:
        raise SystemExit(f"Run {source['id']} used different study code (src tree "
                         f"{source_provenance['src_tree']}, now {src_tree}); its tiers cannot be reused.")
    for tier_id in reuse["tiers"]:
        shutil.copytree(source_output / "tiers" / tier_id, output / "tiers" / tier_id)
    return {"from_run": source["id"], "from_phase": reuse["from_phase"], "tiers": reuse["tiers"],
            "source_code_revision": source["code_revision"], "src_tree": src_tree}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiers", help="Comma-separated subset of tier IDs (operational use only)")
    parser.add_argument("--collect-only", action="store_true", help="Rebuild results from existing outputs")
    parser.add_argument("--allow-dirty", action="store_true", help="Smoke testing only; not for recorded runs")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("study_kind") != "evo2_size_by_use_case":
        raise SystemExit("Unexpected study configuration")
    verify_data(ROOT / "data/manifest.json", ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_inputs.py")], check=True)

    if not args.collect_only:
        tiers = [t for t in config["tiers"] if t["id"] in config["run_tiers"]]
        if args.tiers:
            wanted = set(args.tiers.split(","))
            tiers = [t for t in tiers if t["id"] in wanted]
        if not os.environ.get("RUNPOD_API_KEY"):
            raise SystemExit("RUNPOD_API_KEY is not set")
        args.output.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        bundle = args.output / f"bundle-{stamp}"
        provenance = build_bundle(bundle, args.allow_dirty)
        provenance.update({"started": stamp, "phase": config.get("phase"), "tiers": [t["id"] for t in tiers],
                           "image": config["image"], "config_sha256": sha256(args.config)})
        if config.get("reuse"):
            provenance["reused"] = reuse_tiers(config["reuse"], args.output, provenance["src_tree"])
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        # Tiers sharing a `chain` run in order (T1a then T1b); separate chains run at the same time.
        chains: dict[str, list[dict]] = {}
        for tier in tiers:
            chains.setdefault(tier.get("chain", tier["id"]), []).append(tier)
        with ThreadPoolExecutor(len(chains)) as pool:
            results = list(pool.map(lambda c: run_chain(c, config["image"], bundle, args.output,
                                                        provenance["code_sha256"], config["max_attempts"]),
                                    chains.values()))
        outcomes = [o for chain in results for o in chain]
        (args.output / "launch-outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")

    assemble(config, args.output, ROOT)
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
