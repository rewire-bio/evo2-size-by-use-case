"""Offline tests: window construction, input preparation and statistics. No GPU or torch needed."""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evo2size.genome import centred_window, snv_windows, variant_windows  # noqa: E402
from study.analysis import agreement, auroc, average_precision, grouped_bootstrap  # noqa: E402

WINDOW = 64


def random_chromosome(length: int = 1000, seed: int = 1) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.chrom = random_chromosome()

    def test_every_window_has_the_fixed_length(self):
        pos = 500
        for ref_len, alt_len in [(1, 1), (5, 1), (1, 9), (3, 3), (40, 1)]:
            ref = self.chrom[pos - 1:pos - 1 + ref_len]
            alt = ref[0] + "T" * (alt_len - 1) if alt_len > 1 else ref[0]
            if ref_len == alt_len:
                alt = "".join("A" if b != "A" else "C" for b in ref)
            ref_w, var_w = variant_windows(self.chrom, pos, ref, alt, WINDOW)
            self.assertEqual(len(ref_w), WINDOW)
            self.assertEqual(len(var_w), WINDOW)

    def test_allele_is_centred_and_flanks_come_from_the_genome(self):
        pos, ref, alt = 500, self.chrom[499:504], self.chrom[499]  # a 4 bp deletion with anchor base
        var_w = centred_window(self.chrom, pos, ref, alt, WINDOW)
        left = (WINDOW - len(alt)) // 2
        self.assertEqual(var_w[left:left + len(alt)], alt)
        self.assertEqual(var_w[:left], self.chrom[499 - left:499])
        self.assertEqual(var_w[left + len(alt):], self.chrom[504:504 + WINDOW - left - len(alt)])

    def test_reference_window_is_the_unchanged_genome(self):
        pos, ref = 500, self.chrom[499:502]
        ref_w = centred_window(self.chrom, pos, ref, ref, WINDOW)
        left = (WINDOW - len(ref)) // 2
        self.assertEqual(ref_w, self.chrom[499 - left:499 - left + WINDOW])

    def test_insertion_window(self):
        pos, ref = 500, self.chrom[499]
        alt = ref + "GATTACA"
        var_w = centred_window(self.chrom, pos, ref, alt, WINDOW)
        left = (WINDOW - len(alt)) // 2
        self.assertEqual(var_w[left:left + len(alt)], alt)
        self.assertEqual(var_w[left + len(alt):], self.chrom[500:500 + WINDOW - left - len(alt)])

    def test_reference_mismatch_raises(self):
        wrong = "A" if self.chrom[499] != "A" else "C"
        with self.assertRaises(ValueError):
            centred_window(self.chrom, 500, wrong, "G", WINDOW)

    def test_window_off_the_chromosome_raises(self):
        with self.assertRaises(ValueError):
            centred_window(self.chrom, 3, self.chrom[2], "A", WINDOW)

    def test_snv_windows_match_evo2_cloud_operations_convention(self):
        pos = 500
        ref_w, var_w = snv_windows(self.chrom, pos, self.chrom[pos - 1], "A", WINDOW)
        self.assertEqual(ref_w, self.chrom[pos - 1 - WINDOW // 2:pos - 1 + WINDOW // 2])
        self.assertEqual(var_w[WINDOW // 2], "A")
        self.assertEqual(sum(a != b for a, b in zip(ref_w, var_w)), int(self.chrom[pos - 1] != "A"))


class StatisticsTests(unittest.TestCase):
    def test_auroc_and_average_precision_on_a_known_case(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(auroc(labels, scores), 0.75)
        self.assertAlmostEqual(average_precision(labels, scores), (1 / 1 + 2 / 3) / 2)

    def test_average_precision_handles_ties_as_one_threshold(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(average_precision(labels, scores), 0.5)

    def test_grouped_bootstrap_pairs_differences(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200)
        good = labels + rng.normal(0, 0.5, 200)
        noisy = labels + rng.normal(0, 2.0, 200)
        groups = np.arange(200) // 2
        for metric in ("auroc", "auprc"):
            out = grouped_bootstrap(labels, {"good": good, "noisy": noisy}, groups, n=200, seed=7, metric=metric)
            self.assertGreater(out["good"][metric], out["noisy"][metric])
            low, high = out["noisy"]["difference_vs_good_ci95"]
            self.assertLess(low, high)
            self.assertGreater(low, 0)

    def test_agreement_of_identical_scores(self):
        scores = np.linspace(-1, 0, 50)
        result = agreement(scores, scores.copy())
        self.assertEqual(result["max_abs_difference"], 0.0)
        self.assertAlmostEqual(result["spearman"], 1.0)
        self.assertEqual(result["rank_moves_over_1pct"], 0)


class InputTests(unittest.TestCase):
    """Run against the prepared files when they exist (after make inputs)."""

    def setUp(self):
        self.derived = ROOT / "data/derived"
        if not (self.derived / "brca1_variants.tsv").exists():
            self.skipTest("run make inputs first")

    def test_cpu_subsample_is_20_positions_and_60_snvs(self):
        import pandas as pd
        sub = pd.read_csv(self.derived / "brca1_cpu_subsample.tsv", sep="\t")
        self.assertEqual(len(sub), 60)
        self.assertEqual(sub["pos"].nunique(), 20)
        lof_positions = sub.groupby("pos")["lof"].max()
        self.assertEqual(int(lof_positions.sum()), 10)

    def test_clinvar_selection_and_windows(self):
        import gzip
        import csv
        import pandas as pd
        variants = pd.read_csv(self.derived / "clinvar_noncoding_nonsnv.tsv", sep="\t")
        self.assertEqual(len(variants), 4644)
        self.assertEqual(int(variants["ClinSigSimple"].sum()), 1086)
        self.assertFalse((variants["genomic_element"] == "CDS").any())
        self.assertFalse((variants["variant_type"] == "SNV").any())
        with gzip.open(self.derived / "clinvar_noncoding_nonsnv_windows.tsv.gz", "rt", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual([r["id"] for r in rows], variants["id"].tolist())
        for row in rows:
            self.assertEqual(len(row["ref_window"]), 8192)
            self.assertEqual(len(row["var_window"]), 8192)
            self.assertNotEqual(row["ref_window"].upper(), row["var_window"].upper())


if __name__ == "__main__":
    unittest.main()


class CollectTests(unittest.TestCase):
    """End-to-end check of result assembly on synthetic tier outputs (no cloud runs)."""

    def setUp(self):
        if not (ROOT / "data/derived/clinvar_noncoding_nonsnv.tsv").exists():
            self.skipTest("run make inputs first")

    def test_assemble_on_synthetic_outputs(self):
        import json
        import os
        import tempfile

        import pandas as pd

        from collect import assemble

        from collect import load_config
        config = load_config(ROOT / "configs/full.json")
        config["bootstrap"]["n"] = 50
        brca1 = pd.read_csv(ROOT / "data/derived/brca1_variants.tsv", sep="\t")
        clinvar = pd.read_csv(ROOT / "data/derived/clinvar_noncoding_nonsnv.tsv", sep="\t")
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "root", Path(tmp) / "out"
            (root / "data/reference").mkdir(parents=True)
            os.symlink(ROOT / "data/derived", root / "data/derived")
            truth = -brca1["lof"] + rng.normal(0, 0.5, len(brca1))
            pd.DataFrame({"id": brca1["id"], "delta": truth}).to_csv(
                root / "data/reference/evo2ops_p3_w1_brca1_scores.tsv", sep="\t", index=False)

            def tier(tier_id, results, score_files):
                attempt = out / "tiers" / tier_id / "attempt-1"
                attempt.mkdir(parents=True)
                (attempt / "receipt.json").write_text(json.dumps({
                    "status": "completed", "environment": {}, "timings_s": {}, "results": results}))
                (attempt / "events.jsonl").write_text(
                    '{"event": "pod_created", "utc": "2026-10-10T10:00:00Z"}\n'
                    '{"event": "terminated", "utc": "2026-10-10T11:00:00Z"}\n')
                (attempt / "instance.json").write_text(json.dumps({"cost_per_hr": 0.59, "disk_gb": 100}))
                for name, (ids, delta) in score_files.items():
                    pd.DataFrame({"id": ids, "delta": delta}).to_csv(attempt / f"{name}_scores.tsv", sep="\t", index=False)

            forward = {"mean_loss": 0.3477, "expected_loss": 0.3476563, "passed": True}
            run = {"variants_scored": len(brca1), "scoring_seconds": 3600.0}
            tier("T2", {"forward:evo2_7b": forward, "brca1:evo2_7b": run},
                 {"brca1_evo2_7b": (brca1["id"], truth + rng.normal(0, 1e-4, len(brca1)))})
            label = clinvar["ClinSigSimple"].to_numpy()
            tier("T3", {}, {"indels_evo2_7b": (clinvar["id"], -label + rng.normal(0, 1.5, len(label))),
                            "indels_evo2_20b": (clinvar["id"], -label + rng.normal(0, 0.6, len(label)))})
            tier("T4", {}, {"indels_evo2_40b": (clinvar["id"], -label + rng.normal(0, 0.6, len(label)))})
            assemble(config, out, root)
            results = json.loads((out / "results.json").read_text())
            operations = json.loads((out / "operations.json").read_text())

        g2 = results["g2_l4"]
        self.assertTrue(g2["hardware_equivalent"])
        self.assertGreater(g2["comparison"]["agreement"]["spearman"], 0.999)
        self.assertIsNone(results["g1_cpu"]["hardware_equivalent"])
        g3 = results["g3_indels"]
        self.assertEqual(g3["comparison"]["n"], 4644)
        self.assertFalse(g3["model_adequate"]["evo2_7b"])
        self.assertIn("auroc", g3["comparison"]["splice_site"])
        self.assertAlmostEqual(operations["T2"]["tasks"]["brca1:evo2_7b"]["usd_per_1000_variants"],
                               1000 / len(brca1) * 0.59, places=3)


class WeightTests(unittest.TestCase):
    """Shard joining and hash refusal, with a fake hub laid out like huggingface_hub's cache."""

    def fake_hub(self, root: Path, contents: dict[str, bytes]):
        blobs, snapshot = root / "blobs", root / "snapshot"
        blobs.mkdir(parents=True)
        snapshot.mkdir()
        for filename, data in contents.items():
            blob = blobs / filename
            blob.write_bytes(data)
            (snapshot / filename).symlink_to(blob)

        def download(repo_id, filename, revision, cache_dir):
            return str(snapshot / filename)
        return download

    def test_40b_parts_are_checked_joined_and_freed(self):
        import hashlib
        import tempfile
        from unittest import mock

        from evo2size import pins, weights

        parts = {"evo2_40b.pt.part0": b"first-", "evo2_40b.pt.part1": b"second"}
        pinned = {"repo_id": "x/evo2_40b", "revision": "r",
                  "files": {k: hashlib.sha256(v).hexdigest() for k, v in parts.items()}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(pins.WEIGHTS, {"evo2_40b": pinned}):
            root = Path(tmp)
            joined = weights.fetch_weights("evo2_40b", root, download=self.fake_hub(root / "hub", parts))
            self.assertEqual(joined.read_bytes(), b"first-second")
            self.assertEqual(list((root / "hub/blobs").iterdir()), [])

    def test_hash_mismatch_refuses_to_load(self):
        import tempfile
        from unittest import mock

        from evo2size import pins, weights

        pinned = {"repo_id": "x/evo2_20b", "revision": "r", "files": {"evo2_20b.pt": "0" * 64}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(pins.WEIGHTS, {"evo2_20b": pinned}):
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                weights.fetch_weights("evo2_20b", root, download=self.fake_hub(root / "hub", {"evo2_20b.pt": b"x"}))

    def test_every_pinned_model_has_an_expected_forward_loss(self):
        from evo2size import pins
        self.assertEqual(set(pins.WEIGHTS), set(pins.FORWARD_TEST_LOSS))


class ReuseTests(unittest.TestCase):
    """Phase reuse copies tiers only from a completed run of the right phase, approval and code."""

    def make_run(self, root: Path, ident: str, phase: str, status: str, protocol: str, src_tree: str, finished: str):
        import json
        folder = root / ".research/runs" / ident
        (folder / "output/tiers/T2/attempt-1").mkdir(parents=True)
        (folder / "output/tiers/T2/attempt-1/receipt.json").write_text(json.dumps({"run": ident}))
        (folder / "output/provenance.json").write_text(json.dumps({"src_tree": src_tree}))
        (folder / "manifest.json").write_text(json.dumps({
            "id": ident, "status": status, "config": {"phase": phase}, "protocol_hash": protocol,
            "code_revision": "abc", "finished_at": finished}))

    def test_reuse_picks_the_latest_matching_run_and_refuses_changed_code(self):
        import json
        import tempfile
        from unittest import mock

        import experiment

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".research").mkdir()
            (root / ".research/approval.json").write_text(json.dumps({"protocol_hash": "P"}))
            self.make_run(root, "old", "a", "completed", "P", "TREE", "2026-10-11T01:00:00")
            self.make_run(root, "new", "a", "completed", "P", "TREE", "2026-10-11T02:00:00")
            self.make_run(root, "failed", "a", "failed", "P", "TREE", "2026-10-11T03:00:00")
            self.make_run(root, "other-approval", "a", "completed", "Q", "TREE", "2026-10-11T04:00:00")
            with mock.patch.object(experiment, "ROOT", root):
                out = root / "out"
                record = experiment.reuse_tiers({"from_phase": "a", "tiers": ["T2"]}, out, "TREE")
                self.assertEqual(record["from_run"], "new")
                copied = json.loads((out / "tiers/T2/attempt-1/receipt.json").read_text())
                self.assertEqual(copied["run"], "new")
                with self.assertRaises(SystemExit):
                    experiment.reuse_tiers({"from_phase": "a", "tiers": ["T2"]}, root / "out2", "CHANGED")
                with self.assertRaises(SystemExit):
                    experiment.reuse_tiers({"from_phase": "b", "tiers": ["T2"]}, root / "out3", "TREE")
