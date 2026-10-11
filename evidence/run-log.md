# Run log

Chronological record of every launch and probe, including failures. Times are UTC.

## 2026-10-09, offline preparation (no spend)

- Phase 0 evidence register published (`literature/evidence-register.md`).
- Inputs fetched and hash-checked against `data/manifest.json`. The BRCA1 table is byte-identical to
  evo2-cloud-operations (`92b190ab...`). All 4,644 ClinVar noncoding non-SNV windows matched
  GRCh38. One source row with identical alleles (chr6:112187567 GT>GT) was excluded.
- G1 local wiring probe (`scripts/cpu_wiring_probe.py`) on an Apple M4: a four-layer, random-weight
  copy of the 7B architecture (three Hyena blocks and one attention block) ran a 256-token forward
  pass on CPU through `cpu_patch.py` in 1.4 s with finite output. Without initialised weights the
  output was not finite, as expected for empty tensors. Accuracy is not tested by this probe.

## 2026-10-09, L4 smoke test (outside the recorded run)

- Protocol approved by Tim Richardson at 18:28 (approval `904c29e9`).
- 18:29 to 18:42. Smoke test S1 from committed revision `c476715`: pod qlcttvc3t3r0c4, Runpod
  Secure Cloud L4 (EU-RO-1), USD 0.59/h, about USD 0.13. 7B forward-pass loss 0.34787 (expected
  0.34766, passed); 60 BRCA1 SNVs scored at 3.36 s per window; peak 16.8 GiB allocated. Scores
  matched the H100 reference to within 0.0004 per variant, which confirms window and identifier
  alignment. These outputs are a wiring check and are not used in the analysis. Pod deleted by the
  launcher at 18:42; confirmed absent through the Runpod API.
- Amendment 1 at the owner's request: the CPU tiers (T1a, T1b) wait until the GPU tiers have
  finished. T2's hour limit rises to 7 hours after the smoke-test timing.
- Amendment 1 approved by Tim Richardson (approval `88ee0108`).
- 19:0x. Recorded run held at the owner's request until another study's pod finishes and the
  Runpod balance is topped up. Pre-flight (`scripts/preflight.py`): balance USD 33.80 against USD
  69.60 needed (planned USD 43.60 for T2 to T4 plus one failed T4 attempt); one evo2-cloud-operations
  pod still running; L4, H100 and H200 Secure Cloud stock LOW.

## 2026-10-11, phased runs

- Balance USD 20.81, no pods running. Amendment 2 splits the GPU tiers into phase A (T2, now) and
  phase B (T3, T4, after a top-up). Pre-flight for phase A passed (USD 7.20 needed).
