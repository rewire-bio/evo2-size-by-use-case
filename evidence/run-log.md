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
