# Amendment 1: run the CPU tiers after the GPU tiers

Date: 9 October 2026. No results had been seen; only an L4 smoke test of the pod code was in
progress, outside the recorded run.

## Change

T1a and T1b (G1, 7B on a CPU pod) are marked `deferred` in `configs/full.json`, and
`scripts/experiment.py` skips deferred tiers. The first recorded run covers T2, T3 and T4 only.

T2's hour limit rises from 6 to 7. The L4 smoke test measured 3.36 s per 8 kb window, so the
3,893 BRCA1 SNVs (about 5,200 windows) need about 4.9 hours plus setup, too close to 6.

## Reason

The owner asked to hold the CPU evaluations until the other work has finished.

## Effect on the analysis

None on the design or criteria. The spend cap is unchanged; T2's planned cost rises by about USD 0.60. G1 results stay empty until the CPU tiers run. A later
amendment will remove the deferral and say how the CPU run's outputs are combined with this run's.
