# Amendment 2: run the tiers in phases

Date: 11 October 2026. No results had been seen. The only run so far is the L4 smoke test,
whose outputs are not used.

## Change

The tier definitions move to `configs/tiers.json`, unchanged except that the `deferred` flags of
amendment 1 are removed. Each run configuration lists the tiers it runs:

| Config | Runs | Reuses |
|---|---|---|
| `configs/phase-a.json` | T2 (L4) | none |
| `configs/phase-b.json` | T3 (H100), T4 (H200) | T2 from the latest completed phase A run |
| `configs/phase-c.json` | T1a, T1b (CPU) | T2, T3, T4 from the latest completed phase B run |
| `configs/full.json` | all five tiers | none (used by `make reproduce`) |

A reused tier is copied unchanged from the source run's output. Reuse is refused unless the source
run completed under the current approval and used the same study code (git tree of `src/`). The
copy and its source run are recorded in `provenance.json`. Each phase's `results.json` therefore
covers every tier run so far, and the latest phase is the study's recorded result.

Phase C runs only when the owner asks (amendment 1). `scripts/preflight.py` must pass before each
phase. It checks the balance covers the phase's planned spend plus one failed attempt of its
dearest tier, that no other pods are running, and that each tier's GPU type has stock.

## Reason

The Runpod balance (USD 20.81) covers T2 but not T3 and T4. The owner asked to run part of the
study now.

## Effect on the analysis

None on the design, criteria, budget or cap. Tiers run at different times, so list prices and
host hardware are recorded per tier, as before. Reproduction (`make reproduce`) runs all tiers in
one run with `configs/full.json`.
