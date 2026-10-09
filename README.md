# Which clinical genomics use cases can run on a smaller Evo 2 model?

A reproducible in silico study. For each diagnostic use case raised for the AMP 2026 session on
genomic foundation models, it asks: does a smaller Evo 2 model (1B, 7B or 20B) give the same
answer as 40B, and can it run on a CPU or a cheap GPU instead of an A100 or H100?

**Status: protocol drafted, awaiting approval.** No paid runs have been made.

Published results are cited, not re-run. New runs only fill the gaps that remain after the
literature check.

- [Evidence register](literature/evidence-register.md): use cases, public results by model size
  and hardware, and the gaps
- [Protocol](protocol.md): the three gaps run here (7B on CPU, 7B on an L4, 20B against 40B on
  noncoding indels), criteria and budget
- [Run log](evidence/run-log.md)
- [Sources](literature/sources.json)
- Related study: [evo2-cloud-operations](https://github.com/rewire-bio/evo2-cloud-operations)
  (Evo 2 7B on H100 and L40S across clouds)

## Run it

```sh
make test                                   # offline: fetch inputs, check hashes, run tests
export RUNPOD_API_KEY=...                   # kept outside the repository
make experiment                             # paid: one Runpod pod per tier
```

Each tier's pod deletes itself when the launcher exits or its hour limit passes. Results land in
`results/full/tiers/<ID>/attempt-<n>/`; `results.json` and `operations.json` are rebuilt after
every run.

## Reproduce the cited numbers

```sh
uv run --no-project --with pandas==2.3.3 --with openpyxl==3.1.5 python scripts/summarise_evo2_sd1.py
```

## Licences

Study code: to be set at publication. Cited data stay with their sources and are fetched at run
time, not redistributed.
