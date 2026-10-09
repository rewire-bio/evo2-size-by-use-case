# Which clinical genomics use cases can run on a smaller Evo 2 model?

Status: draft for owner review, 9 October 2026

## Research question

For the diagnostic use cases raised for the AMP 2026 genomic foundation model session, what is
the smallest Evo 2 model that gives the same answer as Evo 2 40B, and what is the cheapest
hardware that reproduces that model's scores?

## Purpose and contribution

Diagnostic teams default to A100 or H100 GPUs for Evo 2. Most of the model-size question is
already answered in public sources (`literature/evidence-register.md`):

- Germline SNVs (BRCA1, BRCA2, ClinVar) and splicing: 7B is within about 0.01 AUROC of 40B.
  Precomputed CADD, AlphaMissense or SpliceAI scores are better than every Evo 2 size.
- 1B: Arc advises against it, and its scores are close to PhyloP.
- Noncoding indels: 40B beats 7B (ClinVar noncoding non-SNVs, AUROC 0.969 against 0.930). This is
  where Evo 2 leads the published baselines.

This study runs only the remaining gaps:

- **G1.** Can 7B run on a CPU? If it can, how fast is it, and do its scores agree with a GPU's?
  No public source reports Evo 2 on a CPU.
- **G2.** Does 7B on a 24 GB L4 GPU give the same scores as an H100, and what does it cost per
  1,000 variants?
- **G3.** Does Evo 2 20B, which fits on one H100, keep the 40B advantage on noncoding indels?
  Arc compared 20B with 40B on TraitGym, BRCA1 and BRCA2, but not on indels.

No result here is a clinical validation. Scores are research outputs, not calibrated
pathogenicity calls.

## Hypotheses and estimands

- G1: whether Arc's 7B forward-pass test passes on CPU; then seconds per 8 kb window, cost per
  1,000 variants, and agreement with the H100 reference on the same variants.
- G2: Spearman correlation and maximum absolute difference of per-variant BRCA1 scores against
  the H100 reference; AUROC difference; seconds per window; cost per 1,000 variants; peak GPU
  memory.
- G3: AUROC and AUPRC of 7B, 20B and 40B on ClinVar noncoding non-SNVs; paired differences
  20B minus 40B and 7B minus 40B, each with a 95% interval.

No hypothesis is tested for significance. Intervals and the pre-set criteria below are
reported.

## Data, provenance, and licences

All inputs are public. No patient data are used. Each file is listed in `data/manifest.json`
with URL, SHA-256, licence and provenance, and is fetched at run time, not redistributed.

- **BRCA1 (G1, G2).** Findlay et al. 2018 saturation genome editing table, 3,893 SNVs, and GRCh37
  chromosome 17, both from Arc's repository at commit `53f1959`. These are the same files and
  windows as evo2-cloud-operations.
- **G1 subsample.** 20 BRCA1 positions (60 SNVs), drawn once with seed 20261009, stratified by
  functional class, and written to a derived file pinned by hash.
- **H100 reference for G1 and G2.** Per-variant 7B BRCA1 scores from evo2-cloud-operations
  recorded run `b75c67aa` (code revision `e93491a`, same image digest and windows), platform P3,
  Runpod H100, forward-pass test passed. The file is copied unchanged into
  `data/reference/` and pinned by hash in `data/manifest.json`.
- **ClinVar indels (G3).** Arc's `evoeval` ClinVar table (MIT licence, Hugging Face revision
  `1ce96bd`, SHA-256 `6aa4f088...`). Rows: `genomic_element` not `CDS`, and `variant_type` in
  deletion, insertion or MNV, excluding one source row whose two alleles are identical. That gives
  4,644 variants: 1,086 pathogenic or likely pathogenic and 3,558 benign or likely benign. All are at least two-star (multiple submitters or expert
  panel).
- **GRCh38 (G3).** UCSC `hg38.fa.gz`, pinned by hash. Windows are built offline into a derived,
  hashed file, so the pods need no genome.
- **Weights.** Hugging Face `arcinstitute`, SHA-256 checked before loading, because the loader
  uses pickle. `evo2_7b` revision `bda0089f` (`c6664592...`), `evo2_20b` revision `8b0f0a9a`
  (`979b345d...`), `evo2_40b` revision `d529aa57` (two parts, `3b74fa4e...` and `bdc4a76e...`).
  Apache-2.0.

## Methods, controls, and baselines

**Software.** The evo2-cloud-operations image
`ghcr.io/rewire-bio/evo2-cloud-operations@sha256:22970720...` (`evo2==0.6.0`, Transformer
Engine present), unchanged on every GPU tier.

**Correctness check (all tiers).** Arc's forward-pass test for each checkpoint. It passes if the
mean loss is within 1e-3 of Arc's expected value: 7B 0.3476563, 20B 0.2166748, 40B 0.2159424.
Failure is reported. It does not stop the tier's other outputs being recorded, because the
failure is itself a finding.

**Variant score.** The 8,192 bp window centred on the variant. Score = log-likelihood of the
variant window minus that of the reference window, forward genomic strand, no reverse-complement
averaging. More negative means predicted more damaging.

- SNVs use the same windows and scoring call as evo2-cloud-operations `w1_brca1`: mean
  log-likelihood, batch size 1, and each distinct reference window scored once.
- Indels and MNVs follow Arc's method (Supplementary Information A.3.12). Both windows are 8,192
  bp and centred on the variant. Context is added or removed at the window edges so the lengths
  match.

**Tiers** (Runpod Secure Cloud, on demand; list price read by API at run time):

| ID | Gap | Hardware | Model and workload |
|---|---|---|---|
| T1a | G1 | CPU pod `cpu3g`, 32 vCPU, 128 GB RAM | 7B forward-pass test on CPU (feasibility) |
| T1b | G1 | same | 7B on the 60-SNV BRCA1 subsample, 4-hour limit |
| T2 | G2 | 1x L4 24 GB | 7B forward-pass test; 7B on all 3,893 BRCA1 SNVs |
| T3 | G3 | 1x H100 80 GB | 7B and 20B forward-pass tests; both on the 4,644 indels |
| T4 | G3 | 1x H200 141 GB | 40B forward-pass test; 40B on the 4,644 indels |

**CPU route (G1).** Arc's code does not run without CUDA. T1a runs it with only the changes in
`src/evo2size/cpu_patch.py`, and the model arithmetic is otherwise unchanged:

1. `torch.cuda.device` contexts become no-ops for CPU devices.
2. Flash attention is off, so Vortex uses its own PyTorch `scaled_dot_product_attention` path. FP8
   input projections are off, which is Arc's documented bfloat16 path for 7B.
3. Rotary embedding uses Vortex's own PyTorch reference function instead of its Triton kernel,
   for fixed-length inference only.
4. The model is cast to float32. Arc's scoring function is called with `device="cpu"`.

A local probe (`scripts/cpu_wiring_probe.py`, 9 October 2026) ran a four-layer, random-weight copy
of the 7B architecture through this path on CPU and gave finite output. Scaling its timing
suggests several minutes per 8 kb window on a CPU, so T1b may stop at its time limit with only
part of the 60 SNVs scored. If T1a fails, T1b is not run, and the result is "not supported without
further porting".

**Baselines.** G3 has no new baseline runs. Published PhyloP and Evo 2 values on Arc's ClinVar
set are cited for context only, because the variant sets differ.

## Metrics and uncertainty

- AUROC and AUPRC. 95% intervals from 2,000 bootstrap resamples, seed 7. BRCA1 is resampled by
  position (the three SNVs at a position stay together). ClinVar is resampled by gene.
  Differences between models are paired on the same resamples.
- Agreement with the reference: Spearman correlation, maximum and median absolute score
  difference, and the number of variants whose rank moves by more than 1% of the list.
- Operations: wall time per window, peak memory (`torch.cuda.max_memory_allocated` or process
  RSS on CPU), billed seconds, and cost (billed seconds times the API list price).
  Cost per 1,000 variants = workload cost / variants x 1,000, excluding image pull and weight
  download, which are reported separately.

**Criteria, fixed before any run.**

- *Hardware-equivalent* (T1b, T2 against the H100 reference): forward-pass test passes,
  Spearman at least 0.99, and absolute AUROC difference at most 0.01.
- *Model-adequate* (G3, 20B or 7B against 40B): the lower bound of the 95% interval of the AUROC
  difference is above -0.02.

**Stratified reporting (G3).** Pathogenic indels concentrate in splice sites and intergenic
regions, and intronic indels are almost all benign. AUROC is therefore also reported within
the splice-site and intergenic strata. A stratum with fewer than 20 positives is reported as a
count only.

## Analysis plan and stopping conditions

Each tier runs once from a clean committed revision through `research run`. Tiers may run at the
same time. Outputs, receipts and logs are kept. Failed attempts are kept and reported.

- Stop a tier after 2 failed attempts. If capacity cannot be obtained within those attempts, the
  tier is reported as unavailable, not substituted.
- Stop the study if spend reaches the cap.
- T1b stops at 4 hours. Variants scored up to that point are reported with their count.

## Reproduction tolerances

Reproduction reruns every tier from a clean checkout with the same image digest and inputs.
Scientific outputs (forward-pass losses, AUROCs, AUPRCs, agreement statistics) are compared with
absolute tolerance 0.01 and relative tolerance 0. Timings and costs are recorded, not compared.

## Budgets

- **Cloud spend cap:** USD 250 for the whole study, including failed attempts and the
  reproduction.
- **Planned spend for the original runs:** about USD 50. T1a under 1, T1b about 5, T2 about 3,
  T3 about 14, T4 about 26. The reproduction costs about the same again.
- **Self-termination:** the launcher deletes each pod when it exits, and a watchdog in the
  launcher deletes it after the tier's hour limit (T1a 2 hours, T4 8 hours to allow for the 82 GB
  40B download, others 6 hours). The image's own in-pod backstop calls Runpod's retired v1
  endpoint and is not relied on. Pods are checked through the Runpod API after every run.
- At most 2 attempts per tier per phase. Under 2 GB of retained evidence. No weights, windows or
  embeddings are committed.

## Limitations and failure interpretation

- One run per tier. Timings are receipts, not a performance benchmark.
- G3 uses Arc's newer ClinVar export, not the October 2024 release used in the paper, so its
  values are not directly comparable with the published 7B and 40B numbers.
- The CPU route uses a patched attention path. If agreement is poor, the patch may be the
  cause, not the CPU itself.
- Evo 2 scores are not calibrated, and ClinVar labels favour well-studied genes.
- A failed forward-pass test or an unavailable tier is a reportable result, not a missing one.

## Approval and amendments

The owner must explicitly approve this protocol and budget with `research approve` before any
paid run. Design changes are recorded in `protocol/amendments/` and need renewed approval.
