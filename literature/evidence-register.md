# Evidence register: Evo 2 model size and hardware by clinical use case

Phase 0 of the study. Retrieved 9 October 2026. Source identifiers refer to
`literature/sources.json`. Every cell is either cited from a public source or marked as a gap.
Only gaps are candidates for new runs.

The use cases come from the AMP 2026 coverage tracker (rewire.it #365). The audience works on
diagnostic DNA and RNA testing for oncology, infectious disease and inherited disorders.

## Summary

1. **7B is enough wherever Evo 2 is a reasonable choice for single-nucleotide variants.** On
   BRCA1, ClinVar SNVs and SpliceVarDB, 7B is within about 0.01 AUROC of 40B, and sometimes
   ahead of it [evo2-sd1, arc-20b].
2. **1B should not be used.** Arc says so in the paper [brixi2026]. On BRCA1 and BRCA2 it scores
   0.732 and 0.749 AUROC against 0.875 and 0.835 for 7B, about the same as a PhyloP
   conservation score [arc-20b].
3. **The larger models matter for noncoding indels and noncoding regulatory variants.** 40B beats
   7B by about 0.04 AUROC on ClinVar noncoding non-SNVs [evo2-sd1], and by 0.17 AUPRC on TraitGym
   Mendelian traits, where 20B matches 40B [arc-20b]. These are the only rows here where Evo 2
   leads every non-Evo method reported.
4. **For most diagnostic SNV questions, a precomputed score beats every Evo 2 size.** These need
   no GPU, just a lookup. Examples: CADD on ClinVar SNVs (0.988 against 0.970 for 40B), AlphaMissense on
   coding SNVs (0.958 against 0.841), SpliceAI on SpliceVarDB (0.961 against 0.827) [evo2-sd1].
5. **Hardware.** 7B runs in bfloat16 on any supported NVIDIA GPU with about 17 GiB peak at 8 kb.
   1B, 20B and 40B need FP8 on a Hopper GPU for correct results [arc-repo, arc-20b]. 40B gives
   chance-level AUROC on an A100 [li2025]. No public source shows Evo 2 running on a CPU, and the
   loader fails without CUDA [arc-issue-67].

## Use cases

### 1. BRCA1/BRCA2 germline interpretation

- **Can Evo 2 do it?** Yes: zero-shot likelihood score, or a probe on embeddings.
- **Model size (public).** BRCA1, all SNVs: 1B 0.732, 7B 0.875, 20B 0.874, 40B 0.874 AUROC
  without reverse-complement averaging [arc-20b]. The paper's own run gives 7B 0.904 and
  40B 0.901 [evo2-sd1]. BRCA2, all SNVs: 1B 0.749, 7B 0.835, 20B 0.842, 40B 0.845 [arc-20b].
  BRCA2 noncoding: 7B 0.874, 40B 0.910. BRCA2 coding: 7B 0.727, 40B 0.768 [evo2-sd1].
- **Baselines (public).** BRCA2 all SNVs: CADD 0.884, above every Evo 2 size. BRCA1 coding:
  AlphaMissense 0.889 against 40B 0.843. BRCA1 noncoding: SpliceAI 0.965, between 7B (0.959)
  and 40B (0.974) [evo2-sd1].
- **Hardware (public).** 7B on H100 and L40S: same scores to within 0.0013, with L40S about 3x
  slower [evo2-ops]. 7B on A100: AUROC 0.88, about 4x slower than an H800 [li2025]. 40B on
  A100: about 0.5 [li2025].
- **Gap.** 7B on a CPU, and on a 24 GB GPU (agreement, throughput, cost).

### 2. Rare-disease ranking and reanalysis (ClinVar SNVs, coding and noncoding)

- **Can Evo 2 do it?** Yes, as one score among the evidence used to rank candidate variants.
- **Model size (public).** ClinVar SNVs: 7B 0.964, 40B 0.970. Coding SNVs: 0.830 and 0.841.
  Noncoding SNVs: 0.978 and 0.987 [evo2-sd1]. No published 1B or 20B values on ClinVar.
- **Baselines (public).** CADD 0.988 on all SNVs. AlphaMissense 0.958 on coding SNVs. SpliceAI
  0.992 on noncoding SNVs [evo2-sd1].
- **Gap.** None for model size. The decision is already clear: 7B is within about 0.01 of 40B,
  and both are below the precomputed baselines.

### 3. Noncoding indels and regulatory variants (inherited disorders)

- **Can Evo 2 do it?** Yes. This is where Evo 2 has an advantage: precomputed SNV tables do
  not cover indels, and Evo 2 scores any sequence change.
- **Model size (public).** ClinVar noncoding non-SNVs: 7B 0.930, 40B 0.969. Noncoding non-SNVs
  far from splice sites: 7B 0.873, 40B 0.923 [evo2-sd1]. TraitGym Mendelian (noncoding, AUPRC):
  1B 0.254, 7B 0.385, 20B 0.558, 40B 0.557. TraitGym complex: 0.116, 0.118, 0.138, 0.137
  [arc-20b].
- **Baselines (public).** PhyloP 241 0.936 on ClinVar noncoding non-SNVs [evo2-sd1]. On
  TraitGym, alignment-based models (CADD, GPN-MSA) do well for Mendelian traits [traitgym]; the
  numbers still need to be extracted.
- **Hardware (public).** 20B: one H100, 38 GB weights, 46 GB peak at 8 kb, 0.7 s per 8 kb
  sequence. 40B: two H100s, 95 GB peak, 1.5 s [arc-20b]. Both need Hopper FP8.
- **Gap.** 20B on ClinVar noncoding indels. It is not known whether the one-GPU 20B keeps the
  40B advantage on indels; Arc's 20B comparison covered TraitGym and BRCA only.

### 4. Splicing follow-up and patient-RNA splicing validation

- **Can Evo 2 do it?** Partly. It can prioritise candidate splice-altering variants. Validating
  them on patient RNA is a separate laboratory and RNA-seq step.
- **Model size (public).** SpliceVarDB SNVs: 7B 0.827, 40B 0.815. Exonic: 0.681 and 0.684.
  Intronic: 0.932 and 0.927 [evo2-sd1]. An independent benchmark gives 7B zero-shot 0.709
  [sun2026].
- **Baselines (public).** SpliceAI 0.961, Pangolin 0.903 [evo2-sd1]. Both run on a CPU, and
  precomputed SpliceAI scores exist.
- **Gap.** None. A larger Evo 2 does not help, and SpliceAI is better.

### 5. Somatic small-variant oncogenicity

- **Can Evo 2 do it?** Not designed for it. Evo 2 is trained on germline genomes, and somatic
  driver status is not an evolutionary-constraint label.
- **Public evidence.** GPN-Star reports Evo 2 at two sizes on COSMIC frequent tumour missense
  variants against common gnomAD missense variants, AUPRC (Fig. 2B, Supplementary Fig. 2)
  [gpn-star]. Values to be extracted. Li et al. scored ovarian tumour SNVs with ClinVar labels on
  private data [li2025]. Arc publishes the COSMIC set, 2,000-row subset, MIT licence [evoeval].
- **Gap.** Pending extraction from GPN-Star. If the two sizes there are 7B and 40B, there is no
  model-size gap.

### 6. DNA pathogen identification

- **Can Evo 2 do it?** Plausibly, through embedding probes. Bacteria and phages are in the
  training data.
- **Public evidence.** Probes on Evo 2 7B (262k) layer-26 activations detect resistance genes in
  metagenome assemblies (region-level ROC-AUC 0.888 linear, 0.977 attention). No comparison with
  AMRFinderPlus or CARD is reported. Code and data are withheld for dual-use reasons [guntoro2026].
- **Gap.** Real, but not testable here without data, and alignment-based tools are the
  diagnostic standard. Not tested.

### 7. RNA pathogen detection

- **Can Evo 2 do it?** No. Viruses that infect eukaryotes were removed from training on purpose,
  and Evo 2 has high perplexity on them [brixi2026].
- **Gap.** None. Not tested.

### 8. Tumour variant calling, RNA fusion detection, ctDNA fragmentomics and methylation, CNV

These tasks work on sequencing reads, fragment patterns, methylation calls or read depth. Evo 2
scores DNA sequences, so it does not address them. Not tested.

### 9. Large-scale diagnostic model execution (rewire-benchmark-data #18)

| Model | Public hardware evidence | Gap |
|---|---|---|
| 1B | FP8 on Hopper required; 0.13 s per sequence on H800 [arc-repo, li2025] | None (not recommended) |
| 7B | H100, L40S [evo2-ops]; H800, A100 [li2025]; RTX 4090 fits, about 22 GB [arc-issue-101] | CPU; 24 GB data-centre GPU (L4) |
| 20B | 1x H100, 0.7 s per 8 kb [arc-20b] | None for hardware |
| 40B | 2x H100 or 1x H200 [nim-prereq, arc-20b]; fails on A100 [li2025] | None |

## Proposed run matrix (gaps only)

| ID | Question | Model | Hardware | Data | Est. cost |
|---|---|---|---|---|---|
| G1 | Does 7B run on a CPU, how fast, and do its scores match a GPU? | 7B | Runpod `cpu3g`, 32 vCPU, 128 GB | BRCA1 subsample, 60 SNVs | about USD 5 |
| G2 | Does 7B on a 24 GB L4 match H100 scores, and what does it cost per 1,000 variants? | 7B | L4 24 GB | BRCA1, 3,893 SNVs | about USD 3 |
| G3 | Does the one-GPU 20B keep the 40B advantage on noncoding indels? | 20B, 40B (7B as a check) | H100; H200 for 40B | ClinVar noncoding indels, matched subset | about USD 20 to 30 |

For G1 and G2, the reference is the per-variant 7B H100 scores from evo2-cloud-operations, once
its recorded run is published. G3 needs 40B, because no per-variant 40B indel scores are public
and a matched comparison needs the same variants.

Not run: anything on 1B, 7B against 40B on SNVs, splicing, RNA viruses, and the read-level
tasks in section 8.

## Still to extract (no spend)

- GPN-Star COSMIC and OMIM values for both Evo 2 sizes, with the CADD and GPN-Star rows.
- TraitGym Mendelian AUPRC for CADD and GPN-MSA, from the leaderboard.
- Licence of the Evo 2 supplementary table, before any of its values are redistributed as data.
