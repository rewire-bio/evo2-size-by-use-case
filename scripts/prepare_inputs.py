"""Build the pods' input files from the verified raw data. Deterministic; no GPU needed.

Writes to data/derived/:
- brca1_variants.tsv: the 3,893 Findlay et al. SNVs (id, pos, ref, alt, labels), as in
  evo2-cloud-operations.
- brca1_cpu_subsample.tsv: the 20-position, 60-SNV subsample for the CPU tier.
- clinvar_noncoding_nonsnv.tsv: the 4,644 ClinVar noncoding non-SNVs with labels and strata.
- clinvar_noncoding_nonsnv_windows.tsv.gz: their 8,192 bp reference and variant windows (GRCh38).
"""
from __future__ import annotations

import csv
import gzip
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evo2size.genome import read_first_fasta_record, snv_windows, variant_windows  # noqa: E402
from evo2size.pins import WINDOW  # noqa: E402

CODING = {"Missense", "Nonsense", "Synonymous"}
SUBSAMPLE_SEED = 20261009
SUBSAMPLE_POSITIONS = 20


def brca1_table(xlsx: Path) -> pd.DataFrame:
    """Same table as evo2-cloud-operations scripts/prepare_inputs.py."""
    raw = pd.read_excel(xlsx, header=2)
    table = pd.DataFrame({
        "id": [f"chr17:{p}:{r}>{a}" for p, r, a in zip(raw["position (hg19)"], raw["reference"], raw["alt"])],
        "pos": raw["position (hg19)"].astype(int),
        "ref": raw["reference"],
        "alt": raw["alt"],
        "consequence": raw["consequence"],
        "func_class": raw["func.class"],
        "function_score": raw["function.score.mean"],
        "cadd": raw["CADD.score"],
        "phylop": raw["phyloP (mammalian)"],
    })
    table["lof"] = (table["func_class"] == "LOF").astype(int)
    table["region"] = ["coding" if c in CODING else "noncoding" for c in table["consequence"]]
    if table["id"].duplicated().any():
        raise ValueError("Duplicate variant identifiers")
    return table


def cpu_subsample(table: pd.DataFrame) -> pd.DataFrame:
    """20 positions with all three SNVs scored: 10 with at least one LOF variant, 10 with none."""
    per_position = table.groupby("pos").agg(n=("id", "size"), any_lof=("lof", "max"))
    complete = per_position[per_position["n"] == 3]
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    chosen = []
    for has_lof in (1, 0):
        candidates = np.sort(complete.index[complete["any_lof"] == has_lof].to_numpy())
        chosen.extend(rng.choice(candidates, SUBSAMPLE_POSITIONS // 2, replace=False).tolist())
    return table[table["pos"].isin(chosen)].sort_values(["pos", "alt"])


def clinvar_noncoding_nonsnv(parquet: Path) -> pd.DataFrame:
    """Arc evoeval ClinVar rows that are noncoding (not CDS) and not single-nucleotide variants."""
    clinvar = pd.read_parquet(parquet)
    # One source row has identical reference and alternate alleles (chr6:112187567 GT>GT); it is
    # not a variant, so rows like it are dropped.
    selected = clinvar[(clinvar["genomic_element"] != "CDS") & (clinvar["variant_type"] != "SNV")
                       & (clinvar["ref"] != clinvar["alt"])].copy()
    selected["id"] = [f"{c}:{p}:{r}>{a}" for c, p, r, a in
                      zip(selected["chrom"], selected["pos"], selected["ref"], selected["alt"])]
    if selected["id"].duplicated().any():
        raise ValueError("Duplicate ClinVar variant identifiers")
    # Bootstrap groups: the gene, or the variant itself where no gene is given.
    has_gene = selected["GeneSymbol"].notna() & (selected["GeneSymbol"] != "")
    selected["group"] = selected["GeneSymbol"].where(has_gene, selected["id"])
    columns = ["id", "chrom", "pos", "ref", "alt", "ClinSigSimple", "genomic_element", "variant_type",
               "feature_lvl2", "consequence", "GeneSymbol", "group", "VariationID", "ReviewStatus"]
    return selected[columns].sort_values(["chrom", "pos", "ref", "alt"]).reset_index(drop=True)


def fasta_records(path: Path, wanted: set[str]):
    """Yield (name, sequence) for the wanted records of a gzipped multi-FASTA, one at a time."""
    name, parts = None, []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name in wanted:
                    yield name, "".join(parts)
                name, parts = line[1:].split()[0], []
            elif name in wanted:
                parts.append(line.strip())
    if name in wanted:
        yield name, "".join(parts)


def write_windows(variants: pd.DataFrame, genome: Path, out: Path) -> None:
    by_chrom = {c: rows for c, rows in variants.groupby("chrom")}
    windows: dict[str, tuple[str, str]] = {}
    for name, sequence in fasta_records(genome, set(by_chrom)):
        for row in by_chrom[name].itertuples():
            windows[row.id] = variant_windows(sequence, int(row.pos), row.ref, row.alt, WINDOW)
    missing = set(variants["id"]) - set(windows)
    if missing:
        raise ValueError(f"{len(missing)} variants have no chromosome in the genome, e.g. {sorted(missing)[:3]}")
    # A fixed gzip timestamp and no stored filename keep the file byte-identical across rebuilds.
    with out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as packed, \
            io.TextIOWrapper(packed, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "ref_window", "var_window"])
        for variant_id in variants["id"]:
            writer.writerow([variant_id, *windows[variant_id]])


def main() -> None:
    raw, derived = ROOT / "data/raw", ROOT / "data/derived"
    derived.mkdir(parents=True, exist_ok=True)

    table = brca1_table(raw / "41586_2018_461_MOESM3_ESM.xlsx")
    chromosome = read_first_fasta_record(raw / "GRCh37.p13_chr17.fna.gz")
    for row in table.itertuples():
        snv_windows(chromosome, row.pos, row.ref, row.alt, WINDOW)  # raises on a reference mismatch
    table.to_csv(derived / "brca1_variants.tsv", sep="\t", index=False)
    subsample = cpu_subsample(table)
    subsample.to_csv(derived / "brca1_cpu_subsample.tsv", sep="\t", index=False)
    print(f"BRCA1: {len(table)} SNVs ({table['lof'].sum()} LOF); CPU subsample {len(subsample)} SNVs "
          f"at {subsample['pos'].nunique()} positions ({subsample['lof'].sum()} LOF)")

    clinvar = clinvar_noncoding_nonsnv(raw / "evoeval_clinvar.parquet")
    clinvar.to_csv(derived / "clinvar_noncoding_nonsnv.tsv", sep="\t", index=False)
    write_windows(clinvar, raw / "hg38.fa.gz", derived / "clinvar_noncoding_nonsnv_windows.tsv.gz")
    print(f"ClinVar noncoding non-SNVs: {len(clinvar)} ({clinvar['ClinSigSimple'].sum()} pathogenic); "
          f"windows checked against GRCh38")


if __name__ == "__main__":
    main()
