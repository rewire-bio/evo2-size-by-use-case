"""Sequence windows shared by the pod job and the offline input preparation. Standard library only."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path


def read_first_fasta_record(path: Path) -> str:
    """Return the first record of a (gzipped) FASTA file as one string, case preserved."""
    opener = gzip.open if path.suffix == ".gz" else open
    parts: list[str] = []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if parts:
                    break
                continue
            parts.append(line.strip())
    return "".join(parts)


def snv_windows(chromosome: str, pos: int, ref: str, alt: str, window: int) -> tuple[str, str]:
    """Reference and variant windows around a 1-based SNV, exactly as evo2-cloud-operations W1.

    Matches Arc's BRCA1 notebook: the window starts window//2 bases before the variant and the
    variant sits at index window//2 (or at pos-1 near the start).
    """
    p = pos - 1
    start = max(0, p - window // 2)
    end = min(len(chromosome), p + window // 2)
    ref_seq = chromosome[start:end]
    offset = min(window // 2, p)
    if ref_seq[offset].upper() != ref.upper():
        raise ValueError(f"Reference mismatch at {pos}: genome has {ref_seq[offset]}, table has {ref}")
    var_seq = ref_seq[:offset] + alt + ref_seq[offset + 1:]
    return ref_seq, var_seq


def centred_window(chromosome: str, pos: int, ref: str, allele: str, window: int) -> str:
    """A window of exactly `window` bases with `allele` in place of `ref`, centred on the allele.

    Arc's non-SNV method (Supplementary Information A.3.12): keep the scored length fixed whatever
    the variant length, by centring on the allele and adding or removing context at both edges.
    `pos` is the 1-based VCF position of the first base of `ref`.
    """
    p = pos - 1
    if chromosome[p:p + len(ref)].upper() != ref.upper():
        raise ValueError(f"Reference mismatch at {pos}: genome has {chromosome[p:p + len(ref)]}, table has {ref}")
    if len(allele) >= window:
        raise ValueError(f"Allele of length {len(allele)} does not fit a {window} bp window")
    left = (window - len(allele)) // 2
    right = window - len(allele) - left
    if p - left < 0 or p + len(ref) + right > len(chromosome):
        raise ValueError(f"Window at {pos} runs off the chromosome")
    return chromosome[p - left:p] + allele + chromosome[p + len(ref):p + len(ref) + right]


def variant_windows(chromosome: str, pos: int, ref: str, alt: str, window: int) -> tuple[str, str]:
    """Reference and variant windows of equal length for any SNV, insertion, deletion or MNV."""
    return centred_window(chromosome, pos, ref, ref, window), centred_window(chromosome, pos, ref, alt, window)


def read_variants(path: Path) -> list[dict]:
    """Read the prepared BRCA1 variant table (id, pos, ref, alt)."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["pos"] = int(row["pos"])
    return rows


def read_window_pairs(path: Path) -> list[dict]:
    """Read prepared window pairs (id, ref_window, var_window) from a gzipped TSV."""
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
