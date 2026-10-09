"""Summarise Evo 2 Supplementary Data 1 by task: top three models plus every Evo 2 size.

Downloads the paper's supplementary bundle from Europe PMC, checks the table's MD5 against the
value printed in the article XML, and prints the rows quoted in literature/evidence-register.md.
"""
import hashlib
import io
import sys
import urllib.request
import zipfile

import pandas as pd

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13128491/supplementaryFiles"
NAME = "41586_2026_10176_MOESM3_ESM.xlsx"
MD5 = "653a2561dc0a31d4f0faf58db833a317"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=600) as response:
        bundle = zipfile.ZipFile(io.BytesIO(response.read()))
    data = bundle.read(NAME)
    if hashlib.md5(data).hexdigest() != MD5:
        sys.exit(f"{NAME}: MD5 does not match {MD5}")
    table = pd.read_excel(io.BytesIO(data))
    keys = ["Data", "Genomic_Feature", "Variant_Type", "Sub_category"]
    table["task"] = table[keys].fillna("").agg("|".join, axis=1)
    for task, rows in table.groupby("task"):
        rows = rows.sort_values("AUROC", ascending=False)
        shown = pd.concat([rows.head(3), rows[rows.Model.str.startswith("Evo 2")]])
        print(f"\n{task} (n={rows.N_variants_total.iloc[0]})")
        for _, row in shown.drop_duplicates("Model").iterrows():
            print(f"  {row.Model:22s} AUROC {row.AUROC:.3f}  AUPRC {row.AUPRC:.3f}")


if __name__ == "__main__":
    main()
