"""Build the paper with an explicit engine and TeX bundle version."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from bootstrap_tectonic import VERSION, bootstrap

# This versioned bundle name is intentionally fixed; never use the default URL.
BUNDLE = "https://relay.fullyjustified.net/default_bundle_v33.tar"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = ["paper/generated/metrics.tex", "paper/generated/table.tex", "paper/figures/convergence.pdf"]
    for name in required:
        if not (root / name).is_file():
            raise SystemExit(f"Missing generated input {name}; run make analysis first")
    executable = bootstrap(root)
    version = subprocess.check_output([str(executable), "--version"], text=True).strip()
    # The official macOS 0.15.0 asset prints both lower- and title-case banners.
    if not re.fullmatch(rf"(?:tectonic {re.escape(VERSION)}\s*)+", version, re.IGNORECASE):
        raise SystemExit(f"Unexpected Tectonic version: {version}")
    output = root / "paper/build"
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "0"
    subprocess.run([str(executable), "--web-bundle", BUNDLE, "--untrusted", "--keep-logs", "--outdir", str(output), "main.tex"], cwd=root / "paper", env=environment, check=True)
    pdf = output / "main.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise SystemExit("Paper build did not produce a valid PDF header")
    print(pdf)


if __name__ == "__main__":
    main()
