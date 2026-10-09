"""Acquire declared datasets explicitly, and always verify their SHA256."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verify_data(manifest: Path, root: Path, fetch: bool = False) -> None:
    entries = json.loads(manifest.read_text())["datasets"]
    for entry in entries:
        for key in ("path", "source", "sha256", "license", "provenance"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"Dataset entry missing {key}")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dataset paths must be relative without parent traversal")
        original = root / relative
        for part in [original, *original.parents]:
            if part == root:
                break
            if part.is_symlink():
                raise ValueError("Dataset symlinks are not permitted")
        path = original.resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Dataset paths must stay within the study directory")
        expected = entry["sha256"]
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Dataset requires a lowercase SHA256 digest")
        if not path.exists() and fetch:
            if not entry["source"].startswith("https://"):
                raise ValueError("Dataset retrieval requires an HTTPS URL")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".download")
            try:
                with urllib.request.urlopen(entry["source"], timeout=300) as source, temporary.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                if _sha256(temporary) != expected:
                    raise ValueError(f"Downloaded dataset checksum mismatch: {entry['path']}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not path.is_file():
            raise ValueError(f"Missing dataset: {entry['path']}; run scripts/data.py --fetch explicitly")
        if _sha256(path) != expected:
            raise ValueError(f"Dataset checksum mismatch: {entry['path']}")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_data(root / "data/manifest.json", root, args.fetch)
