"""Pinned checkpoint download, hash check and shard joining. No torch, so it can be tested offline."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import pins


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hub_download(**kwargs) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def fetch_weights(name: str, cache_dir: Path, download=_hub_download) -> Path:
    """Download a checkpoint at its pinned revision, check every file's hash, and join parts.

    Refuses to return a path unless every hash matches, because an unverified pickle could run code.
    """
    pin = pins.WEIGHTS[name]
    paths = []
    for filename, sha256 in pin["files"].items():
        path = Path(download(repo_id=pin["repo_id"], filename=filename, revision=pin["revision"],
                             cache_dir=str(cache_dir)))
        actual = sha256_file(path)
        if actual != sha256:
            raise RuntimeError(f"SHA-256 mismatch for {pin['repo_id']}/{filename}: expected {sha256}, got {actual}")
        paths.append(path)
    if len(paths) == 1:
        return paths[0]
    joined = cache_dir / f"{name}.pt"
    with joined.open("wb") as out:
        for path in paths:
            with path.open("rb") as part:
                while chunk := part.read(64 * 1024 * 1024):
                    out.write(chunk)
            # The hub returns a symlink into its blob store; free the blob, not just the link.
            blob = path.resolve()
            path.unlink()
            blob.unlink(missing_ok=True)
    return joined
