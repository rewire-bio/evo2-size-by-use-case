"""Download one pinned Tectonic release and verify before execution."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import platform
import tarfile
import urllib.request

VERSION = "0.15.0"
# Computed from official release assets, retrieved 2026-10-01.
ARCHIVES = {
    ("Darwin", "arm64"): ("aarch64-apple-darwin", "24bd46566fa30d41101848405e9cbc4645edb92d8f857c9d21262174fb70cd33"),
    ("Darwin", "x86_64"): ("x86_64-apple-darwin", "dd42576eaa4c0df58c243dd78b7b864d9deb405ffdfcdadd1b79a31faceab747"),
    ("Linux", "aarch64"): ("aarch64-unknown-linux-musl", "1f59f9fb8eb65e8ba18658fc9016767e7d3e12488ded8b8fffa34254e51ce42c"),
    ("Linux", "x86_64"): ("x86_64-unknown-linux-musl", "dfb82876f2986862996e564fa507a9e576e0c1e3bee63c2c1bd677c2543e6407"),
}


def bootstrap(root: Path) -> Path:
    key = platform.system(), platform.machine()
    if key not in ARCHIVES:
        raise RuntimeError(f"No pinned Tectonic binary for {key}")
    target, expected = ARCHIVES[key]
    folder = root / ".tools" / f"tectonic-{VERSION}-{target}"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "release.tar.gz"
    if not archive.exists():
        url = f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{VERSION}/tectonic-{VERSION}-{target}.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read(80 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Tectonic archive checksum mismatch")
        archive.write_bytes(data)
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("Cached Tectonic archive checksum mismatch")
    # Extract only the verified executable, never arbitrary archive paths.
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as package:
        candidates = [m for m in package.getmembers() if Path(m.name).name == "tectonic" and m.isfile()]
        if len(candidates) != 1:
            raise RuntimeError("Unexpected Tectonic archive layout")
        source = package.extractfile(candidates[0])
        assert source is not None
        executable_bytes = source.read()
    executable = folder / "tectonic"
    if not executable.exists() or executable.read_bytes() != executable_bytes:
        executable.write_bytes(executable_bytes)
        executable.chmod(0o755)
    return executable


if __name__ == "__main__":
    print(bootstrap(Path(__file__).resolve().parents[1]))
