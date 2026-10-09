"""Every external artefact the pod loads, pinned by revision and SHA-256."""

# Hugging Face checkpoints. Arc's loader uses torch.load (pickle), so every file is hash-checked
# before loading. evo2_40b is published as two parts that are concatenated into one file.
WEIGHTS = {
    "evo2_7b": {
        "repo_id": "arcinstitute/evo2_7b",
        "revision": "bda0089f92582d5baabf0f22d9fc85f3588f6b58",
        "files": {"evo2_7b.pt": "c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4"},
    },
    "evo2_20b": {
        "repo_id": "arcinstitute/evo2_20b",
        "revision": "8b0f0a9a70c66367ed181a17d049b95699a28fed",
        "files": {"evo2_20b.pt": "979b345dc700a9cae6b63f570d9b3f438e31a68dd8dc0bc0d8d2a0fcd3caf791"},
    },
    "evo2_40b": {
        "repo_id": "arcinstitute/evo2_40b",
        "revision": "d529aa57c30771814217ad89baaeaf6e2315c7d7",
        "files": {
            "evo2_40b.pt.part0": "3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485",
            "evo2_40b.pt.part1": "bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5",
        },
    },
}

# Arc's expected forward-pass losses in evo2/test/test_evo2.py at commit 53f1959.
FORWARD_TEST_LOSS = {
    "evo2_7b": 0.3476563,
    "evo2_20b": 0.2166748046875,
    "evo2_40b": 0.2159424,
}
FORWARD_TEST_TOLERANCE = 1e-3

WINDOW = 8192
