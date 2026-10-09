.PHONY: data inputs test smoke experiment collect analysis paper reproduce

# Offline steps (no cloud spend)
data:
	uv run --frozen python scripts/data.py --fetch

inputs: data
	uv run --frozen python scripts/prepare_inputs.py

test: inputs
	uv run --frozen python -m unittest discover -s tests -v
	bash -n cloud/common.sh cloud/remote-run.sh cloud/runpod/launch.sh

# Wiring check only: offline tests plus result assembly on synthetic outputs. No pods.
smoke: test

collect:
	uv run --frozen python scripts/experiment.py --config configs/full.json --output results/full --collect-only

analysis:
	uv run --frozen python scripts/analyse.py --results results/full

paper:
	uv run --frozen python scripts/build_paper.py

# Paid step: launches Runpod pods for every tier. Needs RUNPOD_API_KEY in the environment.
experiment: test
	uv run --frozen python scripts/experiment.py --config configs/full.json --output results/full

reproduce: experiment analysis paper
