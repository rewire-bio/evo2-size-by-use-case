"""Checks to run just before a paid recorded run, so credit is not spent on a run that cannot finish.

  set -a; source ~/.config/evo2ops/env; set +a
  uv run --frozen python scripts/preflight.py --config configs/full.json

Fails (exit 1) if any of these is not met:
- the Runpod balance covers the planned spend for the tiers to be run, plus one failed attempt of
  the dearest tier (Runpod stops pods when the balance reaches zero, which would lose the attempt);
- no other pods are running on the account (another study's pods share the balance);
- every GPU type the tiers ask for shows Secure Cloud stock for pods.
Read-only: it creates nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# Planned cost per tier in USD, from protocol.md "Budgets" (T2 raised by amendment 1).
PLANNED_USD = {"T1a": 1.0, "T1b": 6.0, "T2": 3.6, "T3": 14.0, "T4": 26.0}


def runpod(url: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                                              "Content-Type": "application/json",
                                              "User-Agent": "evo2-size-by-use-case-preflight/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collect import load_config

    config = load_config(args.config)
    tiers = [t for t in config["tiers"] if t["id"] in config["run_tiers"]]
    problems = []

    planned = sum(PLANNED_USD[t["id"]] for t in tiers)
    needed = planned + max(PLANNED_USD[t["id"]] for t in tiers)
    me = runpod("https://api.runpod.io/graphql", {"query": "query { myself { clientBalance currentSpendPerHr } }"})
    balance = me["data"]["myself"]["clientBalance"]
    print(f"Balance USD {balance:.2f}; planned USD {planned:.2f} for {[t['id'] for t in tiers]}; "
          f"needed with one failed attempt USD {needed:.2f}")
    if balance < needed:
        problems.append(f"balance USD {balance:.2f} is below USD {needed:.2f}: top up first")

    pods = runpod("https://api.runpod.io/v2/pods").get("pods", [])
    if pods:
        names = ", ".join(f"{p['name']} ({p['status']})" for p in pods)
        problems.append(f"{len(pods)} pod(s) already on the account: {names}")

    catalog = runpod("https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE")
    stock = {g["id"]: g.get("availability", "NONE") for g in catalog.get("gpus", catalog if isinstance(catalog, list) else [])}
    for tier in tiers:
        for gpu in json.loads(tier["env"].get("GPU_TYPE_IDS", "[]")):
            level = stock.get(gpu, "NONE")
            print(f"{tier['id']}: {gpu} Secure Cloud stock {level}")
            if level == "NONE":
                problems.append(f"{tier['id']}: no Secure Cloud stock for {gpu}")

    if problems:
        print("\nNot ready:\n- " + "\n- ".join(problems))
        sys.exit(1)
    print("\nReady to run.")


if __name__ == "__main__":
    main()
