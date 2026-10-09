"""Build the paper's figures, tables and numeric macros from results.json and operations.json.

  uv run --frozen python scripts/analyse.py --results results/full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Reference palette (dataviz skill), light mode: categorical slots 1 to 3, fixed order by model.
MODEL_COLOURS = {"evo2_40b": "#2a78d6", "evo2_20b": "#eb6834", "evo2_7b": "#1baf7a"}
MODEL_LABELS = {"evo2_40b": "Evo 2 40B", "evo2_20b": "Evo 2 20B", "evo2_7b": "Evo 2 7B"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STRATA = {"all": "All noncoding non-SNVs", "splice_site": "Splice site", "intergenic": "Intergenic"}


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_model_size(g3: dict, out: Path) -> bool:
    """AUROC with 95% interval for each model, one panel per stratum that has enough positives."""
    comparison = g3.get("comparison")
    if not comparison:
        return False
    panels = [s for s in STRATA if "auroc" in comparison.get(s, {})]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels) + 0.6, 2.2), sharey=True, squeeze=False)
    models = [m for m in MODEL_LABELS if m in g3["models_scored"]]
    for ax, stratum in zip(axes[0], panels):
        entry = comparison[stratum]["auroc"]
        for y, model in enumerate(models):
            point, (low, high) = entry[model]["auroc"], entry[model]["ci95"]
            ax.plot([low, high], [y, y], color=MODEL_COLOURS[model], linewidth=2, solid_capstyle="round")
            ax.plot(point, y, "o", color=MODEL_COLOURS[model], markersize=6, markeredgecolor="white",
                    markeredgewidth=1.5)
            ax.annotate(f"{point:.3f}", (high, y), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK)
        ax.set_title(f"{STRATA[stratum]}\n(n = {comparison[stratum]['n']}, "
                     f"{comparison[stratum]['n_pathogenic']} pathogenic)", fontsize=8, color=INK)
        ax.set_xlabel("AUROC (95% interval)", fontsize=8, color=INK_MUTED)
        style(ax)
    axes[0][0].set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def cost_rows(operations: dict, config: dict) -> list[dict]:
    labels = {t["id"]: t["label"] for t in config["tiers"]}
    rows = []
    for tier_id, op in operations.items():
        for task, entry in op.get("tasks", {}).items():
            if entry.get("usd_per_1000_variants") is None:
                continue
            kind, model = task.split(":")
            rows.append({"tier": tier_id, "label": labels.get(tier_id, tier_id), "task": kind, "model": model,
                         "usd_per_1000": entry["usd_per_1000_variants"],
                         "seconds_per_window": entry["seconds_per_window"],
                         "variants_scored": entry["variants_scored"],
                         "stopped_by_time_limit": entry.get("stopped_by_time_limit", False)})
    return rows


def fig_cost(rows: list[dict], out: Path) -> bool:
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: r["usd_per_1000"])
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(rows) + 0.8))
    names = [f"{MODEL_LABELS[r['model']]}, {r['tier']} ({r['task'].replace('_', ' ')})" for r in rows]
    values = [r["usd_per_1000"] for r in rows]
    # Dots, not bars: the CPU tier can be orders of magnitude dearer, which needs a log axis, and bar
    # length has no meaning on a log axis.
    colours = [MODEL_COLOURS[r["model"]] for r in rows]
    ax.scatter(values, range(len(rows)), c=colours, s=40, edgecolors="white", linewidths=1.5, zorder=3)
    for y, value in enumerate(values):
        ax.annotate(f"USD {value:.2f}", (value, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(min(values) / 2, max(values) * 4)
    ax.set_xlabel("Scoring cost per 1,000 variants, USD (log scale)", fontsize=8, color=INK_MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def tex_number(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def macros(results: dict, rows: list[dict]) -> str:
    lines = ["% Generated by scripts/analyse.py; do not edit."]

    def define(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for gap, prefix in (("g1_cpu", "Cpu"), ("g2_l4", "Lfour")):
        entry = results.get(gap, {})
        comparison = entry.get("comparison") or {}
        define(f"{prefix}ForwardPassed", "yes" if entry.get("forward_test_passed") else "no")
        define(f"{prefix}VariantsScored", str(entry.get("variants_scored") or 0))
        define(f"{prefix}Spearman", tex_number(comparison.get("agreement", {}).get("spearman"), 4))
        define(f"{prefix}AurocDiff", tex_number(comparison.get("auroc_abs_difference"), 4))
        equivalent = entry.get("hardware_equivalent")
        define(f"{prefix}Equivalent", "n/a" if equivalent is None else ("yes" if equivalent else "no"))
    g3 = results.get("g3_indels", {}).get("comparison", {}).get("all", {}).get("auroc", {})
    for model, name in (("evo2_40b", "Forty"), ("evo2_20b", "Twenty"), ("evo2_7b", "Seven")):
        define(f"Indel{name}Auroc", tex_number(g3.get(model, {}).get("auroc")))
        diff = g3.get(model, {}).get("difference_vs_evo2_40b")
        define(f"Indel{name}DiffVsForty", tex_number(None if diff is None else -diff))
    # LaTeX macro names take letters only: \CostTTwoSevenBrca, \CostTThreeTwentyIndels, ...
    words = {"1": "One", "2": "Two", "3": "Three", "4": "Four", "7b": "Seven", "20b": "Twenty", "40b": "Forty"}
    for row in rows:
        tier = "".join(words.get(ch, ch.upper()) for ch in row["tier"])
        model = words[row["model"].replace("evo2_", "")]
        task = "".join(part.capitalize() for part in row["task"].replace("brca1", "brca_one").split("_"))
        define(f"Cost{tier}{model}{task}", f"{row['usd_per_1000']:.2f}")
    return "\n".join(lines) + "\n"


def cost_table(rows: list[dict]) -> str:
    out = ["% Generated by scripts/analyse.py; do not edit.", "\\begin{tabular}{llrrr}", "\\toprule",
           "Tier & Model and task & Variants scored & s per window & USD per 1,000 \\\\", "\\midrule"]
    for r in sorted(rows, key=lambda r: r["usd_per_1000"]):
        note = " (time limit)" if r["stopped_by_time_limit"] else ""
        out.append(f"{r['tier']} & {MODEL_LABELS[r['model']]}, {r['task'].replace('_', ' ')} & "
                   f"{r['variants_scored']:,}{note} & {r['seconds_per_window']:.2f} & {r['usd_per_1000']:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.results / "results.json").read_text())
    operations = json.loads((args.results / "operations.json").read_text())
    config = json.loads((ROOT / "configs/full.json").read_text())
    figures, generated = ROOT / "paper/figures", ROOT / "paper/generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "evo2size",
                         "path.simplify": False})
    rows = cost_rows(operations, config)
    made = {"model_size": fig_model_size(results.get("g3_indels", {}), figures / "model_size_indels.pdf"),
            "cost": fig_cost(rows, figures / "cost_per_1000.pdf")}
    (generated / "values.tex").write_text(macros(results, rows))
    (generated / "cost_table.tex").write_text(cost_table(rows))
    print(json.dumps({"figures": made, "cost_rows": len(rows)}))


if __name__ == "__main__":
    main()
