"""Render scheduling progress and timelines from saved evaluations."""

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.jobshop.plot import draw
from tools.render_gallery import ROOT, RUNS, load, records


def render(root):
    spec = next(item for item in RUNS if item["key"] == "jobshop")
    data = records(spec, root)
    evaluations = [
        load(root / f"{attempt:03d}" / "evaluation.json") for attempt in range(len(data["values"]))
    ]
    best = load(root / "summary.json")["best_attempt"]
    trials = [load(root / name / "evaluation.json")["trials"][0] for name in ("000", best)]
    valid = [i for i, result in enumerate(evaluations) if result["valid"]]
    failed = [i for i, result in enumerate(evaluations) if not result["valid"]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.3), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.22, h_pad=0.22, wspace=0.08, hspace=0.08)
    fig.suptitle("Flint · factory scheduling", x=0.055, ha="left", fontsize=24)
    progress, attempts = axes[0]
    color = spec["accent"]
    progress.step(data["hours"], data["values"], where="post", color=color, lw=2)
    progress.set_title("Best schedule", loc="left")
    progress.set_ylabel("Finishing time · lower is better")
    positions = [(0.04, 0.38), (0.38, 0.51), (0.59, 0.20)]
    for (attempt, title, _), position in zip(spec["milestones"], positions):
        value = data["values"][attempt]
        progress.scatter(data["hours"][attempt], value, color=color, s=24, zorder=3)
        progress.annotate(
            f"{fill(title, width=22)}\n{value:,.0f}",
            (data["hours"][attempt], value),
            xytext=position,
            textcoords="axes fraction",
            fontsize=10,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )
    attempts.step(data["hours"], data["values"], where="post", color=color, lw=2)
    attempts.scatter(
        [data["hours"][i] for i in valid],
        [-evaluations[i]["mean_score"] for i in valid],
        s=15,
        color="#7c8794",
        label="Scored attempt",
    )
    for index, attempt in enumerate(failed):
        attempts.axvline(
            data["hours"][attempt],
            color="#ba534a",
            alpha=0.2,
            label="Failed attempt" if index == 0 else None,
        )
    attempts.set_title(f"Every attempt · {len(valid) - 1} scored, {len(failed)} failed", loc="left")
    attempts.set_ylabel("Finishing time")
    attempts.legend(frameon=False, fontsize=9)
    for axis in axes[0]:
        axis.set_xlabel("Model time (hours)")
        axis.margins(x=0.02)
        axis.grid(alpha=0.12)
    draw(axes[1], trials)
    for axis, title, trial in zip(axes[1], ["Baseline", "Flint"], trials):
        axis.set_title(f"{title} · finishes at {trial['makespan']:,}", loc="left")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "svg"):
        fig.savefig(root / f"search-progress.{extension}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="?", default=ROOT / "examples/2026-10-20-jobshop")
    render(parser.parse_args().run)
