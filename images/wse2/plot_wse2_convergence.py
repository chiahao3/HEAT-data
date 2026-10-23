"""Plot HEAT versus ordinary-GA WSe2 convergence from the exported CSVs.

Edit USER PARAMETERS below, then run:
    python plot_wse2_convergence.py

Both member_metrics.csv and generation_metrics.csv are supported. Total loss
is cumulative best-so-far. Formation energy and the weighted image penalty
belong to that same best-loss candidate; these components need not decrease
monotonically. HEAT stops at generation 128; the conventional curve includes
all complete generations. No reference energy is subtracted.
Use --loss-only to regenerate only the total-loss figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
HEAT_CSV = BASE_DIR / "eat" / "member_metrics.csv"
ORDINARY_CSV = BASE_DIR / "ordinary" / "member_metrics.csv"
OUTPUT_DIR = BASE_DIR
ENERGY_REFERENCE_CSV = BASE_DIR.parents[1] / "scripts/misc/baseline_ptycho_guided_320/objective_comparison.csv"
TOTAL_LOSS_REFERENCE_JSON = BASE_DIR.parents[1] / "scripts/misc/baseline_ptycho_guided_320/run_summary.json"

# Generation zero is the initial population. Optional shared input cap;
# otherwise each method retains its own endpoint.
MAX_GENERATION_OVERRIDE = None
HEAT_FINAL_GENERATION = 128
INCLUDE_PARTIAL_GENERATIONS = False
# Used to identify complete generations when reading member_metrics.csv.
POPULATION_SIZE = 50

LOSS_Y_MIN, LOSS_Y_MAX = None, None
ENERGY_Y_MIN, ENERGY_Y_MAX = None, None
IMAGE_Y_MIN, IMAGE_Y_MAX = None, None

HEAT_COLOR = "#1f1f1f"
ORDINARY_COLOR = "#2171b5"
BEST_LINE_WIDTH = 2.8
FIGURE_SIZE = (4.0, 4.2)
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14
GENERATION_TICK_STEP = 20
OUTPUT_DPI = 300


@dataclass(frozen=True)
class Candidate:
    generation: int
    member_index: int
    score: float
    formation_energy: float
    image_penalty: float


def load_results(csv_path: Path) -> list[Candidate]:
    """Load valid candidates, excluding incomplete generations by default."""
    csv_path = Path(csv_path).expanduser()
    by_generation: dict[int, list[Candidate]] = {}
    complete_flags: dict[int, bool] = {}
    seen = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        generation_csv = "generation_best_score" in (reader.fieldnames or [])
        columns = (
            ("generation_best_score", "best_formation_energy_per_atom", "best_image_term")
            if generation_csv
            else ("score", "formation_energy_per_atom", "image_term")
        )
        required = {"generation", *columns}
        required.add("complete" if generation_csv else "member_index")
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                generation = int(row["generation"])
                member = int(row.get("best_member_index", "0") if generation_csv else row["member_index"])
                values = [float(row[column]) for column in columns]
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric value in {csv_path}, line {line_number}") from exc
            key = (generation, member)
            if key in seen:
                raise ValueError(f"Duplicate generation/member {key} in {csv_path}")
            seen.add(key)
            if generation_csv:
                complete_flags[generation] = row["complete"].strip().lower() in {"true", "1"}
            if row.get("valid", "true").strip().lower() in {"false", "0"}:
                continue
            if not all(np.isfinite(value) for value in values):
                continue
            by_generation.setdefault(generation, []).append(Candidate(generation, member, *values))

    # Count all saved members, including invalid candidates, when deciding
    # whether a generation is complete.
    saved_counts: dict[int, int] = {}
    for generation, _ in seen:
        saved_counts[generation] = saved_counts.get(generation, 0) + 1
    candidates = []
    omitted = []
    for generation, members in sorted(by_generation.items()):
        if MAX_GENERATION_OVERRIDE is not None and generation > MAX_GENERATION_OVERRIDE:
            continue
        complete = (
            complete_flags[generation] if generation_csv
            else saved_counts[generation] == POPULATION_SIZE
        )
        if not INCLUDE_PARTIAL_GENERATIONS and not complete:
            omitted.append(generation)
            continue
        candidates.extend(sorted(members, key=lambda candidate: candidate.member_index))
    if omitted:
        print(f"[incomplete generations omitted] {csv_path.name}: {omitted}")
    if not candidates:
        raise ValueError(f"No usable candidate rows remain in {csv_path}")
    return candidates


def best_candidates_so_far(candidates: list[Candidate]) -> list[Candidate]:
    """Select by total loss and carry all components from the selected member."""
    by_generation: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_generation.setdefault(candidate.generation, []).append(candidate)
    best = None
    selected = []
    for generation, members in sorted(by_generation.items()):
        generation_best = min(members, key=lambda candidate: (candidate.score, candidate.member_index))
        if best is None or generation_best.score < best.score:
            best = generation_best
        # Use the plotted generation while preserving the winner's metrics.
        selected.append(Candidate(generation, best.member_index, best.score, best.formation_energy, best.image_penalty))
    return selected


def style_axis(axis):
    axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def load_energy_reference():
    """Read the recorded energy-only reference from the bundled results."""
    with ENERGY_REFERENCE_CSV.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["structure"] == "previous_energy_only_relaxed"]
    if len(rows) != 1:
        raise ValueError("Expected one recorded energy-only reference")
    return float(rows[0]["formation_energy_per_atom"])


def plot_metric(heat, ordinary, attribute, ylabel, y_limits, output_path, *, ideal_reference=False, endpoint_markers=False):
    """Match the H-sublattice figure style, drawing HEAT over Ordinary."""
    fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    # Draw Ordinary first and give HEAT higher z-order wherever they overlap.
    ordinary_line, = axis.plot(
        [candidate.generation for candidate in ordinary],
        [getattr(candidate, attribute) for candidate in ordinary],
        color=ORDINARY_COLOR, linewidth=BEST_LINE_WIDTH, label="Conventional", zorder=2,
    )
    heat_line, = axis.plot(
        [candidate.generation for candidate in heat],
        [getattr(candidate, attribute) for candidate in heat],
        color=HEAT_COLOR, linewidth=BEST_LINE_WIDTH, label="HEAT", zorder=3,
    )
    if endpoint_markers:
        for candidates, color in ((ordinary, ORDINARY_COLOR), (heat, HEAT_COLOR)):
            end = candidates[-1]
            axis.plot(end.generation, getattr(end, attribute), marker="o",
                      markersize=6.5, color=color, linestyle="none", zorder=5)
    legend_handles = [heat_line, ordinary_line]
    if attribute == "score" and not ideal_reference:
        reference = json.loads(TOTAL_LOSS_REFERENCE_JSON.read_text())
        if reference['status'] != 'complete':
            raise ValueError("The ptychography-guided reference run is not complete")
        reference_line = axis.axhline(
            float(reference['final_score']), color="#d62728", linestyle="--",
            linewidth=2.0, label="Reference", zorder=1,
        )
        legend_handles.append(reference_line)
        print(f"[reference total loss] {reference['final_score']:.12f}")
    if ideal_reference and attribute != "score":
        raise ValueError("The ideal total-loss reference applies only to score")
    if attribute == "formation_energy" or ideal_reference:
        reference_energy = load_energy_reference()
        reference_line = axis.axhline(
            reference_energy, color="#d62728", linestyle="--",
            linewidth=2.0, label="Ideal reference" if ideal_reference else "Reference", zorder=1,
        )
        legend_handles.append(reference_line)
        print(f"[relaxed reference] {reference_energy:.12f} eV/atom")
    axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    axis.set_xlabel("Generation", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.set_ylim(bottom=y_limits[0], top=y_limits[1])
    first = min(heat[0].generation, ordinary[0].generation)
    last = max(heat[-1].generation, ordinary[-1].generation)
    margin = max(0.7, 0.018 * (last - first)) if endpoint_markers else 0.7
    axis.set_xlim(first - margin, last + margin)
    tick_step = GENERATION_TICK_STEP
    if (last - first) / tick_step > 8:
        tick_step = next(step for step in (50, 100, 200, 500, 1000, 2000, 5000)
                         if (last - first) / step <= 6)
    first_tick = int(np.ceil(first / tick_step)) * tick_step
    axis.set_xticks(np.arange(first_tick, last + 1, tick_step))
    style_axis(axis)
    if ideal_reference:
        axis.legend(handles=legend_handles, frameon=False, fontsize=12,
                    handlelength=1.8, loc="center right")
    else:
        axis.legend(handles=legend_handles, frameon=False, fontsize=TICK_LABEL_SIZE)
    fig.savefig(output_path, dpi=OUTPUT_DPI, facecolor="white")
    plt.close(fig)
    print(f"[output] {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-only", action="store_true")
    args = parser.parse_args()
    heat = best_candidates_so_far(load_results(HEAT_CSV))
    ordinary = best_candidates_so_far(load_results(ORDINARY_CSV))
    heat = [candidate for candidate in heat if candidate.generation <= HEAT_FINAL_GENERATION]
    print(f"[generation endpoints] HEAT={heat[-1].generation}, Conventional={ordinary[-1].generation}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, path, candidates in (("HEAT", HEAT_CSV, heat), ("Conventional", ORDINARY_CSV, ordinary)):
        print(f"[{label} source] {Path(path).expanduser().resolve()}")
        print(
            f"[{label}] generations={candidates[0].generation}-{candidates[-1].generation} "
            f"best_loss={candidates[-1].score:.12f} "
            f"formation_energy={candidates[-1].formation_energy:.12f} "
            f"weighted_image_penalty={candidates[-1].image_penalty:.12f}"
        )
    # Each curve ends at its own final generation.
    for attribute, ylabel, limits, filename in (
        ("score", "Loss", (LOSS_Y_MIN, LOSS_Y_MAX), "wse2_total_loss_convergence.png"),
        ("formation_energy", r"$E_{\rm f}$ (eV/atom)", (ENERGY_Y_MIN, ENERGY_Y_MAX), "wse2_formation_energy_convergence.png"),
        ("image_penalty", "Ptycho image penalty", (IMAGE_Y_MIN, IMAGE_Y_MAX), "wse2_image_penalty_convergence.png"),
    ):
        if args.loss_only and attribute != "score":
            continue
        plot_metric(heat, ordinary, attribute, ylabel, limits, OUTPUT_DIR / filename,
                    endpoint_markers=(attribute == "score"))


if __name__ == "__main__":
    main()
