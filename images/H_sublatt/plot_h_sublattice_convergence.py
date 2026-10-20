"""Compare GEAT and ordinary-GA H-sublattice convergence by generation.

Candidate formation energies are read directly from each algorithm's
``genetic_optimization_log.txt``. Each curve is the cumulative minimum
formation-energy gap per H atom to the common relaxed ground-state reference.
Logged energies and reference-hit detection retain their original per-total-atom
normalization; only the plotted energies are converted.
"""

from __future__ import annotations

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
GEAT_GENETIC_LOG = BASE_DIR / "eat" / "genetic_optimization_log.txt"
ORDINARY_GENETIC_LOG = BASE_DIR / "ordinary" / "genetic_optimization_log.txt"
OUTPUT_DIR = BASE_DIR

# The H-sublattice genetic log does not include the reference energy in its
# ga_start record. This is the common relaxed-ground-truth value previously
# recorded for these runs. Update it here if the reference calculation changes.
GROUND_TRUTH_FORMATION_ENERGY_PER_ATOM = -0.2228974413906268
SCAFFOLD_ATOM_COUNT = 90  # 30 Mg + 60 B
REFERENCE_H_ATOM_COUNT = 240

# Set an integer to cap both curves at that generation, inclusive. None shows
# every completed generation available in each source.
MAX_GENERATION_OVERRIDE = 20
# Once both methods have reached the reference, retain this many additional
# generations after the later hit and stop the common plot there.
GENERATIONS_AFTER_BOTH_HIT = 2
# Preserve the GA's zero-based numbering: generation 0 is the initial
# population, and generation 1 is the first evolved population.
GENERATION_NUMBERING = "zero_based"

ENERGY_Y_MIN = -0.01
ENERGY_Y_MAX = None
# A hit means the relaxed EAT-minus-ground-truth difference is strictly < 0.
GROUND_TRUTH_HIT_TOLERANCE_EVPATOM = 1.0e-4

GEAT_COLOR = "#1f1f1f"
ORDINARY_COLOR = "#2171b5"
BEST_LINE_WIDTH = 2.8
FIGURE_SIZE = (4.0, 4.2)
HORIZONTAL_REFERENCE_LINE_WIDTH = 3.0
VERTICAL_REFERENCE_LINE_WIDTH = 2.6
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14

@dataclass(frozen=True)
class Candidate:
    generation: int
    formation_energy: float
    delta_energy: float
    h_atom_count: int


def parse_log_record(line: str) -> tuple[str, dict[str, str]]:
    """Parse one ``event | key=value`` genetic-log record."""
    pieces = [piece.strip() for piece in line.split("|") if piece.strip()]
    if not pieces:
        return "", {}
    fields = {}
    for piece in pieces[1:]:
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        fields[key.strip()] = value.strip()
    return pieces[0], fields


def load_results(genetic_log: Path) -> tuple[float, list[Candidate]]:
    """Load hard-relaxed member formation energies from a genetic log."""
    ground_truth_energy = float(GROUND_TRUTH_FORMATION_ENERGY_PER_ATOM)
    candidates = []
    with genetic_log.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            event, fields = parse_log_record(line)
            if event not in {"member_done", "elite_copied", "imported_seed_used"}:
                continue
            if "generation" not in fields or "formation_energy_per_atom" not in fields:
                continue
            try:
                generation = int(fields["generation"])
                formation_energy = float(fields["formation_energy_per_atom"])
                # HEAT reports realized H atoms separately from its soft sites.
                # In the ordinary GA, every site is a fully occupied H atom.
                h_atom_count = int(fields.get("realized_atom_count", fields["n_sites"]))
            except ValueError as exc:
                raise ValueError(
                    f"Malformed generation or formation energy in {genetic_log} "
                    f"on line {line_number}"
                ) from exc
            if not np.isfinite(formation_energy):
                continue
            if h_atom_count <= 0:
                raise ValueError(f"Cannot normalize by H count {h_atom_count}")
            candidates.append(
                Candidate(
                    generation=generation,
                    formation_energy=formation_energy,
                    delta_energy=formation_energy - ground_truth_energy,
                    h_atom_count=h_atom_count,
                )
            )

    if not candidates:
        raise ValueError(
            f"No records with generation and formation_energy_per_atom found in {genetic_log}"
        )
    if MAX_GENERATION_OVERRIDE is not None:
        generation_limit = int(MAX_GENERATION_OVERRIDE)
        candidates = [
            candidate for candidate in candidates
            if candidate.generation <= generation_limit
        ]
    if not candidates:
        raise ValueError("No candidate rows remained after the generation cap")
    return ground_truth_energy, candidates


def best_energy_so_far(
    candidates: list[Candidate], *, per_h: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return one cumulative-best energy gap for each available generation."""
    best_by_generation: dict[int, float] = {}
    for candidate in candidates:
        delta_energy = candidate.delta_energy
        if per_h:
            reference_per_h = GROUND_TRUTH_FORMATION_ENERGY_PER_ATOM * (
                SCAFFOLD_ATOM_COUNT + REFERENCE_H_ATOM_COUNT
            ) / REFERENCE_H_ATOM_COUNT
            delta_energy = candidate.formation_energy * (
                SCAFFOLD_ATOM_COUNT + candidate.h_atom_count
            ) / candidate.h_atom_count - reference_per_h
        best_by_generation[candidate.generation] = min(
            delta_energy,
            best_by_generation.get(candidate.generation, np.inf),
        )

    generations = np.asarray(sorted(best_by_generation), dtype=int)
    generation_minima = np.asarray(
        [best_by_generation[generation] for generation in generations], dtype=float
    )
    return generations, np.minimum.accumulate(generation_minima)


def ground_truth_hit_generation(candidates: list[Candidate]) -> int | None:
    """Return the first generation whose cumulative gap reaches the tolerance."""
    generations, best = best_energy_so_far(candidates)
    hits = np.flatnonzero(best < GROUND_TRUTH_HIT_TOLERANCE_EVPATOM)
    return int(generations[hits[0]]) if len(hits) else None


def common_plot_horizon(
    geat_candidates: list[Candidate],
    ordinary_candidates: list[Candidate],
) -> int:
    """Choose one endpoint shared by both cumulative-best curves."""
    if MAX_GENERATION_OVERRIDE is not None:
        return int(MAX_GENERATION_OVERRIDE)
    geat_hit = ground_truth_hit_generation(geat_candidates)
    ordinary_hit = ground_truth_hit_generation(ordinary_candidates)
    if geat_hit is not None and ordinary_hit is not None:
        return max(geat_hit, ordinary_hit) + int(GENERATIONS_AFTER_BOTH_HIT)
    return max(
        max(candidate.generation for candidate in geat_candidates),
        max(candidate.generation for candidate in ordinary_candidates),
    )


def best_energy_through_horizon(
    candidates: list[Candidate], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a cumulative-best series and hold its last value to the horizon."""
    clipped = [candidate for candidate in candidates if candidate.generation <= horizon]
    if not clipped:
        raise ValueError(f"No candidate records at or before generation {horizon}")
    generations, best = best_energy_so_far(clipped, per_h=True)
    if generations[-1] < horizon:
        generations = np.append(generations, int(horizon))
        best = np.append(best, best[-1])
    return generations, best


def style_axis(axis):
    axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot(
    geat_candidates: list[Candidate],
    ordinary_candidates: list[Candidate],
    output_path: Path,
):
    horizon = common_plot_horizon(geat_candidates, ordinary_candidates)
    geat_generations, geat_best = best_energy_through_horizon(
        geat_candidates, horizon
    )
    ordinary_generations, ordinary_best = best_energy_through_horizon(
        ordinary_candidates, horizon
    )

    fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    axis.plot(
        geat_generations,
        geat_best,
        color=GEAT_COLOR,
        linewidth=BEST_LINE_WIDTH,
        label="HEAT",
    )
    axis.plot(
        ordinary_generations,
        ordinary_best,
        color=ORDINARY_COLOR,
        linewidth=BEST_LINE_WIDTH,
        label="Conventional",
    )
    axis.axhline(
        0.0,
        color="#c62828",
        linestyle="--",
        linewidth=HORIZONTAL_REFERENCE_LINE_WIDTH,
        zorder=1,
    )
    for candidates, color in (
        (geat_candidates, GEAT_COLOR),
        (ordinary_candidates, ORDINARY_COLOR),
    ):
        hit_generation = ground_truth_hit_generation(candidates)
        if hit_generation is None:
            continue
        axis.axvline(
            hit_generation,
            color=color,
            linestyle=":",
            linewidth=VERTICAL_REFERENCE_LINE_WIDTH,
            alpha=0.55,
        )
        axis.scatter([hit_generation], [0.0], color=color, s=34, zorder=5)
    axis.set_ylabel(r"$E_{\rm best}-E_{\rm ref}$ (eV/H atom)", fontsize=AXIS_LABEL_SIZE)
    axis.set_xlabel("Generation", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.set_ylim(
        bottom=ENERGY_Y_MIN,
        top=(
            float(ENERGY_Y_MAX)
            if ENERGY_Y_MAX is not None
            else max(
                0.01,
                float(np.nanmax(geat_best)) * 1.05,
                float(np.nanmax(ordinary_best)) * 1.05,
            )
        ),
    )
    first_generation = int(min(geat_generations[0], ordinary_generations[0]))
    last_generation = int(horizon)
    axis.set_xlim(first_generation - 0.7, last_generation + 0.7)
    tick_step = max(1, int(np.ceil((last_generation - first_generation) / 7)))
    axis.set_xticks(np.arange(first_generation, last_generation + 1, tick_step))
    style_axis(axis)
    axis.legend(frameon=False, fontsize=TICK_LABEL_SIZE)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    geat_ground_truth, geat_candidates = load_results(GEAT_GENETIC_LOG)
    ordinary_ground_truth, ordinary_candidates = load_results(ORDINARY_GENETIC_LOG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "h_sublattice_convergence.png"
    plot(geat_candidates, ordinary_candidates, output_path)

    geat_generations, geat_best = best_energy_so_far(geat_candidates)
    ordinary_generations, ordinary_best = best_energy_so_far(ordinary_candidates)
    print(f"[GEAT source] {GEAT_GENETIC_LOG.resolve()}")
    print(f"[Ordinary source] {ORDINARY_GENETIC_LOG.resolve()}")
    print(
        f"[GEAT ground truth] formation_energy_per_atom="
        f"{geat_ground_truth:.12f} eV/atom"
    )
    print(
        f"[Ordinary ground truth] formation_energy_per_atom="
        f"{ordinary_ground_truth:.12f} eV/atom"
    )
    print(
        f"[GEAT] candidates={len(geat_candidates)} "
        f"generations={geat_generations[0]}-{geat_generations[-1]} "
        f"hit={ground_truth_hit_generation(geat_candidates)} "
        f"best_gap={geat_best[-1]:.12f} eV/atom"
    )
    print(
        f"[Ordinary] candidates={len(ordinary_candidates)} "
        f"generations={ordinary_generations[0]}-{ordinary_generations[-1]} "
        f"hit={ground_truth_hit_generation(ordinary_candidates)} "
        f"best_gap={ordinary_best[-1]:.12f} eV/atom"
    )
    print(f"[output] {output_path.resolve()}")


if __name__ == "__main__":
    main()
