#!/usr/bin/env python3
"""Shared hard-atom genetic algorithm for fixed-scaffold benchmarks.

This module deliberately reuses the geometry, reference preparation, model
loading, and objective metadata from the matching EAT scripts.  Candidate
structures themselves contain only ordinary, fully occupied atoms.  Children
are produced by periodic real-space cut-and-splice, repaired for close pairs,
mutated, and locally relaxed with ASE FIRE while all scaffold atoms and the
cell remain fixed.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import FIRE


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FILES = {
    "h_sublattice": SCRIPT_DIR.parent / "eat_genetic_scripts" / "eat_genetic_08.27.26_H_sublattice.py",
    "dislocation": SCRIPT_DIR.parent / "eat_genetic_scripts" / "eat_genetic_08.27.26_disloc.py",
}

HARD_DEFAULTS = {
    "candidate_relaxation": {
        "fmax": 0.05,
        "steps": 500,
        "maxstep_angstrom": 0.10,
        "unconverged_penalty": 0.0,
    },
    "hard_atom_genetic": {
        "parent_selection": "tournament",
        "mutation_probability": 0.30,
        "rattle_weight": 0.50,
        "relocate_weight": 0.50,
        "insert_weight": 0.0,
        "delete_weight": 0.0,
        "rattle_atom_fraction": 0.20,
        "initial_count_mode": "fixed",
        "initial_count_min": 1,
        "initial_count_max_factor": 2.0,
        "repair_attempts": 2000,
    },
}

HARD_KEY_MAP = {
    "candidate_relaxation": {
        "fmax": "hard_relax_fmax",
        "steps": "hard_relax_steps",
        "maxstep_angstrom": "hard_relax_maxstep_angstrom",
        "unconverged_penalty": "hard_relax_unconverged_penalty",
    },
    "hard_atom_genetic": {
        "parent_selection": "hard_parent_selection",
        "mutation_probability": "hard_mutation_probability",
        "rattle_weight": "hard_rattle_weight",
        "relocate_weight": "hard_relocate_weight",
        "insert_weight": "hard_insert_weight",
        "delete_weight": "hard_delete_weight",
        "rattle_atom_fraction": "hard_rattle_atom_fraction",
        "initial_count_mode": "hard_initial_count_mode",
        "initial_count_min": "hard_initial_count_min",
        "initial_count_max_factor": "hard_initial_count_max_factor",
        "repair_attempts": "hard_repair_attempts",
    },
}


def read_eat_source_path(config_path):
    """Read paths.eat_source_script without first importing an EAT module."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        return ""
    section = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if indent == 0:
            section = stripped[:-1].strip() if stripped.endswith(":") else None
            continue
        if section != "paths" or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip() != "eat_source_script":
            continue
        value = value.strip()
        if not value or value.lower() in {"null", "none", "~"}:
            return ""
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value
    return ""


def resolve_eat_source_path(mode, config_path):
    config_path = Path(config_path).expanduser().resolve()
    configured = read_eat_source_path(config_path)
    if configured:
        source = Path(os.path.expandvars(configured)).expanduser()
        if not source.is_absolute():
            source = config_path.parent / source
        source = source.resolve()
    else:
        source = SOURCE_FILES[mode].resolve()
    if not source.exists():
        guidance = (
            f"Update paths.eat_source_script in {config_path}"
            if configured
            else f"Add paths.eat_source_script to {config_path}"
        )
        raise FileNotFoundError(
            f"Matching EAT source does not exist: {source}. "
            f"{guidance} so it points to the matching EAT Python script."
        )
    return source


def load_source_module(mode, source_path=None):
    source = Path(source_path).expanduser().resolve() if source_path else SOURCE_FILES[mode].resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"Matching EAT source does not exist: {source}. "
            "Set paths.eat_source_script in the ordinary GA YAML."
        )
    script_dir = str(source.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    # The copied EAT files import their PBFGS helper at module import time.
    # The ordinary GA never constructs or calls that minimizer, so do not make
    # the hard-atom baseline depend on an otherwise unused local module.
    try:
        import minimizer as _unused_minimizer  # noqa: F401
    except ModuleNotFoundError:
        placeholder = types.ModuleType("minimizer")

        class UnavailableEATMinimizer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("The EAT PBFGS minimizer is unavailable in this environment")

        placeholder.minimizer = UnavailableEATMinimizer
        sys.modules["minimizer"] = placeholder
    module_name = f"_ordinary_ga_geometry_{mode}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load matching EAT source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge_dicts(base, override):
    merged = copy.deepcopy(base)
    for section, values in override.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = copy.deepcopy(values)
    return merged


def load_settings(mode, config_path):
    config_path = Path(config_path).expanduser().resolve()
    source_path = resolve_eat_source_path(mode, config_path)
    source = load_source_module(mode, source_path)
    args = source.get_settings(config_path)
    raw = source.read_simple_yaml(config_path) if config_path.exists() else {}
    hard_config = merge_dicts(HARD_DEFAULTS, raw)
    for section, mapping in HARD_KEY_MAP.items():
        values = hard_config.get(section, {})
        for key, attr in mapping.items():
            setattr(args, attr, values[key])
    args.config_path = str(config_path)
    args.ordinary_ga_mode = str(mode)
    args.eat_source_script = str(source_path)
    return source, args, raw


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, SimpleNamespace):
        return {key: json_safe(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def append_line(path, line):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(str(line).rstrip() + "\n")


class ModelEvaluationTracker:
    """Count and time actual FAIR-Chem calculator executions.

    One energy evaluation is one calculator ``calculate`` call. A call that
    jointly returns energy and forces counts once, while later ASE property
    reads served from the calculator cache count zero.
    """

    def __init__(self):
        self.phase = "setup"
        self.total_count = 0
        self.total_seconds = 0.0
        self.by_phase = {}

    def record(self, phase, elapsed_seconds):
        entry = self.by_phase.setdefault(str(phase), {"count": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] += float(elapsed_seconds)
        self.total_count += 1
        self.total_seconds += float(elapsed_seconds)

    def snapshot(self):
        return {
            "count": int(self.total_count),
            "seconds": float(self.total_seconds),
            "by_phase": {
                phase: {"count": int(values["count"]), "seconds": float(values["seconds"])}
                for phase, values in self.by_phase.items()
            },
        }

    @staticmethod
    def difference(after, before):
        phases = set(after["by_phase"]) | set(before["by_phase"])
        return {
            "model_energy_evaluations": int(after["count"] - before["count"]),
            "model_evaluation_seconds": float(after["seconds"] - before["seconds"]),
            "by_phase": {
                phase: {
                    "count": int(after["by_phase"].get(phase, {}).get("count", 0) - before["by_phase"].get(phase, {}).get("count", 0)),
                    "seconds": float(after["by_phase"].get(phase, {}).get("seconds", 0.0) - before["by_phase"].get(phase, {}).get("seconds", 0.0)),
                }
                for phase in sorted(phases)
            },
        }


def ensure_calculator_instrumented(problem):
    tracker = getattr(problem, "model_evaluation_tracker", None)
    if tracker is not None:
        return tracker
    tracker = ModelEvaluationTracker()
    original_calculate = problem.calc.calculate

    def counted_calculate(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_calculate(*args, **kwargs)
        finally:
            tracker.record(tracker.phase, time.perf_counter() - start)

    problem.calc.calculate = counted_calculate
    problem.model_evaluation_tracker = tracker
    return tracker


def terminal_print(args, message):
    if bool(getattr(args, "terminal_verbose", False)):
        print(message, flush=True)


def validate_settings(mode, args):
    if int(args.ga_population_size) <= 0 or int(args.ga_generations) <= 0:
        raise ValueError("genetic.population_size and generations must be positive")
    if int(args.ga_parent_pool_size) < 2:
        raise ValueError("genetic.parent_pool_size must be at least two")
    if int(args.ga_parent_a_tournament_size) <= 0:
        raise ValueError("genetic.parent_a_tournament_size must be positive")
    fractions = np.asarray(
        [args.ga_elite_fraction, args.ga_random_fraction, args.ga_crossover_fraction],
        dtype=float,
    )
    if np.any(fractions < 0.0) or float(fractions.sum()) <= 0.0:
        raise ValueError("GA fractions must be nonnegative and have a positive sum")
    if str(args.ga_execution_backend).strip().lower() != "local":
        raise ValueError("The ordinary fixed-scaffold GA supports local execution only")
    if int(args.hard_relax_steps) <= 0 or float(args.hard_relax_fmax) <= 0.0:
        raise ValueError("candidate_relaxation steps and fmax must be positive")
    if float(args.hard_relax_maxstep_angstrom) <= 0.0:
        raise ValueError("candidate_relaxation.maxstep_angstrom must be positive")
    if float(args.hard_relax_unconverged_penalty) < 0.0:
        raise ValueError("candidate_relaxation.unconverged_penalty must be nonnegative")
    selection = str(args.hard_parent_selection).strip().lower()
    if selection not in ("tournament", "diversity_tournament"):
        raise ValueError(
            "hard_atom_genetic.parent_selection must be tournament or diversity_tournament"
        )
    if not 0.0 <= float(args.hard_mutation_probability) <= 1.0:
        raise ValueError("hard_atom_genetic.mutation_probability must be in [0, 1]")
    if not 0.0 < float(args.hard_rattle_atom_fraction) <= 1.0:
        raise ValueError("hard_atom_genetic.rattle_atom_fraction must be in (0, 1]")
    weights = mutation_weights(args)
    if not weights:
        raise ValueError("At least one applicable hard-atom mutation weight must be positive")
    if int(args.hard_initial_count_min) <= 0:
        raise ValueError("hard_atom_genetic.initial_count_min must be positive")
    if float(args.hard_initial_count_max_factor) < 1.0:
        raise ValueError("hard_atom_genetic.initial_count_max_factor must be at least one")
    if int(args.hard_repair_attempts) <= 0:
        raise ValueError("hard_atom_genetic.repair_attempts must be positive")
    if str(args.ga_live_plot_metric or "score") not in {
        "score",
        "formation_energy_per_atom",
        "composition_term",
        "occupancy_lower_bound_term",
        "final_max_force_eV_per_angstrom",
    }:
        raise ValueError("Unsupported genetic.live_plot_metric for the ordinary GA")
    if mode == "h_sublattice" and str(args.hard_initial_count_mode).lower() != "fixed":
        raise ValueError("The H-sublattice ordinary GA requires initial_count_mode: fixed")
    if mode == "dislocation" and str(args.hard_initial_count_mode).lower() not in (
        "density_poisson",
        "poisson",
    ):
        raise ValueError("The dislocation ordinary GA requires initial_count_mode: density_poisson")


def mutation_weights(args):
    weights = {
        "rattle": float(args.hard_rattle_weight),
        "relocate": float(args.hard_relocate_weight),
        "insert": float(args.hard_insert_weight),
        "delete": float(args.hard_delete_weight),
    }
    return {key: value for key, value in weights.items() if value > 0.0}


def allocate_generation_counts(population_size, args):
    fractions = np.asarray(
        [args.ga_elite_fraction, args.ga_random_fraction, args.ga_crossover_fraction],
        dtype=float,
    )
    raw = fractions / fractions.sum() * int(population_size)
    counts = np.floor(raw).astype(int)
    remainder = int(population_size - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for index in order[:remainder]:
            counts[index] += 1
    return {"elite": int(counts[0]), "random": int(counts[1]), "crossover": int(counts[2])}


def parse_gpu_ids(text):
    values = [piece.strip() for piece in str(text).split(",") if piece.strip()]
    return values or ["0"]


def periodic_delta(reference, points, cell, inv_cell, pbc):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    reference = np.asarray(reference, dtype=float).reshape(1, 3)
    frac = (points - reference) @ np.asarray(inv_cell, dtype=float)
    periodic = np.asarray(pbc, dtype=bool)
    frac[:, periodic] -= np.round(frac[:, periodic])
    return frac @ np.asarray(cell, dtype=float)


def minimum_distances(points, references, problem):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    references = np.asarray(references, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return np.empty(0, dtype=float)
    if len(references) == 0:
        return np.full(len(points), np.inf, dtype=float)
    result = np.full(len(points), np.inf, dtype=float)
    for reference in references:
        delta = periodic_delta(reference, points, problem.cell, problem.inv_cell, problem.pbc)
        result = np.minimum(result, np.linalg.norm(delta, axis=1))
    return result


def wrap_positions(problem, positions):
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(positions) == 0:
        return positions.copy()
    return np.asarray(problem.wrap_positions(positions), dtype=float)


def sample_clear_positions(problem, count, existing, cutoff, attempts):
    existing = np.asarray(existing, dtype=float).reshape(-1, 3)
    accepted = []
    trials = 0
    while len(accepted) < int(count) and trials < int(attempts):
        batch_size = max(8, 2 * (int(count) - len(accepted)))
        sampled = np.asarray(problem.sample_initial_positions(batch_size), dtype=float)
        for point in sampled:
            trials += 1
            references = [problem.fixed_positions, existing]
            if accepted:
                references.append(np.asarray(accepted, dtype=float))
            combined = np.vstack([item for item in references if len(item)])
            if float(cutoff) <= 0.0 or minimum_distances(point[None, :], combined, problem)[0] >= float(cutoff):
                accepted.append(point)
                if len(accepted) >= int(count):
                    break
            if trials >= int(attempts):
                break
    if len(accepted) != int(count):
        raise RuntimeError(
            f"Could place only {len(accepted)}/{int(count)} repaired atoms after {int(attempts)} attempts"
        )
    return wrap_positions(problem, np.asarray(accepted, dtype=float))


def prune_close_positions(problem, positions, cutoff):
    positions = wrap_positions(problem, positions)
    if len(positions) < 2 or float(cutoff) <= 0.0:
        return positions
    order = np.random.permutation(len(positions))
    kept = []
    for index in order:
        point = positions[int(index)]
        if kept and minimum_distances(point[None, :], np.asarray(kept), problem)[0] < float(cutoff):
            continue
        if len(problem.fixed_positions) and minimum_distances(
            point[None, :], problem.fixed_positions, problem
        )[0] < float(cutoff):
            continue
        kept.append(point)
    return np.asarray(kept, dtype=float).reshape(-1, 3)


def count_limits(problem, args, mode):
    if mode == "h_sublattice":
        target = int(problem.target_missing_count)
        return target, target
    lower = max(1, int(args.hard_initial_count_min))
    estimated = max(1.0, float(problem.estimated_search_region_atoms))
    upper = max(lower + 1, int(math.ceil(float(args.hard_initial_count_max_factor) * estimated)))
    return lower, upper


def initial_atom_count(problem, args, mode):
    lower, upper = count_limits(problem, args, mode)
    if mode == "h_sublattice":
        return lower
    sampled = int(np.random.poisson(max(1.0, float(problem.estimated_search_region_atoms))))
    return int(np.clip(sampled, lower, upper))


def repair_positions(problem, positions, args, mode, extras=None):
    cutoff = float(args.ga_superposition_prune_distance_angstrom)
    repaired = prune_close_positions(problem, positions, cutoff)
    lower, upper = count_limits(problem, args, mode)
    target = lower if mode == "h_sublattice" else None
    if target is not None and len(repaired) > target:
        repaired = repaired[np.random.choice(len(repaired), size=target, replace=False)]
    if mode == "dislocation" and len(repaired) > upper:
        repaired = repaired[np.random.choice(len(repaired), size=upper, replace=False)]
    required = (target - len(repaired)) if target is not None else max(0, lower - len(repaired))
    extras_array = np.asarray(extras if extras is not None else [], dtype=float).reshape(-1, 3)
    if required > 0 and len(extras_array):
        extras_array = extras_array[np.random.permutation(len(extras_array))]
        accepted = []
        for point in extras_array:
            references = [problem.fixed_positions, repaired]
            if accepted:
                references.append(np.asarray(accepted, dtype=float))
            combined = np.vstack([item for item in references if len(item)])
            if cutoff <= 0.0 or minimum_distances(point[None, :], combined, problem)[0] >= cutoff:
                accepted.append(point)
                if len(accepted) >= required:
                    break
        if accepted:
            repaired = np.vstack([repaired, np.asarray(accepted, dtype=float)])
            required -= len(accepted)
    if required > 0:
        additions = sample_clear_positions(
            problem,
            required,
            repaired,
            cutoff,
            int(args.hard_repair_attempts),
        )
        repaired = np.vstack([repaired, additions])
    return wrap_positions(problem, repaired)


def make_initial_candidate(problem, args, mode):
    count = initial_atom_count(problem, args, mode)
    problem.set_site_count(count)
    positions = np.asarray(problem.sample_initial_positions(count), dtype=float)
    positions = repair_positions(problem, positions, args, mode)
    return {
        "symbols": [str(problem.species[0])] * len(positions),
        "positions": positions,
    }


def plane_signed_distances(problem, positions, origin_frac, normal):
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    frac = positions @ problem.inv_cell
    delta_frac = frac - np.asarray(origin_frac, dtype=float)[None, :]
    periodic = np.asarray(problem.pbc, dtype=bool)
    delta_frac[:, periodic] -= np.round(delta_frac[:, periodic])
    delta_cart = delta_frac @ problem.cell
    return delta_cart @ np.asarray(normal, dtype=float)


def crossover_candidate(problem, parent_a, parent_b, args, mode):
    pos_a = np.asarray(parent_a["positions"], dtype=float)
    pos_b = np.asarray(parent_b["positions"], dtype=float)
    normal = np.random.normal(size=3)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        normal = np.asarray([1.0, 0.0, 0.0])
    else:
        normal /= norm
    origin_frac = np.random.random(3)
    signed_a = plane_signed_distances(problem, pos_a, origin_frac, normal)
    signed_b = plane_signed_distances(problem, pos_b, origin_frac, normal)
    keep_a = signed_a >= 0.0
    keep_b = signed_b < 0.0
    selected = np.vstack([pos_a[keep_a], pos_b[keep_b]])
    extras = np.vstack([pos_a[~keep_a], pos_b[~keep_b]])
    child_positions = repair_positions(problem, selected, args, mode, extras=extras)
    child = {
        "symbols": [str(problem.species[0])] * len(child_positions),
        "positions": child_positions,
        "crossover_plane": {
            "origin_fractional": origin_frac.tolist(),
            "normal_cartesian": normal.tolist(),
            "selected_from_parent_a": int(np.count_nonzero(keep_a)),
            "selected_from_parent_b": int(np.count_nonzero(keep_b)),
        },
    }
    return child


def mutate_candidate(problem, candidate, args, mode):
    mutated = {
        "symbols": list(candidate["symbols"]),
        "positions": np.asarray(candidate["positions"], dtype=float).copy(),
    }
    mutated["mutation"] = "none"
    if np.random.random() >= float(args.hard_mutation_probability):
        return mutated
    weights = mutation_weights(args)
    names = list(weights)
    probabilities = np.asarray([weights[name] for name in names], dtype=float)
    probabilities /= probabilities.sum()
    operation = str(np.random.choice(names, p=probabilities))
    positions = mutated["positions"]
    lower, upper = count_limits(problem, args, mode)
    if operation == "rattle" and len(positions):
        count = max(1, int(math.ceil(float(args.hard_rattle_atom_fraction) * len(positions))))
        indices = np.random.choice(len(positions), size=min(count, len(positions)), replace=False)
        positions[indices] += float(args.ga_crossover_pos_sigma) * np.random.normal(size=(len(indices), 3))
    elif operation == "relocate" and len(positions):
        index = int(np.random.randint(len(positions)))
        positions[index] = problem.sample_initial_positions(1)[0]
    elif operation == "insert" and mode == "dislocation" and len(positions) < upper:
        addition = sample_clear_positions(
            problem,
            1,
            positions,
            float(args.ga_superposition_prune_distance_angstrom),
            int(args.hard_repair_attempts),
        )
        positions = np.vstack([positions, addition])
    elif operation == "delete" and mode == "dislocation" and len(positions) > lower:
        positions = np.delete(positions, int(np.random.randint(len(positions))), axis=0)
    else:
        operation = "none_inapplicable"
    positions = repair_positions(problem, positions, args, mode)
    mutated["positions"] = positions
    mutated["symbols"] = [str(problem.species[0])] * len(positions)
    mutated["mutation"] = operation
    return mutated


def build_full_atoms(problem, candidate):
    candidate_positions = np.asarray(candidate["positions"], dtype=float).reshape(-1, 3)
    symbols = list(problem.fixed_symbols) + list(candidate["symbols"])
    positions = np.vstack([np.asarray(problem.fixed_positions, dtype=float), candidate_positions])
    atoms = Atoms(symbols=symbols, positions=positions, cell=problem.cell, pbc=problem.pbc)
    atoms.set_constraint(FixAtoms(indices=np.arange(len(problem.fixed_symbols), dtype=int)))
    atoms.calc = problem.calc
    return atoms


def hard_objective(problem, candidate, energy):
    candidate_positions = np.asarray(candidate["positions"], dtype=float).reshape(-1, 3)
    variable_symbols = list(candidate["symbols"])
    all_symbols = list(problem.fixed_symbols) + variable_symbols
    counts = np.asarray(
        [sum(symbol == species for symbol in all_symbols) for species in problem.all_species],
        dtype=float,
    ) if hasattr(problem, "all_species") else np.asarray([len(all_symbols)], dtype=float)
    reference_energies = np.asarray(problem.element_ref_energies, dtype=float)
    n_total = max(float(counts.sum()), 1.0)
    formation_total = float(energy) - float(np.dot(counts, reference_energies))
    formation_per_atom = formation_total / n_total
    occupancies = np.ones((len(candidate_positions), int(problem.n_species)), dtype=float)
    composition_info = problem.composition_penalty_term(occupancies)
    boundary_info = problem.hole_boundary_penalty_term(occupancies, candidate_positions)
    lower_bound_info = problem.occupancy_lower_bound_penalty_term_np(occupancies)
    score = (
        formation_per_atom
        + float(composition_info["objective_term"])
        + float(boundary_info["objective_term"])
        + float(lower_bound_info["objective_term"])
    )
    return {
        "score": float(score),
        "energy_total_eV": float(energy),
        "formation_energy_total_eV": float(formation_total),
        "formation_energy_per_atom": float(formation_per_atom),
        "composition_term": float(composition_info["objective_term"]),
        "hole_boundary_term": float(boundary_info["objective_term"]),
        "occupancy_lower_bound_term": float(lower_bound_info["objective_term"]),
        "occupancy_lower_bound_deficit": float(lower_bound_info["deficit"]),
        "composition_fractions": np.asarray(composition_info["fractions"], dtype=float).tolist(),
        "composition_counts": counts.tolist(),
    }


def evaluate_candidate(problem, candidate, args, seed, generation, member_index, kind, parent_refs):
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    tracker = ensure_calculator_instrumented(problem)
    tracker_before = tracker.snapshot()
    previous_phase = tracker.phase
    member_wall_start = time.perf_counter()
    phase_wall_start = time.perf_counter()
    tracker.phase = "final_FIRE"
    problem.set_site_count(len(candidate["positions"]))
    atoms = build_full_atoms(problem, candidate)
    optimizer = FIRE(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=float(args.hard_relax_maxstep_angstrom),
    )
    try:
        converged = bool(optimizer.run(fmax=float(args.hard_relax_fmax), steps=int(args.hard_relax_steps)))
        atoms.wrap()
        energy = float(atoms.get_potential_energy())
        n_fixed = len(problem.fixed_symbols)
        relaxed_positions = np.asarray(atoms.get_positions()[n_fixed:], dtype=float)
        relaxed = {
            "symbols": list(candidate["symbols"]),
            "positions": relaxed_positions,
        }
        objective = hard_objective(problem, relaxed, energy)
        raw_forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)[n_fixed:]
        max_force = float(np.max(np.linalg.norm(raw_forces, axis=1))) if len(raw_forces) else 0.0
    finally:
        tracker.phase = previous_phase
    final_fire_wall_seconds = time.perf_counter() - phase_wall_start
    compute_metrics = ModelEvaluationTracker.difference(tracker.snapshot(), tracker_before)
    compute_metrics.update({
        "member_wall_seconds": float(time.perf_counter() - member_wall_start),
        "phase_wall_seconds": {"final_FIRE": float(final_fire_wall_seconds)},
    })
    score = float(objective["score"])
    if not converged:
        score += float(args.hard_relax_unconverged_penalty)
    result = {
        "generation": int(generation),
        "member_index": int(member_index),
        "kind": str(kind),
        "parent_refs": list(parent_refs or []),
        "seed": int(seed),
        "symbols": list(relaxed["symbols"]),
        "positions": relaxed_positions,
        "initial_positions": np.asarray(candidate["positions"], dtype=float),
        "n_sites": int(len(relaxed_positions)),
        "score": float(score),
        "native_objective_score": float(objective["score"]),
        "relaxation_converged": bool(converged),
        "relaxation_steps": int(getattr(optimizer, "nsteps", 0)),
        "final_max_force_eV_per_angstrom": float(max_force),
        "elapsed_seconds": float(compute_metrics["member_wall_seconds"]),
        "compute_metrics": compute_metrics,
        "mutation": str(candidate.get("mutation", "none")),
        "crossover_plane": copy.deepcopy(candidate.get("crossover_plane")),
    }
    result.update(objective)
    result["score"] = float(score)
    return result


def member_spatial_vector(problem, member):
    positions = np.asarray(member["positions"], dtype=float).reshape(-1, 3)
    grid = np.asarray(problem.subcell_grid, dtype=int)
    vector = np.zeros(int(np.prod(grid)), dtype=float)
    if not len(positions):
        return vector
    frac = positions @ problem.inv_cell
    lo = np.asarray(problem.hole_min_frac, dtype=float)
    hi = np.asarray(problem.hole_max_frac, dtype=float)
    relative = (frac - lo[None, :]) / np.maximum(hi - lo, 1e-12)[None, :]
    bins = np.floor(relative * grid[None, :]).astype(int)
    valid = np.all((bins >= 0) & (bins < grid[None, :]), axis=1)
    if np.any(valid):
        flat = np.ravel_multi_index(bins[valid].T, tuple(grid.tolist()))
        np.add.at(vector, flat, 1.0)
    return vector


def diversity_distance(problem, member_a, member_b):
    return float(np.sum(np.abs(member_spatial_vector(problem, member_a) - member_spatial_vector(problem, member_b))))


def tournament_parent(parent_pool, tournament_size, exclude=None):
    candidates = [parent for parent in parent_pool if parent is not exclude]
    if not candidates:
        raise ValueError("No eligible tournament parents remain")
    size = min(max(1, int(tournament_size)), len(candidates))
    indices = np.random.choice(len(candidates), size=size, replace=False)
    tournament = [candidates[int(index)] for index in indices]
    return min(tournament, key=lambda parent: float(parent["score"]))


def choose_parent_pair(problem, parent_pool, args):
    if len(parent_pool) < 2:
        raise ValueError("At least two parents are required for crossover")
    tournament_size = int(args.ga_parent_a_tournament_size)
    selection = str(args.hard_parent_selection).strip().lower()
    if selection == "tournament":
        parent_a = tournament_parent(parent_pool, tournament_size)
        parent_b = tournament_parent(parent_pool, tournament_size, exclude=parent_a)
        return parent_a, parent_b
    size = min(max(2, tournament_size), len(parent_pool))
    indices = np.random.choice(len(parent_pool), size=size, replace=False)
    tournament = [parent_pool[int(index)] for index in indices]
    tournament.sort(key=lambda parent: float(parent["score"]))
    parent_a = tournament[0]
    parent_b = max(
        tournament[1:],
        key=lambda parent: diversity_distance(problem, parent_a, parent),
    )
    return parent_a, parent_b


def parent_ref(parent):
    return (
        f"g{int(parent.get('generation', -1))}_"
        f"m{int(parent.get('member_index', -1))}_"
        f"{parent.get('kind', 'parent')}"
    )


def empty_compute_metrics():
    return {
        "model_energy_evaluations": 0,
        "model_evaluation_seconds": 0.0,
        "member_wall_seconds": 0.0,
        "phase_wall_seconds": {},
        "by_phase": {},
    }


def aggregate_compute_metrics(members):
    total = {
        "model_energy_evaluations": 0,
        "model_evaluation_seconds": 0.0,
        "member_wall_seconds_sum": 0.0,
        "by_phase": {},
    }
    for member in members:
        metrics = member.get("compute_metrics", {})
        total["model_energy_evaluations"] += int(metrics.get("model_energy_evaluations", 0))
        total["model_evaluation_seconds"] += float(metrics.get("model_evaluation_seconds", 0.0))
        total["member_wall_seconds_sum"] += float(metrics.get("member_wall_seconds", 0.0))
        for phase, values in metrics.get("by_phase", {}).items():
            entry = total["by_phase"].setdefault(phase, {"count": 0, "seconds": 0.0})
            entry["count"] += int(values.get("count", 0))
            entry["seconds"] += float(values.get("seconds", 0.0))
    return total


def copy_elite(parent, generation, member_index):
    copied = copy.deepcopy(parent)
    copied["generation"] = int(generation)
    copied["member_index"] = int(member_index)
    copied["kind"] = "elite_copy"
    copied["parent_refs"] = [parent_ref(parent)]
    copied["positions"] = np.asarray(parent["positions"], dtype=float).copy()
    copied["initial_positions"] = np.asarray(parent["initial_positions"], dtype=float).copy()
    copied["compute_metrics"] = empty_compute_metrics()
    copied["elapsed_seconds"] = 0.0
    return copied


def result_record(result):
    omitted = {"positions", "initial_positions"}
    record = {key: json_safe(value) for key, value in result.items() if key not in omitted}
    record["positions_angstrom"] = np.asarray(result["positions"], dtype=float).tolist()
    record["initial_positions_angstrom"] = np.asarray(result["initial_positions"], dtype=float).tolist()
    return record


def imported_hard_candidates(problem, args, mode):
    """Load ordinary records or rounded EAT records as hard candidates."""
    path_text = str(getattr(args, "ga_import_structures_json", "") or "").strip()
    if not path_text:
        return []
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("structures", [])
    records = sorted(records, key=lambda record: float(record.get("score", np.inf)))
    requested = int(getattr(args, "ga_import_parent_count", 0) or 0)
    limit = requested if requested > 0 else int(args.ga_population_size)
    candidates = []
    for record in records[:limit]:
        symbols = list(record.get("symbols", []))
        positions = np.asarray(record.get("positions_angstrom", []), dtype=float)
        rounded = record.get("rounded_structure", {})
        if (positions.ndim != 2 or positions.shape[1:] != (3,)) and isinstance(rounded, dict):
            symbols = list(rounded.get("null_symbols", rounded.get("symbols", [])))
            positions = np.asarray(
                rounded.get("null_positions_angstrom", rounded.get("cart_positions_angstrom", [])),
                dtype=float,
            )
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            state = record.get("continuous_state", {})
            occupancies = np.asarray(state.get("species_occupancies", []), dtype=float)
            state_positions = np.asarray(state.get("cart_positions_angstrom", []), dtype=float)
            if (
                occupancies.ndim == 2
                and state_positions.ndim == 2
                and state_positions.shape[1:] == (3,)
                and len(occupancies) == len(state_positions)
            ):
                keep = occupancies.sum(axis=1) > 0.5
                positions = state_positions[keep]
                symbols = [str(problem.species[int(np.argmax(row))]) for row in occupancies[keep]]
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
            continue
        if any(symbol not in problem.species for symbol in symbols) or len(symbols) != len(positions):
            symbols = [str(problem.species[0])] * len(positions)
        positions = repair_positions(problem, positions, args, mode)
        candidates.append({
            "symbols": [str(problem.species[0])] * len(positions),
            "positions": positions,
            "imported_from": str(path),
        })
    return candidates


def ga_log_record(result, event):
    metrics = result.get("compute_metrics", {})
    phases = metrics.get("by_phase", {})
    phase_wall = metrics.get("phase_wall_seconds", {})
    return {
        "event": str(event),
        "generation": int(result["generation"]),
        "member_index": int(result["member_index"]),
        "kind": str(result["kind"]),
        "score": float(result["score"]),
        "formation_energy_per_atom": float(result["formation_energy_per_atom"]),
        "composition_term": float(result["composition_term"]),
        "occupancy_lower_bound_term": float(result["occupancy_lower_bound_term"]),
        "n_sites": int(result["n_sites"]),
        "relaxation_converged": bool(result["relaxation_converged"]),
        "relaxation_steps": int(result["relaxation_steps"]),
        "final_max_force_eV_per_angstrom": float(result["final_max_force_eV_per_angstrom"]),
        "mutation": str(result.get("mutation", "none")),
        "energy_evaluations": int(metrics.get("model_energy_evaluations", 0)),
        "energy_evaluation_seconds": float(metrics.get("model_evaluation_seconds", 0.0)),
        "member_wall_seconds": float(metrics.get("member_wall_seconds", 0.0)),
        "final_FIRE_energy_evaluations": int(phases.get("final_FIRE", {}).get("count", 0)),
        "final_FIRE_energy_evaluation_seconds": float(phases.get("final_FIRE", {}).get("seconds", 0.0)),
        "final_FIRE_wall_seconds": float(phase_wall.get("final_FIRE", 0.0)),
    }


def log_event(path, record):
    pieces = [str(record.get("event", "event"))]
    for key, value in record.items():
        if key == "event":
            continue
        if isinstance(value, float):
            pieces.append(f"{key}={value:+.8f}")
        else:
            pieces.append(f"{key}={value}")
    append_line(path, " | ".join(pieces))


def initialize_population_database(path):
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "representation": "ordinary_hard_atoms",
            "structures": [],
        },
    )


def append_population_record(path, record):
    """Append one structure while retaining a standard, valid JSON database.

    Only the closing tail of the structures array is replaced. Runtime is
    therefore proportional to the new record size, not the accumulated run.
    All writes occur in the main process, so parallel workers cannot race.
    """
    path = Path(path)
    if not path.exists():
        initialize_population_database(path)
    rendered = json.dumps(json_safe(record), indent=2)
    rendered = "\n".join("    " + line for line in rendered.splitlines()).encode("utf-8")
    with path.open("r+b") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        tail_size = min(file_size, 65536)
        handle.seek(file_size - tail_size)
        tail = handle.read(tail_size)
        array_close = tail.rfind(b"]")
        if array_close < 0 or tail[array_close + 1:].strip() != b"}":
            raise ValueError(f"Cannot append to malformed structures database: {path}")
        content_end = array_close
        while content_end > 0 and tail[content_end - 1] in b" \t\r\n":
            content_end -= 1
        if content_end <= 0:
            raise ValueError(f"Cannot locate structures array contents in: {path}")
        is_empty = tail[content_end - 1:content_end] == b"["
        handle.seek(file_size - tail_size + content_end)
        handle.truncate()
        handle.write((b"" if is_empty else b",") + b"\n" + rendered + b"\n  ]\n}\n")
        handle.flush()


def update_live_plot(problem, args, generations, current=None):
    if not bool(args.ga_live_plot_enable):
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    combined = list(generations)
    if current:
        combined.append(list(current))
    metric = str(args.ga_live_plot_metric or "score")
    supported_metrics = {
        "score",
        "formation_energy_per_atom",
        "composition_term",
        "occupancy_lower_bound_term",
        "final_max_force_eV_per_angstrom",
    }
    if metric not in supported_metrics:
        raise ValueError(
            "genetic.live_plot_metric must be one of " + ", ".join(sorted(supported_metrics))
        )
    fig, axis = plt.subplots(figsize=(10, 6))
    colors = {
        "random": "red",
        "random_immigrant": "red",
        "crossover_child": "green",
        "elite_copy": "gray",
        "imported_seed": "gray",
    }
    for generation_index, members in enumerate(combined):
        for member in members:
            jitter = (((generation_index + 1) * 73856093 + (int(member["member_index"]) + 1) * 19349663) % 10000) / 9999.0 - 0.5
            x_value = generation_index + 2.0 * float(args.ga_live_plot_jitter) * jitter
            axis.scatter(
                [x_value],
                [float(member[metric])],
                s=42,
                c=colors.get(str(member["kind"]), "blue"),
                alpha=float(args.ga_live_plot_alpha),
                edgecolors="none",
            )
    if combined:
        best = [min(float(member[metric]) for member in members) for members in combined if members]
        axis.plot(range(len(best)), best, color="black", linewidth=1.6, alpha=0.8)
    axis.set_xlabel("Generation")
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_title("Ordinary hard-atom GA population scores")
    axis.grid(True, alpha=0.22)
    completed_members = [member for members in combined for member in members]
    live_energy_evaluations = sum(int(member.get("compute_metrics", {}).get("model_energy_evaluations", 0)) for member in completed_members)
    live_model_seconds = sum(float(member.get("compute_metrics", {}).get("model_evaluation_seconds", 0.0)) for member in completed_members)
    axis.text(
        0.0,
        -0.14,
        f"energy evaluations={live_energy_evaluations:,}   model time={live_model_seconds / 3600.0:.2f} h",
        transform=axis.transAxes,
        fontsize=10,
        color="0.35",
    )
    path_text = str(args.ga_live_plot_file or "").strip()
    path = Path(path_text).expanduser() if path_text else problem.output_dir / "genetic_optimization_live_scores_by_generation.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def evaluation_task(candidate, seed, generation, member_index, kind, parent_refs=None, metadata=None):
    return {
        "candidate": {
            "symbols": list(candidate["symbols"]),
            "positions": np.asarray(candidate["positions"], dtype=float).tolist(),
            "mutation": str(candidate.get("mutation", "none")),
            "crossover_plane": copy.deepcopy(candidate.get("crossover_plane")),
        },
        "seed": int(seed),
        "generation": int(generation),
        "member_index": int(member_index),
        "kind": str(kind),
        "parent_refs": list(parent_refs or []),
        "metadata": dict(metadata or {}),
    }


def configure_worker_device(args, gpu_id):
    requested = str(getattr(args, "device", "auto") or "auto").strip().lower()
    if requested == "auto" and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA worker requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(int(gpu_id))
        args.device = "cuda"


def worker_loop(mode, worker_index, gpu_id, args_dict, main_output, task_queue, result_queue):
    temporary = None
    try:
        args = SimpleNamespace(**args_dict)
        configure_worker_device(args, gpu_id)
        temporary = tempfile.TemporaryDirectory(prefix=f"ordinary_{mode}_worker_{worker_index}_")
        worker_output = Path(temporary.name)
        if mode == "dislocation":
            for filename in (
                "ground_truth_reference_positions.npy",
                "ground_truth_reference_energy.json",
            ):
                source_path = Path(main_output) / filename
                if source_path.exists():
                    destination = worker_output / filename
                    destination.write_bytes(source_path.read_bytes())
            args.skip_ground_truth_reference = True
        args.output_dir = str(worker_output)
        source = load_source_module(mode, args.eat_source_script)
        problem = source.InpaintingProblem(args)
        result_queue.put({"event": "worker_ready", "worker_index": worker_index, "gpu_id": gpu_id})
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                candidate = dict(task["candidate"])
                candidate["positions"] = np.asarray(candidate["positions"], dtype=float)
                result = evaluate_candidate(
                    problem,
                    candidate,
                    args,
                    task["seed"],
                    task["generation"],
                    task["member_index"],
                    task["kind"],
                    task["parent_refs"],
                )
                result.update(task.get("metadata", {}))
                result_queue.put({
                    "event": "result",
                    "ok": True,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "result": result_record(result),
                })
            except Exception:
                result_queue.put({
                    "event": "result",
                    "ok": False,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "traceback": traceback.format_exc(),
                })
    except Exception:
        result_queue.put({
            "event": "worker_init_error",
            "ok": False,
            "worker_index": worker_index,
            "gpu_id": gpu_id,
            "traceback": traceback.format_exc(),
        })
    finally:
        if temporary is not None:
            temporary.cleanup()


def start_workers(mode, args, count, main_output):
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    workers = []
    gpu_ids = parse_gpu_ids(args.ga_parallel_gpu_ids)
    args_dict = {key: json_safe(value) for key, value in vars(args).items()}
    for worker_index in range(int(count)):
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
        process = context.Process(
            target=worker_loop,
            args=(mode, worker_index, gpu_id, args_dict, str(main_output), task_queue, result_queue),
            daemon=True,
        )
        process.start()
        workers.append(process)
    ready = 0
    while ready < len(workers):
        message = result_queue.get()
        if message.get("event") == "worker_ready":
            ready += 1
            terminal_print(args, f"[worker ready] index={message['worker_index']} gpu={message['gpu_id']}")
            continue
        raise RuntimeError(
            "Ordinary GA worker initialization failed:\n" + str(message.get("traceback", message))
        )
    return task_queue, result_queue, workers


def stop_workers(task_queue, workers):
    if task_queue is not None:
        for _ in workers:
            task_queue.put(None)
    for process in workers:
        process.join()


def restore_worker_result(record):
    restored = dict(record)
    restored["positions"] = np.asarray(restored.pop("positions_angstrom"), dtype=float)
    restored["initial_positions"] = np.asarray(restored.pop("initial_positions_angstrom"), dtype=float)
    return restored


def collect_worker_results(result_queue, expected, args, on_result=None):
    results = []
    last_heartbeat = time.time()
    heartbeat = float(getattr(args, "heartbeat_interval_seconds", 0.0) or 0.0)
    while len(results) < int(expected):
        try:
            timeout = min(60.0, heartbeat) if heartbeat > 0.0 else 60.0
            message = result_queue.get(timeout=max(1.0, timeout))
        except queue_module.Empty:
            if heartbeat > 0.0 and time.time() - last_heartbeat >= heartbeat:
                print(f"[ga] waiting for workers: {len(results)}/{int(expected)} complete", flush=True)
                last_heartbeat = time.time()
            continue
        if message.get("event") != "result" or not message.get("ok", False):
            raise RuntimeError("Ordinary GA worker failed:\n" + str(message.get("traceback", message)))
        result = restore_worker_result(message["result"])
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def release_problem_calculator(problem):
    for name in ("mp_unit_cell", "ground_truth_atoms"):
        atoms = getattr(problem, name, None)
        if atoms is not None:
            atoms.calc = None
    problem.calc = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_ga(problem, args, seed, mode):
    population_size = int(args.ga_population_size)
    generation_count = int(args.ga_generations)
    counts = allocate_generation_counts(population_size, args)
    parallel_workers = min(max(1, int(args.ga_parallel_workers)), population_size)
    output_dir = Path(problem.output_dir)
    ga_log = Path(args.ga_log_file).expanduser() if str(args.ga_log_file or "").strip() else output_dir / "genetic_optimization_log.txt"
    running_log = Path(args.running_log_file).expanduser() if str(args.running_log_file or "").strip() else output_dir / "running_log.txt"
    database_path = output_dir / "structures.json"
    summary_path = output_dir / "genetic_optimization_summary.json"
    ga_log.write_text("", encoding="utf-8")
    initialize_population_database(database_path)
    running_log.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ga_start seed={int(seed)}\n", encoding="utf-8")
    log_event(ga_log, {
        "event": "ga_start",
        "representation": "ordinary_hard_atoms",
        "crossover": "periodic_real_space_cut_and_splice",
        "parent_selection": str(args.hard_parent_selection),
        "population_size": population_size,
        "generations": generation_count,
        "parallel_workers": parallel_workers,
        "hard_relax_fmax": float(args.hard_relax_fmax),
        "hard_relax_steps": int(args.hard_relax_steps),
        "energy_evaluation_definition": "one_actual_FAIRChem_calculator.calculate_call",
    })
    ga_wall_start = time.perf_counter()
    all_generations = []
    previous_population = []
    global_best = None
    imported_candidates = imported_hard_candidates(problem, args, mode)
    task_queue = result_queue = None
    workers = []
    if parallel_workers > 1:
        main_output = problem.output_dir
        release_problem_calculator(problem)
        task_queue, result_queue, workers = start_workers(mode, args, parallel_workers, main_output)

    def evaluate_tasks(tasks, on_result):
        if parallel_workers > 1:
            for task in tasks:
                task_queue.put(task)
            return collect_worker_results(result_queue, len(tasks), args, on_result=on_result)
        results = []
        for task in tasks:
            candidate = task["candidate"]
            candidate["positions"] = np.asarray(candidate["positions"], dtype=float)
            result = evaluate_candidate(
                problem,
                candidate,
                args,
                task["seed"],
                task["generation"],
                task["member_index"],
                task["kind"],
                task["parent_refs"],
            )
            result.update(task.get("metadata", {}))
            results.append(result)
            on_result(result)
        return results

    try:
        for generation in range(generation_count):
            generation_wall_start = time.perf_counter()
            append_line(running_log, f"{time.strftime('%Y-%m-%d %H:%M:%S')} | generation_start generation={generation}")
            generation_results = []
            tasks = []

            def record_member(result):
                generation_results.append(result)
                append_population_record(database_path, result_record(result))
                log_event(ga_log, ga_log_record(result, "member_done"))
                metrics = result.get("compute_metrics", {})
                append_line(
                    running_log,
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} | member_done generation={result['generation']} "
                    f"member={result['member_index']} kind={result['kind']} score={float(result['score']):+.8f} "
                    f"energy_evaluations={int(metrics.get('model_energy_evaluations', 0))} "
                    f"energy_eval_s={float(metrics.get('model_evaluation_seconds', 0.0)):.2f} "
                    f"member_wall_s={float(metrics.get('member_wall_seconds', 0.0)):.2f}",
                )
                # Deliberately refresh after every member so progress remains
                # visible while a generation is still being evaluated.
                update_live_plot(problem, args, all_generations, generation_results)

            if generation == 0:
                for member_index in range(population_size):
                    member_seed = (int(seed) + generation * 1000003 + member_index * 9176) % 2**32
                    np.random.seed(member_seed)
                    if member_index < len(imported_candidates):
                        candidate = copy.deepcopy(imported_candidates[member_index])
                        kind = "imported_seed"
                    else:
                        candidate = make_initial_candidate(problem, args, mode)
                        kind = "random"
                    tasks.append(evaluation_task(candidate, member_seed, generation, member_index, kind))
            else:
                parents = sorted(previous_population, key=lambda member: float(member["score"]))
                parent_pool = parents[: min(int(args.ga_parent_pool_size), len(parents))]
                for member_index, elite in enumerate(parents[: counts["elite"]]):
                    record_member(copy_elite(elite, generation, member_index))
                member_index = int(counts["elite"])
                random_stop = min(population_size, member_index + int(counts["random"]))
                while member_index < random_stop:
                    member_seed = (int(seed) + generation * 1000003 + member_index * 9176) % 2**32
                    np.random.seed(member_seed)
                    candidate = make_initial_candidate(problem, args, mode)
                    tasks.append(evaluation_task(candidate, member_seed, generation, member_index, "random_immigrant"))
                    member_index += 1
                while member_index < population_size:
                    member_seed = (int(seed) + generation * 1000003 + member_index * 9176) % 2**32
                    np.random.seed(member_seed)
                    parent_a, parent_b = choose_parent_pair(problem, parent_pool, args)
                    child = crossover_candidate(problem, parent_a, parent_b, args, mode)
                    child = mutate_candidate(problem, child, args, mode)
                    refs = [parent_ref(parent_a), parent_ref(parent_b)]
                    tasks.append(evaluation_task(
                        child,
                        member_seed,
                        generation,
                        member_index,
                        "crossover_child",
                        parent_refs=refs,
                        metadata={"parent_diversity": diversity_distance(problem, parent_a, parent_b)},
                    ))
                    member_index += 1
            evaluate_tasks(tasks, record_member)
            generation_results.sort(key=lambda member: int(member["member_index"]))
            if len(generation_results) != population_size:
                raise RuntimeError(
                    f"Generation {generation} produced {len(generation_results)}/{population_size} members"
                )
            ranked = sorted(generation_results, key=lambda member: float(member["score"]))
            generation_best = ranked[0]
            if global_best is None or float(generation_best["score"]) < float(global_best["score"]):
                global_best = copy.deepcopy(generation_best)
            previous_population = generation_results
            all_generations.append(generation_results)
            generation_compute = aggregate_compute_metrics(generation_results)
            generation_wall_seconds = time.perf_counter() - generation_wall_start
            log_event(ga_log, {
                "event": "generation_done",
                "generation": generation,
                "best_score": float(generation_best["score"]),
                "global_best_score": float(global_best["score"]),
                "best_n_sites": int(generation_best["n_sites"]),
                "energy_evaluations": int(generation_compute["model_energy_evaluations"]),
                "energy_evaluation_seconds": float(generation_compute["model_evaluation_seconds"]),
                "evaluated_member_wall_seconds_sum": float(generation_compute["member_wall_seconds_sum"]),
                "generation_wall_seconds": float(generation_wall_seconds),
            })
            append_line(
                running_log,
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | generation_done generation={generation} "
                f"best={float(generation_best['score']):+.8f} global_best={float(global_best['score']):+.8f} "
                f"energy_evaluations={generation_compute['model_energy_evaluations']} "
                f"energy_eval_s={generation_compute['model_evaluation_seconds']:.2f} "
                f"generation_wall_s={generation_wall_seconds:.2f}",
            )
    finally:
        if workers:
            stop_workers(task_queue, workers)

    ga_wall_seconds = time.perf_counter() - ga_wall_start
    total_compute = aggregate_compute_metrics(
        [member for generation in all_generations for member in generation]
    )
    log_event(ga_log, {
        "event": "ga_done_compute",
        "energy_evaluations": int(total_compute["model_energy_evaluations"]),
        "energy_evaluation_seconds": float(total_compute["model_evaluation_seconds"]),
        "evaluated_member_wall_seconds_sum": float(total_compute["member_wall_seconds_sum"]),
        "ga_wall_seconds": float(ga_wall_seconds),
    })
    append_line(
        running_log,
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ga_done "
        f"energy_evaluations={total_compute['model_energy_evaluations']} "
        f"energy_eval_s={total_compute['model_evaluation_seconds']:.2f} "
        f"ga_wall_s={ga_wall_seconds:.2f}",
    )
    summary = {
        "generator": "ordinary_fixed_scaffold_ga.py",
        "mode": mode,
        "representation": "ordinary_fully_occupied_atoms",
        "local_optimizer": "ASE FIRE; fixed cell; fixed scaffold atoms",
        "crossover": "periodic real-space cut-and-splice",
        "parent_selection": str(args.hard_parent_selection),
        "compute_metrics_documentation": {
            "energy_evaluation_definition": "One actual FAIR-Chem calculator.calculate execution. A joint energy/forces calculation counts once; ASE cache hits count zero.",
            "energy_evaluation_seconds_definition": "Summed wall time inside FAIR-Chem calculator.calculate across workers. This is aggregate model work time, not elapsed GA wall time.",
            "member_wall_seconds_definition": "Elapsed worker time for one candidate's hard FIRE relaxation and scoring; it excludes main-process plotting and database output.",
            "member_wall_seconds_sum_definition": "Sum over evaluated candidates; with parallel workers it intentionally exceeds elapsed GA wall time.",
            "copied_elites": "Elite copies have zero compute metrics because their relaxation is not repeated.",
            "scope": "Candidate metrics exclude one-time model loading and reference preparation. GA wall time includes worker startup, orchestration, per-member plotting, logging, and database appends.",
        },
        "total_candidate_compute_metrics": total_compute,
        "generation_compute_metrics": [aggregate_compute_metrics(generation) for generation in all_generations],
        "ga_wall_seconds": float(ga_wall_seconds),
        "population_size": population_size,
        "generations": generation_count,
        "elite_fraction": float(args.ga_elite_fraction),
        "random_fraction": float(args.ga_random_fraction),
        "crossover_fraction": float(args.ga_crossover_fraction),
        "parent_pool_size": int(args.ga_parent_pool_size),
        "tournament_size": int(args.ga_parent_a_tournament_size),
        "imported_seed_count": int(len(imported_candidates)),
        "candidate_relaxation": {
            "fmax_eV_per_angstrom": float(args.hard_relax_fmax),
            "steps": int(args.hard_relax_steps),
            "maxstep_angstrom": float(args.hard_relax_maxstep_angstrom),
            "unconverged_penalty": float(args.hard_relax_unconverged_penalty),
        },
        "hard_atom_genetic": {
            "mutation_probability": float(args.hard_mutation_probability),
            "mutation_weights": mutation_weights(args),
            "rattle_atom_fraction": float(args.hard_rattle_atom_fraction),
            "rattle_sigma_angstrom": float(args.ga_crossover_pos_sigma),
            "minimum_distance_angstrom": float(args.ga_superposition_prune_distance_angstrom),
            "initial_count_mode": str(args.hard_initial_count_mode),
            "count_limits": list(count_limits(problem, args, mode)),
        },
        "estimated_search_region_atoms": float(problem.estimated_search_region_atoms),
        "ground_truth_reference_formation_energy_per_atom": float(problem.full_reference_formation_energy_per_atom),
        "best": result_record(global_best),
        "generations_summary": [
            [ga_log_record(member, "member") for member in generation]
            for generation in all_generations
        ],
        "structures_json": str(database_path),
        "ga_log_file": str(ga_log),
    }
    atomic_write_json(summary_path, summary)
    print(
        f"[compute] energy_evaluations={total_compute['model_energy_evaluations']} "
        f"energy_evaluation_time={total_compute['model_evaluation_seconds']:.2f} s "
        f"ga_wall_time={ga_wall_seconds:.2f} s",
        flush=True,
    )
    return global_best


def write_best_outputs(problem, best, args, seed, mode, entry_script):
    candidate = {"symbols": best["symbols"], "positions": best["positions"]}
    atoms = build_full_atoms(problem, candidate)
    atoms.calc = None
    atoms.set_constraint()
    atoms.wrap()
    write(problem.output_dir / "ordinary_ga_best.xyz", atoms)
    write(problem.output_dir / "ordinary_ga_best.xsf", atoms)
    payload = {
        "generator": str(Path(entry_script).resolve()),
        "seed": int(seed),
        "mode": mode,
        "representation": "ordinary_fully_occupied_atoms",
        "fixed_cell": True,
        "fixed_scaffold_atoms": int(len(problem.fixed_symbols)),
        "variable_atoms": int(best["n_sites"]),
        "best": result_record(best),
        "cell_angstrom": np.asarray(problem.cell, dtype=float).tolist(),
        "pbc": np.asarray(problem.pbc, dtype=bool).tolist(),
        "ground_truth_reference_formation_energy_per_atom": float(problem.full_reference_formation_energy_per_atom),
        "files": {
            "xyz": str(problem.output_dir / "ordinary_ga_best.xyz"),
            "xsf": str(problem.output_dir / "ordinary_ga_best.xsf"),
        },
    }
    atomic_write_json(problem.output_dir / "ordinary_ga_result.json", payload)


def write_run_parameters(args, raw_config, entry_script):
    payload = {
        "script": str(Path(entry_script).resolve()),
        "matching_eat_script": str(args.eat_source_script),
        "config_path": str(args.config_path),
        "algorithm": "ordinary hard-atom fixed-scaffold memetic GA",
        "resolved_parameters": {key: json_safe(value) for key, value in sorted(vars(args).items())},
        "config": raw_config,
    }
    atomic_write_json(Path(args.output_dir) / "run_parameters.json", payload)


def parse_cli(default_config, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(default_config), help="YAML configuration path")
    return parser.parse_args()


def main(mode, entry_script):
    entry_script = Path(entry_script).resolve()
    default_config = (entry_script.with_name("ordinary_genetic_disloc_edge_110.yaml")
                      if mode == "dislocation" else entry_script.with_suffix(".yaml"))
    cli = parse_cli(default_config, f"Ordinary hard-atom GA for {mode.replace('_', ' ')}")
    source, args, raw = load_settings(mode, cli.config)
    validate_settings(mode, args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Output directory exists: {output_dir}") from exc
    args.output_dir = str(output_dir)
    write_run_parameters(args, raw, entry_script)
    seed = int(time.time_ns() % 2**32) if args.seed is None else int(args.seed)
    args.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    problem = source.InpaintingProblem(args)
    print(f"[ordinary GA] mode={mode} config={args.config_path}")
    print(f"[seed] {seed}")
    print(f"[device] {problem.device}")
    print(
        f"[population] size={args.ga_population_size} generations={args.ga_generations} "
        f"fractions=(elite={args.ga_elite_fraction}, random={args.ga_random_fraction}, crossover={args.ga_crossover_fraction})"
    )
    print(
        f"[selection] {args.hard_parent_selection} tournament_size={args.ga_parent_a_tournament_size} "
        f"parent_pool={args.ga_parent_pool_size}"
    )
    print(
        f"[relaxation] FIRE fixed_cell=true fixed_scaffold=true fmax={args.hard_relax_fmax} "
        f"steps={args.hard_relax_steps} maxstep={args.hard_relax_maxstep_angstrom} A"
    )
    print(
        f"[variation] periodic_cut_and_splice min_distance={args.ga_superposition_prune_distance_angstrom} A "
        f"mutation_probability={args.hard_mutation_probability} rattle_sigma={args.ga_crossover_pos_sigma} A"
    )
    if mode == "h_sublattice":
        print(f"[atom count] fixed H count={problem.target_missing_count}")
    else:
        lower, upper = count_limits(problem, args, mode)
        print(
            f"[atom count] variable; Poisson mean={problem.estimated_search_region_atoms:.3f} "
            f"limits=[{lower}, {upper}] insert/delete enabled"
        )
    if bool(args.ga_enable):
        best = run_ga(problem, args, seed, mode)
    else:
        candidate = make_initial_candidate(problem, args, mode)
        best = evaluate_candidate(problem, candidate, args, seed, 0, 0, "random", [])
    write_best_outputs(problem, best, args, seed, mode, entry_script)
    print(
        f"[done] best_score={best['score']:+.8f} Eform/atom={best['formation_energy_per_atom']:+.8f} "
        f"variable_atoms={best['n_sites']} converged={best['relaxation_converged']} "
        f"max_force={best['final_max_force_eV_per_angstrom']:.6f} eV/A"
    )
