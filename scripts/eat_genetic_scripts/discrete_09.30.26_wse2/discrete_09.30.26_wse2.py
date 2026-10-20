#!/usr/bin/env python3
"""Hard-atom WSe2 genetic search with microscopy-constrained local relaxation.

Chromosomes contain only real W/Se symbols and Cartesian positions. Atom
counts and identities change through discrete genetic operations; local
PBFGS moves positions only. All file paths in YAML are relative to that YAML.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import write

from gaussian_image_model import atomic_box_bounds, initialize_gaussian_renderer
from position_minimizer import minimizer as BaseMinimizer
from wse2_image_geometry import (
    prepare_experimental_volume, load_double_gaussian_element_fits,
    cylinder_pruned_feasible_sites_zyx, periodic_cubic_bspline_coefficients,
    periodic_cubic_bspline_sample_zyx, torch_image_transform,
    fetch_elemental_reference_atoms,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'discrete_09.30.26_wse2.yaml'
SPECIES = ('W', 'Se')
COLORS = {'random_immigrant': 'red', 'crossover_child': 'green',
          'mutated_child': 'purple', 'elite': 'gray', 'imported': 'orange'}


def reference_api_key(config):
    return str(config['model'].get('mp_api_key') or os.environ.get('MP_API_KEY', '')).strip()


def public_config(config):
    result = copy.deepcopy(config)
    result['model'].pop('mp_api_key', None)
    return result


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(jsonable(payload), f, allow_nan=False)
        f.write('\n')
        tmp = f.name
    os.replace(tmp, path)


def load_config(path):
    path = Path(path).expanduser().resolve()
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text())
    supplied = yaml.safe_load(path.read_text())
    if not isinstance(supplied, dict):
        raise ValueError('The configuration must be a YAML mapping.')
    for section, values in supplied.items():
        if section not in defaults or not isinstance(values, dict):
            raise ValueError(f'Unknown configuration section: {section}')
        for key, value in values.items():
            if key not in defaults[section]:
                raise ValueError(f'Unknown configuration setting: {section}.{key}')
            defaults[section][key] = value
    for section, key in [('paths', 'output_dir'), ('paths', 'import_structures_json'),
                         ('paths', 'element_fit_csv'), ('model', 'reference_energies_json'),
                         ('experimental', 'tiff_file')]:
        value = defaults[section][key]
        if value:
            p = Path(value).expanduser()
            defaults[section][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    validate_config(defaults)
    return defaults


def validate_config(c):
    g, m, h, r = (c[k] for k in ('genetic', 'mutation', 'hard_atoms', 'relaxation'))
    for section, keys in {
        'genetic': ['population_size', 'generations', 'parent_pool_size',
                    'parent_a_tournament_size', 'parent_b_tournament_size'],
        'hard_atoms': ['minimum_count', 'repair_attempts'],
        'runtime': ['parallel_workers', 'cpu_threads', 'checkpoint_interval_iterations'],
        'relaxation': ['max_iterations', 'bfgs_history', 'max_line_search'],
    }.items():
        for key in keys:
            v = c[section][key]
            if isinstance(v, bool) or int(v) != v or v < 1:
                raise ValueError(f'{section}.{key} must be a positive integer.')
    for key in ('import_parent_count',):
        if int(g[key]) != g[key] or not 0 <= g[key] <= g['population_size']:
            raise ValueError(f'Invalid genetic.{key}.')
    if g['import_parent_count'] and not c['paths']['import_structures_json']:
        raise ValueError('Import count requires paths.import_structures_json.')
    fractions = [g[k] for k in ('elite_fraction', 'random_fraction', 'crossover_fraction')]
    if not np.isfinite(fractions).all() or min(fractions) < 0 or not np.isclose(sum(fractions), 1):
        raise ValueError('Elite, random, and crossover fractions must sum to one.')
    if not 0 <= m['probability'] <= 1 or not 0 < m['rattle_atom_fraction'] <= 1:
        raise ValueError('Invalid mutation probability/rattle fraction.')
    weights = [m[k + '_weight'] for k in ('rattle', 'relocate', 'insert', 'delete', 'transmute')]
    if not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError('Mutation weights must be nonnegative with a positive sum.')
    for section, key in [('hard_atoms', 'minimum_distance_angstrom'), ('mutation', 'rattle_sigma_angstrom'),
                         ('imaging', 'penalty'), ('composition', 'penalty'),
                         ('sample_box', 'confinement_penalty'), ('sample_box', 'confinement_margin_angstrom'),
                         ('feasible_region', 'jitter_angstrom'), ('relaxation', 'cost_plateau_tolerance')]:
        if not np.isfinite(c[section][key]) or c[section][key] < 0:
            raise ValueError(f'{section}.{key} must be finite and nonnegative.')
    for section, key in [('sample_box', 'confinement_sharpness'), ('imaging', 'huber_delta'),
                         ('imaging', 'transform_epsilon'), ('relaxation', 'gradient_tolerance'),
                         ('runtime', 'heartbeat_seconds'), ('feasible_region', 'candidate_site_density_per_angstrom3')]:
        if not np.isfinite(c[section][key]) or c[section][key] <= 0:
            raise ValueError(f'{section}.{key} must be finite and positive.')
    if r['cost_plateau_patience'] < 1 or r['cost_plateau_min_iterations'] < 0:
        raise ValueError('Invalid relaxation plateau settings.')
    if not 0 < c['initialization']['occupied_probability'] <= 1:
        raise ValueError('initialization.occupied_probability must lie in (0, 1].')
    for section, key in [('initialization', 'mean_atom_count'), ('hard_atoms', 'maximum_count')]:
        v = c[section][key]
        if v is not None and (not np.isfinite(v) or v < h['minimum_count']):
            raise ValueError(f'{section}.{key} must be at least the minimum atom count.')
    if h['maximum_count'] is not None and int(h['maximum_count']) != h['maximum_count']:
        raise ValueError('hard_atoms.maximum_count must be an integer.')
    target = np.asarray([float(v) for v in str(c['composition']['target']).split(',')])
    if target.shape != (2,) or not np.isfinite(target).all() or min(target) <= 0:
        raise ValueError('composition.target must contain positive W and Se weights.')
    n_elite, n_random, n_cross = generation_counts(c)
    if n_cross and min(g['parent_pool_size'], g['population_size']) < 2:
        raise ValueError('Crossover requires at least two parents.')


def image_args(c):
    e, f, b, i = (c[k] for k in ('experimental', 'feasible_region', 'sample_box', 'imaging'))
    values = dict(e)
    values.pop('tiff_file')
    values.update(experimental_tiff=e['tiff_file'], **c['system'])
    values.update(mask_smoothing_sigma_pixels=f['mask_smoothing_sigma_pixels'],
                  mask_threshold_mode=f['threshold_mode'], mask_threshold=f['threshold'],
                  sample_box_z_min_angstrom=b['z_min_angstrom'],
                  sample_box_z_max_angstrom=b['z_max_angstrom'],
                  sample_box_confinement_margin_angstrom=b['confinement_margin_angstrom'],
                  gaussian_clip_experimental_to_unit_interval=i['clip_experimental_to_unit_interval'],
                  image_target_transform=i['target_transform'], gaussian_transform_epsilon=i['transform_epsilon'],
                  gaussian_shared_sigma_z_angstrom=i['shared_sigma_z_angstrom'],
                  gaussian_additional_lateral_blur_angstrom=i['additional_lateral_blur_angstrom'])
    return SimpleNamespace(**values)


class PositionMinimizer(BaseMinimizer):
    def _construct_B(self, s, y, S_list, Y_list):
        if S_list is None or Y_list is None:
            S_list, Y_list = [], []
        return super()._construct_B(s, y, S_list, Y_list)


class DiscreteProblem:
    def __init__(self, config, device=None, calculator=None, references=None, diagnostics=False):
        self.config = config
        self.args = image_args(config)
        self.output_dir = Path(config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        requested = device or config['runtime']['device']
        if requested == 'auto':
            requested = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device, self.dtype = torch.device(requested), torch.float32
        self.experimental = prepare_experimental_volume(self.args, self.output_dir, write_diagnostics=diagnostics)
        ex = self.experimental
        self.cell = ex['cell']
        self.lengths = np.diag(self.cell).copy()
        self.vacuum_position_shift = ex['vacuum_position_shift_A']
        self.target_shape_zyx, self.target_voxel_size_A = ex['target_shape_zyx'], ex['target_voxel_size_A']
        self.species = list(SPECIES)
        self.gaussian_element_fit_csv, self.gaussian_element_fits = load_double_gaussian_element_fits(config['paths']['element_fit_csv'])
        initialize_gaussian_renderer(self)
        self.target = torch.as_tensor(ex['target_image'], device=self.device, dtype=self.dtype)
        self.mask = torch.as_tensor(ex['image_loss_mask'], device=self.device)
        self.box_lower, self.box_upper = atomic_box_bounds(self.args, ex['sample_cell_lengths_A'], self.vacuum_position_shift)
        self.level_coefficients = torch.as_tensor(periodic_cubic_bspline_coefficients(ex['level_set_A']), device=self.device, dtype=self.dtype)
        ratio = np.asarray([float(v) for v in str(config['composition']['target']).split(',')])
        self.species_probabilities = ratio / ratio.sum()
        f = config['feasible_region']
        sites, summary = cylinder_pruned_feasible_sites_zyx(ex['feasible_mask'], self.target_voxel_size_A,
            f['cylinder_lateral_radius_angstrom'], f['cylinder_z_radius_angstrom'], ex['smoothed_target_sqrt'])
        feasible_volume = float(ex['feasible_mask'].sum() * np.prod(self.target_voxel_size_A))
        count = max(1, round(f['candidate_site_density_per_angstrom3'] * feasible_volume))
        if count > len(sites):
            raise ValueError(f'Candidate density requests {count} positions, but cylinders provide only {len(sites)}.')
        self.candidate_positions = sites[:count, ::-1] * np.asarray(self.target_voxel_size_A)[::-1]
        hard = config['hard_atoms']
        self.minimum_count = int(hard['minimum_count'])
        self.maximum_count = int(hard['maximum_count'] or count)
        mean = config['initialization']['mean_atom_count']
        mean = float(mean) if mean is not None else count * config['initialization']['occupied_probability']
        if not self.minimum_count <= mean <= min(count, self.maximum_count):
            raise ValueError('Starting mean atom count must fit the candidate pool and hard count bounds.')
        self.occupied_probability = mean / count
        self.initialization_metadata = dict(summary, candidate_pool_size=count,
            feasible_volume_A3=feasible_volume, mean_atom_count_before_repair=mean,
            occupied_probability=self.occupied_probability, maximum_count=self.maximum_count)
        if diagnostics:
            atomic_json(self.output_dir / 'initialization_settings.json', self.initialization_metadata)
        self.calc = None
        self.model_evaluations = 0
        self.model_seconds = 0.0
        self.reference_energies = None if references is None else np.asarray(references, dtype=float)
        if calculator is not None:
            self.install_calculator(calculator)

    def install_calculator(self, calculator):
        original = calculator.calculate
        def counted(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.model_evaluations += 1
                self.model_seconds += time.perf_counter() - start
        calculator.calculate = counted
        self.calc = calculator

    def ensure_calculator(self):
        if self.calc is None:
            try:
                from torch._native.registry import deregister_op_overrides
                deregister_op_overrides(disable_op_symbols='bmm')
            except (ImportError, AttributeError):
                pass
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            # Use the ordinary calculator on explicit chemical symbols.
            predictor = pretrained_mlip.get_predict_unit(self.config['model']['model_name'], device=str(self.device))
            self.install_calculator(FAIRChemCalculator(predictor, task_name=self.config['model']['task_name']))

    def ensure_references(self):
        if self.reference_energies is not None:
            return
        model = self.config['model']
        cache = self.output_dir / 'reference_energies.json'
        path = Path(model['reference_energies_json']) if model['reference_energies_json'] else cache
        if path.is_file():
            data = json.loads(path.read_text())
            if data['model_name'] != model['model_name'] or data['task_name'] != model['task_name']:
                raise ValueError('Elemental reference model/task does not match this run.')
            self.reference_energies = np.asarray([data['energy_per_atom_eV'][s] for s in SPECIES], dtype=float)
            if not np.isfinite(self.reference_energies).all():
                raise ValueError('Reference energies must be finite.')
            if path != cache:
                atomic_json(cache, data)
            return
        if model['reference_energies_json']:
            raise FileNotFoundError(path)
        key = reference_api_key(self.config)
        if not key:
            raise RuntimeError('Set model.mp_api_key in YAML, MP_API_KEY in the environment, or model.reference_energies_json for elemental formation-energy references.')
        self.ensure_calculator()
        data = dict(model_name=model['model_name'], task_name=model['task_name'], energy_per_atom_eV={}, elements=[])
        for symbol in SPECIES:
            ref = fetch_elemental_reference_atoms(symbol, key)
            atoms = ref.pop('atoms')
            atoms.calc = self.calc
            energy = float(atoms.get_potential_energy()) / len(atoms)
            data['energy_per_atom_eV'][symbol] = energy
            ref['structure'] = dict(symbols=atoms.get_chemical_symbols(), positions=atoms.positions.tolist(), cell=atoms.cell.array.tolist())
            data['elements'].append(ref)
        self.reference_energies = np.asarray([data['energy_per_atom_eV'][s] for s in SPECIES])
        atomic_json(cache, data)

    def jitter(self, positions, rng):
        original = np.asarray(positions, dtype=float).copy()
        amplitude = self.config['feasible_region']['jitter_angstrom']
        proposal = np.remainder(original + rng.uniform(-amplitude, amplitude, original.shape), self.lengths)
        with torch.no_grad():
            sampled = periodic_cubic_bspline_sample_zyx(self.level_coefficients,
                torch.as_tensor(proposal, device=self.device, dtype=self.dtype), self.target_voxel_size_A).cpu().numpy()
        allowed = (sampled >= 0) & (proposal >= self.box_lower).all(axis=1) & (proposal <= self.box_upper).all(axis=1)
        original[allowed] = proposal[allowed]
        return original

    def sample_position(self, rng):
        index = rng.integers(len(self.candidate_positions))
        return self.jitter(self.candidate_positions[index:index+1], rng)[0]

    def image_and_box(self, symbols, positions, gradient=True):
        pos = torch.as_tensor(positions, device=self.device, dtype=self.dtype).detach().requires_grad_(gradient)
        cell_positions = torch.remainder(pos, pos.new_tensor(self.lengths))
        # One-hot species labels are constants, never optimization variables.
        ids = torch.as_tensor([SPECIES.index(s) for s in symbols], device=self.device)
        labels = torch.nn.functional.one_hot(ids, len(SPECIES)).to(self.dtype)
        linear, _ = self.gaussian_renderer.render(labels, cell_positions)
        imaging = self.config['imaging']
        prediction = torch_image_transform(linear, imaging['target_transform'], imaging['transform_epsilon'])
        loss = torch.nn.functional.huber_loss(prediction[self.mask], self.target[self.mask], delta=imaging['huber_delta'])
        b = self.config['sample_box']
        sharp = b['confinement_sharpness']
        lower = pos.new_tensor(self.box_lower + b['confinement_margin_angstrom'])
        upper = pos.new_tensor(self.box_upper - b['confinement_margin_angstrom'])
        low = torch.nn.functional.softplus(sharp * (lower - cell_positions)) / sharp
        high = torch.nn.functional.softplus(sharp * (cell_positions - upper)) / sharp
        box = (low.square() + high.square()).sum(dim=1).mean()
        objective = imaging['penalty'] * loss + b['confinement_penalty'] * box
        grad = torch.autograd.grad(objective, pos)[0].detach().cpu().numpy() if gradient else None
        with torch.no_grad():
            p, t = prediction[self.mask], self.target[self.mask]
            p, t = p - p.mean(), t - t.mean()
            corr = (p*t).sum() / torch.sqrt(p.square().sum()*t.square().sum() + 1e-20)
        terms = dict(image_loss_raw=float(loss.detach()), image_term=float((imaging['penalty']*loss).detach()),
            sample_box_loss_raw=float(box.detach()), sample_box_term=float((b['confinement_penalty']*box).detach()),
            image_correlation=float(corr))
        return terms, grad

    def objective(self, symbols, positions):
        self.ensure_references()
        self.ensure_calculator()
        pos = np.remainder(positions, self.lengths)
        atoms = Atoms(symbols, positions=pos, cell=self.cell, pbc=True)
        atoms.calc = self.calc
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        counts = np.asarray([symbols.count(s) for s in SPECIES])
        formation = (energy - float(counts @ self.reference_energies)) / len(symbols)
        composition = self.config['composition']['penalty'] * float(np.sum((counts / len(symbols) - self.species_probabilities)**2))
        terms, image_gradient = self.image_and_box(symbols, pos)
        score = formation + composition + terms['image_term'] + terms['sample_box_term']
        gradient = -forces / len(symbols) + image_gradient
        if not np.isfinite(score) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite guided objective or gradient.')
        terms.update(score=score, energy_total_eV=energy, formation_energy_per_atom=formation,
            composition_penalty=composition, atom_count=len(symbols), species_counts=dict(zip(SPECIES, counts.tolist())))
        return score, gradient, terms


def candidate(symbols, positions):
    symbols = list(symbols)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(symbols) != len(positions) or any(s not in SPECIES for s in symbols) or not np.isfinite(positions).all():
        raise ValueError('A candidate must contain finite positions and explicit W/Se symbols.')
    return dict(symbols=symbols, positions=positions.copy())


def separation_ok(pos, existing, lengths, cutoff):
    existing = np.asarray(existing, dtype=float).reshape(-1, 3)
    delta = existing - pos
    delta -= np.rint(delta / lengths) * lengths
    return bool(np.all(np.sum(delta**2, axis=1) >= cutoff**2))


def repair(problem, child, rng):
    h = problem.config['hard_atoms']
    pos = np.remainder(child['positions'], problem.lengths)
    kept, symbols = [], []
    for index in rng.permutation(len(pos)):
        if len(kept) >= problem.maximum_count:
            break
        if separation_ok(pos[index], kept, problem.lengths, h['minimum_distance_angstrom']):
            kept.append(pos[index])
            symbols.append(child['symbols'][index])
    for _ in range(h['repair_attempts']):
        if len(kept) >= problem.minimum_count:
            break
        p = problem.sample_position(rng)
        if separation_ok(p, kept, problem.lengths, h['minimum_distance_angstrom']):
            kept.append(p)
            symbols.append(str(rng.choice(SPECIES, p=problem.species_probabilities)))
    if len(kept) < problem.minimum_count:
        raise ValueError('Could not repair candidate to the minimum atom count without collisions.')
    result = candidate(symbols, kept)
    result['repair_count_change'] = len(symbols) - len(child['symbols'])
    return result


def random_candidate(problem, rng):
    positions = problem.jitter(problem.candidate_positions, rng)
    occupied = rng.random(len(positions)) < problem.occupied_probability
    symbols = rng.choice(SPECIES, size=int(occupied.sum()), p=problem.species_probabilities).tolist()
    raw = candidate(symbols, positions[occupied])
    result = repair(problem, raw, rng)
    result['initial_atom_count_before_repair'] = len(symbols)
    return result


def crossover(problem, a, b, rng):
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    # Plane in the common microscopy frame, no parent translation/alignment.
    origin = rng.uniform(problem.box_lower, problem.box_upper)
    pa, pb = np.asarray(a['positions']), np.asarray(b['positions'])
    keep_a, keep_b = ((pa-origin) @ normal >= 0), ((pb-origin) @ normal < 0)
    symbols = [s for s,k in zip(a['symbols'],keep_a) if k] + [s for s,k in zip(b['symbols'],keep_b) if k]
    child = repair(problem, candidate(symbols, np.concatenate((pa[keep_a],pb[keep_b]))), rng)
    child['crossover_plane'] = dict(origin_xyz_A=origin.tolist(), normal_xyz=normal.tolist(),
        selected_from_parent_a=int(keep_a.sum()), selected_from_parent_b=int(keep_b.sum()))
    return child


def mutate(problem, child, rng):
    result = copy.deepcopy(child)
    m = problem.config['mutation']
    result['mutation'] = 'none'
    result['mutation_attempted'] = bool(rng.random() < m['probability'])
    if not result['mutation_attempted']:
        return result
    names = ('rattle','relocate','insert','delete','transmute')
    weights = np.asarray([m[k+'_weight'] for k in names], dtype=float)
    operation = str(rng.choice(names, p=weights/weights.sum()))
    pos = np.asarray(result['positions']).copy()
    symbols = result['symbols'].copy()
    n = len(symbols)
    changed = False
    if operation == 'rattle':
        idx = rng.choice(n, size=max(1, math.ceil(n*m['rattle_atom_fraction'])), replace=False)
        pos[idx] += rng.normal(0, m['rattle_sigma_angstrom'], (len(idx),3))
        changed = m['rattle_sigma_angstrom'] > 0
    elif operation in ('relocate','insert'):
        if operation != 'insert' or n < problem.maximum_count:
            index = int(rng.integers(n)) if operation == 'relocate' else None
            others = np.delete(pos,index,axis=0) if index is not None else pos
            for _ in range(problem.config['hard_atoms']['repair_attempts']):
                p = problem.sample_position(rng)
                if separation_ok(p,others,problem.lengths,problem.config['hard_atoms']['minimum_distance_angstrom']):
                    if index is not None:
                        pos[index] = p
                    else:
                        pos = np.vstack((pos,p))
                        symbols.append(str(rng.choice(SPECIES,p=problem.species_probabilities)))
                    changed = True
                    break
    elif operation == 'delete' and n > problem.minimum_count:
        index = int(rng.integers(n))
        pos = np.delete(pos,index,axis=0)
        symbols.pop(index)
        changed = True
    elif operation == 'transmute':
        index = int(rng.integers(n))
        symbols[index] = 'Se' if symbols[index] == 'W' else 'W'
        changed = True
    repaired = repair(problem, candidate(symbols,pos), rng)
    result.update(repaired)
    result['mutation'] = operation if changed else 'none_inapplicable'
    result['mutation_operator'] = operation
    return result


class StopRelaxation(Exception):
    pass


def relax_candidate(problem, child, progress_path=None):
    c = problem.config['relaxation']
    symbols = child['symbols'].copy()
    initial = np.remainder(child['positions'], problem.lengths)
    iterations = 0
    prior_metrics = dict(model_energy_evaluations=0, model_evaluation_seconds=0.0,
                         objective_evaluations=0, member_wall_seconds=0.0)
    # Restart from the best saved position, with a fresh BFGS history.
    if progress_path and Path(progress_path).is_file():
        saved = json.loads(Path(progress_path).read_text())
        if saved['symbols'] != symbols:
            raise ValueError('Member progress symbols do not match the planned candidate.')
        initial = np.asarray(saved['positions'],dtype=float)
        iterations = int(saved['iterations'])
        prior_metrics.update(saved.get('compute_metrics', {}))
    x0 = torch.as_tensor(initial.reshape(-1),device=problem.device,dtype=problem.dtype)
    lengths = x0.new_tensor(problem.lengths)
    best = dict(value=float('inf'), x=x0.clone(), terms=None)
    state = dict(iterations=iterations, errors=[], values=[], reason='max_iterations', last_heartbeat=time.monotonic())
    calls_before, seconds_before = problem.model_evaluations, problem.model_seconds
    start = time.perf_counter()
    objective_evaluations = 0
    def project(x):
        return torch.remainder(x.reshape(-1,3),lengths).reshape(-1)
    def fun(x):
        nonlocal objective_evaluations
        value, gradient, terms = problem.objective(symbols,x.detach().cpu().numpy().reshape(-1,3))
        objective_evaluations += 1
        if value < best['value']:
            best.update(value=value,x=x.detach().clone(),terms=terms)
        return x.new_tensor(value), x.new_tensor(gradient.reshape(-1))
    def save_progress():
        if progress_path:
            atomic_json(progress_path,dict(symbols=symbols,positions=best['x'].detach().cpu().numpy().reshape(-1,3),
                iterations=state['iterations'],score=best['value'],compute_metrics=dict(
                    model_energy_evaluations=prior_metrics['model_energy_evaluations']+problem.model_evaluations-calls_before,
                    model_evaluation_seconds=prior_metrics['model_evaluation_seconds']+problem.model_seconds-seconds_before,
                    objective_evaluations=prior_metrics['objective_evaluations']+objective_evaluations,
                    member_wall_seconds=prior_metrics['member_wall_seconds']+time.perf_counter()-start)))
    def callback(x,f,error,k,converged):
        state['iterations'] = iterations + k
        state['values'].append(best['value'])
        state['errors'].append(float(error))
        if state['iterations'] % problem.config['runtime']['checkpoint_interval_iterations'] == 0:
            save_progress()
        now = time.monotonic()
        if now - state['last_heartbeat'] >= problem.config['runtime']['heartbeat_seconds']:
            print(f"  relaxation iteration={state['iterations']} score={float(f):.7g} best={best['value']:.7g}",flush=True)
            state['last_heartbeat'] = now
        if error < c['gradient_tolerance']:
            state['reason'] = 'gradient_tolerance'
            raise StopRelaxation
        patience = c['cost_plateau_patience']
        if c['cost_plateau_tolerance'] > 0 and state['iterations'] >= c['cost_plateau_min_iterations'] and len(state['values']) > patience:
            recent = state['values'][-(patience+1):]
            if recent[0]-recent[-1] <= c['cost_plateau_tolerance']:
                state['reason'] = 'cost_plateau'
                raise StopRelaxation
    try:
        fun(x0)
        remaining = c['max_iterations'] - iterations
        if remaining > 0:
            opt = PositionMinimizer(fun,x0,args=(),project=project,proj_args=(),callback=callback,
                BFGS_hist=c['bfgs_history'],maxit=remaining,tol=c['gradient_tolerance'],max_line_search=c['max_line_search'])
            _, final = opt.PBFGS()
            _, final_grad = fun(final)
            if torch.linalg.norm(project(final-final_grad)-final).item() < c['gradient_tolerance']:
                state['reason'] = 'gradient_tolerance'
    except StopRelaxation:
        pass
    except ValueError as exc:
        if 'No edges found in input system' not in str(exc) or not math.isfinite(best['value']):
            raise
        # A line-search trial disconnected every pair; retain the best valid state.
        state['reason'] = 'invalid_disconnected_trial'
    save_progress()
    final_pos = best['x'].detach().cpu().numpy().reshape(-1,3)
    _, gradient, terms = problem.objective(symbols, final_pos)
    objective_evaluations += 1
    result = dict(terms)
    result.update(structure=dict(symbols=symbols, positions=final_pos.tolist(), cell=problem.cell.tolist(), pbc=[True]*3),
        relaxation=dict(termination_reason=state['reason'],iterations=state['iterations'],
                        gradient_norm=float(np.linalg.norm(gradient)),initial_atom_count=len(symbols)),
        compute_metrics=dict(model_energy_evaluations=prior_metrics['model_energy_evaluations']+problem.model_evaluations-calls_before,
            model_evaluation_seconds=prior_metrics['model_evaluation_seconds']+problem.model_seconds-seconds_before,
            objective_evaluations=prior_metrics['objective_evaluations']+objective_evaluations,
            member_wall_seconds=prior_metrics['member_wall_seconds']+time.perf_counter()-start))
    return result


def generation_counts(c):
    g = c['genetic']
    raw = np.asarray([g[k] for k in ('elite_fraction','random_fraction','crossover_fraction')])*g['population_size']
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw-counts),kind='stable')[:g['population_size']-int(counts.sum())]:
        counts[idx] += 1
    return tuple(int(x) for x in counts)


def parent_pair(pool, rng, a_size, b_size):
    def tournament(indices, size):
        chosen = rng.choice(indices,size=min(size,len(indices)),replace=False)
        return min(chosen, key=lambda j:pool[int(j)]['score'])
    a = int(tournament(np.arange(len(pool)),a_size))
    b = int(tournament(np.asarray([j for j in range(len(pool)) if j != a]),b_size))
    return pool[a], pool[b]


def parent_ref(record):
    return dict(generation=record['generation'],member_index=record['member_index'],score=record['score'])


def import_candidates(problem):
    count = problem.config['genetic']['import_parent_count']
    if count == 0:
        return []
    path = Path(problem.config['paths']['import_structures_json'])
    data = json.loads(path.read_text())
    records = data['structures'] if isinstance(data,dict) else data
    ranked = sorted((r for r in records if math.isfinite(float(r['score']))),key=lambda r:r['score'])
    if len(ranked) < count:
        raise ValueError(f'Requested {count} imported records, but only {len(ranked)} have finite scores.')
    output = []
    for record in ranked[:count]:
        s = record['structure']
        cell = s.get('cell',s.get('cell_angstrom',s.get('cell_vectors_angstrom')))
        if cell is None or not np.allclose(cell,problem.cell,atol=1e-5,rtol=0):
            raise ValueError('Imported structure cell does not match the current padded experimental cell.')
        positions = s.get('positions',s.get('cart_positions_angstrom',s.get('positions_angstrom')))
        child = candidate(s['symbols'],positions)
        if not problem.minimum_count <= len(child['symbols']) <= problem.maximum_count:
            raise ValueError('Imported structure violates hard atom-count bounds.')
        output.append((child,dict(source_path=str(path),source_score=record['score'],
                                 generation=record.get('generation'),member_index=record.get('member_index'))))
    return output


def plan_generation(problem, previous, generation, seed, imports=()):
    c = problem.config
    g = c['genetic']
    rng = np.random.default_rng(np.random.SeedSequence([seed,generation]))
    tasks = []
    n_elite,n_random,n_cross = generation_counts(c)
    if generation == 0:
        for child,ref in imports:
            tasks.append(dict(candidate=child,kind='imported',parent_refs=[ref]))
        for _ in range(g['population_size']-len(tasks)):
            tasks.append(dict(candidate=random_candidate(problem,rng),kind='random_immigrant',parent_refs=[]))
    else:
        pool = sorted(previous,key=lambda r:r['score'])[:g['parent_pool_size']]
        for record in sorted(previous,key=lambda r:r['score'])[:n_elite]:
            elite = copy.deepcopy(record)
            elite.update(kind='elite',parent_refs=[parent_ref(record)],compute_metrics=dict(
                model_energy_evaluations=0,model_evaluation_seconds=0.0,objective_evaluations=0,member_wall_seconds=0.0))
            tasks.append(dict(elite=elite,kind='elite'))
        for _ in range(n_random):
            tasks.append(dict(candidate=random_candidate(problem,rng),kind='random_immigrant',parent_refs=[]))
        for _ in range(n_cross):
            a,b = parent_pair(pool,rng,g['parent_a_tournament_size'],g['parent_b_tournament_size'])
            child = mutate(problem,crossover(problem,a['structure'],b['structure'],rng),rng)
            mutated = child['mutation'] not in ('none','none_inapplicable')
            tasks.append(dict(candidate=child,kind='mutated_child' if mutated else 'crossover_child',parent_refs=[parent_ref(a),parent_ref(b)]))
    for index,task in enumerate(tasks):
        task.update(generation=generation,member_index=index,seed=int(rng.integers(0,2**32)))
    return tasks


WORKER_PROBLEM = None


def worker_init(config, references, device_queue):
    global WORKER_PROBLEM
    device = device_queue.get()
    torch.set_num_threads(config['runtime']['cpu_threads'])
    WORKER_PROBLEM = DiscreteProblem(config, device=device, references=references)


def evaluate_task(task, problem=None):
    problem = problem or WORKER_PROBLEM
    np.random.seed(task['seed'])
    torch.manual_seed(task['seed'])
    out = problem.output_dir
    name = f"g{task['generation']:04d}_m{task['member_index']:04d}"
    path = out/'members'/f'{name}.json'
    if path.is_file():
        return json.loads(path.read_text())
    if 'elite' in task:
        result = copy.deepcopy(task['elite'])
    else:
        child = candidate(task['candidate']['symbols'],task['candidate']['positions'])
        calls_before, seconds_before = problem.model_evaluations, problem.model_seconds
        started = time.perf_counter()
        try:
            result = relax_candidate(problem,child,out/'progress'/f'{name}.json')
        except ValueError as exc:
            if 'No edges found in input system' not in str(exc):
                raise
            # The potential cannot evaluate isolated atoms: explicitly reject
            # this chromosome, while retaining its geometry and failure record.
            result = dict(score=1.0e30,valid=False,error=str(exc),atom_count=len(child['symbols']),
                structure=dict(symbols=child['symbols'],positions=child['positions'].tolist(),
                               cell=problem.cell.tolist(),pbc=[True]*3),
                relaxation=dict(termination_reason='invalid_disconnected_structure'),
                compute_metrics=dict(model_energy_evaluations=problem.model_evaluations-calls_before,
                    model_evaluation_seconds=problem.model_seconds-seconds_before,
                    objective_evaluations=0,member_wall_seconds=time.perf_counter()-started))
        result.setdefault('valid',True)
        result.update(kind=task['kind'],parent_refs=task['parent_refs'],
            genetic_operations={k:v for k,v in task['candidate'].items() if k not in ('symbols','positions')})
    result.update(generation=task['generation'],member_index=task['member_index'],seed=task['seed'])
    atoms = Atoms(**result['structure'])
    (out/'xsf_members').mkdir(exist_ok=True)
    xsf = out/'xsf_members'/f'{name}_{result["kind"]}.xsf'
    write(xsf,atoms)
    result['xsf_path'] = str(xsf)
    atomic_json(path,result)
    (out/'progress'/f'{name}.json').unlink(missing_ok=True)
    return result


def plot_live(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [r for r in records if r.get('valid',True)]
    fig,ax = plt.subplots(figsize=(9,5))
    for kind,color in COLORS.items():
        members = [r for r in records if r['kind']==kind]
        if members:
            x = [r['generation']+0.16*np.sin(17*r['member_index']+r['generation']) for r in members]
            ax.scatter(x,[r['score'] for r in members],s=16,c=color,alpha=.5,label=kind.replace('_',' '))
    gens = sorted(set(r['generation'] for r in records))
    running = []
    best = float('inf')
    for gen in gens:
        best = min(best,min(r['score'] for r in records if r['generation']==gen))
        running.append(best)
    ax.plot(gens,running,color='black',lw=1.2,label='best so far')
    ax.set(xlabel='Generation',ylabel='Guided objective',title='Discrete WSe₂ genetic search')
    ax.legend(fontsize=8)
    fig.tight_layout()
    tmp = out/'live_scores.tmp.png'
    fig.savefig(tmp,dpi=150)
    plt.close(fig)
    os.replace(tmp,out/'discrete_optimization_live_scores_by_generation.png')


def fingerprint(config):
    # Allow increasing requested generations or changing throughput, not science.
    c = public_config(config)
    c['genetic'].pop('generations')
    c['genetic'].pop('live_plot_enable')
    c.pop('runtime')
    blobs = [json.dumps(c,sort_keys=True).encode()]
    for name in ('discrete_09.30.26_wse2.py','wse2_image_geometry.py','gaussian_image_model.py','position_minimizer.py'):
        blobs.append((HERE/name).read_bytes())
    for path in (config['paths']['element_fit_csv'], config['experimental']['tiff_file']):
        blobs.append(Path(path).read_bytes())
    for path in (config['model']['reference_energies_json'], config['paths']['import_structures_json']):
        if path:
            blobs.append(Path(path).read_bytes())
    return hashlib.sha256(b'\0'.join(blobs)).hexdigest()


def update_outputs(problem, records, start, status):
    out = problem.output_dir
    ranked = sorted((r for r in records if r.get('valid',True)),key=lambda r:r['score'])
    atomic_json(out/'structures.json',dict(schema_version=1,structures=records))
    if not ranked:
        return
    best = ranked[0]
    write(out/'best_structure.xsf',Atoms(**best['structure']))
    atomic_json(out/'best_structure.json',best)
    metrics = {key:sum(r['compute_metrics'][key] for r in records) for key in
               ('model_energy_evaluations','model_evaluation_seconds','objective_evaluations','member_wall_seconds')}
    atomic_json(out/'summary.json',dict(status=status,best_score=best['score'],best_member=parent_ref(best),
        evaluated_and_elite_records=len(records),compute_metrics=metrics,current_invocation_wall_seconds=time.perf_counter()-start))
    if problem.config['genetic']['live_plot_enable']:
        plot_live(out,records)


def save_best_images(problem, records):
    import tifffile
    best = min((r for r in records if r.get('valid',True)),key=lambda r:r['score'])['structure']
    ids = torch.as_tensor([SPECIES.index(s) for s in best['symbols']],device=problem.device)
    labels = torch.nn.functional.one_hot(ids,2).to(problem.dtype)
    pos = torch.as_tensor(best['positions'],device=problem.device,dtype=problem.dtype)
    with torch.no_grad():
        prediction,_ = problem.gaussian_renderer.render(labels,pos)
        prediction = torch_image_transform(prediction,problem.config['imaging']['target_transform'],problem.config['imaging']['transform_epsilon'])
    ex = problem.experimental
    roi = tuple(slice(int(a),int(b)) for a,b in zip(ex['sample_roi_start_zyx'],ex['sample_roi_stop_zyx']))
    pred = prediction.cpu().numpy()[roi].astype(np.float32)
    target = ex['target_image'][roi].astype(np.float32)
    for name,data in [('best_predicted',pred),('experimental',target),('best_residual',pred-target)]:
        tifffile.imwrite(problem.output_dir/f'{name}_slices.tif',data,metadata={'axes':'ZYX'})


def run(config, resume=False):
    import fcntl
    out = Path(config['paths']['output_dir'])
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'.driver.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another discrete driver owns {out}.')
    checkpoint = out/'ga_checkpoint.json'
    signature = fingerprint(config)
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text())
        setup_only = old.get('status') == 'initialized' and not any(
            (out/p).exists() for p in ('structures.json','members','plans','progress'))
        if setup_only:
            seed = config['initialization']['seed']
            seed = int(seed) if seed is not None else old['seed']
            atomic_json(checkpoint,dict(fingerprint=signature,seed=seed,status='initialized'))
        else:
            if not resume:
                raise RuntimeError('Output contains a run. Use --resume or a new output directory.')
            if old['fingerprint'] != signature:
                raise ValueError('Cannot resume: science settings, inputs, or code changed.')
            seed = old['seed']
            if config['genetic']['generations'] < old.get('completed_generations', old.get('generation', -1)+1):
                raise ValueError('Cannot resume with fewer generations than already recorded.')
    else:
        if any((out/p).exists() for p in ('structures.json','members','plans')):
            raise RuntimeError('Existing run data without a checkpoint; use a fresh output directory.')
        seed = config['initialization']['seed']
        seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy) % 2**63
        atomic_json(checkpoint,dict(fingerprint=signature,seed=seed,status='initialized'))
    print(f'Discrete WSe2: output={out}, seed={seed}, counts={generation_counts(config)} (elite/random/crossover)',flush=True)
    atomic_json(out/'run_parameters.json',dict(config=public_config(config),seed=seed,code_fingerprint=signature))
    torch.set_num_threads(config['runtime']['cpu_threads'])
    workers = config['runtime']['parallel_workers']
    # The controller needs no GPU or potential when workers own the calculators.
    problem = DiscreteProblem(config,device='cpu' if workers>1 else None,diagnostics=not resume)
    problem.ensure_references()
    imports = import_candidates(problem)
    pool = None
    if workers>1:
        context = mp.get_context('spawn')
        queue = context.Queue()
        requested = config['runtime']['device']
        gpu_ids = [s.strip() for s in str(config['runtime']['gpu_ids']).split(',') if s.strip()]
        if requested == 'auto':
            devices = [f'cuda:{s}' for s in gpu_ids] if torch.cuda.is_available() else ['cpu']
        else:
            devices = [requested]
        if not devices:
            raise ValueError('No worker devices configured.')
        for index in range(workers):
            queue.put(devices[index%len(devices)])
        pool = ProcessPoolExecutor(max_workers=workers,mp_context=context,
            initializer=worker_init,initargs=(config,problem.reference_energies,queue))
    started = time.perf_counter()
    records, previous = [], []
    try:
        for generation in range(config['genetic']['generations']):
            plan_path = out/'plans'/f'g{generation:04d}.json'
            if plan_path.is_file():
                tasks = json.loads(plan_path.read_text())
            else:
                tasks = plan_generation(problem,previous,generation,seed,imports)
                atomic_json(plan_path,tasks)
            current, pending = [], []
            for task in tasks:
                member_path = out/'members'/f'g{generation:04d}_m{task["member_index"]:04d}.json'
                if member_path.is_file():
                    current.append(json.loads(member_path.read_text()))
                else:
                    pending.append(task)
            records.extend(current)
            if pool:
                futures = [pool.submit(evaluate_task,task) for task in pending]
                iterator = (future.result() for future in as_completed(futures))
            else:
                iterator = (evaluate_task(task,problem) for task in pending)
            for result in iterator:
                current.append(result)
                records.append(result)
                print(f"generation={generation} member={result['member_index']} kind={result['kind']} score={result['score']:.8g} atoms={result['atom_count']}",flush=True)
                update_outputs(problem,records,started,'running')
                atomic_json(checkpoint,dict(fingerprint=signature,seed=seed,status='running',generation=generation,completed_members=len(current)))
            previous = sorted(current,key=lambda r:r['member_index'])
            update_outputs(problem,records,started,'running')
            if not any(r.get('valid',True) for r in current):
                raise RuntimeError('All members are invalid; inspect members/*.json and the initialization/count settings.')
        save_best_images(problem,records)
        update_outputs(problem,records,started,'complete')
        atomic_json(checkpoint,dict(fingerprint=signature,seed=seed,status='complete',completed_generations=config['genetic']['generations']))
    finally:
        if pool:
            pool.shutdown(wait=True,cancel_futures=True)
        lock.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--validate-only',action='store_true',help='Check configuration, image geometry, initialization, and hard image gradients without loading FAIRChem.')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config['paths']['output_dir'] = str(args.output_dir.expanduser().resolve())
    if args.validate_only:
        torch.set_num_threads(config['runtime']['cpu_threads'])
        problem = DiscreteProblem(config,device='cpu',diagnostics=True)
        child = random_candidate(problem,np.random.default_rng(123))
        terms,grad = problem.image_and_box(child['symbols'],child['positions'])
        report = dict(initialization=problem.initialization_metadata,drawn_atom_count=len(child['symbols']),
            image_terms=terms,gradient_finite=bool(np.isfinite(grad).all()),generation_counts=generation_counts(config))
        atomic_json(problem.output_dir/'validation.json',report)
        print(json.dumps(report,indent=2))
    else:
        run(config,args.resume)


if __name__ == '__main__':
    main()
