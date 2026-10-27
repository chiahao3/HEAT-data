"""Real-UMA rescoring of bundled results and elemental-reference preparation.

  R1. W/Se elemental energies from the exact MP structures bundled in reference_energies.json vs the recorded values.
  R2. Full WSe2 loss (energy + image + composition + box) of the bundled guided reference, conventional winner and the
      previous energy-only structure, with the production conventional-GA objective, vs the recorded values.
  R3. Fe: bcc single points / relaxation vs mu_Fe implied by the edge log (E_ref/969 - E_f,ref = -8.174070 eV/atom).
  R4. Build and UMA-relax substitute elemental structures for Mg, B, H, Fe (no MP key available) for the MP stub.
Usage: python rescore_real.py REPO OUT_DIR
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.build import bulk, molecule
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

REPO, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
DISC = REPO / "scripts/eat_genetic_scripts/discrete_09.30.26_wse2"
BASE = REPO / "scripts/misc/baseline_ptycho_guided_320"
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
calc = FAIRChemCalculator(predictor, task_name="omat")
res = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}


def energy(atoms):
    a = atoms.copy()
    a.calc = calc
    return float(a.get_potential_energy())


# R1
refs = json.loads((BASE / "inputs/reference_energies.json").read_text())
r1 = {}
for el in refs["elements"]:
    st = el["structure"]
    a = Atoms(st["symbols"], positions=st["positions"], cell=st["cell"], pbc=True)
    e = energy(a) / len(a)
    r1[el["symbol"]] = dict(material_id=el["material_id"], recomputed=e, recorded=refs["energy_per_atom_eV"][el["symbol"]],
                            abs_diff=abs(e - refs["energy_per_atom_eV"][el["symbol"]]))
res["R1_WSe_elemental"] = r1
print("R1", r1, flush=True)

# R2
sys.path.insert(0, str(DISC))
spec = importlib.util.spec_from_file_location("discrete_wse2", DISC / "discrete_09.30.26_wse2.py")
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)
raw = yaml.safe_load((DISC / "discrete_09.30.26_wse2.yaml").read_text())
raw["imaging"]["transform_epsilon"] = float(raw["imaging"]["transform_epsilon"])
raw["paths"].update(output_dir=str(OUT / "problem"), element_fit_csv=str(DISC / raw["paths"]["element_fit_csv"]))
raw["experimental"]["tiff_file"] = str(REPO / "inputs/ptycho_data/objp_zstack_iter0200.tif")
cfg_path = OUT / "config_fixed.yaml"
cfg_path.write_text(yaml.safe_dump(raw))
ga.DEFAULT_CONFIG = cfg_path
config = ga.load_config(cfg_path)
problem = ga.DiscreteProblem(config, device=device, calculator=calc,
                             references=[refs["energy_per_atom_eV"]["W"], refs["energy_per_atom_eV"]["Se"]])
cmp_rows = {r.split(",")[0]: r.split(",") for r in (BASE / "objective_comparison.csv").read_text().splitlines()[1:]}
r2 = {}
cases = {"final_guided_relaxed": json.loads((BASE / "final_result.json").read_text())["structure"],
         "ordinary_winner": json.loads((BASE / "inputs/ordinary_best_structure.json").read_text())["structure"]}
prev = read(BASE / "inputs/previous_energy_only_relaxed.extxyz")
cases["previous_energy_only_relaxed"] = dict(symbols=prev.get_chemical_symbols(), positions=prev.positions.tolist())
for name, st in cases.items():
    score, _, t = problem.objective(list(st["symbols"]), np.asarray(st["positions"], float))
    rec = cmp_rows[name]
    r2[name] = dict(score=score, recorded_score=float(rec[1]), formation_energy=t["formation_energy_per_atom"],
                    recorded_formation_energy=float(rec[2]), abs_diff_score=abs(score - float(rec[1])),
                    abs_diff_formation=abs(t["formation_energy_per_atom"] - float(rec[2])), atoms=t["atom_count"])
    print("R2", name, r2[name], flush=True)
res["R2_WSe2_full_loss"] = r2

# R3: Fe chemical potential implied by the edge log
mu_fe_logged = -7806.832436 / 969 - 0.117481
r3 = dict(mu_Fe_implied_by_edge_log=mu_fe_logged, single_point={})
for a0 in (2.80, 2.82, 2.8334, 2.84, 2.86):
    r3["single_point"][f"{a0:.4f}"] = energy(bulk("Fe", "bcc", a=a0))
res["R3_Fe"] = r3


# R4: substitute elemental structures relaxed with UMA (positions + cell)
def alpha_boron():
    s = Structure.from_spacegroup("R-3m", Lattice.hexagonal(4.9179, 12.5805), ["B", "B"],
                                  [[0.1187, 0.8813, 0.8915], [0.1969, 0.8031, 0.0244]])
    return AseAtomsAdaptor.get_atoms(s.get_primitive_structure())


def h2_box():
    a = molecule("H2")
    a.set_cell([10.0, 10.0, 10.0])
    a.center()
    a.pbc = True
    return a


starts = {"Mg": bulk("Mg", "hcp", a=3.19, c=5.18), "B": alpha_boron(), "H": h2_box(), "Fe": bulk("Fe", "bcc", a=2.8334)}
r4 = {}
for sym, a in starts.items():
    a = a.copy()
    a.calc = calc
    opt = FIRE(a if sym == "H" else FrechetCellFilter(a), logfile=None)
    opt.run(fmax=0.01, steps=300)
    r4[sym] = dict(symbols=a.get_chemical_symbols(), positions=a.positions.tolist(), cell=a.cell.array.tolist(),
                   energy_per_atom=float(a.get_potential_energy()) / len(a), natoms=len(a))
    print("R4", sym, len(a), r4[sym]["energy_per_atom"], flush=True)
res["R3_Fe"]["relaxed"] = r4["Fe"]["energy_per_atom"]
res["R3_Fe"]["relaxed_lattice_A"] = float(np.cbrt(abs(np.linalg.det(r4["Fe"]["cell"])) * 2))
print("R3", res["R3_Fe"], flush=True)
(OUT / "elemental_structures.json").write_text(json.dumps(r4))
res["R4_elemental"] = {k: dict(energy_per_atom=v["energy_per_atom"], natoms=v["natoms"]) for k, v in r4.items()}
(OUT / "rescore_real_results.json").write_text(json.dumps(res, indent=1, default=float))
print("WROTE", OUT)
