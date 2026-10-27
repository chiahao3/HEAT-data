"""Property checks of the EAT-modified FAIRChem (fairchem cce3699 + fairchem_local_changes.patch).

Weights-free: uses fairchem's own fake conserving MoLE checkpoint (UMA architecture, ~random weights), so these
test the *implementation* of effective atoms, not physics. Checks:
  T1 one-hot eat_weights reproduce the integer-species prediction (energy, forces);
  T2 patched fairchem without eat_weights reproduces stock (unpatched) fairchem   [run via --stock subprocess];
  T3 calculator eat_grad (dE/dw) matches central finite differences;
  T4 a fully null site (all weights 0) leaves energy and real-atom forces unchanged (atom effectively absent);
  T5 the H-sublattice driver helpers (set_multi_species_eat_weights, get_energy_gradients_and_forces) agree with T1-T3.
Usage: python eat_patch_checks.py CKPT OUT_JSON [--stock]   (--stock: only print plain E/F as JSON for T2)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from ase.io import read

CKPT, OUT = sys.argv[1], Path(sys.argv[2])
STOCK = "--stock" in sys.argv
REPO = Path(__file__).resolve().parents[2]
from fairchem.core import FAIRChemCalculator  # noqa: E402
from fairchem.core.units.mlip_unit import MLIPPredictUnit  # noqa: E402

torch.manual_seed(0)
if CKPT.startswith("uma-"):  # real pretrained UMA (production path used by the drivers)
    from fairchem.core import pretrained_mlip
    predictor = pretrained_mlip.get_predict_unit(CKPT, device="cuda" if torch.cuda.is_available() else "cpu")
    TASK = "omat"
else:  # fairchem's fake UMA-architecture checkpoint
    predictor = MLIPPredictUnit(CKPT, device="cpu")
    TASK = "oc20"
calc = FAIRChemCalculator(predictor, task_name=TASK)


def structures():
    fe = bulk("Fe", "bcc", a=2.87, cubic=True).repeat((3, 3, 3))
    fe.rattle(0.05, seed=1)
    mgbh = read(REPO / "inputs/MgB2H8/cm803019e_si_003.cif")
    return {"Fe54": fe, "MgB2H8": mgbh}


def plain(atoms):
    a = atoms.copy()
    a.calc = calc
    return float(a.get_potential_energy()), np.asarray(a.get_forces())


if STOCK:
    out = {k: dict(zip(("energy", "forces"), (lambda e, f: (e, f.tolist()))(*plain(a)))) for k, a in structures().items()}
    OUT.write_text(json.dumps(out))
    sys.exit(0)

model = predictor.model
nz = next(m for m in model.modules() if hasattr(m, "sphere_embedding")).sphere_embedding.weight.shape[0]
print("embedding table size", nz)


def eat(atoms, w, grad=False):
    a = atoms.copy()
    a.new_array("eat_weights", np.asarray(w, np.float32))
    if grad:
        a.info["enable_eat_grad"] = True
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces())
    g = np.asarray(a.calc.results["eat_grad"]) if grad else None
    return e, f, g


def onehot(atoms):
    w = np.zeros((len(atoms), nz), np.float32)
    w[np.arange(len(atoms)), atoms.numbers] = 1.0
    return w


res = {}
S = structures()
for name, at in S.items():
    e0, f0 = plain(at)
    e1, f1, _ = eat(at, onehot(at))
    e2, f2, g2 = eat(at, onehot(at), grad=True)
    res[f"T1_{name}"] = dict(E_plain=e0, E_onehot=e1, E_onehot_grad=e2, dE=abs(e1 - e0), dE_grad=abs(e2 - e0),
                             dF_max=float(np.abs(f1 - f0).max()), dF_grad_max=float(np.abs(f2 - f0).max()),
                             F_scale=float(np.abs(f0).max()))
    print("T1", name, res[f"T1_{name}"])

# T3: finite differences on fractional weights (MgB2H8: H sites H/null; Fe: some sites Fe/null)
rng = np.random.default_rng(0)
for name, at in S.items():
    w = onehot(at)
    var_sites = np.where(at.numbers == (1 if name == "MgB2H8" else 26))[0]
    var_sites = rng.choice(var_sites, size=min(12, len(var_sites)), replace=False)
    z = 1 if name == "MgB2H8" else 26
    w[var_sites, z] = rng.uniform(0.3, 0.9, len(var_sites))
    e, _, g = eat(at, w, grad=True)
    h = 2e-2
    rows = []
    for i in var_sites[:8]:
        wp, wm = w.copy(), w.copy()
        wp[i, z] += h
        wm[i, z] -= h
        fd = (eat(at, wp)[0] - eat(at, wm)[0]) / (2 * h)
        rows.append((float(g[i, z]), float(fd)))
    # directional derivative over all variable sites
    v = np.zeros_like(w)
    v[var_sites, z] = rng.normal(size=len(var_sites))
    v /= np.linalg.norm(v)
    dd_fd = (eat(at, w + h * v)[0] - eat(at, w - h * v)[0]) / (2 * h)
    dd_ad = float((g * v).sum())
    an, fd = np.array(rows).T
    res[f"T3_{name}"] = dict(pairs=rows, max_rel_err=float(np.max(np.abs(an - fd) / np.maximum(np.abs(fd), 1e-3))),
                             directional_ad=dd_ad, directional_fd=dd_fd,
                             directional_rel_err=abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-6))
    print("T3", name, {k: v for k, v in res[f"T3_{name}"].items() if k != "pairs"})

# T4: null site invariance
for name, at in S.items():
    base_e, base_f, _ = eat(at, onehot(at))
    rows = {}
    for label, z_ins in (("null_H_symbol", 1), ("null_Fe_symbol", 26)):
        a2 = at.copy()
        p = a2.positions.mean(0) + rng.normal(scale=0.5, size=3)
        a2.append(z_ins)
        a2.positions[-1] = p
        w = onehot(a2)
        w[-1] = 0.0
        e, f, _ = eat(a2, w)
        rows[label] = dict(dE=abs(e - base_e), dF_real_max=float(np.abs(f[:-1] - base_f).max()),
                           F_on_null=float(np.abs(f[-1]).max()))
        for eps in (1e-3, 1e-1):
            w[-1, z_ins] = eps
            rows[label][f"dE_occ_{eps:g}"] = abs(eat(a2, w)[0] - base_e)
    res[f"T4_{name}"] = rows
    print("T4", name, rows)

# T4b: same null-site test with the neighbor cap raised from the fake model's 20 to UMA's production 300.
# With a binding cap, a null site can displace a real neighbour from the graph even though its edges are gated to 0.
caps = [m for m in model.modules() if hasattr(m, "max_neighbors")]
res["T4_fake_model_max_neighbors"] = [int(m.max_neighbors) for m in caps]
for m in caps:
    m.max_neighbors = 300
for name, at in S.items():
    base_e, base_f, _ = eat(at, onehot(at))
    a2 = at.copy()
    a2.append(1)
    a2.positions[-1] = at.positions.mean(0) + rng.normal(scale=0.5, size=3)
    w = onehot(a2)
    w[-1] = 0.0
    e, f, _ = eat(a2, w)
    res[f"T4b_{name}_max_neighbors_300"] = dict(dE=abs(e - base_e), dF_real_max=float(np.abs(f[:-1] - base_f).max()))
    print("T4b", name, res[f"T4b_{name}_max_neighbors_300"])

# T5: the H-sublattice driver's own helpers (MAX_NUM_ELEMENTS layout, species column selection)
import importlib.util  # noqa: E402

drv_path = REPO / "scripts/eat_genetic_scripts/eat_genetic_08.27.26_H_sublattice.py"
sys.path.insert(0, str(drv_path.parent))
spec = importlib.util.spec_from_file_location("heat_h", drv_path)
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
at = S["MgB2H8"].copy()
species_z = [12, 5, 1]
w3 = np.zeros((len(at), 3), np.float32)
w3[np.arange(len(at)), [species_z.index(z) for z in at.numbers]] = 1.0
h_idx = np.where(at.numbers == 1)[0][:20]
w3[h_idx, 2] = 0.6
drv.set_multi_species_eat_weights(at, w3, species_z)
e_d, g_d, f_d = drv.get_energy_gradients_and_forces(at, calc, species_z, want_grad=True)
w_full = np.zeros((len(at), nz), np.float32)
for j, z in enumerate(species_z):
    w_full[:, z] = w3[:, j]
at_plain = at.copy()
del at_plain.arrays["eat_weights"]
e_r, f_r, g_r = eat(at_plain, w_full, grad=True)
res["T5_driver_helpers"] = dict(MAX_NUM_ELEMENTS=drv.MAX_NUM_ELEMENTS, model_table=nz, dE=abs(e_d - e_r),
                               dGrad_max=float(np.abs(g_d - g_r[:, species_z]).max()), dF_max=float(np.abs(f_d - f_r).max()))
print("T5", res["T5_driver_helpers"])

OUT.write_text(json.dumps(res, indent=1))
print("WROTE", OUT)
