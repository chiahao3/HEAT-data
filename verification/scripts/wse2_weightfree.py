"""Weights-free checks of the WSe2 ptychography objective and Fig. 4 claims.

Uses the production conventional-GA DiscreteProblem (same TIFF crop, normalization, Gaussian renderer, image/box/
composition terms as HEAT) with no energy model. Checks:
  A. experimental preprocessing reproduces the bundled 12x280x280 normalized field;
  B. non-energy loss terms of recorded structures (guided reference, conventional winner) are reproduced;
  C. Fig. 4c projection Pearson correlations (bundled TIFFs, and re-rendered from the bundled cropped XSFs);
  D. geometry of the gen-128 HEAT / conventional structures and the guided reference (layering, coordination, twist).
Usage: python wse2_weightfree.py REPO OUT_DIR
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml
from ase.io import read

REPO, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
DISC = REPO / "scripts/eat_genetic_scripts/discrete_09.30.26_wse2"
BASE = REPO / "scripts/misc/baseline_ptycho_guided_320"
FIG = REPO / "images/wse2/figure_280x280"
sys.path.insert(0, str(DISC))
spec = importlib.util.spec_from_file_location("discrete_wse2", DISC / "discrete_09.30.26_wse2.py")
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)
torch.set_num_threads(8)
results = {}

# Corrected copy of the bundled config: `transform_epsilon: 1e-06` is read by PyYAML as a string (see report).
raw = yaml.safe_load((DISC / "discrete_09.30.26_wse2.yaml").read_text())
raw["imaging"]["transform_epsilon"] = float(raw["imaging"]["transform_epsilon"])
raw["paths"]["output_dir"] = str(OUT / "problem")
raw["paths"]["element_fit_csv"] = str(DISC / raw["paths"]["element_fit_csv"])
raw["experimental"]["tiff_file"] = str(REPO / "inputs/ptycho_data/objp_zstack_iter0200.tif")
cfg_path = OUT / "config_fixed.yaml"
cfg_path.write_text(yaml.safe_dump(raw))
orig_default = ga.DEFAULT_CONFIG
ga.DEFAULT_CONFIG = cfg_path  # defaults would otherwise re-introduce the string value
config = ga.load_config(cfg_path)
ga.DEFAULT_CONFIG = orig_default
refs = json.loads((BASE / "inputs/reference_energies.json").read_text())["energy_per_atom_eV"]
problem = ga.DiscreteProblem(config, device="cpu", references=[refs["W"], refs["Se"]])
ex = problem.experimental
mask = np.asarray(ex["image_loss_mask"], bool)
zz, yy, xx = np.nonzero(mask)
z0, z1, y0, y1, x0, x1 = zz.min(), zz.max() + 1, yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
print("target grid", ex["target_shape_zyx"], "measured zyx box", (z0, z1, y0, y1, x0, x1), "cell", np.diag(problem.cell))

# A. preprocessing vs bundled normalized field (central 280x280 of the 320x320 measured crop)
target = np.asarray(ex["target_image"])
disp = (slice(z0, z1), slice(y0 + 20, y1 - 20), slice(x0 + 20, x1 - 20))
mine = target[disp]
bundled = np.load(FIG / "experimental_field_normalized_280x280.npy")
results["A_preprocessing"] = dict(shape=list(mine.shape), bundled_shape=list(bundled.shape),
                                  max_abs_diff=float(np.abs(mine - bundled).max()))
print("A", results["A_preprocessing"])
exp_zsum = mine.sum(0)


def terms_for(symbols, positions):
    symbols = list(symbols)
    t, _ = problem.image_and_box(symbols, np.asarray(positions, float), gradient=False)
    counts = np.array([symbols.count(s) for s in ga.SPECIES])
    t["composition_penalty"] = config["composition"]["penalty"] * float(
        np.sum((counts / len(symbols) - problem.species_probabilities) ** 2))
    t["atom_count"] = len(symbols)
    return t


# B. non-energy terms of recorded full structures
KEYS = ("image_term", "sample_box_term", "composition_penalty", "image_correlation")
for name, path in (("guided_reference", BASE / "final_result.json"),
                   ("conventional_winner_g144", BASE / "inputs/ordinary_best_structure.json")):
    rec = json.loads(path.read_text())
    t = terms_for(rec["structure"]["symbols"], rec["structure"]["positions"])
    row = {k: dict(recomputed=t[k], recorded=rec[k], abs_diff=abs(t[k] - rec[k])) for k in KEYS}
    row["implied_formation_energy"] = rec["score"] - rec["image_term"] - rec["sample_box_term"] - rec["composition_penalty"]
    row["recorded_formation_energy"] = rec["formation_energy_per_atom"]
    results[f"B_{name}"] = row
    print("B", name, {k: (round(v["recomputed"], 8), round(v["recorded"], 8)) for k, v in row.items() if isinstance(v, dict)})


# C. Fig. 4c projections
def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


render_meta = json.loads((FIG / "generation_0128/render_metadata.json").read_text())
for method in ("heat", "ordinary"):
    tif = tifffile.imread(FIG / f"generation_0128/{method}_generation_0128_reconstructed_zsum_280x280.tif").astype(float)
    atoms = read(FIG / f"generation_0128/{method}_generation_0128_cropped_280x280.xsf")
    sym = atoms.get_chemical_symbols()
    pos = atoms.positions.copy()
    # The cropped XSF keeps the same vacuum padding around a smaller footprint: undo the symmetric lateral shift.
    lateral_shift = np.diag(problem.cell)[:2] / 2 - atoms.cell.lengths()[:2] / 2
    pos[:, :2] += lateral_shift
    ids = torch.as_tensor([ga.SPECIES.index(s) for s in sym])
    labels = torch.nn.functional.one_hot(ids, 2).to(problem.dtype)
    with torch.no_grad():
        lin, _ = problem.gaussian_renderer.render(labels, torch.as_tensor(pos, dtype=problem.dtype))
    rer = lin.cpu().numpy()[disp].sum(0)
    counts = {s: sym.count(s) for s in ("W", "Se")}
    results[f"C_{method}"] = dict(
        pearson_bundled_tif_vs_experiment=pearson(tif, exp_zsum),
        pearson_rerendered_from_cropped_xsf=pearson(rer, exp_zsum),
        rerender_over_tif_scale=float((rer * tif).sum() / (tif * tif).sum()),
        rerender_vs_scaled_tif_interior_rel_max=float(np.abs(rer / ((rer * tif).sum() / (tif * tif).sum()) - tif)[15:-15, 15:-15].max() / tif.max()),
        recorded=render_meta["methods"][method]["cropped_zsum_correlation"],
        cropped_atoms=len(sym), species=counts, lateral_shift_A=lateral_shift.tolist())
    print("C", method, results[f"C_{method}"])


# D. geometry
def geometry(symbols, positions, cell_xy, label):
    symbols = np.asarray(symbols)
    pos = np.asarray(positions, float)
    z = pos[:, 2]
    # 1D k-means of heights into 6 planes (two Se-W-Se trilayers), initialised at height quantiles
    centers = np.quantile(z, (np.arange(6) + 0.5) / 6)
    for _ in range(200):
        lab = np.argmin(np.abs(z[:, None] - centers[None, :]), 1)
        new = np.array([z[lab == k].mean() if (lab == k).any() else centers[k] for k in range(6)])
        if np.allclose(new, centers):
            break
        centers = new
    order = np.argsort(centers)
    planes = [dict(z_mean=float(z[lab == k].mean()), z_std=float(z[lab == k].std()), n=int((lab == k).sum()),
                   W=int(((lab == k) & (symbols == "W")).sum()), Se=int(((lab == k) & (symbols == "Se")).sum()))
              for k in order if (lab == k).any()]
    # nearest-neighbour W-Se (minimum image in xy only; z is non-periodic here)
    W, Se = pos[symbols == "W"], pos[symbols == "Se"]
    d = W[:, None, :] - Se[None, :, :]
    d[..., :2] -= cell_xy * np.round(d[..., :2] / cell_xy)
    r = np.linalg.norm(d, axis=-1)
    nn = r.min(1)
    coord = (r < 2.9).sum(1)
    # local trilayer test: each W in an ideal Se-W-Se layer has 3 Se above and 3 below (|dz| 1.2-2.2 A, r < 2.9 A)
    dz = -d[..., 2]  # Se z - W z
    close = r < 2.9
    above = (close & (dz > 1.2) & (dz < 2.2)).sum(1)
    below = (close & (dz < -1.2) & (dz > -2.2)).sum(1)
    trilayer_W = float(((above == 3) & (below == 3)).mean())
    # in-plane W-W orientation per W plane -> twist
    twist = []
    for p in planes:
        if p["W"] < 0.6 * p["n"]:
            continue
        g = [i for i in range(len(pos)) if symbols[i] == "W" and abs(pos[i, 2] - p["z_mean"]) < max(1.0, 2 * p["z_std"])]
        P = pos[g]
        dd = P[:, None, :2] - P[None, :, :2]
        dd -= cell_xy * np.round(dd / cell_xy)
        rr = np.linalg.norm(dd, axis=-1)
        sel = (rr > 2.8) & (rr < 3.7)
        ang = np.degrees(np.arctan2(dd[..., 1][sel], dd[..., 0][sel])) % 60.0
        # circular mean on the 60-degree circle
        th = np.radians(ang * 6)
        twist.append(dict(z=p["z_mean"], angle_mod60=float(np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) / 6 % 60),
                          concentration=float(np.hypot(np.sin(th).mean(), np.cos(th).mean())), n_W=len(g),
                          ww_nn_median=float(np.median(rr[sel])) if sel.any() else None))
    out = dict(label=label, atoms=len(symbols), W=int((symbols == "W").sum()), Se=int((symbols == "Se").sum()),
               n_planes=len(planes), planes=planes,
               W_Se_nn_median=float(np.median(nn)), W_Se_nn_p10_p90=[float(np.percentile(nn, 10)), float(np.percentile(nn, 90))],
               W_with_6_Se_within_2p9=float((coord == 6).mean()), W_with_3_Se_above_and_3_below=trilayer_W,
               z_hist={sp: np.histogram(pos[symbols == sp, 2], bins=np.arange(0, 28.25, 0.25))[0].tolist() for sp in ("W", "Se")}, W_Se_coordination_hist={int(k): int(v) for k, v in zip(*np.unique(coord, return_counts=True))},
               W_layers_orientation=twist)
    if len(twist) == 2:
        dt = (twist[1]["angle_mod60"] - twist[0]["angle_mod60"] + 30) % 60 - 30
        out["twist_between_W_layers_deg"] = float(dt)
    return out


geo = {}
for method in ("heat", "ordinary"):
    atoms = read(FIG / f"generation_0128/{method}_generation_0128_cropped_280x280.xsf")
    geo[method + "_g128_crop"] = geometry(atoms.get_chemical_symbols(), atoms.positions, atoms.cell.lengths()[:2], method)
g0 = read(FIG / "generation_0000_best_loss_cropped_280x280.xsf")
geo["heat_g0_crop"] = geometry(g0.get_chemical_symbols(), g0.positions, g0.cell.lengths()[:2], "heat_g0")
rec = json.loads((BASE / "final_result.json").read_text())["structure"]
P = np.asarray(rec["positions"], float)
lo = (x0 + 20) * problem.target_voxel_size_A[2]
hi = (x1 - 20) * problem.target_voxel_size_A[2]
inside = (P[:, 0] >= lo) & (P[:, 0] < hi) & (P[:, 1] >= lo) & (P[:, 1] < hi)
geo["guided_reference_crop"] = geometry(np.asarray(rec["symbols"])[inside], P[inside], np.diag(problem.cell)[:2], "guided")
results["D_geometry"] = geo
for k, v in geo.items():
    print("D", k, {kk: v[kk] for kk in ("atoms", "W", "Se", "n_planes", "W_Se_nn_median", "W_with_6_Se_within_2p9", "W_with_3_Se_above_and_3_below")},
          "twist", v.get("twist_between_W_layers_deg"))
    for p in v["planes"]:
        print("     plane", p)

(OUT / "wse2_weightfree_results.json").write_text(json.dumps(results, indent=1, default=float))
print("WROTE", OUT / "wse2_weightfree_results.json")
