"""Recompute every quantitative claim of arXiv:2610.28996 that the bundled logs/metrics can support.

Pure parsing of images/*/{eat,ordinary}/genetic_optimization_log.txt and images/wse2/*/*.csv; no model needed.
Usage: python audit_recorded_results.py REPO OUT_JSON
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
IMG = REPO / "images"


def parse(path):
    rows = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        fields = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
        rows.append((parts[0], fields))
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def fixed_scaffold_case(case, ref_eform, n_scaffold, n_ref_active, per_label, per_count):
    out = {}
    for method in ("eat", "ordinary"):
        recs = parse(IMG / case / method / "genetic_optimization_log.txt")
        start = dict(next(f for e, f in recs if e == "ga_start"))
        gens = {int(f["generation"]): {k: num(v) for k, v in f.items()} for e, f in recs if e == "generation_done"}
        members = [{k: num(v) for k, v in f.items()} for e, f in recs if e in ("member_done", "elite_copied")
                   and "formation_energy_per_atom" in f]
        g_sorted = sorted(gens)
        cum_evals = np.cumsum([gens[g]["energy_evaluations"] for g in g_sorted])
        cum_wall = np.cumsum([gens[g]["generation_wall_seconds"] for g in g_sorted])
        # cumulative-best formation energy (the quantity plotted), and the candidate that sets it
        best, best_member, curve, hit, best_members = np.inf, None, [], None, []
        for g in g_sorted:
            for m in members:
                if int(m["generation"]) == g and m["formation_energy_per_atom"] < best:
                    best, best_member = m["formation_energy_per_atom"], m
            curve.append(best - ref_eform)
            best_members.append(best_member)
            if hit is None and best - ref_eform < 1e-4:
                hit = dict(generation=g, member=best_member)
        flagged = [m for m in members if "final_relaxation_converged" in m]
        unconv = [m for m in flagged if str(m["final_relaxation_converged"]) == "False"]
        info = dict(seed=start.get("seed"), generations_logged=len(g_sorted), last_generation=g_sorted[-1],
                    members_logged=len(members),
                    final_relax_unconverged_fraction=(len(unconv) / len(flagged)) if flagged else "not logged")
        if hit:
            g = hit["generation"]
            i = g_sorted.index(g)
            m = hit["member"]
            n_act = int(m.get("realized_atom_count", m.get("n_sites")))
            info["hit"] = dict(generation=g, cumulative_evaluations=int(cum_evals[i]), cumulative_wall_hours=cum_wall[i] / 3600,
                               formation_energy=m["formation_energy_per_atom"], gap_eV_per_atom=m["formation_energy_per_atom"] - ref_eform,
                               active_atom_count=n_act, reference_active_count=n_ref_active,
                               final_relaxation_converged=m.get("final_relaxation_converged"),
                               final_max_force=m.get("final_max_force_eV_per_angstrom"))
        else:
            info["hit"] = None
        g20 = min(20, g_sorted[-1])
        i20 = g_sorted.index(g20)
        gap = curve[i20]
        m = best_members[i20]
        n_act = int(m.get("realized_atom_count", m.get("n_sites")))
        e_total_gap = (m["formation_energy_per_atom"] * (n_scaffold + n_act) - ref_eform * (n_scaffold + n_ref_active))
        info[f"at_generation_{g20}"] = dict(cumulative_evaluations=int(cum_evals[i20]), cumulative_wall_hours=cum_wall[i20] / 3600,
                                             gap_eV_per_atom=gap, best_active_atom_count=n_act,
                                             total_energy_gap_eV=e_total_gap, **{f"gap_eV_per_{per_label}": e_total_gap / per_count})
        info["per_generation_wall_s_mean_gens_1_to_last"] = float(np.mean([gens[g]["generation_wall_seconds"] for g in g_sorted[1:]]))
        info["_wall"] = {g: gens[g]["generation_wall_seconds"] for g in g_sorted}
        info["cumulative_best_gap_curve"] = [float(c) for c in curve]
        out[method] = info
    # matched-window time per generation (generations 1..min(last_eat, last_ordinary, 20))
    last = min(out["eat"]["last_generation"], out["ordinary"]["last_generation"], 20)
    for method in out:
        wall = out[method].pop("_wall")
        out[method]["matched_window"] = f"1-{last}"
        out[method]["per_generation_wall_s_matched_window"] = float(np.mean([wall[g] for g in range(1, last + 1)]))
    return out


res = {}
# H sublattice: reference hard-coded in the plot script (not in the log)
res["H_sublattice"] = fixed_scaffold_case("H_sublatt", -0.2228974413906268, 90, 240, "H_atom", 240)
for method in ("eat", "ordinary"):
    # plot_h_sublattice_convergence.py: E_f(cand)*(90+n)/n - E_f(ref)*330/240, i.e. total formation-energy gap per H atom
    r = res["H_sublattice"][method]["at_generation_20"]
    r["gap_per_atom_times_330_over_240"] = r["gap_eV_per_atom"] * 330 / 240
# edge: reference from ga_start record; per-core normalisation of the paper uses 48 core atoms (8 per layer x 6)
edge_ref = float(dict(next(f for e, f in parse(IMG / "edge/eat/genetic_optimization_log.txt") if e == "ga_start"))[
    "ground_truth_reference_formation_energy_per_atom"])
res["edge"] = fixed_scaffold_case("edge", edge_ref, 605, 364, "core_atom", 48)
res["edge"]["reference_formation_energy_per_atom"] = edge_ref


def read_csv(path):
    with open(path) as f:
        return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(f)]


w = {}
for method in ("eat", "ordinary"):
    gm = [r for r in read_csv(IMG / f"wse2/{method}/generation_metrics.csv") if str(r["complete"]) == "True"]
    gm.sort(key=lambda r: int(r["generation"]))
    cum = np.cumsum([r["model_energy_evaluations"] for r in gm])
    best = [r["best_so_far_score"] for r in gm]

    def row_at(g):
        # best-so-far member details at generation g (the generation's best row if it set the record, else earlier)
        k = max(i for i, r in enumerate(gm) if int(r["generation"]) <= g and r["generation_best_score"] <= r["best_so_far_score"] + 1e-12)
        r = gm[k]
        return dict(generation=g, set_at_generation=int(r["generation"]), best_loss=r["best_so_far_score"],
                    formation_energy=r["best_formation_energy_per_atom"], weighted_image_penalty=r["best_image_term"],
                    atoms=int(r["best_atom_count"]), W=int(r["best_W"]), Se=int(r["best_Se"]),
                    cumulative_evaluations=int(cum[[int(x["generation"]) for x in gm].index(g)]))

    first_035 = next((int(r["generation"]) for r in gm if r["best_so_far_score"] <= -0.35), None)
    w[method] = dict(last_generation=int(gm[-1]["generation"]), generation0_best=best[0], first_generation_loss_le_m0p35=first_035,
                     at_128=row_at(128), at_last=row_at(int(gm[-1]["generation"])),
                     total_evaluations=int(cum[-1]))
    if method == "eat":
        # budget-matched comparison: first HEAT generation whose cumulative evaluations reach the conventional GA's 128-gen budget
        w[method]["_cum"] = cum.tolist()
        w[method]["_gens"] = [int(r["generation"]) for r in gm]
    w[method]["cumulative_best_curve"] = [float(b) for b in best]
    w[method]["cumulative_evaluations_curve"] = [int(c) for c in cum]
conv128 = w["ordinary"]["at_128"]["cumulative_evaluations"]
gens, cum = w["eat"].pop("_gens"), np.asarray(w["eat"].pop("_cum"))
g_budget = gens[int(np.searchsorted(cum, conv128))]
w["eat"]["budget_matched"] = dict(conventional_evaluations_through_128=conv128, heat_generation=g_budget,
                                  heat_row=w["eat"]["at_128"] if g_budget == 128 else None)
w["eat"]["budget_matched"]["heat_row"] = None
mm = read_csv(IMG / "wse2/eat/member_metrics.csv")
best_by_g = {}
bm = None
for g in range(0, g_budget + 1):
    for r in mm:
        if int(r["generation"]) == g and str(r["valid"]) == "True" and (bm is None or r["score"] < bm["score"]):
            bm = r
w["eat"]["budget_matched"]["heat_best_through_budget_generation"] = dict(
    score=bm["score"], formation_energy=bm["formation_energy_per_atom"], weighted_image_penalty=bm["image_term"],
    generation_found=int(bm["generation"]))
for g in (42, 43, 44):
    b = None
    for r in mm:
        if int(r["generation"]) <= g and str(r["valid"]) == "True" and (b is None or r["score"] < b["score"]):
            b = r
    w["eat"]["budget_matched"][f"heat_best_through_gen_{g}"] = dict(
        cumulative_evaluations=int(cum[gens.index(g)]), score=b["score"], formation_energy=b["formation_energy_per_atom"],
        weighted_image_penalty=b["image_term"], generation_found=int(b["generation"]))
final = w["eat"]["at_128"]["best_loss"]
w["eat"]["first_generation_reaching_final_best"] = next(int(r["generation"]) for r in mm
                                                        if str(r["valid"]) == "True" and abs(r["score"] - final) < 1e-9)
res["wse2"] = w
OUT.write_text(json.dumps(res, indent=1, default=float))


def show(d, indent=0):
    for k, v in d.items():
        if isinstance(v, list):
            continue
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            show(v, indent + 2)
        else:
            print(" " * indent + f"{k}: {v}")


show(res)
print("WROTE", OUT)
