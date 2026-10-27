"""Collect every verification result into one JSON consumed by build_report.py.

Reads the result tree written by the cluster jobs (~/heat_verify/results on altas, sshfs-mounted locally) plus the
archive's own logs. Usage: python collect_report_data.py RESULTS_DIR REPO OUT_JSON
"""
import csv
import json
import re
import sys
from pathlib import Path

RES, REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def load(p):
    p = RES / p
    return json.loads(p.read_text()) if p.exists() else None


def parse_log(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if parts:
            rows.append((parts[0], dict(p.split("=", 1) for p in parts[1:] if "=" in p)))
    return rows


def run_curve(log_path, ref_eform):
    """Cumulative best formation-energy gap per generation, cumulative evaluations and wall hours."""
    recs = parse_log(log_path)
    members = [f for e, f in recs if e in ("member_done", "elite_copied") and "formation_energy_per_atom" in f]
    gens = {int(f["generation"]): f for e, f in recs if e == "generation_done"}
    out, best, evals, wall = [], 1e9, 0, 0.0
    for g in sorted(gens):
        for m in members:
            if int(m["generation"]) == g:
                best = min(best, float(m["formation_energy_per_atom"]))
        evals += int(float(gens[g].get("energy_evaluations", 0)))
        wall += float(gens[g].get("generation_wall_seconds", 0))
        out.append(dict(g=g, gap=best - ref_eform, evals=evals, hours=wall / 3600))
    partial = [m for m in members if int(m["generation"]) not in gens]
    return dict(curve=out, members_logged=len(members), partial_generation_members=len(partial),
                partial_best_gap=(min(float(m["formation_energy_per_atom"]) for m in partial) - ref_eform) if partial else None)


data = {}
data["audit"] = load("audit_recorded_results.json")
data["wse2_free"] = load("wse2_weightfree/wse2_weightfree_results.json")
data["eat_fake"] = load("eat_patch/eat_patch_results.json")
data["eat_real"] = load("eat_patch_real/eat_patch_results.json")
data["rescore"] = load("rescore_real/rescore_real_results.json")
data["mp_mu"] = load("mp_elements/mp_mu_check.json")
data["config_loading"] = load("config_loading.json")

# launcher: first root error per case
launcher = {}
for f in sorted((RES / "launcher").glob("*.log")):
    errs = [l for l in f.read_text().splitlines() if re.match(r"^[A-Za-z_.]+(Error|Exception):", l) and "CalledProcessError" not in l]
    launcher[f.stem] = errs[-1][:200] if errs else "no error captured"
data["launcher"] = launcher

# repo unit tests
tests = {}
for f in sorted((RES / "repo_tests").glob("test_*.log")):
    t = f.read_text()
    ran = re.search(r"Ran (\d+) tests?", t)
    status = "OK" if re.search(r"^OK", t, re.M) else re.search(r"^FAILED.*$", t, re.M).group(0) if re.search(r"^FAILED", t, re.M) else "?"
    tests[f.stem] = dict(ran=int(ran.group(1)) if ran else 0, status=status)
data["repo_tests"] = tests

# archive convergence curves (recorded runs) for the charts
H_REF = -0.2228974413906268
edge_ref = float(dict(next(f for e, f in parse_log(REPO / "images/edge/eat/genetic_optimization_log.txt") if e == "ga_start"))[
    "ground_truth_reference_formation_energy_per_atom"])
data["recorded"] = {
    "H": {m: run_curve(REPO / f"images/H_sublatt/{m}/genetic_optimization_log.txt", H_REF) for m in ("eat", "ordinary")},
    "edge": {m: run_curve(REPO / f"images/edge/{m}/genetic_optimization_log.txt", edge_ref) for m in ("eat", "ordinary")},
}
wse2 = {}
for m in ("eat", "ordinary"):
    with open(REPO / f"images/wse2/{m}/generation_metrics.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["complete"] == "True"]
    ev, hrs, curve = 0, 0.0, []
    for r in sorted(rows, key=lambda r: int(r["generation"])):
        ev += int(float(r["model_energy_evaluations"]))
        hrs += float(r["summed_member_wall_seconds"]) / 3600
        curve.append(dict(g=int(r["generation"]), loss=float(r["best_so_far_score"]), evals=ev, hours=hrs))
    wse2[m] = curve
data["recorded"]["wse2"] = wse2

# reruns with real UMA
reruns = {}
for d in sorted((RES / "real_runs").glob("*_job*")):
    log = next(iter(d.rglob("genetic_optimization_log.txt")), None)
    info = dict(dir=d.name)
    gt = next(iter(d.rglob("ground_truth_reference_energy.json")), None)
    if log is not None:
        start = dict(next((f for e, f in parse_log(log) if e == "ga_start"), {}))
        if "ground_truth_reference_formation_energy_per_atom" in start:
            ref = float(start["ground_truth_reference_formation_energy_per_atom"])
            info["reference_total_energy"] = float(start.get("ground_truth_reference_total_energy", "nan"))
        elif gt is not None:
            ref = json.loads(gt.read_text())["formation_energy_per_atom_eV"]
        else:
            ref = None
        info["reference_formation_energy"] = ref
        info["seed"] = start.get("seed")
        if ref is not None:
            info.update(run_curve(log, ref))
        else:  # WSe2 HEAT driver: loss curve from generation_done + per-member evaluation counts
            evals_by_g = {}
            for line in (log.parent / "running_log.txt").read_text().splitlines():
                m = re.search(r"member_done generation=(\d+) .*energy_evaluations=(\d+)", line)
                if m:
                    evals_by_g[int(m.group(1))] = evals_by_g.get(int(m.group(1)), 0) + int(m.group(2))
            ev, curve = 0, []
            for e, fdict in parse_log(log):
                if e == "generation_done":
                    ev += evals_by_g.get(int(fdict["generation"]), 0)
                    curve.append(dict(g=int(fdict["generation"]), loss=float(fdict["global_best_score"]), evals=ev))
            info["loss_curve"] = curve
            info["population"] = int(json.loads(next(iter(d.rglob("run_parameters.json"))).read_text()).get("resolved_parameters", {}).get("ga_population_size", 0) or 0) \
                if next(iter(d.rglob("run_parameters.json")), None) else None
    gm = next(iter(d.rglob("generation_metrics.csv")), None)
    if gm is not None:
        with open(gm) as fh:
            info["generation_metrics"] = [{k: r[k] for k in ("generation", "best_so_far_score", "best_formation_energy_per_atom",
                                                              "best_image_term", "best_atom_count", "model_energy_evaluations")}
                                          for r in csv.DictReader(fh)]
    mem_dir = next(iter(d.rglob("members")), None)
    if gm is None and mem_dir is not None and mem_dir.is_dir():
        # discrete WSe2 driver: rebuild per-generation metrics from its per-member records
        mem = [json.loads(p.read_text()) for p in sorted(mem_dir.glob("g*_m*.json"))]
        by_g, best, rows = {}, None, []
        for m in mem:
            by_g.setdefault(int(m["generation"]), []).append(m)
        for g in sorted(by_g):
            valid = [m for m in by_g[g] if m.get("valid")]
            gb = min(valid, key=lambda m: m["score"])
            best = gb if best is None or gb["score"] < best["score"] else best
            rows.append(dict(generation=g, members=len(by_g[g]), best_so_far_score=best["score"], best_formation_energy_per_atom=best["formation_energy_per_atom"],
                             best_image_term=best["image_term"], best_atom_count=best["atom_count"],
                             model_energy_evaluations=sum(m["compute_metrics"].get("model_energy_evaluations", 0) for m in by_g[g])))
        info["generation_metrics"] = rows
    summ = next(iter(d.rglob("summary.json")), None)
    if summ is not None:
        info["summary"] = json.loads(summ.read_text())
    reruns[d.name] = info
data["reruns"] = reruns
OUT.write_text(json.dumps(data, indent=1, default=float))
print("WROTE", OUT, "reruns:", list(reruns))
