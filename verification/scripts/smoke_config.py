"""Write a tiny-run portable_config.yaml (same format as run_search.py) and print the driver command.

Usage: python smoke_config.py REPO METHOD CASE OUT_DIR
"""
import json
import sys
from pathlib import Path

REPO, METHOD, CASE, OUT = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], Path(sys.argv[4]).resolve()
entry = json.loads((REPO / "configs" / f"{METHOD}_{CASE}.json").read_text())
cfg = entry["config"]
for section, key in entry["repository_paths"]:
    cfg[section][key] = str((REPO / cfg[section][key]).resolve())
cfg["paths"]["output_dir"] = str(OUT)
cfg["model"]["task_name"] = "oc20"  # the fake checkpoint has oc20/omol heads only
g = cfg["genetic"]
g.update(population_size=6, generations=2, parent_pool_size=6)
if "parallel_workers" in g:
    g.update(parallel_workers=1, parallel_gpu_ids="0")
OVERRIDES = {
    "ground_truth_relaxation": dict(steps=5),
    "candidate_relaxation": dict(steps=20),
    "objective": dict(reference_relax_steps=5),
    "random_immigrant_image_prefit": dict(soft_steps=20),
    "minimizer": dict(maxit=10, final_pbfgs_maxit=20, final_pbfgs_cost_plateau_min_iterations=5),
    "relaxation": dict(max_iterations=20, cost_plateau_min_iterations=5),
    "runtime": dict(gpu_ids="0", device="cuda"),
}
for section, values in OVERRIDES.items():
    if section in cfg:
        for k, v in values.items():
            if k in cfg[section]:
                cfg[section][k] = v
OUT.mkdir(parents=True, exist_ok=True)
lines = ["# smoke-test config (tiny run, fake model)"]
for section, values in cfg.items():
    lines.append(section + ":")
    for key, value in values.items():
        text = json.dumps(value)
        if key == "transform_epsilon" and "--fix-yaml" in sys.argv:
            text = "1.0e-06"  # see report: PyYAML reads 1e-06 as a string
        lines.append(f"  {key}: {text}")
(OUT / "portable_config.yaml").write_text("\n".join(lines) + "\n")
print(REPO / entry["script"], OUT / "portable_config.yaml")
