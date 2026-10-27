"""Write a production portable_config.yaml (as run_search.py does) with only run-length/hardware overrides.

Usage: python real_config.py REPO METHOD CASE OUT_DIR GENERATIONS WORKERS GPU_IDS [--fix-yaml]
"""
import json
import os
import sys
from pathlib import Path

REPO, METHOD, CASE, OUT = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], Path(sys.argv[4]).resolve()
GENS, WORKERS, GPUS = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
entry = json.loads((REPO / "configs" / f"{METHOD}_{CASE}.json").read_text())
cfg = entry["config"]
for section, key in entry["repository_paths"]:
    cfg[section][key] = str((REPO / cfg[section][key]).resolve())
cfg["paths"]["output_dir"] = str(OUT / "run")  # several drivers refuse an existing output directory
g = cfg["genetic"]
g["generations"] = GENS
if os.environ.get("HEAT_POPULATION"):  # reduced population for a functional check (recorded as such in the report)
    g.update(population_size=int(os.environ["HEAT_POPULATION"]), parent_pool_size=int(os.environ["HEAT_POPULATION"]))
if "parallel_workers" in g:
    g.update(parallel_workers=WORKERS, parallel_gpu_ids=GPUS)
if "runtime" in cfg and "gpu_ids" in cfg["runtime"]:
    cfg["runtime"].update(gpu_ids=GPUS, parallel_workers=WORKERS)
OUT.mkdir(parents=True, exist_ok=True)
lines = ["# production config; only generations / workers / GPU ids overridden for verification"]
for section, values in cfg.items():
    lines.append(section + ":")
    for key, value in values.items():
        text = json.dumps(value)
        if key == "transform_epsilon" and "--fix-yaml" in sys.argv:
            text = "1.0e-06"  # PyYAML reads 1e-06 as a string (see report)
        lines.append(f"  {key}: {text}")
(OUT / "portable_config.yaml").write_text("\n".join(lines) + "\n")
print(REPO / entry["script"], OUT / "portable_config.yaml")
