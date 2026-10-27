"""Load each launcher-written portable_config.yaml through its driver's own config loader (no model, no search).

Checks that every value in configs/METHOD_CASE.json arrives in the driver with the right value and numeric type.
Usage: python check_config_loading.py REPO WORK_DIR OUT_JSON
"""
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO, WORK, OUT = (Path(p).resolve() for p in sys.argv[1:4])
CASES = [(m, c) for m in ("heat", "conventional") for c in ("h", "edge", "wse2")]


def load_module(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    if isinstance(b, bool) or isinstance(a, bool):
        return str(a).lower() == str(b).lower()
    if isinstance(b, (int, float)):
        return isinstance(a, (int, float)) and abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
    return str(a).strip() == str(b).strip() or b in ("", None) and a in ("", None)


results = {}
for method, case in CASES:
    key = f"{method}_{case}"
    out_dir = WORK / key
    r = subprocess.run([sys.executable, str(REPO / "run_search.py"), method, case, "--prepare-only", "--output-dir", str(out_dir)],
                       capture_output=True, text=True)
    entry = json.loads((REPO / "configs" / f"{key}.json").read_text())
    yaml_path = out_dir / "portable_config.yaml"
    row = dict(prepare_rc=r.returncode, script=entry["script"])
    try:
        script = REPO / entry["script"]
        if script.parent.name == "ordinary_genetic_scripts":  # the entry scripts are 9-line wrappers
            script = script.parent / "ordinary_fixed_scaffold_ga.py"
        mod = load_module(script, f"drv_{key}")
        if script.name.startswith("discrete"):
            cfg = mod.load_config(yaml_path)
            lookup = lambda s, k: cfg[s][k]  # noqa: E731
        elif script.parent.name == "ordinary_genetic_scripts":
            mode = "h_sublattice" if case == "h" else "dislocation"
            src, args, raw = mod.load_settings(mode, yaml_path)
            maps = [src.CONFIG_KEY_MAP, mod.HARD_KEY_MAP]
            lookup = lambda s, k: getattr(args, next(m[s][k] for m in maps if s in m and k in m[s]))  # noqa: E731
        else:
            args = mod.get_settings(yaml_path)
            lookup = lambda s, k: getattr(args, mod.CONFIG_KEY_MAP[s][k])  # noqa: E731
        mism, unmapped, n = [], [], 0
        for s, vals in entry["config"].items():
            for k, v in vals.items():
                try:
                    got = lookup(s, k)
                except (KeyError, StopIteration, AttributeError):
                    unmapped.append(f"{s}.{k}")
                    continue
                n += 1
                if [s, k] in entry["repository_paths"] or k == "output_dir":
                    continue
                if not same(got, v):
                    mism.append(dict(key=f"{s}.{k}", json=v, loaded=got, loaded_type=type(got).__name__))
        row.update(status="loaded", checked=n, mismatches=mism, unmapped=unmapped)
    except Exception as exc:  # report, do not stop
        row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
    results[key] = row
    print(key, {k: v for k, v in row.items() if k != "traceback"})

OUT.write_text(json.dumps(results, indent=1, default=str))
print("WROTE", OUT)
