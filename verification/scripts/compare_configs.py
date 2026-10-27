"""Compare portable configs/*.json against recorded parameters/*/*/run_parameters.json."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {"heat_h": "eat/H_sublatt", "heat_edge": "eat/edge", "heat_wse2": "eat/wse2",
         "conventional_h": "ordinary/H_sublatt", "conventional_edge": "ordinary/edge",
         "conventional_wse2": "ordinary/wse2"}
PATHLIKE = ("dir", "file", "path", "json", "csv", "tiff", "script", "log")

def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, prefix))
        else:
            out[k] = v
    return out

def flat_cfg(cfg, record):
    """Map each config key to the record key it corresponds to (bare or section-prefixed)."""
    out = {}
    for section, values in cfg.items():
        for k, v in values.items():
            for cand in (k, f"{section}_{k}", f"ga_{k}"):
                if cand in record:
                    out[cand] = v
                    break
            else:
                out[f"{section}.{k}"] = v
    return out

def norm(v):
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return float(s)
        except ValueError:
            return s.replace(" ", "")
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return float(v)
    return v

for cfg, rec in PAIRS.items():
    r = json.loads((ROOT / "parameters" / rec / "run_parameters.json").read_text())
    r = flat(r.get("resolved_parameters") or r.get("config") or r)
    c = flat_cfg(json.loads((ROOT / "configs" / f"{cfg}.json").read_text())["config"], r)
    shared = sorted(set(c) & set(r))
    diffs = [(k, c[k], r[k]) for k in shared
             if norm(c[k]) != norm(r[k]) and not any(p in k for p in PATHLIKE) and k != "mp_api_key"]
    print(f"== {cfg}: {len(c)} config keys, {len(shared)} shared with record, {len(diffs)} value differences")
    for k, a, b in diffs:
        print(f"   {k}: config={a!r} recorded={b!r}")
    missing = sorted(set(c) - set(r))
    if missing:
        print(f"   config-only keys ({len(missing)}): {missing[:15]}")
