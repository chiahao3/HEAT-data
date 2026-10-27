#!/bin/bash
## EAT patch property checks with the real UMA-s-1p1 weights (incl. stock-fairchem comparison).
OUT=~/heat_verify/results/eat_patch_real; mkdir -p $OUT
S=~/workspace/HEAT-data/verification/scripts/eat_patch_checks.py
PYTHONPATH=~/heat_verify/fairchem_stock/src python $S uma-s-1p1 $OUT/stock_ef.json --stock || exit 1
python $S uma-s-1p1 $OUT/eat_patch_results.json || exit 1
python - <<PY
import json; from pathlib import Path
out = Path("$OUT"); stock = json.loads((out/"stock_ef.json").read_text()); res = json.loads((out/"eat_patch_results.json").read_text())
for k, v in stock.items():
    import numpy as np
    t1 = res[f"T1_{k}"]
    res[f"T2_{k}"] = dict(E_stock=v["energy"], E_patched=t1["E_plain"], dE=abs(v["energy"] - t1["E_plain"]))
    print("T2", k, res[f"T2_{k}"])
(out/"eat_patch_results.json").write_text(json.dumps(res, indent=1))
PY
