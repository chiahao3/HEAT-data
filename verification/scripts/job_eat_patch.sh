#!/bin/bash
## Job 2: build fairchem's fake UMA-architecture checkpoint, then EAT patch property checks (incl. stock comparison).
set -u
OUT=~/heat_verify/results/eat_patch; mkdir -p $OUT
CK=$OUT/fake_uma_conserving_mole.pt
if [ ! -f $CK ]; then
  python ~/workspace/HEAT-data/verification/scripts/make_fake_uma_checkpoint.py ~/heat_verify/fairchem $OUT > $OUT/make_ckpt.log 2>&1 || { tail -40 $OUT/make_ckpt.log; exit 1; }
  tail -2 $OUT/make_ckpt.log
fi
# stock (unpatched) fairchem source at the same revision, put first on the path
[ -d ~/heat_verify/fairchem_stock ] || git -C ~/heat_verify/fairchem worktree add -q ~/heat_verify/fairchem_stock cce3699dc64d01c0a3bb5c6d5609caab96b985fa
PYTHONPATH=~/heat_verify/fairchem_stock/src python -c "import fairchem.core,sys; print('stock fairchem from', fairchem.core.__file__)"
PYTHONPATH=~/heat_verify/fairchem_stock/src python ~/workspace/HEAT-data/verification/scripts/eat_patch_checks.py $CK $OUT/stock_ef.json --stock || exit 1
python ~/workspace/HEAT-data/verification/scripts/eat_patch_checks.py $CK $OUT/eat_patch_results.json || exit 1
python - <<PY
import json, numpy as np
from pathlib import Path
out = Path("$OUT")
stock = json.loads((out/"stock_ef.json").read_text()); res = json.loads((out/"eat_patch_results.json").read_text())
for k, v in stock.items():
    t1 = res[f"T1_{k}"]
    print("T2", k, "stock E", v["energy"], "patched-plain E", t1["E_plain"], "dE", abs(v["energy"] - t1["E_plain"]))
    res[f"T2_{k}"] = dict(E_stock=v["energy"], E_patched=t1["E_plain"], dE=abs(v["energy"] - t1["E_plain"]))
(out/"eat_patch_results.json").write_text(json.dumps(res, indent=1))
PY
