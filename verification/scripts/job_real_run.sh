#!/bin/bash
## Production-settings search with real UMA weights; only generation count / workers / GPU ids changed.
## Usage: job_real_run.sh METHOD CASE GENERATIONS WORKERS GPU_IDS [--fix-yaml]
METHOD=$1; CASE=$2; GENS=$3; WORKERS=$4; GPUS=$5; FIX=${6:-}
REPO=~/workspace/HEAT-data
OUT=~/heat_verify/results/real_runs/${METHOD}_${CASE}_job${SLURM_JOB_ID}
mkdir -p $OUT
python - <<PY
import json
from pathlib import Path
el = json.loads(Path("$HOME/heat_verify/results/rescore_real/elemental_structures.json").read_text())
ref = json.loads(Path("$REPO/scripts/misc/baseline_ptycho_guided_320/inputs/reference_energies.json").read_text())
for e in ref["elements"]:
    el[e["symbol"]] = dict(e["structure"], material_id=e["material_id"])
Path("$OUT/elemental_structures_served.json").write_text(json.dumps(el))
PY
export HEAT_ELEMENTAL_JSON=$OUT/elemental_structures_served.json MP_API_KEY=stub-no-network
export PYTHONPATH=$REPO/verification/smoke${PYTHONPATH:+:$PYTHONPATH}
read SCRIPT CFG < <(python $REPO/verification/scripts/real_config.py $REPO $METHOD $CASE $OUT $GENS $WORKERS $GPUS $FIX)
cd $REPO
echo "script=$SCRIPT config=$CFG out=$OUT"
start=$(date +%s)
python $SCRIPT --config $CFG > $OUT/driver_stdout.txt 2>&1; rc=$?
echo "DRIVER_RC=$rc elapsed_s=$(( $(date +%s) - start ))"
tail -30 $OUT/driver_stdout.txt
exit $rc
