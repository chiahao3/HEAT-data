#!/bin/bash
## Production-settings search with real UMA weights; only generation count / workers / GPU ids changed.
## Usage: job_real_run.sh METHOD CASE GENERATIONS WORKERS GPU_IDS [--fix-yaml]
METHOD=$1; CASE=$2; GENS=$3; WORKERS=$4; GPUS=$5; FIX=${6:-}
REPO=~/workspace/HEAT-data
OUT=~/heat_verify/results/real_runs/${METHOD}_${CASE}_job${SLURM_JOB_ID}
mkdir -p $OUT
# Exact Materials Project elemental structures (fetched once with the user's key; JSON holds no credentials)
cp ~/heat_verify/results/mp_elements/mp_elemental_structures.json $OUT/elemental_structures_served.json
export HEAT_ELEMENTAL_JSON=$OUT/elemental_structures_served.json MP_API_KEY=stub-no-network
export PYTHONPATH=$REPO/verification/smoke${PYTHONPATH:+:$PYTHONPATH}
read SCRIPT CFG < <(python $REPO/verification/scripts/real_config.py $REPO $METHOD $CASE $OUT $GENS $WORKERS $GPUS $FIX)
cd $REPO
echo "script=$SCRIPT config=$CFG out=$OUT"
start=$(date +%s)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python $SCRIPT --config $CFG > $OUT/driver_stdout.txt 2>&1 &
drv=$!
## Watchdog: the drivers hang if a worker fails (see report). If no log line for WATCHDOG_MIN minutes,
## kill the worker processes so the main process can surface the worker's traceback and exit.
( while kill -0 $drv 2>/dev/null; do
    sleep 300
    log=$(find $OUT -name running_log.txt | head -1)
    if [ -n "$log" ] && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt $(( ${WATCHDOG_MIN:-60} * 60 )) ]; then
      echo "WATCHDOG: no progress for ${WATCHDOG_MIN:-60} min; killing workers of $drv"; pkill -TERM -P $drv; sleep 120; kill -TERM $drv
    fi
  done ) &
wait $drv; rc=$?
echo "DRIVER_RC=$rc elapsed_s=$(( $(date +%s) - start ))"
tail -30 $OUT/driver_stdout.txt
exit $rc
