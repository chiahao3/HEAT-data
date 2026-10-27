#!/bin/bash
## Smoke run of one search driver end-to-end with the fake UMA-architecture model and stubbed MP client.
## Usage: job_smoke.sh METHOD CASE [--fix-yaml]
METHOD=$1; CASE=$2; FIX=${3:-}
REPO=~/workspace/HEAT-data
OUT=~/heat_verify/results/smoke/${METHOD}_${CASE}${FIX:+_fixed}
rm -rf $OUT; mkdir -p $OUT
export HEAT_SMOKE_CKPT=~/heat_verify/results/eat_patch/fake_uma_conserving_mole.pt MP_API_KEY=fake-key-for-stub
export PYTHONPATH=$REPO/verification/smoke${PYTHONPATH:+:$PYTHONPATH}
read SCRIPT CFG < <(python $REPO/verification/scripts/smoke_config.py $REPO $METHOD $CASE $OUT $FIX)
cd $REPO
start=$(date +%s)
timeout 3h python $SCRIPT --config $CFG > $OUT/driver_stdout.txt 2>&1; rc=$?
echo "DRIVER_RC=$rc elapsed_s=$(( $(date +%s) - start ))"
tail -25 $OUT/driver_stdout.txt
echo "--- output files"; find $OUT -maxdepth 2 -type f | sed "s|$OUT/||" | sort | head -40
exit $rc
