#!/bin/bash
## Run the documented launcher (without --prepare-only) for every method/case and record how far it gets.
## CPU job with GPUs hidden and no model access: a launch that reaches model loading is counted as "started".
REPO=~/workspace/HEAT-data; OUT=~/heat_verify/results/launcher; rm -rf $OUT; mkdir -p $OUT
cd $REPO
for m in heat conventional; do for c in h edge wse2; do
  timeout 300 python run_search.py $m $c --output-dir $OUT/$m-$c > $OUT/$m-$c.log 2>&1; rc=$?
  echo "== $m $c rc=$rc"; grep -E "Error|error:|Traceback" $OUT/$m-$c.log | tail -3
done; done
