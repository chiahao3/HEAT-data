#!/bin/bash
## Job 1: the repository's own unit tests + the documented convergence-plot reproduction.
## Runs against the copy at ~/heat_verify/repo so tracked images are not overwritten.
REPO=~/heat_verify/repo
OUT=~/heat_verify/results/repo_tests
mkdir -p $OUT
fail=0
cd $REPO/scripts/eat_genetic_scripts/genetic_09.30.26_wse2
for t in test_gaussian_09_30 test_ga_mutations test_ga_import_seeding test_ga_generation_extension; do
  echo "===== $t"
  python -m unittest -v $t > $OUT/$t.log 2>&1; rc=$?
  tail -4 $OUT/$t.log; echo "RC $t $rc"; [ $rc -ne 0 ] && fail=1
done
cd $REPO/scripts/eat_genetic_scripts/discrete_09.30.26_wse2
echo "===== test_discrete_wse2"
python -m unittest -v test_discrete_wse2 > $OUT/test_discrete_wse2.log 2>&1; rc=$?
tail -4 $OUT/test_discrete_wse2.log; echo "RC test_discrete_wse2 $rc"; [ $rc -ne 0 ] && fail=1

cd $REPO
for s in images/H_sublatt/plot_h_sublattice_convergence.py images/edge/plot_dislocation_convergence.py images/wse2/plot_wse2_convergence.py; do
  echo "===== $s"
  python $s > $OUT/$(basename $s .py).log 2>&1; rc=$?
  cat $OUT/$(basename $s .py).log; echo "RC $s $rc"; [ $rc -ne 0 ] && fail=1
done
cp $REPO/images/H_sublatt/*.png $REPO/images/edge/*.png $OUT/ 2>/dev/null
cp $REPO/images/wse2/wse2_*convergence*.png $OUT/ 2>/dev/null
exit $fail
