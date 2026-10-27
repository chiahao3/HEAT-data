#!/bin/bash
## Usage: submit.sh NAME [cpu|2g|a100] -- command args...
## Refuses to submit when 4 or more heat-* jobs (running or pending) already exist for this user.
set -euo pipefail
NAME=$1; KIND=$2; shift 2; [ "$1" = "--" ] && shift
MAX=4
n=$(squeue -u "$USER" -h -o "%j" | grep -c "^heat-" || true)
if [ "$n" -ge "$MAX" ]; then echo "REFUSED: $n heat-* jobs already queued/running (max $MAX)"; exit 2; fi
case "$KIND" in
  cpu)  RES=(--cpus-per-task=8 --mem=32G) ;;
  2g)   RES=(--gres=gpu:2g.20gb:1 --cpus-per-task=8 --mem=32G) ;;
  a100) RES=(--gres=gpu:a100:1 --cpus-per-task=16 --mem=64G) ;;
  *) echo "kind must be cpu, 2g or a100"; exit 1 ;;
esac
mkdir -p ~/heat_verify/logs
sbatch --parsable --job-name="heat-$NAME" "${RES[@]}" ${SBATCH_EXTRA:-} "$(dirname "$0")/heat_job.sub" "$@"
