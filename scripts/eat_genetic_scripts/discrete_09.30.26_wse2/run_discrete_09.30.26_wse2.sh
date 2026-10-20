#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${DISCRETE_PYTHON:-python3}"
exec "$python_bin" "$script_dir/discrete_09.30.26_wse2.py" --config "$script_dir/discrete_09.30.26_wse2.yaml" "$@"
