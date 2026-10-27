"""Train fairchem's own 2-step fake conserving MoLE (UMA-architecture) checkpoint, as in its test suite.

The weights are essentially random. This exercises the patched code path only; energies are not physical.
Usage: python make_fake_uma_checkpoint.py FAIRCHEM_ROOT OUT_DIR
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

fairchem_root, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
os.chdir(fairchem_root)
sys.path.insert(0, str(fairchem_root))  # hydra targets such as tests.core...GradSaveOptimizer
sys.path.insert(0, str(fairchem_root / "tests" / "core" / "units" / "mlip_unit"))
from create_fake_dataset import create_fake_uma_dataset  # noqa: E402
from fairchem.core._cli import main as launch_main  # noqa: E402
from fairchem.core.units.mlip_unit.mlip_unit import UNIT_INFERENCE_CHECKPOINT  # noqa: E402

data_dir = tempfile.mkdtemp()
run_dir = tempfile.mkdtemp()
create_fake_uma_dataset(data_dir)
sys.argv[1:] = [
    "--config", "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml",
    "num_experts=8",
    "heads.energyandforcehead.module=fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
    "checkpoint_every=10000", "datasets=aselmdb_conserving",
    f"+job.run_dir={run_dir}", f"datasets.data_root_dir={data_dir}",
    "job.device_type=CPU", "+job.timestamp_id=12345", "optimizer=savegrad",
    "max_steps=2", "max_epochs=null", "expected_loss=null", "act_type=gate", "ff_type=spectral",
]
launch_main()  # fairchem's CLI reads sys.argv, as tests/core/testing_utils.launch_main does
ckpt = Path(run_dir) / "12345" / "checkpoints" / "step_0" / UNIT_INFERENCE_CHECKPOINT
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(ckpt, out_dir / "fake_uma_conserving_mole.pt")
print("CHECKPOINT", out_dir / "fake_uma_conserving_mole.pt")
