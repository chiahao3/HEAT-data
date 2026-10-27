"""UMA single-point energy per atom of each fetched MP elemental structure (the drivers' mu) vs recorded values."""
import json
import sys

from ase import Atoms
from fairchem.core import FAIRChemCalculator, pretrained_mlip

calc = FAIRChemCalculator(pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda"), task_name="omat")
recorded = {"W": -12.952261009011435, "Se": -3.5006994563118683, "Fe": -7806.832436 / 969 - 0.117481}
res = {}
for sym, e in json.loads(open(sys.argv[1]).read()).items():
    a = Atoms(e["symbols"], positions=e["positions"], cell=e["cell"], pbc=True)
    a.calc = calc
    mu = a.get_potential_energy() / len(a)
    res[sym] = dict(material_id=e["material_id"], mu=mu, recorded=recorded.get(sym))
    print(sym, res[sym], "diff" if sym in recorded else "", abs(mu - recorded[sym]) if sym in recorded else "")
json.dump(res, open(sys.argv[2], "w"), indent=1)
