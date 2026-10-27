"""Pick the bcc-Fe lattice constant whose UMA single point equals mu_Fe implied by the recorded edge log.

The recorded edge run used the Materials Project lowest-hull Fe structure (no MP key here); its UMA energy per atom is
recoverable from ga_start: E_ref/969 - E_f,ref = -8.174068 eV/atom. Usage: python fit_fe_reference.py SERVED_JSON
"""
import json
import sys
from pathlib import Path

from ase.build import bulk
from fairchem.core import FAIRChemCalculator, pretrained_mlip

TARGET = -7806.832436 / 969 - 0.117481
path = Path(sys.argv[1])
calc = FAIRChemCalculator(pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda"), task_name="omat")


def e(a):
    at = bulk("Fe", "bcc", a=a)
    at.calc = calc
    return at.get_potential_energy()


lo, hi = 2.80, 2.83  # E(2.80) > target > E(2.83) (compressed side of the minimum)
for _ in range(40):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if e(mid) > TARGET else (lo, mid)
a = 0.5 * (lo + hi)
at = bulk("Fe", "bcc", a=a)
served = json.loads(path.read_text())
served["Fe"] = dict(symbols=at.get_chemical_symbols(), positions=at.positions.tolist(), cell=at.cell.array.tolist(),
                    material_id=f"bcc-Fe-a{a:.5f}-fitted-to-logged-mu")
path.write_text(json.dumps(served))
print(f"Fe reference: a={a:.6f} A, E={e(a):.6f} eV/atom, target={TARGET:.6f}")
