"""Stubs loaded by every Python process (incl. spawned GA workers) when this dir is on PYTHONPATH.

HEAT_ELEMENTAL_JSON set: Materials Project queries return the structures in that JSON (no MP key available):
    UMA-relaxed substitutes for Mg, B, H, Fe and the exact bundled MP structures for W, Se.
HEAT_SMOKE_CKPT set: additionally replace UMA by fairchem's fake checkpoint and serve ASE placeholders (not physical).
"""
import json
import os

if os.environ.get("HEAT_ELEMENTAL_JSON") and not os.environ.get("HEAT_SMOKE_CKPT"):
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    import mp_api.client as _mpc

    _ELEMENTS = json.loads(open(os.environ["HEAT_ELEMENTAL_JSON"]).read())

    class _Summary:
        def search(self, formula=None, material_ids=None, fields=None, chemsys=None, **kw):
            formula = formula or chemsys
            e = _ELEMENTS[str(formula)]
            atoms = Atoms(e["symbols"], positions=e["positions"], cell=e["cell"], pbc=True)
            return [{"material_id": e.get("material_id", f"substitute-{formula}"), "formula_pretty": str(formula),
                     "energy_above_hull": 0.0, "formation_energy_per_atom": 0.0,
                     "structure": AseAtomsAdaptor.get_structure(atoms)}]

    class _StubMPRester:
        def __init__(self, *a, **k):
            self.materials = type("M", (), {"summary": _Summary()})()
            self.summary = self.materials.summary  # newer mp_api spelling used by wse2_image_geometry.py
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    _mpc.MPRester = _StubMPRester

if os.environ.get("HEAT_SMOKE_CKPT"):
    import numpy as np
    from ase import Atoms
    from ase.build import bulk, molecule
    from pymatgen.io.ase import AseAtomsAdaptor
    import fairchem.core.pretrained_mlip as _pm
    from fairchem.core.units.mlip_unit import MLIPPredictUnit
    import mp_api.client as _mpc

    def _fake_predict_unit(model_name, *args, device=None, **kwargs):
        dev = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        return MLIPPredictUnit(os.environ["HEAT_SMOKE_CKPT"], device=str(dev))

    _pm.get_predict_unit = _fake_predict_unit

    def _element(symbol):
        if symbol == "H":
            a = molecule("H2")
            a.set_cell([10, 10, 10]); a.center(); a.pbc = True
            return a
        table = {"Mg": ("hcp", 3.21), "Fe": ("bcc", 2.87), "W": ("bcc", 3.17), "B": ("sc", 1.8), "Se": ("sc", 2.6)}
        lattice, a = table[symbol]
        return bulk(symbol, lattice, a=a)

    class _Summary:
        def search(self, formula=None, material_ids=None, fields=None, **kw):
            sym = str(formula)
            return [{"material_id": f"fake-{sym}", "formula_pretty": sym, "energy_above_hull": 0.0,
                     "formation_energy_per_atom": 0.0, "structure": AseAtomsAdaptor.get_structure(_element(sym))}]

    class _FakeMPRester:
        def __init__(self, *a, **k):
            self.materials = type("M", (), {"summary": _Summary()})()
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    _mpc.MPRester = _FakeMPRester
