"""Fetch the elemental reference structures each driver would select from Materials Project, as credential-free JSON.

The API key is read from a file at run time and never printed or written. Selection mirrors the drivers:
  H/edge drivers:  summary.search(formula=X), sort by (energy_above_hull, material_id)
  WSe2 drivers:    summary.search(chemsys=X), pure-element docs, sort by (energy_above_hull, formation_energy_per_atom)
Usage: python fetch_mp_elements.py KEY_FILE OUT_JSON
"""
import json
import sys
from pathlib import Path

from mp_api.client import MPRester

key = Path(sys.argv[1]).expanduser().read_text().strip().splitlines()[0].strip()
FIELDS = ["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom", "structure"]
BIG = 1e9


def big(v):
    return BIG if v is None else float(v)


out = {}
with MPRester(key) as mpr:
    for sym in ("Mg", "B", "H", "Fe", "W", "Se"):
        by_formula = sorted(mpr.materials.summary.search(formula=sym, fields=FIELDS),
                            key=lambda d: (big(d.energy_above_hull), str(d.material_id)))
        by_chemsys = [d for d in mpr.materials.summary.search(chemsys=sym, fields=FIELDS)
                      if set(d.structure.composition.get_el_amt_dict()) == {sym}]
        by_chemsys.sort(key=lambda d: (big(d.energy_above_hull), big(d.formation_energy_per_atom)))
        d = by_formula[0]
        s = d.structure
        out[sym] = dict(material_id=str(d.material_id), selected_by_chemsys_rule=str(by_chemsys[0].material_id),
                        energy_above_hull=d.energy_above_hull,
                        candidates=[(str(x.material_id), x.energy_above_hull) for x in by_formula[:5]],
                        symbols=[str(site.specie) for site in s.sites], positions=s.cart_coords.tolist(),
                        cell=s.lattice.matrix.tolist(), natoms=len(s))
        print(sym, out[sym]["material_id"], "chemsys-rule:", out[sym]["selected_by_chemsys_rule"],
              "natoms", len(s), "top candidates", out[sym]["candidates"][:3], flush=True)
Path(sys.argv[2]).write_text(json.dumps(out))
print("WROTE", sys.argv[2])
