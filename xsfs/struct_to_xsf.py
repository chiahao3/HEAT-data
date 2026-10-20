#!/usr/bin/env python3
"""Write the lowest-formation-energy member of every GA generation as XSF.

The converter supports both self-contained EAT structure records and the
``ordinary_hard_atoms`` database written by ``ordinary_fixed_scaffold_ga.py``.
Ordinary-GA records contain only the variable atoms, so this script combines
them with the run's fixed-scaffold XSF before writing each complete structure.

The only required command-line argument is the input ``structures.json``.
Records with negative generations (for example imported parent records) are
not GA generations and are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


CELL_KEYS = (
    "cell_angstrom",
    "lattice_vectors_angstrom",
    "cell_vectors_angstrom",
    "cell",
)
SYMBOL_KEYS = ("symbols", "all_symbols", "chemical_symbols", "atom_symbols")
POSITION_KEYS = (
    "cart_positions_angstrom",
    "positions_angstrom",
    "all_positions_angstrom",
    "all_cart_positions_angstrom",
    "cartesian_positions_angstrom",
    "positions",
)
FIXED_SYMBOL_KEYS = ("fixed_symbols", "scaffold_symbols")
FIXED_POSITION_KEYS = (
    "fixed_positions_angstrom",
    "scaffold_positions_angstrom",
    "fixed_cart_positions_angstrom",
)
AUTO_SCAFFOLD_FILENAMES = (
    "fixed_heavy_atom_scaffold.xsf",  # H-sublattice runs
    "fixed_scaffold.xsf",             # dislocation runs
    "fixed_Mg_Fe_scaffold.xsf",       # legacy H-sublattice name
)


@dataclass(frozen=True)
class StructureContext:
    representation: str
    cell: list[list[float]] | None
    fixed_symbols: list[str]
    fixed_positions: list[list[float]]
    scaffold_source: Path | None

    @property
    def is_ordinary_hard_atoms(self):
        return self.representation.strip().lower() in {
            "ordinary_hard_atoms",
            "ordinary_fully_occupied_atoms",
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structures_json",
        help="Path to the structures.json produced by a genetic EAT run.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="",
        help=(
            "Output folder. Defaults to "
            "<structures_json_dir>/lowest_formation_energy_by_generation."
        ),
    )
    parser.add_argument(
        "--scaffold-xsf",
        default="",
        help=(
            "Fixed-scaffold XSF for ordinary_hard_atoms records. By default, "
            "the script searches beside structures.json for "
            "fixed_heavy_atom_scaffold.xsf or fixed_scaffold.xsf."
        ),
    )
    return parser.parse_args()


def load_structure_payload(path):
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get("structures", [])
        metadata = payload
    elif isinstance(payload, list):
        records = payload
        metadata = {}
    else:
        raise ValueError(f"Expected JSON object or list in {path}")
    if not isinstance(records, list):
        raise ValueError("structures field must be a list.")
    return metadata, records


def clean_float_for_name(value):
    text = f"{float(value):+.8f}"
    return text.replace("+", "p").replace("-", "m").replace(".", "p")


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")


def first_present(mapping, keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def first_present_any(mappings, keys):
    for mapping in mappings:
        value = first_present(mapping, keys)
        if value is not None:
            return value
    return None


def validate_structure_arrays(symbols, positions, cell):
    symbols = [str(symbol) for symbol in (symbols or [])]
    positions = list(positions or [])
    cell = list(cell or [])
    if len(symbols) != len(positions):
        raise ValueError(
            f"symbols and positions lengths do not match: "
            f"{len(symbols)} symbols, {len(positions)} positions."
        )
    if any(len(position) < 3 for position in positions):
        raise ValueError("Every Cartesian position must contain three coordinates.")
    if len(cell) != 3 or any(len(row) < 3 for row in cell):
        raise ValueError("cell_angstrom must be a 3x3 matrix.")
    positions = [[float(value) for value in position[:3]] for position in positions]
    cell = [[float(value) for value in row[:3]] for row in cell]
    return symbols, positions, cell


def read_xsf_structure(path):
    """Read the first CRYSTAL/PRIMCOORD structure from a plain XSF file."""
    path = Path(path).expanduser().resolve()
    data_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    upper_lines = [line.upper() for line in data_lines]
    try:
        vector_index = upper_lines.index("PRIMVEC")
        coordinate_index = upper_lines.index("PRIMCOORD")
    except ValueError as exc:
        raise ValueError(
            f"{path} must contain PRIMVEC and PRIMCOORD sections."
        ) from exc

    try:
        cell = [
            [float(value) for value in data_lines[vector_index + offset].split()[:3]]
            for offset in (1, 2, 3)
        ]
        atom_count = int(data_lines[coordinate_index + 1].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed PRIMVEC or PRIMCOORD header in {path}") from exc

    symbols = []
    positions = []
    atom_lines = data_lines[coordinate_index + 2 : coordinate_index + 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(
            f"Expected {atom_count} atom rows in {path}, found {len(atom_lines)}."
        )
    for line_number, line in enumerate(atom_lines, start=1):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed atom row {line_number} in {path}: {line!r}")
        symbols.append(fields[0])
        try:
            positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise ValueError(
                f"Invalid coordinates on atom row {line_number} in {path}"
            ) from exc
    return validate_structure_arrays(symbols, positions, cell)


def find_scaffold_xsf(structures_path, explicit_path):
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scaffold XSF not found: {path}")
        return path
    for filename in AUTO_SCAFFOLD_FILENAMES:
        candidate = structures_path.parent / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def build_structure_context(metadata, structures_path, scaffold_xsf=""):
    representation = str(metadata.get("representation", ""))
    cell = first_present(metadata, CELL_KEYS)
    fixed_symbols = first_present(metadata, FIXED_SYMBOL_KEYS)
    fixed_positions = first_present(metadata, FIXED_POSITION_KEYS)
    scaffold_source = find_scaffold_xsf(structures_path, scaffold_xsf)

    if scaffold_source is not None:
        xsf_symbols, xsf_positions, xsf_cell = read_xsf_structure(scaffold_source)
        if cell is None:
            cell = xsf_cell
        if fixed_symbols is None:
            fixed_symbols = xsf_symbols
        if fixed_positions is None:
            fixed_positions = xsf_positions

    fixed_symbols = list(fixed_symbols or [])
    fixed_positions = list(fixed_positions or [])
    if len(fixed_symbols) != len(fixed_positions):
        raise ValueError(
            "Top-level/scaffold fixed symbols and positions have different lengths: "
            f"{len(fixed_symbols)} and {len(fixed_positions)}."
        )
    if cell is not None:
        _, _, cell = validate_structure_arrays([], [], cell)
    return StructureContext(
        representation=representation,
        cell=cell,
        fixed_symbols=[str(symbol) for symbol in fixed_symbols],
        fixed_positions=[
            [float(value) for value in position[:3]] for position in fixed_positions
        ],
        scaffold_source=scaffold_source,
    )


def get_structure(record, context):
    structures = [
        record.get("structure"),
        record.get("rounded_structure"),
        record.get("final_rounded_structure"),
        record,
    ]
    structures = [item for item in structures if isinstance(item, dict)]

    for structure in structures:
        symbols = first_present(structure, SYMBOL_KEYS)
        positions = first_present(structure, POSITION_KEYS)
        cell = first_present_any([structure, record], CELL_KEYS)
        if cell is None:
            cell = context.cell
        if symbols is None or positions is None or cell is None:
            continue

        variable_only = (
            context.is_ordinary_hard_atoms
            and len(symbols) == int(record.get("n_sites", len(symbols)))
        )
        if variable_only:
            if not context.fixed_symbols:
                raise ValueError(
                    "ordinary_hard_atoms records contain only variable atoms, but no "
                    "fixed scaffold was found. Put fixed_heavy_atom_scaffold.xsf "
                    "(H sublattice) or fixed_scaffold.xsf (dislocation) beside "
                    "structures.json, or pass --scaffold-xsf PATH."
                )
            symbols = list(context.fixed_symbols) + list(symbols)
            positions = list(context.fixed_positions) + list(positions)
        return validate_structure_arrays(symbols, positions, cell)

    for structure in structures:
        fixed_symbols = first_present_any(
            [structure, record], FIXED_SYMBOL_KEYS
        )
        fixed_positions = first_present_any(
            [structure, record], FIXED_POSITION_KEYS
        )
        null_symbols = first_present_any(
            [structure, record],
            ("null_symbols", "ghost_symbols", "mobile_symbols", "realized_symbols"),
        )
        null_positions = first_present_any(
            [structure, record],
            (
                "null_positions_angstrom",
                "ghost_positions_angstrom",
                "mobile_positions_angstrom",
                "realized_positions_angstrom",
            ),
        )
        cell = first_present_any([structure, record], CELL_KEYS)
        if cell is None:
            cell = context.cell
        if (
            fixed_symbols is not None
            and fixed_positions is not None
            and null_symbols is not None
            and null_positions is not None
            and cell is not None
        ):
            symbols = list(fixed_symbols) + list(null_symbols)
            positions = list(fixed_positions) + list(null_positions)
            return validate_structure_arrays(symbols, positions, cell)

    available_keys = sorted({key for structure in structures for key in structure.keys()})
    raise ValueError(
        "Record structure is missing recognizable symbols/positions/cell fields. "
        f"Available keys include: {available_keys[:40]}"
    )


def composition_value(record):
    value = first_present(
        record,
        ("composition_penalty", "composition_term", "composition_objective_term"),
    )
    return 0.0 if value is None else float(value)


def formation_energy_value(record):
    """Return the per-atom formation energy used by the 08.27 GA writers."""
    value = first_present(
        record,
        (
            "formation_energy_per_atom",
            "formation_energy_per_atom_eV",
            "eform_per_atom",
        ),
    )
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def finite_float_or_default(value, default=float("inf")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def select_lowest_formation_energy_by_generation(records):
    """Choose one deterministic minimum-energy record for each GA generation."""
    selected = {}
    skipped = {
        "not_a_mapping": 0,
        "missing_or_invalid_generation": 0,
        "negative_generation": 0,
        "missing_or_invalid_formation_energy": 0,
    }
    for original_index, record in enumerate(records):
        if not isinstance(record, dict):
            skipped["not_a_mapping"] += 1
            continue
        try:
            generation = int(record["generation"])
        except (KeyError, TypeError, ValueError, OverflowError):
            skipped["missing_or_invalid_generation"] += 1
            continue
        if generation < 0:
            skipped["negative_generation"] += 1
            continue
        formation_energy = formation_energy_value(record)
        if formation_energy is None:
            skipped["missing_or_invalid_formation_energy"] += 1
            continue
        # Formation energy is authoritative. Score, member index, and input
        # order only make exact-energy ties reproducible.
        tie_key = (
            formation_energy,
            finite_float_or_default(record.get("score")),
            int(record.get("member_index", -1)),
            original_index,
        )
        incumbent = selected.get(generation)
        if incumbent is None or tie_key < incumbent[0]:
            selected[generation] = (
                tie_key,
                original_index,
                record,
                formation_energy,
            )
    return selected, skipped


def write_xsf(path, record, formation_energy, context):
    symbols, positions, cell = get_structure(record, context)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# Generated from structures.json\n")
        handle.write(
            "# selection = lowest_formation_energy_per_atom_in_generation\n"
        )
        handle.write(f"# generation = {int(record.get('generation', -1))}\n")
        handle.write(
            "# formation_energy_per_atom = "
            f"{float(formation_energy):.12f}\n"
        )
        score = finite_float_or_default(record.get("score"), default=float("nan"))
        handle.write(f"# score = {score:.12f}\n")
        handle.write(f"# composition_penalty = {composition_value(record):.12f}\n")
        handle.write(f"# member_index = {int(record.get('member_index', -1))}\n")
        handle.write(f"# kind = {record.get('kind', '')}\n")
        handle.write("CRYSTAL\n")
        handle.write("PRIMVEC\n")
        for row in cell:
            handle.write(
                f"  {float(row[0]): .12f}  {float(row[1]): .12f}  "
                f"{float(row[2]): .12f}\n"
            )
        handle.write("PRIMCOORD\n")
        handle.write(f"  {len(symbols)} 1\n")
        for symbol, position in zip(symbols, positions):
            handle.write(
                f"  {str(symbol):2s}  {float(position[0]): .12f}  "
                f"{float(position[1]): .12f}  {float(position[2]): .12f}\n"
            )


def main():
    args = parse_args()
    structures_path = Path(args.structures_json).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else structures_path.parent / "lowest_formation_energy_by_generation"
    )
    metadata, records = load_structure_payload(structures_path)
    context = build_structure_context(metadata, structures_path, args.scaffold_xsf)
    selected, skipped = select_lowest_formation_energy_by_generation(records)
    if not selected:
        raise ValueError(
            "No nonnegative-generation records with a finite "
            "formation_energy_per_atom were found."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "generation",
            "filename",
            "original_index",
            "formation_energy_per_atom",
            "score",
            "composition_penalty",
            "member_index",
            "kind",
            "parent_refs",
        ])
        for generation in sorted(selected):
            _, original_index, record, formation_energy = selected[generation]
            score = finite_float_or_default(record.get("score"), default=float("nan"))
            member_index = int(record.get("member_index", -1))
            kind = safe_name(record.get("kind", "structure"))
            filename = (
                f"generation_{generation:03d}_"
                f"eform_{clean_float_for_name(formation_energy)}_"
                f"member_{member_index:03d}_{kind}.xsf"
            )
            write_xsf(
                output_dir / filename,
                record,
                formation_energy,
                context,
            )
            writer.writerow([
                generation,
                filename,
                original_index,
                formation_energy,
                score,
                composition_value(record),
                member_index,
                record.get("kind", ""),
                ";".join(str(item) for item in record.get("parent_refs", [])),
            ])
    print(
        f"Selected {len(selected)} lowest-formation-energy members from "
        f"{len(records)} records."
    )
    print(f"Wrote {len(selected)} XSF files to {output_dir}")
    print(f"Wrote index: {index_path}")
    skipped_total = sum(skipped.values())
    if skipped_total:
        details = ", ".join(
            f"{key}={value}" for key, value in skipped.items() if value
        )
        print(f"Skipped {skipped_total} non-generation/invalid records ({details}).")
    if context.scaffold_source is not None:
        print(f"Used fixed scaffold: {context.scaffold_source}")


if __name__ == "__main__":
    main()
