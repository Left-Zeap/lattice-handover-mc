from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from ..constants import AMU, C, PI

@dataclass(frozen=True)
class Line:
    wavelength_m: float
    gamma_rad_s: float

    @property
    def omega(self):
        return 2.0 * PI * C / self.wavelength_m

@dataclass(frozen=True)
class Atom:
    name: str
    mass_kg: float
    d1: Line
    d2: Line

def load_atom(species: str) -> Atom:
    path = Path(__file__).resolve().parents[1] / "data" / "atoms.json"
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
    try:
        d = db[species.lower()]
    except KeyError:
        raise KeyError(f"unknown species {species!r}; available: {list(db)}")

    def line(x):
        return Line(
            wavelength_m=x["wavelength_nm"] * 1e-9,
            gamma_rad_s=2.0 * PI * x["linewidth_MHz"] * 1e6,
        )

    return Atom(
        name=species.lower(),
        mass_kg=d["mass_u"] * AMU,
        d1=line(d["D1"]),
        d2=line(d["D2"]),
    )
