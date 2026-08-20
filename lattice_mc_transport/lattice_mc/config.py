from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_config(path):
    cfg = load_json(path)
    required = ["species", "laser", "initial", "geometry", "simulation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    return cfg

def clone_with_overrides(cfg, detuning_GHz=None, power_w=None, n_atoms=None):
    out = deepcopy(cfg)
    if detuning_GHz is not None:
        out["laser"]["d1_red_detuning_GHz"] = float(detuning_GHz)
    if power_w is not None:
        out["geometry"]["l1"]["power_w"] = float(power_w)
        out["geometry"]["l2"]["power_w"] = float(power_w)
    if n_atoms is not None:
        out["initial"]["n_atoms"] = int(n_atoms)
    return out
