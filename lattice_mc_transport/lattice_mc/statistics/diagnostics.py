from __future__ import annotations
from ..constants import KB
from ..backend import to_cpu

def kinetic_temperature(xp, v, alive, mass):
    idx = alive
    n = int(to_cpu(xp.sum(idx)))
    if n < 2:
        return {
            "Tx_K": float("nan"),
            "Ty_K": float("nan"),
            "Tz_K": float("nan"),
            "T_K": float("nan"),
            "n_alive": n,
        }

    vv = v[idx]
    mean = xp.mean(vv, axis=0)
    var = xp.mean((vv - mean[None, :])**2, axis=0)
    Txyz = mass * var / KB
    vals = to_cpu(Txyz)
    return {
        "Tx_K": float(vals[0]),
        "Ty_K": float(vals[1]),
        "Tz_K": float(vals[2]),
        "T_K": float(vals.mean()),
        "n_alive": n,
    }
