"""All the settings live here so I only have one place to look.
On the command line I can change any of them with --set key=value."""
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass
class Config:
    # network: 5 x 100 is a size I took from other PINN examples, I did not tune it
    hidden_layers: int = 5
    hidden_units: int = 100
    # physics
    case: str = "cavity"            # "cavity" or "venturi"
    nu: float = 0.01                # true viscosity (Re = 100)
    inverse: bool = False           # True: nu becomes a trainable parameter
    nu_init: float = 0.02           # starting guess for the inverse problem
    # loss weights: the boundary term gets 10x so the network does not ignore the walls
    # (not tuned yet, experiments/run_tuning.py is for that)
    lambda_data: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_out: float = 1.0         # outlet pressure p = 0 (venturi only)
    # sampling
    n_colloc: int = 10000
    resample_every: int = 500       # new random points every 500 Adam steps so it cannot overfit one set
    n_bc_per_wall: int = 400
    # optimisation: Adam first to get close, then L-BFGS to polish
    adam_iters: int = 20000
    adam_lr: float = 1e-3
    lr_schedule: str = "cosine"     # "cosine", "step" or "none"
    lr_min_factor: float = 0.01
    lr_step_size: int = 5000
    lr_gamma: float = 0.5
    lbfgs_iters: int = 3000
    lbfgs_history: int = 50
    # run control
    seed: int = 0
    device: str = "auto"            # cpu, cuda or auto
    dtype: str = "float32"
    ckpt_every: int = 1000
    log_every: int = 100
    quick: bool = False
    snapshot_every: int = 0         # if > 0, save predictions on a coarse grid (for the videos)

    def for_kind(self, kind):
        # kind is 'pinn', 'data_only' or 'pinn_inverse'.
        # For the data-only baseline I just set the physics weights to 0, so it is the same network and training.
        if kind == "pinn":
            return replace(self, inverse=False)
        if kind == "data_only":
            return replace(self, lambda_pde=0.0, lambda_bc=0.0, lambda_out=0.0, inverse=False)
        if kind == "pinn_inverse":
            return replace(self, inverse=True)
        raise ValueError("unknown model kind: " + kind)

    def as_quick(self):
        # small settings so I can check on my laptop that everything runs
        return replace(self, quick=True, adam_iters=2000, lbfgs_iters=200, n_colloc=2000,
                       resample_every=200, ckpt_every=500, log_every=50)

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def apply_overrides(cfg, items):
    # Turns a list like ['adam_iters=500', 'lambda_bc=5'] into a changed config.
    # I look at the type of each field so the strings from the command line get the right type.
    types = {f.name: (f.type if isinstance(f.type, str) else f.type.__name__) for f in fields(Config)}
    changes = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in types:
            raise KeyError("unknown config key: " + key)
        if types[key] == "bool":
            changes[key] = value.lower() in ("1", "true", "yes")
        elif types[key] == "int":
            changes[key] = int(value)
        elif types[key] == "float":
            changes[key] = float(value)
        else:
            changes[key] = value
    return replace(cfg, **changes)
