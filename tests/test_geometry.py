import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn.config import Config
from pinn.data_utils import make_sensors
from pinn.geometry import get_geometry
from pinn.train import train


@pytest.mark.parametrize("case", ["cavity", "venturi"])
def test_interior_samples_are_inside_and_deterministic(case):
    g = get_geometry(case)
    a = g.sample_interior(500, seed=3)
    b = g.sample_interior(500, seed=3)
    assert torch.equal(a, b) and a.shape == (500, 2) and g.inside(a.numpy()).all()
    assert g.inside(g.snapshot_points()).all()


def test_venturi_boundary_targets():
    g = get_geometry("venturi")
    b = g.boundary(100)
    xy, uv = b["vel"]
    walls = xy[:600]
    assert torch.allclose(walls[:, 1].abs(), torch.as_tensor(g.half_height(walls[:, 0].numpy())))   # on the wall
    assert (uv[:600] == 0).all()
    inlet_u = uv[600:, 0]
    assert inlet_u.mean().item() == pytest.approx(1.0, abs=1e-3)       # mean inlet velocity = 1
    assert torch.allclose(b["out"][:, 0], torch.full((100,), 12.0, dtype=torch.float64))


def test_cavity_boundary_excludes_lid_corners():
    xy, uv = get_geometry("cavity").boundary(50)["vel"]
    lid = xy[:50]
    assert lid[:, 0].min() > 0 and lid[:, 0].max() < 1 and (uv[:50, 0] == 1).all()


@pytest.mark.parametrize("case", ["cavity", "venturi"])
# a very small run, only to see that the training loop works and the loss goes down (not a real result)
def test_tiny_training_reduces_loss(case):
    g = get_geometry(case)
    pts = g.sample_interior(40, seed=1).numpy()
    ref = {"x": pts[:, 0], "y": pts[:, 1], "u": np.sin(pts[:, 0]), "v": np.cos(pts[:, 1]), "p": pts[:, 0]}
    sens = make_sensors(ref, 20, 1, seed=0)
    cfg = Config(case=case, hidden_layers=2, hidden_units=16, n_colloc=200, n_bc_per_wall=20, adam_iters=60,
                 lbfgs_iters=0, log_every=10, device="cpu", ckpt_every=1000, resample_every=30)
    res = train(cfg, sens, verbose=False)
    h = res["history"]["total"]
    assert res["status"] == "ok" and h[-1] < h[0]


def test_command_line_overrides_use_the_right_types():
    from pinn.config import apply_overrides
    cfg = apply_overrides(Config(), ["adam_iters=60", "lambda_bc=2.5", "case=venturi", "inverse=true"])
    assert cfg.adam_iters == 60 and cfg.lambda_bc == 2.5 and cfg.case == "venturi" and cfg.inverse is True
