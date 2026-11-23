"""Tests for my OpenFOAM file reader, the centerlines, the Ghia loader and the sensor files.
The numbers in here are made up for the tests. They are not benchmark data."""
import numpy as np
import pytest

from pinn.data_utils import centerline_u, centerline_v, grid_view, load_ghia, make_sensors, sensor_path
from pinn.foam_io import parse_internal_field

SCALAR = """FoamFile {}
dimensions [0 2 -2 0 0 0 0];
internalField   nonuniform List<scalar>
4
(
1.5
-2e-3
3
4
)
;
boundaryField { movingWall { type zeroGradient; } }
"""
VECTOR = """internalField   nonuniform List<vector> 2
(
(1 2 0)
(-3.5e-2 4 0)
)
;
boundaryField { a { type fixedValue; value uniform (9 9 9); } }
"""


def synthetic_ref(N=64):
    g = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(g, g)
    x, y = X.ravel(), Y.ravel()
    perm = np.random.default_rng(1).permutation(N * N)  # I shuffle the cells on purpose, OpenFOAM does not promise an order
    x, y = x[perm], y[perm]
    return {"x": x, "y": y, "u": y.copy(), "v": x * (1 - x), "p": x + y}


def test_parse_scalar_and_vector():
    assert np.allclose(parse_internal_field(SCALAR), [1.5, -2e-3, 3, 4])
    assert np.allclose(parse_internal_field(VECTOR), [[1, 2, 0], [-3.5e-2, 4, 0]])


def test_parse_uniform():
    assert np.allclose(parse_internal_field("internalField uniform 0;", 3), 0)
    assert parse_internal_field("internalField uniform (1 0 0);", 3).shape == (3, 3)


def test_grid_view_handles_scrambled_order():
    ref = synthetic_ref(16)
    xs, ys, g = grid_view(ref)
    X, Y = np.meshgrid(xs, ys)
    assert np.allclose(g["u"], Y) and np.allclose(g["v"], X * (1 - X))


def test_centerlines_of_known_fields():
    ref = synthetic_ref(64)
    yq = np.linspace(0, 1, 37)
    assert np.allclose(centerline_u(ref, yq, lid_u=1.0), yq, atol=1e-12)  # u = y, wall values consistent
    xq = np.linspace(0, 1, 37)
    assert np.allclose(centerline_v(ref, xq), xq * (1 - xq), atol=1e-3)


def test_missing_ghia_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ghia"):
        load_ghia(tmp_path / "nope.csv")


def test_ghia_loader_parses_format(tmp_path):
    f = tmp_path / "g.csv"  # synthetic test values only
    f.write_text("profile,coordinate,value\nu,0.5,0.1\nu,0.25,0.05\nv,0.5,-0.2\nv,0.75,0.3\n")
    g = load_ghia(f)
    assert np.allclose(g["u"][0], [0.25, 0.5]) and np.allclose(g["u"][1], [0.05, 0.1])
    assert np.allclose(g["v"][0], [0.5, 0.75])


def test_sensors_deterministic_noise_free_and_no_pressure():
    ref = synthetic_ref(32)
    a, b = make_sensors(ref, 50, 1, seed=3), make_sensors(ref, 50, 1, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    z = make_sensors(ref, 50, 0, seed=3)
    assert np.array_equal(z["u"], ref["u"][z["cell_index"]])
    assert len(np.unique(z["cell_index"])) == 50
    assert not any(k.startswith("p") for k in a)
    assert sensor_path("venturi", 50, 1, 3).name == "N50_noise1_seed3.npz"
    assert sensor_path("venturi", 50, 1, 3).parent.name == "sensors"


# the std of the noise should be noise_pct percent of the RMS, that is how I defined it in make_sensors
def test_noise_std_matches_definition():
    ref = synthetic_ref(64)
    s = make_sensors(ref, 1000, 5, seed=0)
    rms = np.sqrt(np.mean(ref["u"] ** 2))
    assert s["noise_std_u"] == pytest.approx(0.05 * rms)
    assert np.std(s["u"] - s["u_clean"]) == pytest.approx(0.05 * rms, rel=0.15)
