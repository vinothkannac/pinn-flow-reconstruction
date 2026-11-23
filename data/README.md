# data

| path | what it is |
|---|---|
| `cavity/openfoam_N{32,64,128}.npz` | OpenFOAM cavity fields at the cell centres (x, y, u, v, p). Written by `openfoam/cavity/Allrun`. N = 128 is the reference. |
| `venturi/openfoam_k{1,2,4}.npz` | OpenFOAM Venturi fields at the cell centres. k is the refinement level (4,800 / 19,200 / 76,800 cells). Level 4 is the reference. Written by `openfoam/venturi/Allrun`. |
| `<case>/sensors/N{N}_noise{pct}_seed{s}.npz` | Synthetic sensors: random cell centres with noise, only u and v. Written by `scripts/preprocessing/make_sensors.py`. |
| `cavity/ghia_re100.csv` | Centerline values from Ghia, Ghia and Shin (1982), Re = 100. Typed in by hand from the paper. |

## Format of `ghia_re100.csv`

```
profile,coordinate,value
u,<y>,<u>      u along x = 0.5   (Table I, Re = 100 column)
v,<x>,<v>      v along y = 0.5   (Table II, Re = 100 column)
```
