# openfoam/cavity

Lid-driven cavity, Re = 100, `icoFoam`. `./Allrun` runs the 32, 64 and 128 cell meshes one after the other, checks that each one
is steady (`scripts/openfoam/check_steady.py`) and exports the fields to `data/cavity/`.
`system/meshParams` is a template that `Allrun` fills in for each mesh.
