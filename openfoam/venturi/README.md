# openfoam/venturi

Venturi passage, Re = 100, `simpleFoam`. `./Allrun` writes the `blockMeshDict` with `scripts/openfoam/generate_venturi_mesh.py`,
runs the three meshes, exports the fields to `data/venturi/` and runs `scripts/verification/verify_venturi.py`.
