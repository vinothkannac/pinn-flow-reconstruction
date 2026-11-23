"""Writes the blockMeshDict of the Venturi passage, so I do not have to type the vertices by hand.

At x = 0, 2, 4, 5, 9, 12 the full heights are 1, 1, 0.5, 0.5, 1, 1 (inlet height = 1):
straight inlet, contraction over 2, throat of length 1, diffuser over 4 (a gentle 3.6 degrees), straight outlet.
Refinement level k gives 20k cells across and 20k cells per unit length (so k = 1, 2, 4 for three meshes).
The mesh is one cell thick in z with empty front and back, which is how OpenFOAM does 2D.
"""
import argparse
from pathlib import Path

X = [0.0, 2.0, 4.0, 5.0, 9.0, 12.0]
H = [1.0, 1.0, 0.5, 0.5, 1.0, 1.0]
Z = 0.1   # thickness in z (one cell, empty front/back)

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ny = 20 * args.k
n = len(X)
# vertices: bottom row 0..n-1, top row n..2n-1 at z = 0, then the same again at z = Z
verts = [(x, -h / 2, 0) for x, h in zip(X, H)] + [(x, h / 2, 0) for x, h in zip(X, H)]
verts += [(x, y, Z) for x, y, _ in verts]
bot = lambda s: s
top = lambda s: n + s
back = 2 * n

lines = ["FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }",
         "scale 1;", "vertices", "("]
lines += ["    (%g %g %g)" % v for v in verts]
lines += [");", "blocks", "("]
for s in range(n - 1):
    nx = round((X[s + 1] - X[s]) * 20 * args.k)
    v = [bot(s), bot(s + 1), top(s + 1), top(s), bot(s) + back, bot(s + 1) + back, top(s + 1) + back, top(s) + back]
    lines.append("    hex (%s) (%d %d 1) simpleGrading (1 1 1)" % (" ".join(map(str, v)), nx, ny))
lines += [");", "edges ();", "boundary", "("]
lines.append("    inlet { type patch; faces ((%d %d %d %d)); }" % (bot(0), bot(0) + back, top(0) + back, top(0)))
lines.append("    outlet { type patch; faces ((%d %d %d %d)); }" %
             (bot(n - 1), top(n - 1), top(n - 1) + back, bot(n - 1) + back))
lines.append("    bottomWall { type wall; faces (" + " ".join(
    "(%d %d %d %d)" % (bot(s), bot(s + 1), bot(s + 1) + back, bot(s) + back) for s in range(n - 1)) + "); }")
lines.append("    topWall { type wall; faces (" + " ".join(
    "(%d %d %d %d)" % (top(s), top(s) + back, top(s + 1) + back, top(s + 1)) for s in range(n - 1)) + "); }")
faces = []
for s in range(n - 1):
    faces.append("(%d %d %d %d)" % (bot(s), top(s), top(s + 1), bot(s + 1)))
    faces.append("(%d %d %d %d)" % (bot(s) + back, bot(s + 1) + back, top(s + 1) + back, top(s) + back))
lines += ["    frontAndBack { type empty; faces (" + " ".join(faces) + "); }", ");", "mergePatchPairs ();"]
Path(args.out).write_text("\n".join(lines) + "\n")
