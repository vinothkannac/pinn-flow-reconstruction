"""A small reader for OpenFOAM field files (ASCII). It only reads the internalField part.
I wrote it myself so I would not need an extra package just to read U and p."""
import itertools
import re
from pathlib import Path

import numpy as np

_HEADER = re.compile(r"internalField\s+(uniform|nonuniform)\s*(?:List<(\w+)>)?")
_VEC = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")


def parse_internal_field(text, ncells=None):
    # Gives an array of shape (n,) for scalars or (n, 3) for vectors.
    # "uniform" fields store only one value, so for those I need to know the number of cells.
    m = _HEADER.search(text)
    if m is None:
        raise ValueError("no internalField entry found")
    kind, typ = m.group(1), m.group(2)
    rest = text[m.end():]

    if kind == "uniform":
        mm = re.match(r"\s*(\([^)]*\)|[^;\s]+)\s*;", rest)
        if mm is None:
            raise ValueError("cannot parse uniform internalField")
        val = np.array(mm.group(1).strip("()").split(), dtype=float)
        if ncells is None:
            raise ValueError("ncells is required for a uniform field")
        return np.tile(val, (ncells, 1)) if val.size > 1 else np.full(ncells, val[0])

    mc = re.match(r"\s*(\d+)\s*\(", rest)
    if mc is None:
        raise ValueError("cannot parse nonuniform list header")
    n = int(mc.group(1))
    body = rest[mc.end():]
    if typ in (None, "scalar"):
        arr = np.array(body[: body.index(")")].split(), dtype=float)
        if arr.size != n:
            raise ValueError(f"expected {n} scalars, found {arr.size}")
        return arr
    if typ == "vector":
        rows = [tuple(map(float, g.groups())) for g in itertools.islice(_VEC.finditer(body), n)]
        if len(rows) != n:
            raise ValueError(f"expected {n} vectors, found {len(rows)}")
        return np.array(rows, dtype=float)
    raise ValueError(f"unsupported list type {typ!r}")


def read_field(path, ncells=None):
    return parse_internal_field(Path(path).read_text(), ncells)


def time_dirs(case):
    # All folders of a case that are named like a number (0, 100, ...), sorted by time.
    # Other folders like constant and system are skipped.
    out = []
    for d in Path(case).iterdir():
        if d.is_dir():
            try:
                out.append((float(d.name), d))
            except ValueError:
                pass
    return sorted(out)
