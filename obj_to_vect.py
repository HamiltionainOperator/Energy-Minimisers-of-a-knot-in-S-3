#!/usr/bin/env python3
"""
Convert a repulsive-curves OBJ output file to .vect format.

Handles:
  - Standard Wavefront OBJ with 'v' (vertex) and 'l' (line/edge) entries.
  - Reconstructs the closed loop ordering from the edge adjacency graph.
  - Single-component curves only (multi-component OBJs are warned about).

Usage:
    python3 obj_to_vect.py input.obj output.vect
"""

import argparse
import os
import sys
from typing import List, Tuple


# ---------------------------------------------------------------------------
# OBJ parsing
# ---------------------------------------------------------------------------

def parse_obj(path: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
    """
    Parse an OBJ file and return (vertices, edges).

    Vertices are 0-indexed. Edge entries ('l') may list more than two indices
    (a polyline); in that case all consecutive pairs are emitted as edges.
    """
    vertices: List[Tuple[float, float, float]] = []
    edges:    List[Tuple[int, int]]            = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "l":
                indices = [int(x.split("/")[0]) - 1 for x in parts[1:]]  # 0-indexed
                for a, b in zip(indices, indices[1:]):
                    edges.append((a, b))

    return vertices, edges


# ---------------------------------------------------------------------------
# Loop reconstruction
# ---------------------------------------------------------------------------

def edges_to_ordered_loop(n_verts: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    Reconstruct a single ordered closed loop from an unordered edge set.

    Falls back to the vertex index sequence 0..n-1 when no edges are present.
    """
    if not edges:
        return list(range(n_verts))

    # Build adjacency list (undirected)
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Walk from an arbitrary start vertex
    start = edges[0][0]
    loop  = [start]
    prev  = None
    cur   = start

    while True:
        neighbors = adj.get(cur, [])
        nxt = None
        for nb in neighbors:
            if nb != prev:
                nxt = nb
                break
        if nxt is None or nxt == start:
            break
        loop.append(nxt)
        prev = cur
        cur  = nxt

    if len(loop) < len(adj):
        # Graph has more than one component; warn and keep the traversed piece
        print(
            f"Warning: edge graph has multiple components. "
            f"Traversed {len(loop)} / {len(adj)} vertices.",
            file=sys.stderr,
        )

    return loop


# ---------------------------------------------------------------------------
# VECT writer
# ---------------------------------------------------------------------------

def write_vect(vertices: List[Tuple[float, float, float]],
               loop: List[int],
               path: str) -> None:
    n = len(loop)
    with open(path, "w") as f:
        f.write("1\n")
        f.write(f"{n}\n")
        for idx in loop:
            x, y, z = vertices[idx]
            f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert repulsive-curves OBJ output to .vect format"
    )
    parser.add_argument("input",  help="Input .obj file")
    parser.add_argument("output", help="Output .vect file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    vertices, edges = parse_obj(args.input)

    if not vertices:
        print("Error: no vertices found in OBJ file", file=sys.stderr)
        sys.exit(1)

    loop = edges_to_ordered_loop(len(vertices), edges)
    write_vect(vertices, loop, args.output)

    print(f"Written {len(loop)} points → {args.output}")


if __name__ == "__main__":
    main()
