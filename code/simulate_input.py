#!/usr/bin/env python3
"""
Simulated input for stimplan.py (NOT data): F footprints x 4 manually selected electrodes on the MaxTwo grid.

Geometry (assumed, adjust to your axon-tracking results):
  L1 central soma    : soma positions over the organoid contact area, somas >= soma_sep apart
  L2 axon hillock    : 20-50 um from the soma, in the axon direction
  L3 central axon    : 250-500 um from the soma
  L4 terminal axon   : 600-1200 um from the soma
Axons bend slightly (angle jitter). Electrode index = row*220 + col, 17.5 um pitch.

  python3 simulate_input.py --F 20 --seed 1 --out sim_electrodes.csv
"""
import argparse, csv, math, random

PITCH, NC, NR = 17.5, 220, 120
W, H = NC * PITCH, NR * PITCH


def simulate(F=20, seed=1, soma_sep=250.0, max_tries=20000):
    rng = random.Random(seed)
    cx, cy, rx, ry = W / 2, H / 2, 1500, 800          # organoid contact area (ellipse)
    somas, rows, used = [], [], set()
    tries = 0
    while len(somas) < F:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not place footprints; lower soma_sep or F")
        u, v = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if u * u + v * v > 1:
            continue
        sx, sy = cx + u * rx, cy + v * ry
        if any(math.hypot(sx - a, sy - b) < soma_sep for a, b in somas):
            continue
        ang = rng.uniform(0, 2 * math.pi)
        pts = {"L1": (sx, sy)}
        for L, (lo, hi) in (("L2", (20, 50)), ("L3", (250, 500)), ("L4", (600, 1200))):
            dist = rng.uniform(lo, hi); a = ang + rng.gauss(0, 0.2)
            pts[L] = (sx + dist * math.cos(a), sy + dist * math.sin(a))
        cr = {L: (int(round(x / PITCH)), int(round(y / PITCH))) for L, (x, y) in pts.items()}
        if not all(0 <= c < NC and 0 <= r < NR for c, r in cr.values()):
            continue
        es = {L: r * NC + c for L, (c, r) in cr.items()}
        if len(set(es.values())) < 4 or set(es.values()) & used:
            continue
        used |= set(es.values()); somas.append((sx, sy))
        k = len(somas)
        for L in ("L1", "L2", "L3", "L4"):
            rows.append(dict(electrode=es[L], footprint=f"f{k:02d}", location=L))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=int, default=20); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--soma-sep", type=float, default=250.0); ap.add_argument("--out", default="sim_electrodes.csv")
    p = ap.parse_args()
    rows = simulate(p.F, p.seed, p.soma_sep)
    with open(p.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["electrode", "footprint", "location"]); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} electrodes -> {p.out}")
