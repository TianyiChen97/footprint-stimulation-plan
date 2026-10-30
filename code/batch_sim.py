"""Batch check of step 1 on simulated wells: does G reach the lower bound, and is L balanced?"""
import csv, io, time, sys
from stimplan import configure, CAP
from simulate_input import simulate, PITCH, NC

def els_from(rows):
    return [dict(electrode=r["electrode"], x=(r["electrode"] % NC) * PITCH, y=(r["electrode"] // NC) * PITCH,
                 footprint=r["footprint"], location=r["location"]) for r in rows]

out = []
for F, sep in ((20, 250.0), (40, 200.0)):
    for seed in range(1, 31):
        els = els_from(simulate(F, 1000 + seed, sep))
        t0 = time.time(); _, G, rep = configure(els, 200.0, CAP, seed); dt = time.time() - t0
        out.append(dict(F=F, seed=seed, G=G, lower_bound=rep["lower_bound"], lb_capacity=rep["lb_capacity"],
                        lb_closest_group=rep["lb_clique"], at_lower_bound=G == rep["lower_bound"],
                        L_balance_residual=rep["residual_imbalance"],
                        min_dist_um=round(rep["min_same_config_distance"], 1), seconds=round(dt, 2)))
        print(out[-1], flush=True)
with open("batch_results.csv", "w", newline="") as fh:  # written to the current directory
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
