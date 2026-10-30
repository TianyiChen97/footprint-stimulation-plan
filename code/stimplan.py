#!/usr/bin/env python3
"""
stimplan: from manually selected stimulation electrodes to the stimulation table.

Step 1  configure : label each selected electrode with a routing configuration g = 1..G
                    - as few configurations as possible (G - 1 re-routes, each causing a global artifact)
                    - at most CAP (=32 stimulation units) electrodes per configuration
                    - any two electrodes in the same configuration are >= d_min apart
                    - L1..L4 balanced across configurations (each configuration gets floor/ceil of n_L / G)
                  -> configuration image + electrode table
Step 2  table     : rows = every electrode x waveform x amplitude x repetition (time empty);
                    configurations in random order; rows shuffled within each configuration;
                    each stimulus slot = [baseline b][stimulus d][recovery tau], so
                    t = start_g + b + (q-1)(b + d + tau); after each configuration: re-route, then eps
                  -> stimulation spreadsheet (location, waveform, amplitude, time) + CSV

Input CSV columns: electrode (0-26399) [, footprint, location (L1..L4)] [, x_um, y_um]
If x_um/y_um are absent, positions are computed as x = (e mod 220) * 17.5, y = (e div 220) * 17.5 (VERIFY).

  python3 stimplan.py electrodes.csv --dmin 200 --tau 1 --eps 2 --out well0
"""
import argparse, csv, itertools, math, random
from collections import Counter, defaultdict

import numpy as np
import networkx as nx

DELTA = 50e-6           # s, stimulation time resolution (MaxLab DelaySamples)
PITCH = 17.5            # um
NCOL, NROW = 220, 120
CAP = 32                # stimulation units per well
ARRAY_W, ARRAY_H = NCOL * PITCH, NROW * PITCH


# ===================================================================================== input
def load_electrodes(path):
    rows = list(csv.DictReader(open(path)))
    els = []
    for r in rows:
        e = int(r["electrode"])
        if not 0 <= e < NCOL * NROW:
            raise ValueError(f"electrode {e} out of range")
        x = float(r["x_um"]) if r.get("x_um") not in (None, "") else (e % NCOL) * PITCH
        y = float(r["y_um"]) if r.get("y_um") not in (None, "") else (e // NCOL) * PITCH
        els.append(dict(electrode=e, x=x, y=y, footprint=r.get("footprint", "") or "",
                        location=r.get("location", "") or ""))
    ids = [d["electrode"] for d in els]
    if len(set(ids)) != len(ids):
        dup = [e for e, c in Counter(ids).items() if c > 1]
        raise ValueError(f"electrodes selected more than once: {dup}")
    return els


# ===================================================================================== step 1
def conflict_graph(els, dmin):
    P = np.array([[d["x"], d["y"]] for d in els])
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    G = nx.Graph(); G.add_nodes_from(range(len(els)))
    i, j = np.where(np.triu(D < dmin, 1))
    G.add_edges_from(zip(i.tolist(), j.tolist()))
    return G, D


def lower_bound(els, cg, cap):
    clique = max(nx.find_cliques(cg), key=len) if cg.number_of_edges() else [0]
    return max(math.ceil(len(els) / cap), len(clique)), clique


def _targets(els, G):
    """Allowed count range per (label, g) and per g: floor..ceil of an even split."""
    n_lab = Counter(d["location"] for d in els)
    lab_rng = {L: (c // G, -(-c // G)) for L, c in n_lab.items()}
    n = len(els)
    return lab_rng, (n // G, -(-n // G))


def _penalty(state, els, cg, G, cap, lab_rng, size_rng, balance):
    col = state
    conf = sum(1 for u, v in cg.edges if col[u] == col[v])
    size = Counter(col)
    over = sum(max(0, size[g] - cap) for g in range(G))
    pen = 1000 * conf + 1000 * over
    if balance:
        cnt = Counter((els[i]["location"], col[i]) for i in range(len(els)))
        for L, (lo, hi) in lab_rng.items():
            for g in range(G):
                c = cnt[(L, g)]
                pen += 10 * (max(0, lo - c) + max(0, c - hi))
    lo, hi = size_rng
    pen += sum(max(0, lo - size[g]) + max(0, size[g] - hi) for g in range(G))
    return pen


def try_G(els, cg, G, cap, balance, rng, iters=20000):
    n = len(els)
    lab_rng, size_rng = _targets(els, G)
    # greedy start: highest degree first, put in the least-loaded configuration without conflict
    order = sorted(range(n), key=lambda i: (-cg.degree[i], rng.random()))
    col = [-1] * n
    for i in order:
        ok = [g for g in range(G) if all(col[j] != g for j in cg[i])]
        cand = ok or list(range(G))
        cnt = Counter(col[j] for j in range(n) if col[j] >= 0 and els[j]["location"] == els[i]["location"])
        size = Counter(c for c in col if c >= 0)
        col[i] = min(cand, key=lambda g: (cnt[g], size[g], rng.random()))
    pen = _penalty(col, els, cg, G, cap, lab_rng, size_rng, balance)
    T0 = 5.0
    for it in range(iters):
        if pen == 0:
            break
        if rng.random() < 0.5:                                   # move one electrode
            i = rng.randrange(n); g = rng.randrange(G)
            if g == col[i]:
                continue
            old = col[i]; col[i] = g
            new = _penalty(col, els, cg, G, cap, lab_rng, size_rng, balance)
            undo = lambda: col.__setitem__(i, old)
        else:                                                    # swap two electrodes
            i, j = rng.randrange(n), rng.randrange(n)
            if col[i] == col[j]:
                continue
            col[i], col[j] = col[j], col[i]
            new = _penalty(col, els, cg, G, cap, lab_rng, size_rng, balance)
            ci, cj = col[i], col[j]
            undo = lambda ci=ci, cj=cj: (col.__setitem__(i, cj), col.__setitem__(j, ci))
        T = T0 * (1 - it / iters) + 1e-3
        if new <= pen or rng.random() < math.exp((pen - new) / T):
            pen = new
        else:
            undo()
    return col, pen


def configure(els, dmin=200.0, cap=CAP, seed=0, restarts=20, max_extra=6):
    """Return (labels 1..G, G, report)."""
    cg, D = conflict_graph(els, dmin)
    lb, clique = lower_bound(els, cg, cap)
    balance = all(d["location"] for d in els)
    rng = random.Random(seed)
    best = None
    for G in range(lb, lb + max_extra + 1):
        lab_rng, size_rng = _targets(els, G)
        for r in range(restarts):
            col, pen = try_G(els, cg, G, cap, balance, rng)
            hard = sum(1 for u, v in cg.edges if col[u] == col[v]) + \
                sum(max(0, c - cap) for c in Counter(col).values())
            if hard == 0 and (best is None or pen < best[1]):
                best = (col[:], pen, G)
            if hard == 0 and pen == 0:
                break
        if best is not None and best[2] == G:
            break
    if best is None:
        raise RuntimeError("no valid configuration found; reduce d_min or check the electrode list")
    col, pen, G = best
    # relabel configurations in a random order (this is also the order they are run in step 2)
    perm = list(range(G)); rng.shuffle(perm)
    labels = [perm[c] + 1 for c in col]
    # report
    same = [min((D[i, j] for j in range(len(els)) if j != i and labels[j] == labels[i]), default=float("inf"))
            for i in range(len(els))]
    rep = dict(n=len(els), G=G, lower_bound=lb, lb_capacity=math.ceil(len(els) / cap), lb_clique=len(clique),
               clique=[els[i]["electrode"] for i in clique], dmin=dmin, balanced=balance,
               min_same_config_distance=min(same), residual_imbalance=pen if balance else None,
               sizes=dict(sorted(Counter(labels).items())),
               loc_counts={g: dict(sorted(Counter(els[i]["location"] for i in range(len(els))
                                                  if labels[i] == g).items())) for g in range(1, G + 1)})
    for i, d in enumerate(els):
        d["configuration"] = labels[i]
        d["nearest_same_config_um"] = round(same[i], 2)
    return labels, G, rep


# ===================================================================================== step 2
def build_table(els, G, waveforms, amplitudes_mv, R, tau, b, eps, d=0.0004, seed=0):
    """Rows: electrode x W x A x rep, shuffled within configuration, times assigned. Times exact in frames.

    Each stimulus occupies one slot  [ baseline b ][ stimulus d ][ recovery tau ];
    onsets within a configuration are b + d + tau apart; after the last slot of a configuration: re-route, eps.
    """
    fr = lambda s: s / DELTA
    for name, v in (("tau", tau), ("b", b), ("eps", eps), ("d", d)):
        if abs(fr(v) - round(fr(v))) > 1e-9:
            raise ValueError(f"{name} = {v} s is not a multiple of {DELTA*1e6:.0f} us")
    tau_f, b_f, eps_f, d_f = round(fr(tau)), round(fr(b)), round(fr(eps)), round(fr(d))
    slot_f = b_f + d_f + tau_f
    rng = random.Random(seed + 1)
    rows, configs = [], []
    start = 0
    for g in range(1, G + 1):
        mem = [e for e in els if e["configuration"] == g]
        block = [dict(location=e["electrode"], waveform=w, amplitude_mv=amplitudes_mv[a], amp_label=a,
                      configuration=g, footprint=e["footprint"], loc_label=e["location"], rep=r)
                 for e in mem for w in waveforms for a in amplitudes_mv for r in range(1, R + 1)]
        rng.shuffle(block)
        for q, row in enumerate(block):
            row["t_frames"] = start + b_f + q * slot_f                # onset after its own baseline
            row["index_in_config"] = q + 1
        end = start + len(block) * slot_f                              # recovery after the last stimulus ends here
        configs.append(dict(configuration=g, n_electrodes=len(mem), n_stimuli=len(block),
                            start_frames=start, first_onset_frames=start + b_f, end_frames=end,
                            reroute_at_frames=end if g < G else "",
                            electrodes=" ".join(str(e["electrode"]) for e in sorted(mem, key=lambda e: e["electrode"])),
                            **{f"n_{L}": sum(1 for e in mem if e["location"] == L)
                               for L in sorted({e["location"] for e in els if e["location"]})}))
        rows += block
        start = end + eps_f                                            # re-route at `end`, artifact recovery eps
    T_frames = configs[-1]["end_frames"]
    return rows, configs, T_frames


# ===================================================================================== outputs
CFG_STYLE = [("#1f77b4", "o"), ("#d62728", "s"), ("#2ca02c", "^"), ("#9467bd", "D"),
             ("#ff7f0e", "v"), ("#8c564b", "P"), ("#e377c2", "X"), ("#17becf", "*")]


def plot_configurations(els, G, rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
    r = rep["dmin"] / 2
    fig = plt.figure(figsize=(12, 5.2 + 2.6 * math.ceil(G / 3)))
    gs = fig.add_gridspec(1 + math.ceil(G / 3), 3, height_ratios=[2.2] + [1] * math.ceil(G / 3), hspace=0.35)
    ax = fig.add_subplot(gs[0, :])

    def base(a, title):
        a.add_patch(Rectangle((0, 0), ARRAY_W, ARRAY_H, fc="#f7f7f7", ec="#999", lw=0.8))
        a.set_xlim(-40, ARRAY_W + 40); a.set_ylim(ARRAY_H + 40, -40); a.set_aspect("equal")
        a.set_title(title, loc="left")
        a.set_xlabel(r"$x$ ($\mu$m)"); a.set_ylabel(r"$y$ ($\mu$m)")

    base(ax, f"All {rep['n']} selected electrodes: {G} configurations "
             f"(lower bound {rep['lower_bound']}), $d_{{min}}$ = {rep['dmin']:.0f} $\\mu$m, "
             f"min same-configuration distance = {rep['min_same_config_distance']:.0f} $\\mu$m")
    byfp = defaultdict(list)
    for d in els:
        if d["footprint"]:
            byfp[d["footprint"]].append(d)
    for f, ds in byfp.items():
        ds = sorted(ds, key=lambda d: d["location"])
        ax.plot([d["x"] for d in ds], [d["y"] for d in ds], color="#bbb", lw=0.7, zorder=1)
    for d in els:
        c, m = CFG_STYLE[(d["configuration"] - 1) % len(CFG_STYLE)]
        ax.plot(d["x"], d["y"], marker=m, color=c, mec="k", mew=0.4, ms=7, zorder=3)
        if d["location"]:
            ax.text(d["x"] + 25, d["y"] - 25, f"${d['location'][0]}_{d['location'][1:]}$", fontsize=6, zorder=4)
    h = [Line2D([], [], marker=CFG_STYLE[g][1], color=CFG_STYLE[g][0], mec="k", mew=0.4, ls="",
                ms=7, label=f"configuration {g+1} ({rep['sizes'][g+1]} el.)") for g in range(G)]
    h.append(Line2D([], [], color="#bbb", lw=0.7, label="same footprint"))
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)

    for g in range(1, G + 1):
        a = fig.add_subplot(gs[1 + (g - 1) // 3, (g - 1) % 3])
        c, m = CFG_STYLE[(g - 1) % len(CFG_STYLE)]
        lc = rep["loc_counts"][g]
        base(a, f"configuration {g}: {rep['sizes'][g]} electrodes" +
             (("\n" + ", ".join(f"${L[0]}_{L[1:]}$:{v}" for L, v in lc.items())) if rep["balanced"] else ""))
        for d in els:
            if d["configuration"] == g:
                a.add_patch(Circle((d["x"], d["y"]), r, fc=c, ec=c, alpha=0.18, lw=0.6))
                a.plot(d["x"], d["y"], marker=m, color=c, mec="k", mew=0.4, ms=5)
            else:
                a.plot(d["x"], d["y"], marker=".", color="#ccc", ms=2)
        a.tick_params(labelsize=7); a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
    fig.text(0.01, 0.005, f"Small panels: discs of radius $d_{{min}}/2$ = {r:.0f} $\\mu$m; within a configuration "
                          "no two discs overlap, i.e. every pair is at least $d_{min}$ apart.", fontsize=8)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")


def write_outputs(els, rows, configs, T_frames, p, rep, out):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    A = "Arial"
    fb, fh, fi = Font(name=A, size=10), Font(name=A, size=10, bold=True, color="FFFFFF"), Font(name=A, size=10, color="0000FF")
    hf = PatternFill("solid", fgColor="374151"); grey = PatternFill("solid", fgColor="EDEDED")

    def head(ws, names, widths):
        for j, (n, w) in enumerate(zip(names, widths), 1):
            c = ws.cell(row=1, column=j, value=n); c.font = fh; c.fill = hf
            c.alignment = Alignment(horizontal="center", wrap_text=True)
            ws.column_dimensions[get_column_letter(j)].width = w
        ws.freeze_panes = "A2"

    wb = Workbook()
    # ---- Stimulation: the specification table (sorted by time) + annotations
    ws = wb.active; ws.title = "Stimulation"
    head(ws, ["location (electrode)", "waveform", "amplitude (mV)", "time (s)", "time (frames, 50 us)", "",
              "configuration", "footprint", "L", "amplitude label", "rep", "index in configuration"],
         [12, 10, 13, 12, 14, 3, 13, 10, 6, 12, 6, 12])
    for c in ws[1][6:]:
        c.fill = PatternFill("solid", fgColor="6B7280")
    rows_t = sorted(rows, key=lambda r: r["t_frames"])
    for i, r in enumerate(rows_t, 2):
        vals = [r["location"], r["waveform"], r["amplitude_mv"], None, r["t_frames"], None,
                r["configuration"], r["footprint"], r["loc_label"], r["amp_label"], r["rep"], r["index_in_config"]]
        for j, v in enumerate(vals, 1):
            ws.cell(row=i, column=j, value=v).font = fb
        ws.cell(row=i, column=4, value=f"=E{i}*Parameters!$B$4").font = fb
        ws.cell(row=i, column=4).number_format = "0.00000"
        for j in range(7, 13):
            ws.cell(row=i, column=j).fill = grey
    n = len(rows_t) + 1

    # ---- Configurations
    wc = wb.create_sheet("Configurations")
    keys = list(configs[0].keys())
    head(wc, keys, [13 if k != "electrodes" else 80 for k in keys])
    for i, c in enumerate(configs, 2):
        for j, k in enumerate(keys, 1):
            wc.cell(row=i, column=j, value=c[k]).font = fb

    # ---- Electrodes
    we = wb.create_sheet("Electrodes")
    ek = ["electrode", "x", "y", "footprint", "location", "configuration", "nearest_same_config_um"]
    head(we, ["electrode", "x (um)", "y (um)", "footprint", "L", "configuration", "nearest same-config electrode (um)"],
         [11, 9, 9, 10, 6, 13, 20])
    for i, d in enumerate(sorted(els, key=lambda d: (d["configuration"], d["electrode"])), 2):
        for j, k in enumerate(ek, 1):
            we.cell(row=i, column=j, value=d[k]).font = fb
    ne = len(els) + 1

    # ---- Parameters + checks
    wp = wb.create_sheet("Parameters", 0)
    wp.column_dimensions["A"].width = 44; wp.column_dimensions["B"].width = 16; wp.column_dimensions["C"].width = 70
    prm = [("Delta (s), time resolution", DELTA, "MaxLab DelaySamples: 50 us"),                                # B4
           ("tau (s), recovery after each stimulus", p.tau, "gap so the neurons recover. PLACEHOLDER until pilot"),  # B5
           ("b (s), baseline before each stimulus", p.b, "draft: 350 ms"),                                       # B6
           ("eps (s), wait after each re-route", p.eps, "artifact decay. PLACEHOLDER until pilot"),              # B7
           ("d (s), stimulus duration", p.d, "0.4 ms; every waveform must fit in d"),                                    # B8
           ("d_min (um)", p.dmin, "minimum distance within a configuration (practitioner)"),                      # B9
           ("R (repetitions)", p.R, ""),                                                                          # B10
           ("Waveforms", ", ".join(p.waveforms), "defined in the waveform library"),                              # B11
           ("Amplitudes (mV)", ", ".join(f"{k}={v:g}" for k, v in p.amplitudes.items()), "PLACEHOLDER values"),  # B12
           ("Random seed", p.seed, "reproduces step 1 and step 2")]                                              # B13
    wp["A1"] = "Stimulation plan"; wp["A1"].font = Font(name=A, size=13, bold=True)
    wp["A2"] = "Each stimulus slot: [baseline b][stimulus d][recovery tau]; after each re-route: eps."; wp["A2"].font = fb
    for i, (k, v, m) in enumerate(prm, 4):
        wp.cell(row=i, column=1, value=k).font = fb
        wp.cell(row=i, column=2, value=v).font = fi
        wp.cell(row=i, column=3, value=m).font = fb
    G = len(configs)
    chk = [("Selected electrodes", f"=COUNT(Electrodes!A2:A{ne})", ""),                                           # 16
           ("Configurations G", f"=MAX(Electrodes!F2:F{ne})", f"lower bound {rep['lower_bound']} "
            f"(capacity {rep['lb_capacity']}, closest-group {rep['lb_clique']})"),                               # 17
           ("Stimuli N", f"=COUNT(Stimulation!A2:A{n})", "electrodes x waveforms x amplitudes x R"),             # 18
           ("Onset-to-onset within a configuration (s)", "=B6+B8+B5", "b + d + tau"),                            # 19
           ("T, closed form (s)", "=B18*B19+(B17-1)*B7", "N (b + d + tau) + (G-1) eps"),                         # 20
           ("T, from table (s)", f"=MAX(Stimulation!D2:D{n})+B8+B5", "last onset + d + tau"),                   # 21
           ("T (h)", "=B21/3600", ""),                                                                           # 22
           ("Closed form = table?", '=IF(ABS(B20-B21)<0.000001,"yes","NO")', ""),                                # 23
           ("Min same-configuration distance (um)", f"=MIN(Electrodes!G2:G{ne})", ""),                           # 24
           ("Distance constraint met?", '=IF(B24>=B9,"yes","NO")', ""),                                          # 25
           ("Max electrodes in a configuration", f"=MAX(Configurations!B2:B{G+1})", "must be <= 32"),             # 26
           ("Capacity constraint met?", '=IF(B26<=32,"yes","NO")', "")]                                          # 27
    wp["A15"] = "Derived and checks (formulas)"; wp["A15"].font = Font(name=A, size=10, bold=True)
    for i, (k, f, m) in enumerate(chk, 16):
        wp.cell(row=i, column=1, value=k).font = fb
        wp.cell(row=i, column=2, value=f).font = fb
        wp.cell(row=i, column=3, value=m).font = fb
    wp["B22"].number_format = "0.000"
    wp["A29"] = "Stimulation sheet: columns A-E are the specification (one row per stimulus); grey columns are " \
                "annotations. Time = frames x Delta, measured from the session-start marker theta."
    wp["A29"].font = fb
    wb.save(out + "_stimulation.xlsx")

    with open(out + "_stimulation.csv", "w", newline="") as fh_:
        w = csv.writer(fh_)
        w.writerow(["location", "waveform", "amplitude_mv", "time_s", "time_frames", "configuration"])
        for r in rows_t:
            w.writerow([r["location"], r["waveform"], r["amplitude_mv"], f"{r['t_frames']*DELTA:.5f}",
                        r["t_frames"], r["configuration"]])


# ===================================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("electrodes")
    ap.add_argument("--dmin", type=float, default=200.0)
    ap.add_argument("--tau", type=float, default=1.0); ap.add_argument("--b", type=float, default=0.35)
    ap.add_argument("--eps", type=float, default=2.0); ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--d", type=float, default=0.0004, help="stimulus duration (s), every waveform must fit in it")
    ap.add_argument("--waveforms", default="W1,W2,W3")
    ap.add_argument("--amplitudes", default="A1=100,A2=200,A3=400", help="label=mV,...")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", default="well")
    p = ap.parse_args()
    p.waveforms = p.waveforms.split(",")
    p.amplitudes = {k: float(v) for k, v in (s.split("=") for s in p.amplitudes.split(","))}

    els = load_electrodes(p.electrodes)
    labels, G, rep = configure(els, p.dmin, CAP, p.seed)
    plot_configurations(els, G, rep, p.out + "_configurations")
    rows, configs, T_f = build_table(els, G, p.waveforms, p.amplitudes, p.R, p.tau, p.b, p.eps, p.d, p.seed)
    write_outputs(els, rows, configs, T_f, p, rep, p.out)
    print(f"step 1: {rep['n']} electrodes -> G = {G} configurations (lower bound {rep['lower_bound']}: "
          f"capacity {rep['lb_capacity']}, closest group {rep['lb_clique']}), sizes {rep['sizes']}, "
          f"min same-config distance {rep['min_same_config_distance']:.1f} um (d_min {p.dmin:g}), "
          f"L balance residual {rep['residual_imbalance']}")
    for g, c in rep["loc_counts"].items():
        print(f"        configuration {g}: {c}")
    print(f"step 2: {len(rows)} stimuli, T = {T_f*DELTA:.5f} s = {T_f*DELTA/3600:.4f} h")


if __name__ == "__main__":
    main()
