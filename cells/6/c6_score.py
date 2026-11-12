#!/usr/bin/env python
"""Cell 6 v1 scoring: per-run readouts and the pre-registered decision table (cells/6/prereg.md v1).

    python cells/6/c6_score.py --run runs/c6/H-act_S_s0      # readouts of one run -> scores.npz, scores.json
    python cells/6/c6_score.py --aggregate                     # decision table over all runs -> cells/6/result_data.json
    python cells/6/c6_score.py --oracle H-act                  # the same readouts on ground-truth oracles (no model)

Pilot runs (c6_run.json: pilot = true) are refused: a pilot only reports training loss and the gate.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c6lib as c                                                     # noqa: E402
from analysis import holonomy as ho                                   # noqa: E402

ROOT = c.ROOT
RUNS = ROOT / "runs" / "c6"
PROBES = ho.probe_set()
THRESH = {"auc": 0.90, "prec": 0.75, "exceed_ub": 0.10, "model_check": 0.95}
MARGIN = {"auc": 0.05, "prec": 0.25}
N_BOOT = 2000


# ============================================================================ query sets (ground truth only)
def sets_for(cond: str) -> dict:
    """Evaluation and calibration query sets for the world of ``cond`` (cached; deterministic in the seeds)."""
    key = json.dumps(c.CONDITIONS[cond]["world"], sort_keys=True)
    import hashlib
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    path = RUNS / "_sets" / f"{h}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    w = c.make_world(cond)
    s = {"eval": c.analysis_set(w, c.EVAL_SEED, c.N_EVAL_Z, full=True),
         "cal": c.analysis_set(w, c.CAL_SEED, c.N_CAL_Z, full=False), "world": c.CONDITIONS[cond]["world"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(s))
    return s


def _pad(seqs):
    L = np.array([len(s) for s in seqs])
    A = np.zeros((len(seqs), max(1, L.max())), np.int64)
    for i, s in enumerate(seqs):
        A[i, :len(s)] = s
    return A, L


def z_latents(pred, zs):
    A, L = _pad([z["prefix"] for z in zs])
    return pred.run(pred.start(np.stack([z["s0"] for z in zs])), A, L)


def branch(pred, Zz, zidx, seqs):
    A, L = _pad(seqs)
    return pred.run(pred.take(Zz, np.asarray(zidx)), A, L)


def compare(pred, w, ZA, ZB, true_states_A=None, probes=PROBES, chunk=256):
    """H_obs (max and mean over probes) for each pair; optional model check (predicted vs true positions along
    the probes from branch A)."""
    N = ZA.shape[0]
    hmax, hmean = np.empty(N), np.empty(N)
    ok = tot = ok_s = tot_s = 0
    tp = ho.TruePredictor(w)
    for i in range(0, N, chunk):
        idx = np.arange(i, min(N, i + chunk))
        pa = ho.probe_positions(pred, pred.take(ZA, idx), probes)
        pb = ho.probe_positions(pred, pred.take(ZB, idx), probes)
        diff = (pa != pb).any(-1).mean(-1)
        hmax[idx], hmean[idx] = diff.max(1), diff.mean(1)
        if true_states_A is not None:
            sA = np.asarray(true_states_A[idx])
            pt = ho.probe_positions(tp, sA, probes)[..., :2]
            flip = sA.copy()
            flip[:, 2] ^= 1
            pf = ho.probe_positions(tp, flip, probes)[..., :2]
            hit = (pa[..., :2] == pt).all(-1)
            sens = (pt != pf).any(-1)                      # frames whose true position depends on the door bit
            ok += int(hit.sum())
            tot += hit.size
            ok_s += int(hit[sens].sum())
            tot_s += int(sens.sum())
    out = {"max": hmax, "mean": hmean}
    if true_states_A is not None:
        out["model_check_all"] = ok / max(tot, 1)
        out["model_check_bit_sensitive"] = ok_s / max(tot_s, 1)
        out["model_check_bit_sensitive_frames"] = tot_s
        out["model_check"] = min(out["model_check_all"], out["model_check_bit_sensitive"])
    return out


def oracle_door_probe(w, sz):
    """Position-dependent oracle probe: shortest path to the cell left of the door, then three 'right' moves."""
    ax, ay = w.doors[0][0] - 1, w.doors[0][1]
    row = w.cell_row[ay * w.W + ax]
    s, acts = np.asarray(sz, dtype=np.int64).copy(), []
    while (int(s[0]), int(s[1])) != (ax, ay):
        d = w.D[row, w.NS[w.encode(s)]]
        a = int(np.argmin(d))
        acts.append(a)
        s = w.step(s, a)
    return np.array(acts + [3, 3, 3], dtype=np.int64)


# ============================================================================ per-run readouts
def readouts(pred, w, sets, latent_scale=None) -> tuple[dict, dict]:
    """All readouts for one predictor. Returns (arrays, summary)."""
    ev, cal = sets["eval"]["zs"], sets["cal"]["zs"]
    arr, summ = {}, {}
    # ---------------- loops (P6.1, P6.3 within-model, P6.5, model check)
    Zz = z_latents(pred, ev)
    rows = [(i, lp) for i, z in enumerate(ev) for lp in z["loops"]]
    zi = np.array([i for i, _ in rows])
    ZA = branch(pred, Zz, zi, [lp["gamma"] for _, lp in rows])
    ZB = branch(pred, Zz, zi, [lp["ref"] for _, lp in rows])
    trueA = np.stack([w.trajectory(ev[i]["sz"], lp["gamma"])[-1] for i, lp in rows])
    r = compare(pred, w, ZA, ZB, trueA)
    arr.update(loop_h=r["max"], loop_hmean=r["mean"], loop_z=zi,
               loop_cls=np.array([lp["cls"] for _, lp in rows]), loop_len=np.array([lp["length"] for _, lp in rows]),
               loop_n=np.array([lp["n"] for _, lp in rows]), loop_true=np.array([lp["true_holonomy"] for _, lp in rows]),
               loop_obs_closed=np.array([lp["obs_closed"] for _, lp in rows]))
    for k in ("model_check", "model_check_all", "model_check_bit_sensitive", "model_check_bit_sensitive_frames"):
        summ[k] = r[k]
    if latent_scale is not None:
        za, zb = (np.asarray(x.cpu()) if hasattr(x, "cpu") else np.asarray(x, float) for x in (ZA, ZB))
        with np.errstate(invalid="ignore", divide="ignore"):
            arr["loop_m1"] = np.linalg.norm(za - zb, axis=1) / (latent_scale or np.nan)
    # oracle probe (secondary): same single position-dependent probe from both branches
    oh = np.empty(len(rows))
    for k in range(0, len(rows), 512):
        idx = np.arange(k, min(len(rows), k + 512))
        probes =[oracle_door_probe(w, ev[zi[j]]["sz"]) for j in idx]
        A, L = _pad(probes)
        pa = _positions_along(pred, pred.take(ZA, idx), A, L)
        pb = _positions_along(pred, pred.take(ZB, idx), A, L)
        for t, j in enumerate(idx):
            oh[j] = float((pa[t, :L[t]] != pb[t, :L[t]]).any(-1).mean())
    arr["loop_oracle"] = oh
    # ---------------- calibration loops (P6.3 threshold)
    Zc = z_latents(pred, cal)
    crow = [(i, lp) for i, z in enumerate(cal) for lp in z["loops"] if lp["cls"] == "even2"]
    ci = np.array([i for i, _ in crow])
    rc = compare(pred, w, branch(pred, Zc, ci, [lp["gamma"] for _, lp in crow]),
                 branch(pred, Zc, ci, [lp["ref"] for _, lp in crow]))
    arr["cal_even2_h"] = rc["max"]
    # ---------------- plaquettes (P6.2)
    for room in (0, 1):
        key = f"plaq_R{room}"
        prow = [(i, k, pq) for i, z in enumerate(ev) for k, pq in enumerate(z[key]) if pq is not None]
        pi = np.array([i for i, _, _ in prow])
        rp = compare(pred, w, branch(pred, Zz, pi, [pq["gamma"] for _, _, pq in prow]),
                     branch(pred, Zz, pi, [pq["ref"] for _, _, pq in prow]))
        npl = len(ev[0][key])
        sc = np.full(npl, np.nan)
        kk = np.array([k for _, k, _ in prow])
        for k in range(npl):
            if (kk == k).any():
                sc[k] = rp["max"][kk == k].mean()
        arr[f"{key}_score"] = sc
        arr[f"{key}_n"] = np.array([(kk == k).sum() for k in range(npl)])
    # ---------------- baseline: detection pairs (P6.6, detection)
    drow = [(i, p) for i, z in enumerate(ev) for p in z["det_pairs"]]
    di = np.array([i for i, _ in drow])
    rd = compare(pred, w, branch(pred, Zz, di, [p["h1"] for _, p in drow]), branch(pred, Zz, di, [p["h2"] for _, p in drow]))
    arr.update(det_h=rd["max"], det_z=di, det_cls=np.array([p["cls"] for _, p in drow]),
               det_len=np.array([p["length"] for _, p in drow]), det_true=np.array([p["true_differs"] for _, p in drow]),
               det_obs_closed=np.array([p["obs_closed"] for _, p in drow]))
    # ---------------- baseline: generic pairs + attribution (P6.6, localisation)
    grow = [(i, p) for i, z in enumerate(ev) for p in z["gen_pairs"]]
    gi = np.array([i for i, _ in grow])
    rg = compare(pred, w, branch(pred, Zz, gi, [p["h1"] for _, p in grow]), branch(pred, Zz, gi, [p["h2"] for _, p in grow]))
    arr["gen_h"] = rg["max"]
    arr["gen_parity"] = np.stack([p["parity_diff"] for _, p in grow])
    return arr, summ


def _positions_along(pred, Z, A, L):
    out = np.zeros((A.shape[0], A.shape[1], 3 if pred.door_visible else 2), np.int64)
    for t in range(A.shape[1]):
        m = t < L
        Z = pred.where(m, pred.step(Z, A[:, t]), Z)
        out[:, t] = pred.decode(Z)
    return out


# ============================================================================ statistics on the arrays
def attribution_scores(w, gen_h, gen_parity, corners):
    """Non-geometric localiser: per room-0 cell, Pearson correlation between its entry-parity difference and the
    pair's H_obs; a plaquette's score is the sum over its four cells."""
    cell_score = np.zeros(len(w.cells))
    hz = gen_h - gen_h.mean()
    for r in range(len(w.cells)):
        x = gen_parity[:, r].astype(float)
        if x.std() == 0 or hz.std() == 0:
            continue
        cell_score[r] = float(np.mean((x - x.mean()) * hz) / (x.std() * hz.std()))
    sc = []
    for x, y in corners:
        cells = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
        sc.append(sum(cell_score[w.cell_row[cy * w.W + cx]] for cx, cy in cells))
    return np.array(sc)


def run_summary(cond: str, arr: dict, summ: dict, w) -> dict:
    s = dict(summ)
    m = arr["loop_cls"]
    lab = np.where(m == "odd", 1, 0)
    use = (m == "odd") | (m == "even2")
    s["auc_strat"] = ho.stratified_auc(arr["loop_h"][use], lab[use], arr["loop_len"][use])
    boot = ho.cluster_bootstrap(lambda ix: ho.stratified_auc(arr["loop_h"][use][ix], lab[use][ix], arr["loop_len"][use][ix]),
                                arr["loop_z"][use], N_BOOT, seed=1)
    s["auc_strat_ci"] = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    s["auc_pooled"] = ho.auc(arr["loop_h"][m == "odd"], arr["loop_h"][(m == "even2") | (m == "null")])
    s["auc_strat_mean_probe"] = ho.stratified_auc(arr["loop_hmean"][use], lab[use], arr["loop_len"][use])
    s["auc_oracle_probe"] = ho.stratified_auc(arr["loop_oracle"][use], lab[use], arr["loop_len"][use])
    if "loop_m1" in arr:
        s["auc_m1_latent"] = ho.stratified_auc(arr["loop_m1"][use], lab[use], arr["loop_len"][use])
    s["tau"] = float(np.percentile(arr["cal_even2_h"], 95))
    corners0, corners1 = ho.plaquette_corners(w, 0), ho.plaquette_corners(w, 1)
    sw0, sw1 = ho.switch_plaquette_mask(w, corners0), ho.switch_plaquette_mask(w, corners1)
    da0 = ho.door_adjacent_plaquette_mask(w, corners0)
    sc0 = np.nan_to_num(arr["plaq_R0_score"], nan=-1.0)
    s["prec4_R0"] = ho.precision_at_k(sc0, sw0, 4, seed=2)
    rk = ho.ranks_desc(sc0, seed=2)
    s["ranks_switch_R0"] = rk[sw0].tolist()
    s["ranks_door_adjacent_R0"] = rk[da0].tolist()
    s["door_adjacent_in_top4"] = bool((rk[da0] <= 4).any())
    sc1 = np.nan_to_num(arr["plaq_R1_score"], nan=-1.0)
    s["prec4_R1"] = ho.precision_at_k(sc1, sw1, 4, seed=2) if sw1.any() else None
    s["plaq_R1_coverage"] = arr["plaq_R1_n"].tolist()
    dm = arr["det_cls"]
    dl = np.where(dm == "odd", 1, 0)
    du = (dm == "odd") | (dm == "even2")
    s["base_auc_strat"] = ho.stratified_auc(arr["det_h"][du], dl[du], arr["det_len"][du])
    gsc = attribution_scores(w, arr["gen_h"], arr["gen_parity"], corners0)
    s["base_prec4_R0"] = ho.precision_at_k(gsc, sw0, 4, seed=2)
    s["base_ranks_switch_R0"] = ho.ranks_desc(gsc, seed=2)[sw0].tolist()
    return s


def exceedance(h, z, tau):
    """(fraction of loops with H > tau, upper 95 % bound from a z-cluster bootstrap)."""
    h, z = np.asarray(h), np.asarray(z)
    if len(h) == 0:
        return float("nan"), float("nan")
    frac = float((h > tau).mean())
    boot = ho.cluster_bootstrap(lambda ix: float((h[ix] > tau).mean()), z, N_BOOT, seed=3)
    return frac, float(np.percentile(boot, 95))


# ============================================================================ driver: one run
def latent_scale(pred, w):
    gate = c.gate_set(w, c.GATE_SEED)
    acts = [g["actions"] for g in gate[:200]]
    A, L = _pad(acts)
    Z = pred.start(np.stack([g["s0"] for g in gate[:200]]))
    steps = []
    for t in range(A.shape[1]):
        m = t < L
        Zn = pred.step(Z, A[:, t])
        d = np.linalg.norm(np.asarray(Zn.cpu()) - np.asarray(Z.cpu()), axis=1) if hasattr(Z, "cpu") else \
            np.linalg.norm(np.asarray(Zn, float) - np.asarray(Z, float), axis=1)
        steps.extend(d[m].tolist())
        Z = pred.where(m, Zn, Z)
    return float(np.median(steps))


def score_run(run: Path) -> dict:
    meta = json.loads((run / "c6_run.json").read_text())
    if meta.get("pilot"):
        raise SystemExit("refused: pilot runs report training loss and the gate only (prereg v1, pilot rule)")
    import torch
    from baselines.encode import load
    torch.set_num_threads(int(__import__("os").environ.get("C6_THREADS", "1")))
    model, normalizer, _ = load(run)
    w = c.make_world(meta["cond"])
    pred = ho.WMPredictor(model, normalizer, w)
    sets = sets_for(meta["cond"])
    arr, summ = readouts(pred, w, sets, latent_scale(pred, w))
    s = run_summary(meta["cond"], arr, summ, w)
    s.update(meta)
    np.savez_compressed(run / "scores.npz", **{k: v for k, v in arr.items()})
    (run / "scores.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def score_oracle(cond: str, kind: str = "true") -> dict:
    w = c.make_world(cond)
    pred = {"true": ho.TruePredictor, "memoryless": ho.MemorylessPredictor, "anyentry": ho.AnyEntryPredictor}[kind](w)
    sets = sets_for(cond)
    arr, summ = readouts(pred, w, sets)
    s = run_summary(cond, arr, summ, w)
    s.update({"cond": cond, "oracle": kind})
    return s, arr


# ============================================================================ decision table
def _seed_ok(xs, k=2):
    return sum(bool(x) for x in xs) >= k


def aggregate(runs_dir: Path = RUNS) -> dict:
    R = {}
    for d in sorted(runs_dir.glob("*")):
        if not (d / "c6_run.json").exists():
            continue
        meta = json.loads((d / "c6_run.json").read_text())
        if meta.get("pilot"):
            continue
        g = json.loads((d / "gate.json").read_text())
        s = json.loads((d / "scores.json").read_text()) if (d / "scores.json").exists() else None
        R[(meta["cond"], meta["size"], meta["seed"], meta["n_train"])] = {"gate": g, "s": s, "dir": str(d)}
    out = {"per_size": {}, "runs": {f"{k[0]}_{k[1]}_s{k[2]}_n{k[3]}": {"gate_pass": v["gate"]["pass"],
                                                                       **({} if v["s"] is None else v["s"])}
                                    for k, v in R.items()}}
    size_state = {}
    for size in ("S", "M"):
        prim = [R.get(("H-act", size, sd, c.N_TRAIN_MAIN)) for sd in (0, 1, 2)]
        if any(p is None or p["s"] is None for p in prim):
            size_state[size] = "MISSING"
            continue
        gp = [p["gate"]["pass"] for p in prim]
        evaluable = _seed_ok(gp)
        rec = {"gate_pass": gp, "evaluable": evaluable}
        p61 = [g and p["s"]["auc_strat"] >= THRESH["auc"] for g, p in zip(gp, prim)]
        p62 = [g and p["s"]["prec4_R0"] >= THRESH["prec"] for g, p in zip(gp, prim)]
        p63, exc = [], []
        for sd, (g, p) in enumerate(zip(gp, prim)):
            tau = p["s"]["tau"]
            arr = np.load(Path(p["dir"]) / "scores.npz")
            m = (arr["loop_cls"] == "even2") | (arr["loop_cls"] == "null")
            e_in = exceedance(arr["loop_h"][m], arr["loop_z"][m], tau)
            res = {"within": e_in}
            ok = g and e_in[1] <= THRESH["exceed_ub"]
            for ctl in ("C-open", "C-vis"):
                cr = R.get((ctl, size, sd, c.N_TRAIN_MAIN))
                if cr is None or cr["s"] is None:
                    res[ctl] = None
                    ok = False
                    continue
                ca = np.load(Path(cr["dir"]) / "scores.npz")
                mm = ca["loop_obs_closed"].astype(bool)
                e = exceedance(ca["loop_h"][mm], ca["loop_z"][mm], tau)
                res[ctl] = {"exceed": e, "gate_pass": cr["gate"]["pass"]}
                ok = ok and cr["gate"]["pass"] and e[1] <= THRESH["exceed_ub"]
            p63.append(bool(ok))
            exc.append(res)
        mc = [p["s"]["model_check"] >= THRESH["model_check"] for p in prim]
        rec.update(P61=p61, P62=p62, P63=p63, exceedance=exc, model_check=mc,
                   P61_size=_seed_ok(p61), P62_size=_seed_ok(p62), P63_size=_seed_ok(p63), model_check_size=_seed_ok(mc))
        if not evaluable:
            st = "NE"
        elif rec["P61_size"] and rec["P62_size"] and rec["P63_size"]:
            st = "PASS"
        else:
            st = "FAIL_CHECKED" if rec["model_check_size"] else "FAIL_UNCHECKED"
        rec["state"] = st
        size_state[size] = st
        out["per_size"][size] = rec
    a, b = size_state.get("S"), size_state.get("M")
    if "MISSING" in (a, b):
        verdict = "INCOMPLETE"
    elif a == b == "PASS":
        verdict = "READOUT_WORKS"
    elif "PASS" in (a, b):
        verdict = "INCONCLUSIVE"
    elif "FAIL_CHECKED" in (a, b):
        verdict = "READOUT_DEAD"
    else:
        verdict = "INCONCLUSIVE"
    out["readout_verdict"] = verdict
    # ---------------- P6.6 (only meaningful if the readout works; always computed for the record)
    p66 = {}
    for size in ("S", "M"):
        prim = [R.get(("H-act", size, sd, c.N_TRAIN_MAIN)) for sd in (0, 1, 2)]
        if any(p is None or p["s"] is None for p in prim):
            continue
        ok = [p for p in prim if p["gate"]["pass"]]
        if len(ok) < 2:
            p66[size] = None
            continue
        dauc = float(np.median([p["s"]["auc_strat"] - p["s"]["base_auc_strat"] for p in ok]))
        dprec = float(np.median([p["s"]["prec4_R0"] - p["s"]["base_prec4_R0"] for p in ok]))
        need = {}
        for name, key, thr in (("holo_det", "auc_strat", THRESH["auc"]), ("base_det", "base_auc_strat", THRESH["auc"]),
                               ("holo_loc", "prec4_R0", THRESH["prec"]), ("base_loc", "base_prec4_R0", THRESH["prec"])):
            need[name] = None
            for n in (250, 500, 2000):
                rs = [R.get(("H-act", size, sd, n)) for sd in (0, 1, 2)]
                if any(r is None or r["s"] is None for r in rs):
                    continue
                if _seed_ok([r["s"][key] >= thr for r in rs]):        # no gate here: both readouts read the same models
                    need[name] = n
                    break

        def wins(h, bse):
            return h is not None and (bse is None or h <= bse / 2)
        p66[size] = {"d_auc": dauc, "d_prec": dprec, "n_required": need,
                     "win_auc": dauc >= MARGIN["auc"], "win_prec": dprec >= MARGIN["prec"],
                     "win_data_det": wins(need["holo_det"], need["base_det"]),
                     "win_data_loc": wins(need["holo_loc"], need["base_loc"])}
    out["P66"] = p66
    geo = None
    if verdict == "READOUT_WORKS" and all(p66.get(s) for s in ("S", "M")):
        metrics = ("win_auc", "win_prec", "win_data_det", "win_data_loc")
        geo = any(p66["S"][m] and p66["M"][m] for m in metrics)
    out["geometric_claim_survives"] = geo
    out["status"] = {"READOUT_WORKS": ("alive" if geo else "merged_into:HCDP (Zhang et al. 2026-07)"),
                     "READOUT_DEAD": "dead", "INCONCLUSIVE": "unchanged (inconclusive)",
                     "INCOMPLETE": "unchanged (runs missing)"}[verdict]
    # ---------------- P6.5 (secondary)
    p65 = {}
    for size in ("S", "M"):
        def med(cond, key="auc_strat"):
            rs = [R.get((cond, size, sd, c.N_TRAIN_MAIN)) for sd in (0, 1, 2)]
            v = [r["s"][key] for r in rs if r is not None and r["s"] is not None]
            return float(np.median(v)) if v else None
        act, pc, pn = med("H-act"), med("H-pc"), med("H-pn")
        if None in (act, pc, pn):
            continue
        p65[size] = {"auc_act": act, "auc_pc": pc, "auc_pn": pn, "holds": abs(act - pc) <= 0.05 and pn <= 0.6,
                     "falsified": act - pc > 0.10, "pn_red_flag": pn > 0.6}
    out["P65"] = p65
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--oracle")
    ap.add_argument("--kind", default="true")
    a = ap.parse_args(argv)
    if a.run:
        print(json.dumps(score_run(Path(a.run)), indent=1, default=float))
    elif a.aggregate:
        res = aggregate()
        (HERE / "result_data.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: res[k] for k in ("readout_verdict", "geometric_claim_survives", "status")}, indent=1))
    elif a.oracle:
        s, _ = score_oracle(a.oracle, a.kind)
        print(json.dumps(s, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
