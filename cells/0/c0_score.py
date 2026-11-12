#!/usr/bin/env python
"""Cell 0 v2 scoring: per-run measurements (condition-blind) and the pre-registered decision table.

    python cells/0/c0_score.py --run runs/c0/raw-jepa-sym_M_s0     # one trained run -> measures.json
    python cells/0/c0_score.py --null angle_raw --seed 0             # untrained-encoder null -> runs/c0/null-...
    python cells/0/c0_score.py --data-gate                           # pick the analysis set (ground truth only)
    python cells/0/c0_score.py --aggregate                           # decision table -> cells/0/result_data.json

Blindness (prereg v2 item 10): ``measure`` sees only the latent cloud, the analysis set and the observation mode
(needed to load data and decode). It is identical for every arm; arm labels enter only in ``aggregate``.
Every measurement is a frozen instrument from analysis/ (thresholds from calibration.md, seam ratio v2 from
calibration.md section 11).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c0lib as c                                                       # noqa: E402

import zoo                                                              # noqa: E402
from analysis import gates as G                                         # noqa: E402
from analysis import seam as SM                                         # noqa: E402
from analysis import smoothness as SMO                                  # noqa: E402
from analysis import topology as T                                      # noqa: E402

ROOT = c.ROOT
RUNS = ROOT / "runs" / "c0"
TAU, N_FPS, R_STAB, RHO_MIN = 2.8, 600, 4, 0.5
KNN_MIN, QUAD_MIN, LIP_MAX = 0.966, 0.972, 4.13
THETA_GATE = 0.99
RECON_NMSE_MAX = 0.05
STD_MIN, ERANK_MIN, COPY_EPS = 0.1, 2.0, 1e-6
KAPPA_PH_MAX = SM.KAPPA_PH_MAX


# ============================================================================ analysis set and data gate
def analysis_traj(world: str, index: int, obs: str):
    w = zoo.make("pendulum")
    kw = c.WORLD if world == "main" else c.NEG_WORLD
    return w.sample(seed=zoo.split_seed("analysis", index), horizon=c.HORIZON, n_traj=c.N_ANALYSIS,
                    obs_mode=obs, **kw), w


def h1_state(Z, theta, seed) -> dict:
    """Prereg section 3.3 H1 rule (FPS n = 600, death/birth >= 2.8, R = 4 half-subsamples unanimous) plus the
    winding condition (|rho| >= 0.5): state 'ring' / 'ring_not_wrapping' / 'none' / 'unstable'."""
    rng = np.random.default_rng(seed)
    idx = T.fps_subsample(Z, N_FPS, int(rng.integers(2 ** 31)))
    dg = T.h1_diagram(Z[idx], pi_need=6.0, coeff=47, do_cocycles=True)
    top = T.top_pi(dg, 4)
    b_main = int((top >= TAU).sum())
    b_stab = []
    for _ in range(R_STAB):
        half = T.random_subsample(Z, len(Z) // 2, int(rng.integers(2 ** 31)))
        sub = half[T.fps_subsample(Z[half], N_FPS, int(rng.integers(2 ** 31)))]
        b_stab.append(int((T.top_pi(T.h1_diagram(Z[sub], pi_need=6.0), 4) >= TAU).sum()))
    rho = np.nan
    if len(dg["pi"]):
        rho = float(T.winding_check(dg, theta[idx])["abs_rho"])
    if b_main == 1 and all(b == 1 for b in b_stab):
        st = "ring" if (np.isfinite(rho) and rho >= RHO_MIN) else "ring_not_wrapping"
    elif b_main == 0 and all(b == 0 for b in b_stab):
        st = "none"
    else:
        st = "unstable"
    wrapping_classes = int(any(b == 1 for b in [b_main]) and np.isfinite(rho) and rho >= RHO_MIN)
    return dict(state=st, b_main=b_main, b_stab=b_stab, top_pi=[float(x) for x in top[:3]],
                abs_rho=None if not np.isfinite(rho) else rho, wrapping_class=wrapping_classes)


def data_gate(world: str, max_tries: int = 3) -> dict:
    """Prereg section 3.6: the ground-truth embedding (cos, sin, omega) of the analysis set must reproduce
    ground_truth()['region_betti'][1] and the wrapping count; else the next index, at most 3 (then STOP)."""
    log = []
    for idx in range(max_tries):
        tr, w = analysis_traj(world, idx, "state")
        S = tr.state[:, 1:].reshape(-1, 2)
        gt = w.ground_truth(**tr.meta["params"])
        Z = np.stack([np.cos(S[:, 0]), np.sin(S[:, 0]), S[:, 1]], 1)
        h = h1_state(Z, S[:, 0], seed=7000 + idx)
        b1, wrap = int(gt["region_betti"][1]), int(gt["region_b1_wrapping"])
        want = "ring" if (b1 == 1 and wrap == 1) else ("ring_not_wrapping" if b1 == 1 else "none")
        ok = h["state"] == want
        log.append(dict(index=idx, truth=dict(b1=b1, wrap=wrap), h1=h, ok=ok))
        if ok:
            return dict(world=world, index=idx, ok=True, log=log)
    return dict(world=world, index=None, ok=False, log=log)


# ============================================================================ measurements (blind)
def latents(model, normalizer, traj):
    from baselines.encode import encode
    z, t = encode(model, traj, normalizer)                    # [N, 64, d], t = 1..64 (history 2)
    return z.reshape(-1, z.shape[-1]).astype(np.float64), t


def measure(Z, S, Sprev, obs: str, seed: int = 0, decoded_angle=None) -> dict:
    """Every v2 measurement of one latent cloud Z [N, d] with true states S [N, 2] (theta_t, omega_t) and the
    window's older frame Sprev [N, 2]."""
    th, om, thp = S[:, 0], S[:, 1], Sprev[:, 0]
    E = om ** 2 / 2 - np.cos(th)
    out = {}
    h = h1_state(Z, th, seed=11 + seed)
    out["h1"] = h
    sr = SM.seam_ratio_v2(Z, th, om, thp, groups=np.repeat(np.arange(len(Z) // c.HORIZON), c.HORIZON),
                          n_boot=200, seed=seed)
    fl = SM.seam_flags_v2(sr)
    out["seam"] = dict(ratio=sr["ratio"], ci95=list(sr["ci95"]), status=sr["status"], verdict3=fl["verdict3"],
                       kappa=sr["kappa_first_pass"], fold_index=fl["fold_index"], folded=fl["folded_at_cut"],
                       misalignment_suspect=fl["misalignment_suspect"], n_straddle_dropped=sr["n_straddle_dropped"])
    out["consistency"] = SM.seam_ph_consistency(h["state"], fl["verdict3"], sr["kappa_first_pass"], fl["folded_at_cut"])
    out["h1_none_high_kappa"] = bool(h["state"] == "none" and sr["kappa_first_pass"] > KAPPA_PH_MAX)   # item 6
    for n in (8000, 2000):                                       # Z's choice: n = 8000 primary, 2000 sensitivity
        g = G.theta_decodability(Z, th, n=n, seed=seed)
        out[f"theta_gate_n{n}"] = dict(r2_min=g["r2_min"], passed=bool(g["r2_min"] >= THETA_GATE))
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(Z), size=min(2000, len(Z)), replace=False)
    knn = SMO.knn_probe_r2(Z[sel], E[sel], k=10, seed=seed)
    quad = SMO.ridge_probe_r2(Z[sel], E[sel], degree=2, seed=seed)
    lin = SMO.ridge_probe_r2(Z[sel], E[sel], degree=1, seed=seed)
    lip = SMO.lipschitz_tail(Z[sel], E[sel], k=10)
    decodable = knn >= KNN_MIN if obs == "angle_raw" else (knn >= KNN_MIN and quad >= QUAD_MIN)   # item 8
    out["energy"] = dict(knn_r2=knn, quad_r2=quad, linear_r2=lin, lipschitz_tail=lip, decodable=bool(decodable),
                         smooth=bool(lip <= LIP_MAX))
    if decoded_angle is not None:                                 # P0.1 gate: pointwise theorem condition
        pc = G.pointwise_angle_check(decoded_angle, th, tol=np.pi / 2)
        out["pointwise"] = dict(frac_ok=pc["frac_ok"], holds=pc["all_ok"], n_bad=pc["n_bad"], max_err=pc["max_err"])
    # collapse diagnostics on the analysis latents (jepa exclusion gate)
    std = Z.std(0)
    ev = np.linalg.eigvalsh(np.cov(Z.T))
    ev = np.clip(ev, 0, None)
    p = ev / ev.sum() if ev.sum() > 0 else ev
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if ev.sum() > 0 else 0.0
    out["collapse"] = dict(min_std=float(std.min()), effective_rank=erank,
                           collapsed=bool(std.min() <= STD_MIN or erank < ERANK_MIN))
    return out


def decoded_angles(model, normalizer, traj, obs):
    """Decoded angle of the current frame of every analysis window (recon models only)."""
    import torch
    from baselines.data import frames_to_tensor
    out = []
    with torch.no_grad():
        for i in range(0, traj.n, 50):
            x = frames_to_tensor(traj.obs[i:i + 50], "cpu", normalizer)
            win = model.windows(x)
            z = model.encode(win.flatten(0, 1))
            f = model.decode(z).cpu().numpy()
            if normalizer is not None:
                f = f * normalizer.std + normalizer.mean
            out.append(f)
    f = np.concatenate(out)
    if obs == "angle_cs":
        return G.angle_from_cs(f.reshape(len(f), 2))
    if obs == "angle_raw":
        return np.mod(f.reshape(-1) + np.pi, 2 * np.pi) - np.pi
    from zoo.pendulum import render
    grid = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    bank = render(grid, quantize=False).astype(np.float64)              # [720, 64, 64] in [0, 1] = model units
    return G.nearest_frame_angle(f.reshape(len(f), -1), bank.reshape(len(bank), -1), grid)


def cross_cut_excess(model, normalizer, traj, S, Sprev, variant) -> dict:
    """Training-loss diagnostic (prereg v2 item 5): per-window loss of straddle windows over that of
    non-straddle windows in the same |omega| bin (recon: current-frame error; jepa: one-step latent error)."""
    import torch
    from baselines.data import frames_to_tensor
    errs = []
    with torch.no_grad():
        for i in range(0, traj.n, 50):
            x = frames_to_tensor(traj.obs[i:i + 50], "cpu", normalizer)
            win = model.windows(x)                                    # [n, T, h, *obs]
            z = model.encode(win.flatten(0, 1)).view(win.shape[0], win.shape[1], -1)
            if variant == "recon":
                pred = model.decode(z.flatten(0, 1)).view(win.shape[0], win.shape[1], -1)
                tgt = win[:, :, -1].reshape(win.shape[0], win.shape[1], -1)
                e = model.recon_error(pred, tgt).mean(-1)
            else:
                nxt = model.step(z[:, :-1].flatten(0, 1)).view(win.shape[0], win.shape[1] - 1, -1)
                e = ((nxt - z[:, 1:]) ** 2).mean(-1)
                e = torch.cat([e, e[:, -1:]], 1)
            errs.append(e.cpu().numpy())
    err = np.concatenate(errs).reshape(-1)
    strad = SM.straddle_mask(S[:, 0], Sprev[:, 0])
    bins = np.digitize(np.abs(S[:, 1]), [0.5, 1.0, 1.6, 2.2])
    ratios, w = [], []
    for b in np.unique(bins):
        m = bins == b
        if (m & strad).sum() >= 20 and (m & ~strad).sum() >= 20:
            ratios.append(np.median(err[m & strad]) / max(np.median(err[m & ~strad]), 1e-12))
            w.append((m & strad).sum())
    return dict(excess_ratio=float(np.average(ratios, weights=w)) if ratios else None, n_straddle=int(strad.sum()))


def convergence(metrics: dict, variant: str) -> dict:
    """Prereg v2 item 11: converged iff the validation loss component (recon: 'recon'; jepa: 'inv') changes by
    < 10 % over the last 25 % of training (pred_err[8] is only recorded at the end, so the val loss is used)."""
    key = "recon" if variant == "recon" else "inv"
    cv = [r for r in metrics.get("curves", {}).get("val", []) if key in r]
    if len(cv) < 4:
        return dict(converged=None)
    steps = np.array([r["step"] for r in cv])
    v = np.array([r[key] for r in cv])
    tail = steps >= steps.max() * 0.75
    a, b = v[tail][0], v[tail][-1]
    rel = abs(b - a) / max(abs(a), 1e-12)
    return dict(converged=bool(rel < 0.10), rel_change=float(rel))


def exclusion_gate(metrics: dict, variant: str, loss: str, coll: dict) -> dict:
    """Prereg v2 item 4: runs are excluded only for collapse or for not having learned. recon: val nmse < 0.05
    (circular loss: the wrapped nmse); jepa: not collapsed, and latent_pred_err[8] < latent_copy_err[8] unless the
    copy error is ~0."""
    fm = metrics.get("final", {}) or {}
    if variant == "recon":
        vr = fm.get("val_recon", {}) or {}
        nm = vr.get("nmse_wrapped") if loss == "circular" else vr.get("nmse")
        learned = nm is not None and nm < RECON_NMSE_MAX
        return dict(learned=bool(learned), collapsed=False, nmse=nm, excluded=not learned)
    pe = (fm.get("latent_pred_err") or {}).get("8")
    ce = (fm.get("latent_copy_err") or {}).get("8")
    learned = pe is not None and ce is not None and (pe < ce or ce <= COPY_EPS)
    return dict(learned=bool(learned), collapsed=coll["collapsed"], pred8=pe, copy8=ce,
                excluded=bool(coll["collapsed"] or not learned))


def score_run(run: Path, gate_index: int) -> dict:
    import torch
    from baselines.encode import load
    torch.set_num_threads(1)
    cfg = json.loads((run / "config.json").read_text())["train"]
    metrics = json.loads((run / "metrics.json").read_text())
    obs, variant, loss = cfg["obs"], cfg["variant"], cfg["loss"]
    world = "neg" if cfg["world_kwargs"].get("E_hi") is not None else "main"
    model, normalizer, _ = load(run)
    traj, _ = analysis_traj(world, gate_index, obs)
    Z, _ = latents(model, normalizer, traj)
    S = traj.state[:, 1:].reshape(-1, 2)
    Sprev = traj.state[:, :-1].reshape(-1, 2)
    dec = decoded_angles(model, normalizer, traj, obs) if variant == "recon" else None
    m = measure(Z, S, Sprev, obs, seed=int(cfg["seed"]), decoded_angle=dec)
    m["exclusion"] = exclusion_gate(metrics, variant, loss, m["collapse"])
    m["convergence"] = convergence(metrics, variant)
    if obs == "angle_raw":
        m["cross_cut"] = cross_cut_excess(model, normalizer, traj, S, Sprev, variant)
    m["meta"] = dict(run=run.name, analysis_index=gate_index, device=json.loads((run / "config.json").read_text())
                     ["env"]["device"], steps=metrics.get("steps_done"))
    (run / "measures.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def score_null(obs: str, seed: int, gate_index: int) -> dict:
    """Untrained (initialised) encoder of size M: the 'inherited topology' null (Z's choice 3)."""
    import torch
    from baselines.mlp_wm import build_wm
    w = zoo.make("pendulum")
    torch.manual_seed(seed)
    model = build_wm(w.spec(obs), "recon", "M", c.LATENT, c.HISTORY, c.K_MAIN).eval()
    traj, _ = analysis_traj("main", gate_index, obs)
    Z, _ = latents(model, None, traj)
    S = traj.state[:, 1:].reshape(-1, 2)
    Sprev = traj.state[:, :-1].reshape(-1, 2)
    m = measure(Z, S, Sprev, obs, seed=seed)
    d = RUNS / f"null-{obs}_M_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "measures.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-gate", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args(argv)
    gate_file = RUNS / "data_gate.json"
    if a.data_gate:
        res = {w: data_gate(w) for w in ("main", "neg")}
        RUNS.mkdir(parents=True, exist_ok=True)
        gate_file.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({w: (r["ok"], r["index"]) for w, r in res.items()}))
        return 0 if all(r["ok"] for r in res.values()) else 2
    if a.aggregate:
        from c0_aggregate import aggregate
        res = aggregate()
        (HERE / "result_data.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res["verdicts"], indent=1, default=float))
        return 0
    gates = json.loads(gate_file.read_text())
    if a.run:
        run = Path(a.run)
        world = "neg" if "_neg" in run.name else "main"
        if not gates[world]["ok"]:
            raise SystemExit("data gate failed: STOP (prereg section 3.6)")
        print(json.dumps(score_run(run, gates[world]["index"]), default=float)[:2000])
    elif a.null:
        print(json.dumps(score_null(a.null, a.seed, gates["main"]["index"]), default=float)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
