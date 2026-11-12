"""Cell 0 v2.1 decision table (cells/0/prereg.md v2.1, sections 4-7). Arm labels enter only here.

Run classes (angle_raw primary readout = the PH x SR consistency table, analysis.seam.seam_ph_consistency):
  excluded            collapse (RankMe < 2 or a latent std <= 0.1) or not learned (incl. diverged) -> out
  inconclusive        the consistency table could not decide, or the theta gate failed with kappa > 5 (outside the
                      gate's calibrated range) -> out of the denominator (v2.1, review B1)
  not_carrying_theta  theta gate (n = 8000) failed with kappa <= 5 -> decided, counts as neither glued nor torn
  glued / torn        decided by the consistency table
Arm level (S, M, L x 3 seeds = 9 runs): rates over the DECIDED runs; the arm is evaluable iff all 9 runs are there
and >= 6 are decided (else the prediction is I). With this denominator a prediction is F only if decided runs
show the opposite outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import c0lib as c

RUNS = c.ROOT / "runs" / "c0"
TWO_THIRDS = 2 / 3
KAPPA_GATE_MAX = 5.0
DECIDED = ("glued", "torn", "not_carrying_theta")


def load_measures() -> dict:
    out = {}
    for d in sorted(RUNS.glob("*")):
        f = d / "measures.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def run_class(m: dict, gate_key: str = "theta_gate_n8000") -> str:
    if m.get("exclusion", {}).get("excluded"):
        return "excluded"
    if not m[gate_key]["passed"]:
        k = m.get("seam", {}).get("kappa")
        if k is None or not np.isfinite(k) or k > KAPPA_GATE_MAX:
            return "inconclusive"
        return "not_carrying_theta"
    return m["consistency"]


def h1_class(m: dict) -> str:
    """For P0.1 / P0.2 / P0.5: ring / not_ring / inconclusive (unstable; none with kappa > 5)."""
    st = m["h1"]["state"]
    if st == "unstable" or m.get("h1_none_high_kappa"):
        return "inconclusive"
    return "ring" if st == "ring" else "not_ring"


def arm_runs(M: dict, arm: str, sizes=c.SCORED_SIZES, world="main", K=c.K_MAIN, lam=0.1) -> list:
    rows = [r for r in c.run_table() if r["arm"] == arm and r["size"] in sizes and r["world"] == world
            and r["K"] == K and r["role"] != "device_control" and (r["lambda_lat"] == lam or r["variant"] != "recon")]
    return [(r, M.get(r["name"])) for r in rows]


def arm_rates(M: dict, arm: str, gate_key: str = "theta_gate_n8000", **kw) -> dict:
    rs = arm_runs(M, arm, **kw)
    missing = [r["name"] for r, m in rs if m is None]
    cls = [run_class(m, gate_key) for _, m in rs if m is not None]
    dec = [x for x in cls if x in DECIDED]
    n = len(dec)
    seams = [m["seam"]["ratio"] for _, m in rs if m is not None and not m["exclusion"]["excluded"]]
    return dict(n_runs=len(rs), missing=missing, classes=cls, n_decided=n, evaluable=(not missing) and n >= 6,
                glued_rate=(dec.count("glued") / n) if n else None, torn_rate=(dec.count("torn") / n) if n else None,
                seam_median=float(np.nanmedian(seams)) if seams else None)


def tri(cond_true: bool, evaluable: bool) -> str:
    return "I" if not evaluable else ("T" if cond_true else "F")


def instrument_check(M: dict, groups: dict, world: str, want: str) -> dict:
    """P0.1 (want ring) / P0.2 (want not ring), pooled per observation over the listed arms (9 runs each): at most
    1 of the usable runs may fail; usable = not excluded, (P0.1) pointwise check holds, H1 not inconclusive.
    Fewer than 6 usable -> insufficient; any run missing -> missing."""
    res = {}
    for obs, arms in groups.items():
        sizes = c.SCORED_SIZES if world == "main" else ("M",)
        rs = [x for arm in arms for x in arm_runs(M, arm, sizes=sizes, world=world)]
        if any(m is None for _, m in rs):
            res[obs] = dict(status="missing")
            continue
        fails, used = 0, 0
        for r, m in rs:
            if m["exclusion"]["excluded"]:
                continue
            if want == "ring" and not (m.get("pointwise") or {}).get("holds", True):
                continue
            hc = h1_class(m)
            if hc == "inconclusive":
                continue
            used += 1
            fails += int((hc == "ring") != (want == "ring"))
        res[obs] = dict(status="pass" if fails <= 1 and used >= 6 else ("insufficient" if used < 6 else "fail"),
                        fails=fails, used=used)
    return res


def p05_cell(M: dict, arm: str) -> dict:
    """P0.5 for one (observation, variant), M + L (6 runs): ring rate over the decided runs (ring / not ring;
    excluded and H1-inconclusive runs out; v2.1 review B5); I unless >= 5 decided."""
    rs = arm_runs(M, arm, sizes=("M", "L"))
    if any(m is None for _, m in rs):
        return dict(status="missing")
    cl = [h1_class(m) for _, m in rs if not m["exclusion"]["excluded"]]
    dec = [x for x in cl if x != "inconclusive"]
    if len(dec) < 5:
        return dict(status="I", decided=len(dec))
    rate = dec.count("ring") / len(dec)
    return dict(status="decided", ring_rate=rate, decided=len(dec), original_le4of6=rate <= 4 / 6,
                mine_ge5of6=rate >= 5 / 6)


def aggregate() -> dict:
    M = load_measures()
    out = {"n_measured": len(M)}
    p01 = instrument_check(M, {"angle_cs": ["cs-recon"], "pixels64": ["pix-recon"]}, "main", "ring")
    p02 = instrument_check(M, {"angle_cs": ["cs-recon", "cs-jepa-sg", "cs-jepa-sym"],
                               "pixels64": ["pix-recon", "pix-jepa-sg", "pix-jepa-sym"]}, "neg", "not_ring")
    out["P01"], out["P02"] = p01, p02
    # ---------------- angle_raw arms (primary)
    arms = ("raw-recon-mse", "raw-recon-circ", "raw-jepa-sg", "raw-jepa-sym")
    A = {arm: arm_rates(M, arm) for arm in arms}
    A2000 = {arm: arm_rates(M, arm, gate_key="theta_gate_n2000") for arm in arms}     # sensitivity (Z's choice 2)
    out["angle_raw_arms"], out["angle_raw_arms_theta_n2000"] = A, A2000

    def preds(A):
        mse, circ = A["raw-recon-mse"], A["raw-recon-circ"]
        ev3 = mse["evaluable"] and circ["evaluable"]
        p03 = tri(ev3 and mse["torn_rate"] >= TWO_THIRDS and circ["glued_rate"] >= TWO_THIRDS
                  and circ["glued_rate"] - mse["glued_rate"] >= 0.5, ev3)
        p04 = {}
        for v in ("sg", "sym"):
            j = A[f"raw-jepa-{v}"]
            ev = j["evaluable"] and mse["evaluable"]
            p04[v] = tri(ev and j["glued_rate"] >= TWO_THIRDS and j["glued_rate"] - mse["glued_rate"] >= 0.5, ev)
        return p03, p04
    p03, p04 = preds(A)
    out["sensitivity_theta_n2000"] = dict(zip(("P03", "P04"), preds(A2000)))
    nb = all(A[a]["evaluable"] for a in A)
    p04p = tri(nb and all(A[a]["torn_rate"] >= TWO_THIRDS for a in ("raw-recon-mse", "raw-recon-circ", "raw-jepa-sg"))
               and A["raw-jepa-sym"]["seam_median"] < 5, nb)
    lam = {arm: {str(l): [run_class(m) if m else None for _, m in arm_runs(M, arm, sizes=("M",), lam=l)]
                 for l in (0.0, 0.1)} for arm in ("raw-recon-mse", "raw-recon-circ")}
    p05 = {f"{obs}-{v}": p05_cell(M, f"{pre}-jepa-{v}") for obs, pre in (("angle_cs", "cs"), ("pixels64", "pix"))
           for v in ("sg", "sym")}
    p06 = {}
    for arm in ("cs-recon", "raw-recon-mse", "raw-recon-circ", "pix-recon"):
        rs = [m for _, m in arm_runs(M, arm, sizes=("L",)) if m is not None and not m["exclusion"]["excluded"]]
        if len(rs) < 2:
            p06[arm] = dict(status="missing_or_excluded")
            continue
        dec = [m["energy"]["decodable"] for m in rs]
        smo = [m["energy"]["smooth"] for m in rs]
        p06[arm] = dict(mine_decodable_and_smooth=sum(d and s for d, s in zip(dec, smo)) >= 2,
                        original_decodable_not_smooth=sum(d and not s for d, s in zip(dec, smo)) >= 2, n=len(rs))
    p07 = {arm: {size: [run_class(m) if arm.startswith("raw") else h1_class(m)
                        for _, m in arm_runs(M, arm, sizes=(size,)) if m is not None]
                 for size in ("S", "M", "L", "XL")} for arm in c.ARMS}
    nulls = {k: dict(h1=v["h1"]["state"], seam=v["seam"]["ratio"], consistency=v["consistency"],
                     theta_r2=v["theta_gate_n8000"]["r2_min"]) for k, v in M.items() if k.startswith("null-")}
    out.update(P03=p03, P04=p04, P04_prime_nonblind=p04p, lambda_lat=lam, P05=p05, P06=p06, P07=p07,
               null_encoders=nulls)
    # ---------------- decision table (prereg v2.1 section 7)
    cs01, cs02 = p01["angle_cs"].get("status"), p02["angle_cs"].get("status")
    if "fail" in (cs01, cs02):
        decision = dict(state="STOP", action="angle_cs instrument check failed: recalibrate on ground truth only, then "
                        "rerun with new model and analysis seeds; no neighbour updates")
    elif cs01 != "pass" or cs02 != "pass":                       # review B2: no verdict without passing checks
        decision = dict(state="INSTRUMENT_INSUFFICIENT", action="P0.1 / P0.2 (angle_cs) not established: no updates")
    else:
        d = {}
        if p03 == "T":
            d["cell0.status"] = "alive"
            d["cell4.must"] = "+ 输出损失的几何决定潜空间拓扑：格 4 的实验必须把输出几何当成受控变量"
        elif p03 == "F":
            d["cell0.confidence.exists"] = "high -> mid（'输出几何决定拓扑'不成立；status 不变，待 P0.4 / P0.5）"
        notes = {}
        for v in ("sg", "sym"):
            j = A[f"raw-jepa-{v}"]
            if p04[v] == "T":
                notes[v] = "粘合"
            elif p04[v] == "F":
                notes[v] = "继承切口（撕开率 >= 2/3）" if j["torn_rate"] >= TWO_THIRDS else "混合（既未粘合也未稳定撕开）"
        if notes:
            d["cell0.notes.mechanism"] = notes
        c4 = 0
        if "I" not in (p04["sg"], p04["sym"]):
            if "T" in (p04["sg"], p04["sym"]):
                c4 -= 1
            elif all(A[f"raw-jepa-{v}"]["torn_rate"] >= TWO_THIRDS for v in ("sg", "sym")):
                c4 += 1
        px01, px02 = p01["pixels64"].get("status"), p02["pixels64"].get("status")
        if "fail" in (px01, px02):
            d["P05"] = "STOP for pixels: P0.5 not judged; the phase-1 updates above stand"
        elif px01 != "pass" or px02 != "pass" or any(v.get("status") != "decided" for v in p05.values()):
            d["P05"] = "deferred or I: needs pixel M / L and passing pixel instrument checks (item 12)"
        else:
            mine = all(v["mine_ge5of6"] for v in p05.values())
            orig = all(v["original_le4of6"] for v in p05.values())
            c4 += -1 if mine else (1 if orig else 0)
            d["P05"] = "mine (all four >= 5/6)" if mine else ("original (all four <= 4/6)" if orig else "mixed: no update")
        c4 = int(np.clip(c4, -1, 1))
        if c4:
            d["cell4.confidence.yields"] = f"{'up' if c4 > 0 else 'down'} one tier (P0.4 and P0.5 combined, capped at one)"
        d["cell15.declared_output"] = "格 0 的检查点与二级能量漂移（measures.json 的 energy_drift）"
        state = "DECIDED" if p03 != "I" and "I" not in p04.values() else "PARTIAL"
        decision = dict(state=state, updates=d)
    out["verdicts"] = dict(P01={k: v.get("status") for k, v in p01.items()},
                           P02={k: v.get("status") for k, v in p02.items()}, P03=p03, P04=p04,
                           P04_prime_nonblind=p04p, decision=decision)
    return out
