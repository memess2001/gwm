"""Cell 0 v2 decision table (cells/0/prereg.md v2, sections 4-5). Arm labels enter only here.

Run-level classes (angle_raw primary readout = the PH x SR consistency table, analysis.seam.seam_ph_consistency):
  excluded            collapse or not learned (prereg v2 item 4) -> out of the denominator
  not_carrying_theta  theta gate (n = 8000) failed -> stays in the denominator, counts as neither glued nor torn
  glued / torn / inconclusive   from the consistency table (H1 = 0 with kappa > 5 is inconclusive, item 6)
Arm-level (S, M, L x 3 seeds = 9 runs): an arm is evaluable iff >= 6 of 9 runs are not excluded; rates are over
the non-excluded runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import c0lib as c

RUNS = c.ROOT / "runs" / "c0"
TWO_THIRDS = 2 / 3


def load_measures() -> dict:
    out = {}
    for d in sorted(RUNS.glob("*")):
        f = d / "measures.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def run_class(m: dict) -> str:
    if m.get("exclusion", {}).get("excluded"):
        return "excluded"
    if not m["theta_gate_n8000"]["passed"]:
        return "not_carrying_theta"
    return m["consistency"]


def h1_class(m: dict) -> str:
    """For P0.1 / P0.2 / P0.5 (angle_cs, pixels): ring / not_ring / inconclusive (unstable; none with kappa > 5)."""
    st = m["h1"]["state"]
    if st == "unstable" or m.get("h1_none_high_kappa"):
        return "inconclusive"
    return "ring" if st == "ring" else "not_ring"


def arm_runs(M: dict, arm: str, sizes=c.SCORED_SIZES, world="main", K=c.K_MAIN, lam=0.1) -> list:
    rows = [r for r in c.run_table() if r["arm"] == arm and r["size"] in sizes and r["world"] == world
            and r["K"] == K and (r["lambda_lat"] == lam or r["variant"] != "recon")]
    return [(r, M.get(r["name"])) for r in rows]


def arm_rates(M: dict, arm: str, **kw) -> dict:
    rs = arm_runs(M, arm, **kw)
    missing = [r["name"] for r, m in rs if m is None]
    cls = [run_class(m) for _, m in rs if m is not None]
    ok = [x for x in cls if x != "excluded"]
    n = len(ok)
    res = dict(n_runs=len(rs), missing=missing, n_not_excluded=n, classes=cls,
               evaluable=(not missing) and n >= 6,
               glued_rate=(ok.count("glued") / n) if n else None, torn_rate=(ok.count("torn") / n) if n else None,
               seam_median=float(np.nanmedian([m["seam"]["ratio"] for _, m in rs if m is not None]))
               if any(m is not None for _, m in rs) else None)
    return res


def tri(cond_true: bool, evaluable: bool) -> str:
    return "I" if not evaluable else ("T" if cond_true else "F")


def instrument_check(M: dict, groups: dict, world: str, want: str) -> dict:
    """P0.1 (want ring) / P0.2 (want not ring), pooled per observation over the listed arms (9 runs each): at most
    1 of the gate-passing, non-inconclusive runs may fail (prereg v2 item 8); fewer than 6 usable -> insufficient."""
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
                continue                                   # P0.1 gate: pointwise theorem condition
            hc = h1_class(m)
            if hc == "inconclusive":
                continue
            used += 1
            fails += int((hc == "ring") != (want == "ring"))
        res[obs] = dict(status="pass" if fails <= 1 and used >= 6 else ("insufficient" if used < 6 else "fail"),
                        fails=fails, used=used)
    return res


def aggregate() -> dict:
    M = load_measures()
    out = {"n_measured": len(M)}
    # ---------------- instrument checks (phase 1: angle_cs; phase 2 adds pixels)
    p01 = instrument_check(M, {"angle_cs": ["cs-recon"], "pixels64": ["pix-recon"]}, "main", "ring")
    p02 = instrument_check(M, {"angle_cs": ["cs-recon", "cs-jepa-sg", "cs-jepa-sym"],
                               "pixels64": ["pix-recon", "pix-jepa-sg", "pix-jepa-sym"]}, "neg", "not_ring")
    out["P01"], out["P02"] = p01, p02
    stop = any(v.get("status") == "fail" for v in list(p01.values()) + list(p02.values()))
    # ---------------- angle_raw arms
    A = {arm: arm_rates(M, arm) for arm in ("raw-recon-mse", "raw-recon-circ", "raw-jepa-sg", "raw-jepa-sym")}
    out["angle_raw_arms"] = A
    mse, circ = A["raw-recon-mse"], A["raw-recon-circ"]
    ev3 = mse["evaluable"] and circ["evaluable"]
    p03 = tri(ev3 and mse["torn_rate"] >= TWO_THIRDS and circ["glued_rate"] >= TWO_THIRDS
              and circ["glued_rate"] - mse["glued_rate"] >= 0.5, ev3)
    p04 = {}
    for v in ("sg", "sym"):
        j = A[f"raw-jepa-{v}"]
        ev = j["evaluable"] and mse["evaluable"]
        p04[v] = tri(ev and j["glued_rate"] >= TWO_THIRDS and j["glued_rate"] - mse["glued_rate"] >= 0.5, ev)
    # P0.4' (non-blind, reported only, never in a decision)
    nb = all(A[a]["evaluable"] for a in A)
    p04p = tri(nb and all(A[a]["torn_rate"] >= TWO_THIRDS for a in ("raw-recon-mse", "raw-recon-circ", "raw-jepa-sg"))
               and A["raw-jepa-sym"]["seam_median"] < 5, nb)
    # lambda_lat factor (item 5): recon arms at M with lambda_lat = 0 vs 0.1 (descriptive)
    lam = {}
    for arm in ("raw-recon-mse", "raw-recon-circ"):
        lam[arm] = {str(l): [run_class(m) if m else None for _, m in arm_runs(M, arm, sizes=("M",), lam=l)]
                    for l in (0.0, 0.1)}
    # ---------------- P0.5 (decision only once pixel M and L exist: item 12)
    p05 = {}
    for obs, pre in (("angle_cs", "cs"), ("pixels64", "pix")):
        for v in ("sg", "sym"):
            rs = arm_runs(M, f"{pre}-jepa-{v}", sizes=("M", "L"))
            if any(m is None for _, m in rs):
                p05[f"{obs}-{v}"] = dict(status="missing")
                continue
            cl = [h1_class(m) for _, m in rs if not m["exclusion"]["excluded"]]
            k = cl.count("ring")
            p05[f"{obs}-{v}"] = dict(ring=k, of=len(cl), original_le4of6=k <= 4, mine_ge5of6=k >= 5)
    pixels_ready = all(not v.get("status") for k, v in p05.items() if k.startswith("pixels64"))
    # ---------------- P0.6 (recon at L; jepa descriptive)
    p06 = {}
    for arm in ("cs-recon", "raw-recon-mse", "raw-recon-circ", "pix-recon"):
        rs = [m for _, m in arm_runs(M, arm, sizes=("L",)) if m is not None and not m["exclusion"]["excluded"]]
        if len(rs) < 2:
            p06[arm] = dict(status="missing")
            continue
        dec = [m["energy"]["decodable"] for m in rs]
        smo = [m["energy"]["smooth"] for m in rs]
        p06[arm] = dict(mine_decodable_and_smooth=sum(d and s for d, s in zip(dec, smo)) >= 2,
                        original_decodable_not_smooth=sum(d and not s for d, s in zip(dec, smo)) >= 2, n=len(rs))
    # ---------------- P0.7 (descriptive): ring / glued rates by size
    p07 = {}
    for arm in c.ARMS:
        p07[arm] = {}
        for size in ("S", "M", "L", "XL"):
            ms = [m for _, m in arm_runs(M, arm, sizes=(size,)) if m is not None]
            if ms:
                p07[arm][size] = [run_class(m) if arm.startswith("raw") else h1_class(m) for m in ms]
    nulls = {k: dict(h1=v["h1"]["state"], seam=v["seam"]["ratio"], consistency=v["consistency"])
             for k, v in M.items() if k.startswith("null-")}
    out.update(P03=p03, P04=p04, P04_prime_nonblind=p04p, lambda_lat=lam, P05=p05, P06=p06, P07=p07,
               null_encoders=nulls)
    # ---------------- decision table (prereg v2 section 5)
    if stop:
        decision = dict(state="STOP", action="instrument or theory problem: recalibrate on ground truth only, then "
                        "rerun with new model and analysis seeds; no neighbour updates (item 7)")
    else:
        d = {}
        d["cell0.status"] = {"T": "alive", "F": "dead", "I": "unchanged"}[p03]
        d["cell0.confidence.exists"] = {"T": "high (unchanged)", "F": "high -> mid", "I": "unchanged"}[p03]
        if p03 == "T":
            d["cell4.must"] = "+ 输出损失的几何决定潜空间拓扑：格 4 的实验必须把输出几何当成受控变量"
        mech = (p04["sg"], p04["sym"])
        d["cell0.notes.mechanism"] = {
            ("F", "T"): "梯度能同时拉动切口两侧（对称 JEPA）才粘合；stop-grad 继承输入切口",
            ("T", "T"): "只做潜空间预测就会粘合，与 stop-grad 无关",
            ("F", "F"): "无解码器的预测也不粘合：潜空间继承输入编码的拓扑",
            ("T", "F"): "stop-grad 粘合而对称不粘合：与 pilot 相反，需要单独解释",
        }.get(mech, "至少一支不确定")
        if "I" not in mech:
            d["cell4.confidence.yields"] = ("down one tier (在 JEPA 潜空间里'先发现拓扑'不那么必要)"
                                            if "T" in mech else "up one tier (拓扑发现需要显式机制)")
        d["P05"] = ("deferred: needs pixel M and L (item 12)" if not pixels_ready else
                    "see P05: original (<= 4/6) -> cell 4 yields up one tier; mine (>= 5/6) -> cell 4 yields down "
                    "one tier; cell 15 notes only")
        d["cell15.declared_output"] = "格 0 的检查点与二级能量漂移结果（保守世界里的模型集，用于标定虚假收缩）"
        decision = dict(state="DECIDED" if p03 != "I" else "INCONCLUSIVE", updates=d)
    out["verdicts"] = dict(P01={k: v.get("status") for k, v in p01.items()},
                           P02={k: v.get("status") for k, v in p02.items()}, P03=p03, P04=p04,
                           P04_prime_nonblind=p04p, decision=decision)
    return out
