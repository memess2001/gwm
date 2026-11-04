"""Tests of the cell-0 instruments (analysis/topology.py, smoothness.py, controls.py, seam.py, gates.py).

Only synthetic clouds and ground-truth pendulum embeddings are used (instrument checks, no model latents).
"""
import numpy as np
import pytest
from ripser import ripser

from analysis import controls as C
from analysis import smoothness as SM
from analysis import topology as T

TAU = 3.0          # a pi threshold well inside the gap between noise (< 2.5) and the loops used here


def circle(n, radius=1.0, centre=(0.0, 0.0), noise=0.0, seed=0, dim=2):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    X = np.zeros((n, dim))
    X[:, 0], X[:, 1] = centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)
    return X + noise * rng.standard_normal(X.shape), t


def count(X, n=None, method="random", seed=0):
    idx = np.arange(len(X)) if n is None else (T.fps_subsample(X, n, seed) if method == "fps"
                                                 else T.random_subsample(X, n, seed))
    dg = T.h1_diagram(X[idx], bs_alpha=0.05)
    return T.count_pi_threshold(T.top_pi(dg), TAU), T.bs_test(dg)["n_sig"], dg


# ----------------------------------------------------------------------------------------------- subsampling
def test_fps_is_a_greedy_net_and_deterministic():
    X = np.random.default_rng(1).uniform(size=(3000, 3))
    i1, r1 = T.fps_subsample(X, 200, seed=5, return_radius=True)
    i2, r2 = T.fps_subsample(X, 400, seed=5, return_radius=True)
    assert len(set(i1)) == 200 and np.array_equal(i1, i2[:200])          # nested, distinct
    assert np.array_equal(i1, T.fps_subsample(X, 200, seed=5))            # seeded
    assert r2 < r1
    d = np.min(np.linalg.norm(X[:, None, :] - X[i1][None], axis=2), axis=1)
    assert np.isclose(d.max(), r1)                                        # covering radius is exact
    assert np.array_equal(T.random_subsample(X, 50, 3), T.random_subsample(X, 50, 3))


# ----------------------------------------------------------------------------------------------- Betti numbers
def test_circle_has_one_h1_class():
    X, _ = circle(400, noise=0.03, seed=2)
    c_thr, c_bs, dg = count(X, 300)
    assert c_thr == 1 and c_bs == 1
    top = T.top_pi(dg)
    assert top[0] >= 6.0 and top[1] < 2.5          # top[0] may be a certified lower bound (censored bar)
    assert T.count_pi_gap(top, 2.0) == 1 and T.count_pi_threshold_gap(top, TAU, 2.0) == 1


def test_two_circles_have_two_h1_classes():
    A, _ = circle(300, 1.0, (0, 0), 0.02, seed=3)
    B, _ = circle(300, 0.7, (3.5, 0), 0.02, seed=4)
    X = np.vstack([A, B])
    for method in ("random", "fps"):
        c_thr, c_bs, dg = count(X, 400, method)
        assert c_thr == 2, method
        assert T.count_pi_gap(T.top_pi(dg), 2.0) == 2
    assert count(X, 400, "random")[1] == 2                # B&S (valid for i.i.d. subsamples)


def test_blob_has_no_h1_class():
    X = np.random.default_rng(5).standard_normal((2000, 8))
    for method in ("random", "fps"):
        c_thr, c_bs, dg = count(X, 300, method)
        assert c_thr == 0 and T.top_pi(dg)[0] < 2.0, method
        assert T.count_pi_gap(T.top_pi(dg), 2.0) == 0
    assert count(X, 300, "random")[1] == 0


def test_pendulum_cylinder_disk_and_torn_chart():
    S, info = C.pendulum_states(3_999_000, n_traj=200, horizon=32)
    assert info["b1"] == 1 and info["wrap"] == 1
    assert count(C.emb_ref(S), 300, "fps")[0] == 1
    assert count(C.emb_torn(S), 300, "fps")[0] == 0                       # cut open at +-pi
    assert count(C.emb_folded(S), 300, "fps")[0] == 0                     # folded onto [-1, 1]
    Sd, infod = C.pendulum_states(3_999_001, n_traj=200, horizon=32, E_hi=0.8)
    assert infod["b1"] == 0 and count(C.emb_ref(Sd), 300, "fps")[0] == 0  # libration-only disk


# ----------------------------------------------------------------------------------------------- diagram exactness
def test_adaptive_threshold_matches_full_filtration():
    S, _ = C.pendulum_states(3_999_002, n_traj=200, horizon=32)
    Y = C.emb_ref(S, 3.0)[T.random_subsample(S, 300, 0)]
    full = ripser(Y, maxdim=1)["dgms"][1]
    pi_full = np.sort(full[:, 1] / full[:, 0])[::-1]
    dg = T.h1_diagram(Y, pi_need=4.0)
    assert dg["thresh"] < np.inf and dg["censored"].any()                 # the long loop was censored ...
    assert np.all(dg["pi"][dg["censored"]] >= 4.0)                        # ... with a certified lower bound
    exact = np.sort(dg["pi"][~dg["censored"]])[::-1]
    assert np.allclose(exact, pi_full[1:len(exact) + 1], rtol=1e-5)       # every other bar is exact
    assert len(dg["pi"]) == len(pi_full)


# ----------------------------------------------------------------------------------------------- Bobrowski-Skraba
def test_bs_constants_and_null():
    assert T.BS_A == {"rips": 1.0, "cech": 0.5}                          # Sci. Rep. 13:12274, Eq. 2
    assert np.isclose(T.lgumbel_cdf(np.log(np.log(2.0))), 0.5)            # LGumbel median
    pi = 1.0 + np.random.default_rng(0).exponential(0.2, 500)
    ell, B, m = T.bs_ell(pi)
    assert m == 500 and np.isclose(ell.mean(), -np.euler_gamma)           # B centres l at the LGumbel mean
    # i.i.d. uniform square: l-values close to LGumbel (sd pi / sqrt 6 = 1.28)
    X = np.random.default_rng(7).uniform(size=(800, 2))
    dg = T.h1_diagram(X)
    ell = T.bs_test(dg)["ell"]
    ell = ell[np.isfinite(ell)]
    assert 1.0 < ell.std() < 1.6
    assert T.bs_pi_star(dg["pi"]) > T.top_pi(dg)[0]                     # nothing significant


# ----------------------------------------------------------------------------------------------- rules
def test_count_rules_on_synthetic_bars():
    assert T.count_pi_threshold([7.0, 1.8, 1.7, 1.6], 3.0) == 1
    assert T.count_pi_threshold([7.0, 6.0, 1.7, 1.6], 3.0) == 2
    assert T.count_pi_gap([1.9, 1.8, 1.7, 1.6], 2.0) == 0
    assert T.count_pi_gap([7.0, 6.0, 1.7, 1.6], 2.0) == 2
    assert T.count_pi_threshold_gap([3.2, 2.9, 1.7, 1.6], 3.0, 1.5) == -1    # cut inside a cluster: ambiguous
    assert T.count_pi_threshold_gap([1.9, 1.8], 3.0, 1.5) == 0
    assert T.stable_count([1, 1, 1, 0, 1], 0.8) == 1 and T.stable_count([1, 0, 1, 0, 1], 0.8) == -1
    assert T.stability([1, 1, 0, 1], 1) == 0.75


# ----------------------------------------------------------------------------------------------- winding
def test_winding_of_circle_and_of_a_non_wrapping_loop():
    X, t = circle(400, noise=0.02, seed=8)
    dg = T.h1_diagram(X, coeff=47, do_cocycles=True)
    w = T.winding_check(dg, t)
    assert w["abs_rho"] > 0.95 and abs(w["degree"]) == 1
    # the same loop against an angle that only oscillates (like theta on the libration ring): no winding
    w0 = T.winding_check(dg, 1.2 * np.sin(t))
    assert w0["abs_rho"] < 0.3 and w0["degree"] == 0
    assert np.isclose(T.fisher_lee(t, -t + 1.0), -1.0) and np.isclose(T.fisher_lee(t, t + 2.0), 1.0)


# ----------------------------------------------------------------------------------------------- whitening / embeddings
def test_whitening_undoes_linear_thinning():
    S, _ = C.pendulum_states(3_999_003, sampling="box", n_traj=100, horizon=40)
    thin = C.emb_thin(S, 1 / 30)
    assert count(thin, 300, "fps")[0] == 0                                # below the raw detection limit
    assert count(T.pca_whiten(thin)[0], 300, "fps")[0] == 1               # affine invariance restores it
    Xw, info = T.pca_whiten(np.random.default_rng(0).standard_normal((500, 5)) * [5, 3, 1, 1, 0.1], 3)
    assert Xw.shape == (500, 3) and np.allclose(Xw.var(0, ddof=1), 1.0)


def test_random_smooth_embedding_keeps_the_loop():
    X, _ = circle(800, noise=0.0, seed=9)
    for level in ("mild", "medium", "strong"):
        Z = C.RandomSmoothEmbedding(2, 8, level, seed=1)(X)
        assert Z.shape == (800, 8)
        assert count(Z, 300, "fps")[0] == 1, level
    assert C.local_distortion(X, C.RandomSmoothEmbedding(2, 8, "mild", seed=1)(X))["spread"] < 1.5


# ----------------------------------------------------------------------------------------------- energy
@pytest.fixture(scope="module")
def pend():
    S, info = C.pendulum_states(3_999_004, n_traj=250, horizon=64)
    idx = np.random.default_rng(0).permutation(len(S))[:2000]
    return S[idx], info["energy"][idx]


def test_identity_is_smooth_and_permuted_is_not(pend):
    S, E = pend
    ref = C.emb_ref(S)
    s_id = SM.smoothness_score(ref, E, ref, k=10, n_perm=5)
    assert np.isclose(s_id["s"], 1.0) and np.isclose(s_id["D_perm"], 2 * len(E) / (len(E) - 1))
    assert abs(s_id["D_perm_emp"] - s_id["D_perm"]) < 5 * s_id["D_perm_sd"] + 1e-3
    perm = ref[np.random.default_rng(1).permutation(len(ref))]
    s_perm = SM.smoothness_score(perm, E, ref, k=10)
    assert abs(s_perm["s"]) < 0.1
    assert s_perm["lip_tail"] > 2 * s_id["lip_tail"]


def test_probes_on_reference_folds_and_permutation(pend):
    S, E = pend
    ref = C.emb_ref(S)
    assert SM.ridge_probe_r2(ref, E, degree=2) > 0.999                    # E = omega^2/2 - cos(theta): quadratic
    assert SM.ridge_probe_r2(ref, E, degree=1) < 0.5                      # ... but not linear in this chart
    assert SM.knn_probe_r2(ref, E, 10) > 0.97
    assert SM.krr_probe_r2(ref, E, max_n=400)["r2"] > 0.99
    for Z in (C.emb_fold_theta(S), C.emb_fold_omega(S), C.emb_folded(S)):  # folds along E's symmetries
        assert SM.knn_probe_r2(Z, E, 10) > 0.97
        assert SM.smoothness_score(Z, E, ref, k=10)["s"] > 0.95
    perm = ref[np.random.default_rng(2).permutation(len(ref))]
    assert SM.knn_probe_r2(perm, E, 10) < 0.05 and abs(SM.ridge_probe_r2(perm, E, degree=2)) < 0.05


def test_non_smooth_controls_are_flagged(pend):
    S, E = pend
    ref = C.emb_ref(S)
    sin10 = C.emb_sin_omega(S, 10)
    assert SM.knn_probe_r2(sin10, E, 10) < 0.5
    assert SM.smoothness_score(sin10, E, ref, k=10)["s"] < 0.5
    patch = C.patch_shuffle(S, ref, 8, seed=3)
    tail_ref = SM.lipschitz_tail(ref, E, 10)
    assert SM.lipschitz_tail(patch, E, 10) > 2 * tail_ref                 # tears show up in the slope tail


# ----------------------------------------------------------------------------------------------- seam ratio
from analysis import gates as G      # noqa: E402
from analysis import seam as SE      # noqa: E402


@pytest.fixture(scope="module")
def pend_full():
    """250 conservative trajectories x 64 steps of the cell-0 world (seed reserved for these tests)."""
    S, info = C.pendulum_states(3_999_010, n_traj=250, horizon=64)
    return S, info["energy"], np.arange(len(S)) // 64


def test_group_weighted_median_matches_repeated_values():
    rng = np.random.default_rng(0)
    vals = np.sort(rng.normal(size=7)), np.sort(rng.normal(size=4))
    w = rng.integers(0, 4, size=7).astype(float), rng.integers(1, 4, size=4).astype(float)
    m, tot = SE._group_wmedian(np.concatenate(vals), np.array([0, 7]), np.array([7, 11]), np.concatenate(w))
    for g in range(2):
        rep = np.repeat(vals[g], w[g].astype(int))
        assert tot[g] == len(rep) and m[g] == rep[(len(rep) - 1) // 2]       # lower median of the expanded sample


def test_seam_ratio_glued_vs_torn_and_scale_free(pend_full):
    S, E, grp = pend_full
    th, om = S[:, 0], S[:, 1]
    glued = SE.seam_ratio(C.emb_ref(S), th, om, groups=grp, n_boot=50, seed=1)
    torn = SE.seam_ratio(C.emb_torn(S), th, om, groups=grp, n_boot=50, seed=1)
    assert glued["status"] == torn["status"] == "ok"
    assert 0.8 < glued["ratio"] < 1.25 and torn["ratio"] > 20
    assert glued["ci95"][1] < 5.0 < torn["ci95"][0]                       # P0.4' threshold far from both
    assert SE.classify_seam(glued) == "glued" and SE.classify_seam(torn) == "torn"
    for r in (glued, torn):                                               # placebo cuts elsewhere: ~1 on both
        assert len(r["placebo"]) == 7 and r["placebo_max"] < 1.2
    # invariant under rotation, translation and global rescaling of the latent (exactly the same pairs)
    Q = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))[0]
    moved = SE.seam_ratio(3.7 * C.emb_ref(S) @ Q.T + 5.0, th, om, n_boot=0)
    assert np.isclose(moved["ratio"], glued["ratio"], rtol=1e-9)
    # the arc chart at alpha = 1 is a rigid motion of the reference chart, at alpha = 0 it is the torn chart
    assert np.isclose(SE.seam_ratio(C.emb_arc(S, 1.0), th, om, n_boot=0)["ratio"], glued["ratio"], rtol=1e-9)
    assert np.isclose(SE.seam_ratio(C.emb_arc(S, 0.0), th, om, n_boot=0)["ratio"], torn["ratio"], rtol=1e-9)


def test_seam_ratio_grows_with_the_gap_and_profiles_partial_gluing(pend_full):
    S, E, grp = pend_full
    th, om = S[:, 0], S[:, 1]
    assert np.isclose(C.arc_gap(0.5), 2 * np.sin(np.pi * 0.5) / 0.5) and np.isclose(C.arc_gap(0.0), 2 * np.pi)
    Z = C.emb_arc(S, 0.9)                                                 # chord across the cut = arc_gap
    i = np.argmin(np.abs(th - np.pi + 1e-9))
    P = np.array([[np.pi - 1e-12, om[i]], [-np.pi, om[i]]])
    assert np.isclose(np.linalg.norm(np.diff(C.emb_arc(P, 0.9), axis=0)), C.arc_gap(0.9), rtol=1e-6)
    sr = [SE.seam_ratio(C.emb_arc(S, a), th, om, n_boot=0)["ratio"] for a in (1.0, 0.98, 0.9, 0.6, 0.3)]
    assert np.all(np.diff(sr) > 0) and sr[0] < 1.25 and sr[-1] > 20
    # glued only for fast rotations: the omega profile shows it, the pooled ratio is dominated by the torn part
    r = SE.seam_ratio(C.emb_arc_omega(S, glue_fast=True), th, om, n_boot=0)
    prof = {float(k.split("..")[0]): v["ratio"] for k, v in r["by_omega"].items()}
    fast = [v for lo, v in prof.items() if lo >= 1.75 or lo + 0.25 <= -1.75]
    slow = [v for lo, v in prof.items() if -0.75 <= lo and lo + 0.25 <= 0.75]
    assert fast and slow and max(fast) < 1.3 and min(slow) > 10
    assert r["ratio"] > 5.0                                               # all omega (the preregistered definition)
    rf = SE.seam_ratio(C.emb_arc_omega(S, glue_fast=True), th, om, n_boot=0, omega_abs_min=1.6)
    assert rf["ratio"] < 1.3 and rf["settings"]["omega_abs_min"] == 1.6   # fast rotations only: glued


def test_seam_ratio_needs_data_on_both_sides_of_the_cut():
    S, _ = C.pendulum_states(3_999_011, n_traj=100, horizon=64, E_hi=0.5)    # libration only: never reaches pi
    r = SE.seam_ratio(C.emb_ref(S), S[:, 0], S[:, 1], n_boot=10)
    assert r["status"] == "insufficient" and np.isnan(r["ratio"]) and SE.classify_seam(r) == "insufficient"
    with pytest.raises(ValueError):
        SE.seam_ratio(C.emb_ref(S), S[:, 0], S[:, 1], control_angles=(3.0,))   # control window overlaps the cut
    with pytest.raises(ValueError):
        SE.seam_ratio(C.emb_ref(S), S[:-1, 0], S[:, 1])


# ----------------------------------------------------------------------------------------------- gates
def test_theta_gate_passes_injective_and_rejects_conserved_quantity_latents(pend_full):
    S, E, _ = pend_full
    th, om = S[:, 0], S[:, 1]
    for Z in (C.emb_ref(S), C.emb_torn(S), C.RandomSmoothEmbedding(3, 8, "medium", seed=3)(C.emb_ref(S))):
        g = G.theta_decodability(Z, th, seed=0)
        assert g["passed"] and g["r2_min"] >= 0.99 and g["n"] == 2000 and g["k"] == 10
    folded = G.theta_decodability(np.stack([np.cos(th), np.abs(om)], 1), th, seed=0)
    assert folded["r2_cos"] > 0.99 and folded["r2_sin"] < 0.3 and not folded["passed"]   # sin theta folded away
    energy = G.theta_decodability(np.stack([E, np.sign(om)], 1), th, seed=0)            # (E, sign omega)
    assert energy["r2_min"] < 0.3 and not energy["passed"]
    assert not G.theta_decodability(E, th, seed=0)["passed"]                           # 1-D latent accepted


def test_pointwise_angle_check_wraps_and_is_strict():
    th = np.array([np.pi - 0.1, -np.pi + 0.05, 0.0, 1.0])
    hat = np.array([-np.pi + 0.1, np.pi - 0.05, np.pi / 2 - 1e-9, 1.0 - np.pi / 2])
    assert np.allclose(G.circular_error(hat, th), [0.2, 0.1, np.pi / 2 - 1e-9, np.pi / 2])
    r = G.pointwise_angle_check(hat, th)
    assert r["n_bad"] == 1 and np.isclose(r["frac_ok"], 0.75) and not r["all_ok"]     # error = pi/2 is not < pi/2
    assert G.pointwise_angle_check(hat[:3], th[:3])["all_ok"]
    bad = G.pointwise_angle_check(np.array([np.nan, 0.0]), np.array([0.0, 0.0]))
    assert bad["n_nonfinite"] == 1 and bad["frac_ok"] == 0.5 and not bad["all_ok"]


def test_decoded_angles_from_cs_and_from_a_frame_bank():
    rng = np.random.default_rng(4)
    th = rng.uniform(-np.pi, np.pi, 500)
    assert np.allclose(G.circular_error(G.angle_from_cs(np.stack([np.cos(th), np.sin(th)], 1)), th), 0, atol=1e-12)
    grid = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    frames = lambda a: np.stack([np.cos(a), np.sin(a), np.cos(2 * a), np.sin(3 * a)], 1)  # any injective "render"
    dec = G.nearest_frame_angle(frames(th) + 0.001 * rng.normal(size=(500, 4)), frames(grid), grid, batch=128)
    assert G.circular_error(dec, th).max() < 2 * np.pi / 720
    assert G.pointwise_angle_check(dec, th)["all_ok"]


# ----------------------------------------------------------------------------------------------- review (round 2)
def test_circular_coordinate_cocycle_is_consistent_and_invariant():
    """The Z_47 cocycle that circular_coordinate reads (i) lists edges as (larger, smaller) vertex, the orientation
    convention the code relies on, (ii) restricted to the Rips complex at r and lifted to integers it is an integer
    cocycle (zero on every triangle), and (iii) the coordinate is invariant under rotation, translation and
    global rescaling of the cloud."""
    rng = np.random.default_rng(11)
    t = rng.uniform(-np.pi, np.pi, 1500)
    X = np.stack([np.cos(t), np.sin(t), rng.uniform(-1, 1, 1500)], 1)       # annulus S^1 x [-1, 1]
    idx = T.fps_subsample(X, 250, 0)
    dg = T.h1_diagram(X[idx], coeff=47, do_cocycles=True)
    i_bar = int(np.argmax(dg["pi"]))
    coc = np.asarray(dg["cocycles"][i_bar]).astype(np.int64)
    assert np.all(coc[:, 0] > coc[:, 1])
    cc = T.circular_coordinate(dg)
    val = coc[:, 2] % 47
    val[val > 23] -= 47
    alpha = {(int(b), int(a)): int(v) for a, b, v in zip(coc[:, 0], coc[:, 1], val)}   # oriented small -> large
    from scipy.spatial import cKDTree
    Y = dg["points"]
    nb = [set() for _ in range(len(Y))]
    for a, b in cKDTree(Y).query_pairs(cc["r"]):
        nb[a].add(b), nb[b].add(a)
    n_tri = 0
    for a in range(len(Y)):
        for b in (x for x in nb[a] if x > a):
            for c in (x for x in nb[a] & nb[b] if x > b):
                n_tri += 1
                assert alpha.get((a, b), 0) + alpha.get((b, c), 0) - alpha.get((a, c), 0) == 0
    assert n_tri > 1000
    w = T.winding_check(dg, t[idx])
    assert w["abs_rho"] > 0.95 and abs(w["degree"]) == 1
    Q = np.linalg.qr(rng.normal(size=(5, 5)))[0][:, :3]
    for s in (1e-3, 1e3):
        Xm = s * X @ Q.T - 4.0
        im = T.fps_subsample(Xm, 250, 0)
        assert np.array_equal(im, idx)
        dm = T.h1_diagram(Xm[im], coeff=47, do_cocycles=True)
        wm = T.winding_check(dm, t[im])
        assert np.isclose(wm["abs_rho"], w["abs_rho"], rtol=1e-9) and wm["degree"] == w["degree"]


def test_seam_omega_aspect_and_fold_flags(pend_full):
    S, E, grp = pend_full
    th, om = S[:, 0], S[:, 1]
    torn100 = SE.seam_ratio(np.stack([th, 100.0 * om], 1), th, om, groups=grp, n_boot=0)
    assert np.isclose(torn100["omega_aspect"], 100.0, rtol=1e-6)            # exact for (theta, c omega)
    assert torn100["ratio"] < 5.0                                           # a torn latent that reads "glued" ...
    f = SE.seam_flags(torn100)
    assert f["verdict"] == "glued" and f["omega_dominated"] and not f["in_range"]     # ... is flagged
    ref = SE.seam_ratio(C.emb_ref(S), th, om, groups=grp, n_boot=0)
    torn = SE.seam_ratio(C.emb_torn(S), th, om, groups=grp, n_boot=0)
    assert abs(ref["omega_aspect"] - 1.0) < 0.02 and np.isclose(torn["omega_aspect"], 1.0, rtol=1e-6)
    for r in (ref, torn):
        assert r["fold_index"] > 0.85 and SE.seam_flags(r)["in_range"]
        assert min(r["fold_index_ctrl"]) > 0.85                             # internal null at the control angles
    # glued but folded at the cut: (|theta|, omega) and (cos theta, |sin theta|, omega) read "glued" (SR ~ 0.5) ...
    for Z in (C.emb_fold_abs(S), C.emb_fold_theta(S)):
        r = SE.seam_ratio(Z, th, om, groups=grp, n_boot=0)
        f = SE.seam_flags(r)
        assert r["ratio"] < 1.0 and f["verdict"] == "glued"
        assert r["fold_index"] < SE.FOLD_INDEX_MIN and f["folded_at_cut"] and not f["in_range"]   # ... flagged
    # both diagnostics are scale free and rotation invariant
    Q = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    moved = SE.seam_ratio(0.01 * C.emb_ref(S) @ Q.T + 2.0, th, om, groups=grp, n_boot=0)
    assert np.isclose(moved["omega_aspect"], ref["omega_aspect"], rtol=1e-6)
    assert np.isclose(moved["fold_index"], ref["fold_index"], rtol=1e-9)
    ci = SE.seam_flags(dict(ratio=4.0, status="ok", ci95=(3.0, 6.0), omega_aspect=1.0, fold_index=1.0))
    assert ci["ci_straddles"] and ci["in_range"] and ci["verdict"] == "glued"


def test_seam_ratio_counts_the_trajectories_at_the_cut(pend_full):
    S, E, grp = pend_full
    th, om = S[:, 0], S[:, 1]
    r = SE.seam_ratio(C.emb_torn(S), th, om, groups=grp, n_boot=0)
    k = r["n_cut_groups"]
    crossing = len(np.unique(grp[np.abs(SE.wrap_angle(th - np.pi)) < 0.12]))
    assert r["status"] == "ok" and 10 <= k <= crossing
    assert SE.seam_ratio(C.emb_torn(S), th, om, groups=grp, n_boot=0, min_cut_groups=k)["status"] == "ok"
    few = SE.seam_ratio(C.emb_torn(S), th, om, groups=grp, n_boot=0, min_cut_groups=k + 1)
    assert few["status"] == "insufficient" and SE.classify_seam(few) == "insufficient"
    assert SE.seam_ratio(C.emb_torn(S), th, om, n_boot=0)["n_cut_groups"] is None      # no groups: no guard


def test_partial_fold_and_fold_abs_controls(pend_full):
    S, E, _ = pend_full
    th = S[:, 0]
    assert np.allclose(C.emb_partial_fold(S, E, E.min() - 1.0), C.emb_ref(S))          # nothing folded
    assert np.allclose(C.emb_partial_fold(S, E, E.max() + 1.0), C.emb_fold_theta(S))   # everything folded
    r2 = [G.theta_decodability(C.emb_partial_fold(S, E, np.quantile(E, q)), th, seed=0)["r2_min"]
          for q in (0.0, 0.2, 1.0)]
    assert r2[0] > 0.99 > r2[1] > r2[2] and r2[2] < 0.1                                # graded, monotone
    P = np.array([[np.pi - 1e-3, 0.5], [-np.pi + 1e-3, 0.5], [0.3, 0.5], [-0.3, 0.5]])
    Zf = C.emb_fold_abs(P)
    assert np.linalg.norm(Zf[0] - Zf[1]) < 1e-9 and np.linalg.norm(Zf[2] - Zf[3]) < 1e-12   # glued and folded
