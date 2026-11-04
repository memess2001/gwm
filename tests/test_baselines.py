"""Baseline world models: size ladder, shapes, data alignment (against the zoo dynamics), training, checkpoints,
determinism, ledger, jepa-sym / jepa-sg gradient paths, the recon open-loop objective, the gate quantities of
metrics.json, split / normalisation hygiene, ground-truth guards, host-side dtype handling (MPS), whole-episode
open-loop windows (cell 6), the CPU benchmark script."""
import ast
import importlib.util
import json
import math
import shutil
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import zoo
from baselines import ledger
from baselines.data import Normalizer, WindowDataset, episode_K, frames_to_tensor, make_splits
from baselines.encode import encode, load
from baselines.mlp_wm import (SIZE_TOL, SIZES, VARIANTS, WMConfig, WorldModel, build_wm, canonical_variant,
                              count_params, describe, effective_rank, is_jepa, sliding_windows, solve_widths,
                              train_flops_per_sample, vicreg_terms)
from baselines.train import TrainConfig, main, run, save_checkpoint
from zoo.base import SPLIT_SEEDS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    """Tiny models: one thread is fastest and robust to other jobs loading the CPU (no OpenMP oversubscription)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)

# (world, world kwargs, obs mode): every obs mode the baselines must handle
CASES = [("pendulum", {}, "angle_cs"), ("pendulum", {}, "angle_raw"), ("pendulum", {}, "pixels64"),
         ("toggle_grid", {}, "grid"), ("wright_fisher", {}, "freq"), ("pendulum", {"policy": "random"}, "angle_cs")]
LADDER_CASES = CASES[:5]


def spec_of(world, kw, mode):
    return zoo.make(world, **kw).spec(mode)


def tiny_data(world="pendulum", kw=None, mode="angle_cs", n_train=32, n_val=8, horizon=30):
    return make_splits(world, mode, horizon, {"train": n_train, "val": n_val}, kw or {})


def tiny_cfg(**over):
    base = dict(steps=60, batch=32, log_every=10, eval_every=30, n_eval=64, n_eval_final=64, quiet=True,
                threads=1, horizon=40, rollout_k=(1, 8, 32), latent_k=(1, 8))
    base.update(over)
    return TrainConfig(**base)


# ------------------------------------------------------------------ size ladder
@pytest.mark.parametrize("world,kw,mode", LADDER_CASES)
def test_param_ladder_targets(world, kw, mode):
    spec = spec_of(world, kw, mode)
    prev = 0
    for size, target in SIZES.items():
        sol = solve_widths(spec, size)
        assert abs(sol["params"] / target - 1) <= SIZE_TOL, (mode, size, sol)
        assert sol["params"] > prev
        prev = sol["params"]
        if size in ("S", "M"):                 # real build agrees with the meta-device count
            rec, jep = build_wm(spec, "recon", size), build_wm(spec, "jepa", size)
            jsg = build_wm(spec, "jepa-sg", size)
            assert count_params(rec) == sol["params"]
            assert (rec.cfg.width, rec.cfg.channels) == (jep.cfg.width, jep.cfg.channels)
            assert count_params(jsg) == count_params(jep) and describe(jsg)["params_recon"] == count_params(rec)
            d = describe(rec)
            assert d["params_recon"] == count_params(rec) and d["params_jepa"] == count_params(jep)
            assert count_params(jep) == count_params(rec) - d["params_decoder"]
            assert count_params(jep.encoder) == count_params(rec.encoder)
            assert count_params(jep.transition) == count_params(rec.transition)


def test_param_ladder_cell6_widest_grid():
    """Cell 6 trains with history 1 on grids of up to 3 doors (the widest input): every size still hits its target,
    and all three variants are built with one set of widths (sizes are matched on the recon model)."""
    spec = spec_of("toggle_grid", {"n_doors": 3, "switch_visible": False}, "grid")
    for size in SIZES:
        sol = solve_widths(spec, size, history=1)
        assert abs(sol["rel_err"]) <= SIZE_TOL, (size, sol)
        with torch.device("meta"):
            ms = {v: build_wm(spec, v, size, history=1, K=63) for v in VARIANTS}
        assert {(m.cfg.width, m.cfg.channels) for m in ms.values()} == {(sol["width"], 0)}
        assert count_params(ms["recon"]) == sol["params"]
        assert len({count_params(m.encoder) for m in ms.values()}) == 1
        assert len({count_params(m.transition) for m in ms.values()}) == 1


def test_pixel_widths_scale_jointly():
    spec = spec_of("pendulum", {}, "pixels64")
    sols = [solve_widths(spec, s) for s in SIZES]
    for a, b in zip(sols, sols[1:]):
        assert b["channels"] > a["channels"] and b["width"] > a["width"]
    for s in sols:
        assert abs(s["width"] - 4 * s["channels"]) <= 3


def test_flops_reported():
    f_vec = train_flops_per_sample(build_wm(spec_of("pendulum", {}, "angle_cs"), "recon", "S").cfg, batch=2)
    f_pix = train_flops_per_sample(build_wm(spec_of("pendulum", {}, "pixels64"), "recon", "S").cfg, batch=2)
    assert 0 < f_vec["train_flops_per_sample"] < f_pix["train_flops_per_sample"]
    assert f_pix["fwd_flops_encoder"] > 0 and f_pix["fwd_flops_decoder"] > 0
    f_j = train_flops_per_sample(build_wm(spec_of("pendulum", {}, "pixels64"), "jepa", "S").cfg, batch=2)
    assert f_j["fwd_flops_decoder"] is None


# ------------------------------------------------------------------ shapes
SHAPE_CASES = [(c, v, "mse") for c in CASES for v in VARIANTS] + [(CASES[1], "recon", "circular")]


@pytest.mark.parametrize("case,variant,loss", SHAPE_CASES)
def test_forward_backward_shapes(case, variant, loss):
    world, kw, mode = case
    spec, sp = tiny_data(world, kw, mode, n_train=6, n_val=0, horizon=16)
    torch.manual_seed(0)
    m = build_wm(spec, variant, None, width=16, channels=2, loss=loss)
    ds = WindowDataset(sp["train"], 2, 8, spec["action"])
    b = ds.sample_batch(np.random.default_rng(0), 5)
    B, obs_shape, d = 5, tuple(spec["obs_shape"]), m.cfg.latent_dim
    assert b["frames"].shape == (B, 10) + obs_shape and b["frames"].dtype == torch.float32
    assert b["hist"].shape == (B, 2) + obs_shape and b["future"].shape == (B, 8) + obs_shape
    A = {"none": 0, "discrete": 4, "continuous": 1}[spec["action"]["type"]]
    assert (b["actions"] is None) == (A == 0)
    if A:
        assert b["actions"].shape == (B, 8, A)
    win = m.windows(b["frames"])
    assert win.shape == (B, 9, 2) + obs_shape
    assert all(torch.equal(win[:, j], b["frames"][:, j:j + 2]) for j in range(9))
    assert torch.equal(sliding_windows(b["frames"], 2, use_view=False), win)        # the MPS code path
    z = m.encode(b["hist"])
    assert z.shape == (B, d)
    Z = m.rollout(z, b["actions"], 8)
    assert Z.shape == (B, 9, d) and torch.equal(Z[:, 0], z)
    if variant == "recon":
        assert m.decode(z).shape == (B,) + obs_shape and torch.isfinite(m.decode(z)).all()
    else:
        assert m.decoder is None
        with pytest.raises(RuntimeError):
            m.decode(z)
    loss_v, logs = m.loss(b["frames"], b["actions"])
    assert loss_v.ndim == 0 and torch.isfinite(loss_v)
    loss_v.backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert float(m.transition[-1].weight.grad.abs().sum()) > 0          # zero-init last layer still learns
    assert float(sum(p.grad.abs().sum() for p in m.encoder.parameters())) > 0
    if variant != "recon":
        assert logs["latent_std"].shape == (d,) and {"inv", "var", "cov"} <= set(logs)


def test_channels_last_matches_default_layout():
    spec = spec_of("pendulum", {}, "pixels64")
    torch.manual_seed(0)
    m = build_wm(spec, "recon", None, width=16, channels=4)
    frames = torch.rand(6, 10, 64, 64)
    with torch.no_grad():
        ref = m.loss(frames)[0]
        m.set_channels_last(True)
        assert m.encoder.conv[0].weight.is_contiguous(memory_format=torch.channels_last)
        assert torch.allclose(m.loss(frames)[0], ref, rtol=1e-4)
        assert m.decode(torch.zeros(2, 8)).shape == (2, 64, 64)


def test_transition_starts_as_identity():
    m = build_wm(spec_of("toggle_grid", {}, "grid"), "recon", "S")
    z, a = torch.randn(3, 8), torch.eye(4)[[0, 1, 2]]
    assert torch.equal(m.step(z, a), z)
    with pytest.raises(ValueError):
        m.step(z)                                                        # discrete-action world needs actions


def test_invalid_configs():
    sp_raw, sp_pix = spec_of("pendulum", {}, "angle_raw"), spec_of("pendulum", {}, "pixels64")
    with pytest.raises(ValueError):
        build_wm(sp_raw, "jepa", "S", loss="circular")
    with pytest.raises(ValueError):
        build_wm(sp_pix, "recon", "S", loss="circular")
    with pytest.raises(ValueError):
        build_wm(sp_raw, "rssm", "S")
    with pytest.raises(ValueError):
        build_wm(sp_raw, "jepa-sg", "S", loss="circular")
    with pytest.raises(ValueError):
        build_wm(sp_pix, "recon", "S", pixel_out="tanh")


def test_circular_loss_is_periodic():
    m = build_wm(spec_of("pendulum", {}, "angle_raw"), "recon", None, width=8, loss="circular")
    p, t = torch.rand(50, 1) * 6 - 3, torch.rand(50, 1) * 6 - 3
    e = m.recon_error(p, t)
    assert torch.allclose(e, m.recon_error(p + 2 * math.pi, t), atol=1e-5)
    assert torch.allclose(e, 1 - torch.cos(p - t))
    mse = build_wm(spec_of("pendulum", {}, "angle_raw"), "recon", None, width=8)
    assert not torch.allclose(mse.recon_error(p + 2 * math.pi, t), mse.recon_error(p, t))   # the wrap matters


# ------------------------------------------------------------------ data
def test_window_alignment_with_actions():
    """a_t maps state_t -> state_{t+1}: windows, futures and actions line up with the zoo convention."""
    w = zoo.make("toggle_grid")
    tr = w.sample(seed=5, horizon=30, n_traj=7, obs_mode="grid")
    h, K = 3, 5
    ds = WindowDataset(tr, h, K, w.spec("grid")["action"])
    assert len(ds) == 7 * (30 - K - h + 2)
    idx = np.arange(len(ds))
    b = ds.get_batch(idx)
    n, t = b["index"]
    assert t.min() == h - 1 and t.max() == 30 - K
    for i in range(0, len(idx), 13):
        ni, ti = n[i], t[i]
        assert np.array_equal(b["hist"][i].numpy(), tr.obs[ni, ti - h + 1:ti + 1])
        assert np.array_equal(b["future"][i].numpy(), tr.obs[ni, ti + 1:ti + K + 1])
        assert np.array_equal(b["actions"][i].numpy().argmax(-1), tr.actions[ni, ti:ti + K])
        assert np.array_equal(b["actions"][i].sum(-1).numpy(), np.ones(K))          # one-hot
        for j in range(K):                                                            # zoo convention check
            assert np.array_equal(w.step(tr.state[ni, ti + j], tr.actions[ni, ti + j]), tr.state[ni, ti + j + 1])
    item = ds[17]
    assert item["hist"].shape == (h, 4, 7, 13) and item["actions"].shape == (K, 4)


def test_window_alignment_matches_zoo_dynamics_pendulum():
    """Continuous torques: re-simulating each window from its ground-truth start state with the window's actions
    (zoo dynamics) reproduces the states behind its frames; the same check with the actions shifted by one step
    fails, so the test would catch an off-by-one (hold=1: a new torque every step)."""
    w = zoo.make("pendulum", policy="random", hold=1)
    tr = w.sample(seed=11, horizon=24, n_traj=4, obs_mode="angle_cs")
    act, shifted = w.spec("angle_cs")["action"], []
    for h, K, anchors in ((2, 6, "all"), (1, None, "episode")):
        ds = WindowDataset(tr, h, K, act, anchors=anchors)
        b = ds.get_batch(np.arange(len(ds)))
        n, t = b["index"]
        for i in range(0, len(ds), 3):
            ni, ti, a = n[i], t[i], b["actions"][i].numpy()             # a_t .. a_{t+K-1}, raw torques [K, 1]
            truth = tr.state[ni, ti:ti + ds.K + 1]
            sim = w.rollout(tr.state[ni, ti], ds.K, a[None])[0]         # states t .. t+K under the zoo dynamics
            assert np.allclose(np.exp(1j * sim[:, 0]), np.exp(1j * truth[:, 0]), atol=1e-9)
            assert np.allclose(sim[:, 1], truth[:, 1], atol=1e-9)
            assert np.allclose(w.observe(sim[1:], "angle_cs"), b["future"][i].numpy(), atol=1e-6)
            assert np.allclose(w.observe(sim[0], "angle_cs"), b["hist"][i, -1].numpy(), atol=1e-6)
            if ti + ds.K < tr.horizon:                                    # a_{t+1} .. a_{t+K}: off by one
                wrong = w.rollout(tr.state[ni, ti], ds.K, tr.actions[ni, ti + 1:ti + ds.K + 1][None])[0]
                shifted.append(np.abs(wrong[:, 1] - truth[:, 1]).max())
        if anchors == "episode":
            assert (t == 0).all() and ds.K == 24 and b["frames"].shape[1] == 25
    assert len(shifted) > 5 and min(shifted) > 1e-3


def test_continuous_actions_raw_and_start_anchors():
    w = zoo.make("pendulum", policy="random")
    tr = w.sample(seed=1, horizon=20, n_traj=4, obs_mode="angle_cs")
    ds = WindowDataset(tr, 2, 19, w.spec("angle_cs")["action"], anchors="start")
    assert len(ds) == 4
    b = ds.get_batch(np.arange(4))
    assert np.array_equal(b["index"][1], np.ones(4, dtype=int))                       # t = h-1 only
    assert np.allclose(b["actions"].numpy(), tr.actions[:, 1:20])
    with pytest.raises(ValueError):
        WindowDataset(tr, 2, 20, w.spec("angle_cs")["action"])                         # horizon too short


def test_pixels_stay_uint8():
    spec, sp = tiny_data("pendulum", {}, "pixels64", n_train=3, n_val=0, horizon=12)
    tr = sp["train"]
    ds = WindowDataset(tr, 2, 4, spec["action"])
    assert ds.obs is tr.obs and ds.obs.dtype == np.uint8                               # no float copy kept
    b = ds.get_batch(np.array([0, 5]))
    n, t = b["index"]
    assert b["frames"].dtype == torch.float32 and 0.0 <= float(b["frames"].min()) <= float(b["frames"].max()) <= 1.0
    assert np.allclose(b["frames"][1].numpy(), tr.obs[n[1], t[1] - 1:t[1] + 5] / 255.0)
    with pytest.raises(ValueError):
        Normalizer.fit(tr)


def test_standardisation_uses_train_statistics():
    spec, sp = tiny_data("pendulum", {}, "angle_raw", n_train=20, n_val=10, horizon=30)
    nz = Normalizer.fit(sp["train"])
    x = sp["train"].obs.reshape(-1, 1).astype(np.float64)
    assert np.allclose(nz.mean, x.mean(0), atol=1e-6) and np.allclose(nz.std, x.std(0), atol=1e-6)
    val_own = Normalizer.fit(sp["val"])
    assert not np.allclose(val_own.mean, nz.mean)
    ds = WindowDataset(sp["val"], 2, 3, spec["action"], nz)
    b = ds.get_batch(np.array([4]))
    n, t = b["index"]
    raw = sp["val"].obs[n[0], t[0] - 1:t[0] + 4]
    assert np.allclose(b["frames"][0].numpy(), (raw - nz.mean) / nz.std, atol=1e-6)
    res = run(tiny_cfg(obs="angle_raw", steps=2, standardize=True, loss="circular"), data=(spec, sp))
    assert np.allclose(res["config"]["data"]["normalizer"]["mean"], nz.mean)
    assert res["model"].cfg.angle_scale == pytest.approx(float(nz.std[0]))


def test_frames_to_tensor_casts_on_the_host(monkeypatch):
    """Only uint8 / float32 arrays ever become tensors that are moved to the device (MPS has no float64): float64 and
    integer inputs are cast to float32 first, and a Normalizer always holds float32."""
    seen, real = [], torch.from_numpy
    monkeypatch.setattr(torch, "from_numpy", lambda a: seen.append(a.dtype) or real(a))
    x64 = np.random.default_rng(0).normal(size=(3, 5, 2))
    nz = Normalizer(mean=np.zeros(2), std=np.full(2, 2.0))
    assert nz.mean.dtype == np.float32 and nz.std.dtype == np.float32
    t = frames_to_tensor(x64, "cpu", nz)
    assert t.dtype == torch.float32 and np.allclose(t.numpy(), x64 / 2.0, atol=1e-6)
    u8 = frames_to_tensor(np.full((2, 4, 4), 255, np.uint8), "cpu")
    assert u8.dtype == torch.float32 and float(u8.max()) == 1.0
    assert frames_to_tensor(np.arange(6).reshape(2, 3), "cpu").dtype == torch.float32
    assert set(seen) <= {np.dtype(np.float32), np.dtype(np.uint8)}, seen


def test_val_split_never_affects_training():
    """Training reads the train split only: replacing the val split (other seeds) leaves the trained weights bitwise
    unchanged, with standardisation on (train statistics) - val is used for evaluation only, never for selection."""
    spec, sp = tiny_data("pendulum", {}, "angle_raw", n_train=24, n_val=8)
    _, other = make_splits("pendulum", "angle_raw", 30, {"val": 8}, index=7)
    assert not np.array_equal(other["val"].obs, sp["val"].obs)
    for variant in ("recon", "jepa-sg"):
        cfg = dict(obs="angle_raw", variant=variant, steps=12, standardize=True)
        a = run(tiny_cfg(**cfg), data=(spec, sp))
        b = run(tiny_cfg(**cfg), data=(spec, {"train": sp["train"], "val": other["val"]}))
        sa, sb = a["model"].state_dict(), b["model"].state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
        assert np.array_equal(a["normalizer"].mean, b["normalizer"].mean)
        assert a["metrics"]["final"] != b["metrics"]["final"]                       # the val metrics did change


def test_splits_use_disjoint_seeds():
    _, sp = make_splits("wright_fisher", "freq", 10, {"train": 3, "val": 2, "test": 2}, index=4)
    assert {s: t.meta["seed"] for s, t in sp.items()} == {s: SPLIT_SEEDS[s] + 4 for s in sp}
    assert not np.allclose(sp["train"].obs[:2], sp["val"].obs)


def test_pipeline_never_reads_ground_truth():
    """Training and encoding work with state / info removed: models only ever see obs and actions."""
    spec, sp = tiny_data("toggle_grid", {}, "grid", n_train=16, n_val=6, horizon=40)
    for tr in sp.values():
        tr.state, tr.info = None, {}
    res = run(tiny_cfg(world="toggle_grid", obs="grid", steps=3), data=(spec, sp))
    assert res["metrics"]["steps_done"] == 3
    forbidden_mods = {"ripser", "gudhi", "sklearn", "persim", "analysis"}
    for f in ("train.py", "data.py", "mlp_wm.py", "encode.py"):
        tree = ast.parse((ROOT / "baselines" / f).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("state", "info"), f"{f} reads ground truth .{node.attr}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                assert not any(str(n).split(".")[0] in forbidden_mods for n in names), f


# ------------------------------------------------------------------ losses / diagnostics
def test_vicreg_terms_and_effective_rank():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4096, 3, 8, generator=g) * 2.0
    v, c, std = vicreg_terms(z)
    assert float(v) == 0.0 and float(c) < 0.05 and std.shape == (3, 8)
    zc = torch.randn(4096, 1, 1, generator=g).expand(4096, 1, 8) * 0.5
    v2, c2, _ = vicreg_terms(zc)
    assert float(v2) == pytest.approx(0.5, abs=0.03) and float(c2) > 0.3
    assert effective_rank(z[:, 0]) > 7.8
    assert effective_rank(zc[:, 0]) == pytest.approx(1.0, abs=1e-3)


def _manual_recon_loss(m, frames, actions):
    """Independent re-implementation of the recon objective: explicit open-loop latent rollout from E(w_t)."""
    h = m.cfg.history
    K = frames.shape[1] - h
    zh = m.encode(frames[:, :h])
    rec, lat = ((m.decode(zh) - frames[:, h - 1]) ** 2).mean(), 0.0                 # j = 0: o_t itself
    for j in range(1, K + 1):
        zh = m.step(zh, None if actions is None else actions[:, j - 1])              # never re-encodes o_{t+j}
        rec = rec + ((m.decode(zh) - frames[:, h - 1 + j]) ** 2).mean()
        lat = lat + ((zh - m.encode(frames[:, j:j + h]).detach()) ** 2).mean()
    return rec + m.cfg.lambda_lat * lat


@pytest.mark.parametrize("world,kw,mode", [("pendulum", {"policy": "random"}, "angle_cs"), ("toggle_grid", {}, "grid"),
                                           ("pendulum", {}, "pixels64")])
def test_recon_loss_is_open_loop_over_j0_to_K(world, kw, mode):
    """recon = sum_{j=0..K} mse(D(F^j(E(w_t))), o_{t+j}) + lambda * sum_{j=1..K} mse(F^j(E(w_t)), sg(E(w_{t+j}))):
    same value and parameter gradients as a hand-written loop, and the frames after w_t reach the loss ONLY as
    targets (their gradient is exactly the MSE-target gradient of the open-loop predictions)."""
    spec, sp = tiny_data(world, kw, mode, n_train=6, n_val=0, horizon=16)
    for h, K in ((2, 5), (1, 7)):
        ds = WindowDataset(sp["train"], h, K, spec["action"])
        b = ds.sample_batch(np.random.default_rng(0), 8)
        torch.manual_seed(0)
        m = build_wm(spec, "recon", None, width=16, channels=2, history=h, K=K)
        with torch.no_grad():
            m.transition[-1].weight.normal_(0.0, 0.3)            # F != identity: open and closed loop now differ
        frames = b["frames"].clone().requires_grad_(True)
        loss, logs = m.loss(frames, b["actions"])
        ref = _manual_recon_loss(m, b["frames"], b["actions"])
        assert torch.allclose(loss, ref, rtol=1e-5)
        params = list(m.parameters())
        g_frames, *g_model = torch.autograd.grad(loss, [frames] + params)
        g_ref = torch.autograd.grad(ref, params)
        scale = max(float(g.abs().max()) for g in g_ref)
        assert all(torch.allclose(a, c, rtol=1e-4, atol=1e-6 * scale) for a, c in zip(g_model, g_ref))
        with torch.no_grad():
            Z = m.rollout(m.encode(b["frames"][:, :h]), b["actions"], K)                # from w_t only
            pred = torch.stack([m.decode(Z[:, j]) for j in range(K + 1)], 1)
            assert torch.allclose(logs["recon_j0"], ((pred[:, 0] - b["frames"][:, h - 1]) ** 2).mean(), rtol=1e-5)
        expect = -2.0 * (pred[:, 1:] - b["frames"][:, h:]) / b["frames"][:, 0].numel()
        assert torch.allclose(g_frames[:, h:], expect, rtol=1e-4, atol=1e-6 * float(expect.abs().max()))


@pytest.mark.parametrize("variant", ["recon", "jepa-sym", "jepa-sg"])
def test_tiny_training_reduces_loss(variant):
    """recon / jepa-sym: the loss halves within 120 steps. jepa-sg is not monotone at the start: the variance hinge
    first spreads the latents (std ~0.5 -> 1), which raises the ABSOLUTE invariance error, and with stop-gradient
    targets only the online branch can shrink it again; it is therefore checked against its peak over 300 steps
    and, scale-free, by the latent prediction beating the copy predictor z_{t+k} ~ z_t."""
    spec, sp = tiny_data()
    sg = variant == "jepa-sg"
    res = run(tiny_cfg(variant=variant, steps=300 if sg else 120), data=(spec, sp))
    tr = res["metrics"]["curves"]["train"]
    val = res["metrics"]["curves"]["val"]
    assert not res["metrics"]["diverged"]
    if sg:
        assert tr[-1]["loss"] < 0.5 * max(r["loss"] for r in tr), [r["loss"] for r in tr]
        assert val[-1]["loss"] < 0.5 * max(r["loss"] for r in val)
    else:
        assert tr[-1]["loss"] < 0.5 * tr[0]["loss"], [r["loss"] for r in tr]
        assert val[-1]["loss"] < val[0]["loss"]
    f = res["metrics"]["final"]
    if variant == "recon":
        assert set(f["rollout"]) == {"1", "8"} and f["rollout_k_skipped"] == [32]          # horizon 30 < 1 + 32
        assert f["val_recon"]["nmse"] is not None
    else:
        assert set(f["latent_pred_err"]) == {"1", "8"} and len(f["latent_std"]) == 8
        assert f["effective_rank"] >= 1.0
        assert "effective_rank" in tr[-1] and len(tr[-1]["latent_std"]) == 8
        assert f["latent_pred_err"]["1"] < f["latent_copy_err"]["1"]
        assert f["latent_pred_err"]["8"] < 0.6 * f["latent_copy_err"]["8"]


def test_final_metrics_match_manual_computation():
    """The gate quantities in metrics.json, recomputed by hand from the val trajectories: recon val nmse (TRAIN
    per-feature variance, raw units after un-standardising) and open-loop rollout errors from the episode start with
    the right torques; jepa latent prediction / copy errors, per-dim std and effective rank of z_t."""
    kw = {"policy": "random", "hold": 1}
    spec, sp = tiny_data("pendulum", kw, "angle_cs", n_train=16, n_val=5, horizon=20)
    val, h, T = sp["val"], 2, 20
    tv_train = sp["train"].obs.reshape(-1, 2).astype(np.float64).var(0).sum()
    a = torch.as_tensor(val.actions)                                                # [N, T, 1]
    for variant in ("recon", "jepa-sg"):
        res = run(tiny_cfg(world_kwargs=kw, variant=variant, steps=30, standardize=True, n_eval_final=10 ** 6,
                           rollout_k=(1, 8), latent_k=(1, 8)), data=(spec, sp))
        m, nz, f = res["model"].eval(), res["normalizer"], res["metrics"]["final"]
        X = torch.as_tensor((val.obs - nz.mean) / nz.std)                          # model units
        with torch.no_grad():
            Z = torch.stack([m.encode(X[:, t - h + 1:t + 1]) for t in range(h - 1, T + 1)], 1)    # Z[:, i] = z_{h-1+i}
            if variant == "recon":
                rec = torch.stack([m.decode(Z[:, i]) for i in range(Z.shape[1])], 1).numpy() * nz.std + nz.mean
                mse = ((rec[:, :-1] - val.obs[:, h - 1:-1]) ** 2).mean()           # windows with t <= T-1 (K = 1)
                assert f["val_recon"]["mse"] == pytest.approx(mse, rel=1e-4)
                assert f["val_recon"]["nmse"] == pytest.approx(mse * 2 / tv_train, rel=1e-4)
                for k in (1, 8):                                                   # from w_{h-1}, torques a_{h-1}..
                    zk = m.rollout(Z[:, 0], a[:, h - 1:h - 1 + k], k)[:, k]
                    ok = m.decode(zk).numpy() * nz.std + nz.mean
                    assert f["rollout"][str(k)]["mse"] == pytest.approx(((ok - val.obs[:, h - 1 + k]) ** 2).mean(),
                                                                        rel=1e-4)
            else:
                anchors = range(0, T - 8 - h + 2)                                  # t = h-1 .. T-kmax
                z0 = torch.cat([Z[:, i] for i in anchors])
                tv = float(z0.double().var(0).sum())
                for k in (1, 8):
                    err = torch.cat([((m.rollout(Z[:, i], a[:, h - 1 + i:h - 1 + i + k], k)[:, k] - Z[:, i + k]) ** 2)
                                     .sum(-1) for i in anchors])
                    cp = torch.cat([((Z[:, i] - Z[:, i + k]) ** 2).sum(-1) for i in anchors])
                    assert f["latent_pred_err"][str(k)] == pytest.approx(float(err.mean()) / tv, rel=1e-4)
                    assert f["latent_copy_err"][str(k)] == pytest.approx(float(cp.mean()) / tv, rel=1e-4)
                assert f["latent_std"] == pytest.approx(z0.double().std(0).tolist(), rel=1e-4)
                assert f["effective_rank"] == pytest.approx(effective_rank(z0), rel=1e-4)


def test_tiny_training_with_discrete_actions_reduces_loss():
    spec, sp = tiny_data("toggle_grid", {}, "grid", n_train=32, n_val=8, horizon=40)
    res = run(tiny_cfg(world="toggle_grid", obs="grid", steps=100), data=(spec, sp))
    tr = res["metrics"]["curves"]["train"]
    assert tr[-1]["loss"] < 0.5 * tr[0]["loss"]


# ------------------------------------------------------------------ jepa-sym / jepa-sg
def test_variant_names_and_alias():
    assert VARIANTS == ("recon", "jepa-sym", "jepa-sg")
    assert canonical_variant("jepa") == "jepa-sym" and canonical_variant("jepa-sg") == "jepa-sg"
    assert is_jepa("jepa") and is_jepa("jepa-sg") and not is_jepa("recon")
    spec = spec_of("pendulum", {}, "angle_cs")
    assert build_wm(spec, "jepa", "S").cfg.variant == "jepa-sym"
    assert TrainConfig(variant="jepa").variant == "jepa-sym"
    old = build_wm(spec, "jepa-sym", None, width=8).cfg.to_dict()
    old["variant"] = "jepa"                                              # a config written before the split
    assert WMConfig.from_dict(old).variant == "jepa-sym"
    with pytest.raises(ValueError):
        canonical_variant("jepa-ema")


def _manual_jepa_loss(m, frames, actions, stop_grad):
    """Independent re-implementation of the two jepa objectives (explicit loops, explicit detach)."""
    cfg, h = m.cfg, m.cfg.history
    K = frames.shape[1] - h
    z = [m.encode(frames[:, j:j + h]) for j in range(K + 1)]            # E(w_{t+j}), j = 0..K
    zh, inv = z[0], 0.0
    for j in range(1, K + 1):
        zh = m.step(zh, None if actions is None else actions[:, j - 1])
        tgt = z[j].detach() if stop_grad else z[j]
        inv = inv + ((zh - tgt) ** 2).mean()

    def var_cov(x):                                                     # x [B, d]
        xc = x - x.mean(0)
        v = xc.pow(2).sum(0) / (len(x) - 1)
        c = xc.T @ xc / (len(x) - 1)
        off = c - torch.diag(torch.diag(c))
        return torch.relu(cfg.vic_gamma - torch.sqrt(v + cfg.vic_eps)).mean(), off.pow(2).sum() / x.shape[1]

    slices = [z[0]] if stop_grad else z
    vc = [var_cov(x) for x in slices]
    l_var = sum(v for v, _ in vc) / len(vc)
    l_cov = sum(c for _, c in vc) / len(vc)
    wi, wv, wc = cfg.vicreg
    return wi * inv + wv * l_var + wc * l_cov


@pytest.mark.parametrize("world,kw,mode", [("pendulum", {}, "angle_cs"), ("toggle_grid", {}, "grid"),
                                           ("pendulum", {}, "pixels64")])
def test_jepa_gradient_paths(world, kw, mode):
    """jepa-sg: no gradient reaches the prediction targets; jepa-sym: it does. Both match a hand-written loss."""
    spec, sp = tiny_data(world, kw, mode, n_train=6, n_val=0, horizon=16)
    ds = WindowDataset(sp["train"], 2, 5, spec["action"])
    b = ds.sample_batch(np.random.default_rng(1), 12)
    h = 2
    for variant, sg in (("jepa-sg", True), ("jepa-sym", False)):
        torch.manual_seed(0)
        m = build_wm(spec, variant, None, width=16, channels=2)
        frames = b["frames"].clone().requires_grad_(True)
        loss, _ = m.loss(frames, b["actions"])
        g_frames, = torch.autograd.grad(loss, frames, retain_graph=True)
        # frames h.. only enter through the target windows w_{t+1..t+K} (window 0 = frames 0..h-1)
        target_only = g_frames[:, h:].abs().sum()
        assert float(g_frames[:, :h].abs().sum()) > 0
        if sg:
            assert float(target_only) == 0.0
        else:
            assert float(target_only) > 0
        params = [p for p in m.parameters()]
        g_model = torch.autograd.grad(loss, params)
        ref = _manual_jepa_loss(m, b["frames"], b["actions"], stop_grad=sg)
        assert torch.allclose(loss, ref, rtol=1e-5, atol=1e-6)
        g_ref = torch.autograd.grad(ref, params)
        # absolute tolerance on the scale of the largest gradient: some entries are analytically zero (e.g. the last
        # encoder bias of jepa-sym at init: F = identity and VICReg is centred) and are float noise in both paths
        scale = max(float(g.abs().max()) for g in g_ref)
        for gm, gr in zip(g_model, g_ref):
            assert torch.allclose(gm, gr, rtol=1e-4, atol=1e-5 * scale)
    # the stop-gradient changes the gradient, not the invariance value (same weights, VICReg terms off)
    losses, grads = {}, {}
    for variant in ("jepa-sg", "jepa-sym"):
        torch.manual_seed(0)
        m = build_wm(spec, variant, None, width=16, channels=2, vicreg=(1.0, 0.0, 0.0))
        loss, _ = m.loss(b["frames"], b["actions"])
        loss.backward()
        losses[variant], grads[variant] = float(loss.detach()), m.encoder[0].weight.grad.clone() if mode != "pixels64" \
            else m.encoder.conv[0].weight.grad.clone()
    assert losses["jepa-sg"] == pytest.approx(losses["jepa-sym"], rel=1e-6)
    assert not torch.allclose(grads["jepa-sg"], grads["jepa-sym"])


def test_jepa_sg_targets_cost_no_backward():
    spec = spec_of("pendulum", {}, "pixels64")
    f = {v: train_flops_per_sample(build_wm(spec, v, "S").cfg, batch=2)["train_flops_per_sample"]
         for v in ("jepa-sym", "jepa-sg")}
    assert f["jepa-sg"] < 0.6 * f["jepa-sym"]


def test_jepa_logs_collapse_diagnostics():
    spec, sp = tiny_data()
    for variant in ("jepa-sym", "jepa-sg"):
        res = run(tiny_cfg(variant=variant, steps=20, log_every=10, eval_every=20), data=(spec, sp))
        rows = res["metrics"]["curves"]["train"]
        assert all(len(r["latent_std"]) == 8 and r["effective_rank"] >= 1.0 for r in rows)
        f = res["metrics"]["final"]
        assert len(f["latent_std"]) == 8 and 1.0 <= f["effective_rank"] <= 8.0 and f["latent_std_min"] > 0
        assert res["config"]["model"]["variant"] == variant


# ------------------------------------------------------------------ pixel decoder output
def test_pixel_decoder_output_modes(tmp_path):
    spec = spec_of("pendulum", {}, "pixels64")
    torch.manual_seed(0)
    lin = build_wm(spec, "recon", None, width=8, channels=2)
    assert lin.cfg.pixel_out == "linear"
    torch.manual_seed(0)
    sig = build_wm(spec, "recon", None, width=8, channels=2, pixel_out="sigmoid")
    z = torch.randn(4, 8) * 3
    with torch.no_grad():
        y_lin, y_sig = lin.decode(z), sig.decode(z)
    assert torch.allclose(torch.sigmoid(y_lin), y_sig, atol=1e-6)      # same weights, only the squashing differs
    assert float(y_sig.min()) >= 0 and float(y_sig.max()) <= 1
    # a checkpoint written before the option existed (no "pixel_out" key) loads with the sigmoid it was trained with
    save_checkpoint(tmp_path / "model.pt", sig, None, {}, 0)
    ck = torch.load(tmp_path / "model.pt", weights_only=True)
    del ck["model_config"]["pixel_out"]
    torch.save(ck, tmp_path / "model.pt")
    old, _, _ = load(tmp_path)
    assert old.cfg.pixel_out == "sigmoid"
    with torch.no_grad():
        assert torch.allclose(old.decode(z), y_sig, atol=1e-6)


def test_pixel_init_latents_depend_on_input():
    """Signal-preserving init: at initialisation pixel latents vary across inputs (the torch default gave ~0.01)."""
    spec, sp = tiny_data("pendulum", {}, "pixels64", n_train=20, n_val=0, horizon=30)
    b = WindowDataset(sp["train"], 2, 1, spec["action"]).sample_batch(np.random.default_rng(0), 64)
    torch.manual_seed(0)
    m = build_wm(spec, "recon", "S")
    with torch.no_grad():
        z = m.encode(b["hist"])
        y = m.decode(z)
    assert float(z.std(0).min()) > 0.1
    assert float(y.std(0).mean()) > 1e-3


# ------------------------------------------------------------------ whole-episode open-loop windows (cell 6)
def test_episode_windows_cover_whole_episode():
    w = zoo.make("toggle_grid", switch_visible=False)
    tr = w.sample(seed=2, horizon=20, n_traj=5, obs_mode="grid")
    act = w.spec("grid")["action"]
    ds = WindowDataset(tr, 1, None, act, anchors="episode")
    assert ds.K == 20 == episode_K(20, 1) and len(ds) == 5
    b = ds.get_batch(np.arange(5))
    assert np.array_equal(b["index"][1], np.zeros(5, dtype=int))                    # starts at t = 0
    assert b["frames"].shape == (5, 21, 4, 7, 13) and np.array_equal(b["frames"].numpy(), tr.obs)
    assert b["actions"].shape == (5, 20, 4) and np.array_equal(b["actions"].numpy().argmax(-1), tr.actions)
    assert np.array_equal(b["actions"].sum(-1).numpy(), np.ones((5, 20)))          # one-hot
    assert float(b["frames"][:, :, 2].abs().sum()) == 0.0                          # switch markers hidden
    ds2 = WindowDataset(tr, 2, None, act, anchors="episode")
    assert ds2.K == 19 and np.array_equal(ds2.get_batch(np.arange(5))["hist"].numpy(), tr.obs[:, :2])
    assert WindowDataset(tr, 1, 20, act, anchors="episode").K == 20                 # explicit full K is fine
    with pytest.raises(ValueError):
        WindowDataset(tr, 1, 8, act, anchors="episode")                              # not the full episode
    with pytest.raises(ValueError):
        WindowDataset(tr, 1, None, act)                                              # K only optional for episode
    with pytest.raises(ValueError):
        WindowDataset(tr, 1, 4, act, anchors="middle")


def test_episode_train_config():
    assert TrainConfig(anchors="episode", history=1, horizon=64).K == 64
    assert TrainConfig(anchors="episode", history=1, horizon=63).K == 63
    assert TrainConfig(anchors="episode", history=2, horizon=64, K=63).K == 63
    assert TrainConfig().K == 8 and TrainConfig(anchors="start", K=12).K == 12
    with pytest.raises(ValueError):
        TrainConfig(anchors="episode", history=1, horizon=64, K=8)
    with pytest.raises(ValueError):
        TrainConfig(anchors="sometimes")
    spec, sp = tiny_data("toggle_grid", {}, "grid", n_train=4, n_val=2, horizon=20)
    with pytest.raises(ValueError):                                                  # data horizon != cfg.horizon
        run(tiny_cfg(world="toggle_grid", obs="grid", anchors="episode", history=1, horizon=24, steps=1),
            data=(spec, sp))


@pytest.mark.parametrize("kw", [{}, {"switch_visible": False}, {"switches_enabled": False},
                                {"switch_visible": False, "policy": "loops", "n_doors": 2}])
def test_episode_training_toggle_grid(kw):
    """history 1, open loop from o_0 over the whole episode, discrete one-hot actions: the loss goes down."""
    H = 24
    spec, sp = tiny_data("toggle_grid", kw, "grid", n_train=48, n_val=8, horizon=H)
    res = run(tiny_cfg(world="toggle_grid", obs="grid", world_kwargs=kw, anchors="episode", history=1,
                       horizon=H, steps=80, batch=16, log_every=10, eval_every=40, lambda_lat=0.0),
              data=(spec, sp))
    m, met = res["model"], res["metrics"]
    assert m.cfg.K == H and m.cfg.history == 1 and m.cfg.action_dim == 4
    assert res["config"]["data"]["windows"] == {"train": 48, "val": 8}
    tr = met["curves"]["train"]
    assert not met["diverged"] and tr[-1]["loss"] < 0.5 * tr[0]["loss"], [r["loss"] for r in tr]
    assert str(H) in met["final"]["rollout"] and "8" in met["final"]["rollout"]     # k = K added for episodes
    assert res["config"]["data"]["world_params"]["switch_visible"] == kw.get("switch_visible", True)


def test_episode_cli(tmp_path):
    out = tmp_path / "ep"
    code = main(["--world", "toggle_grid", "--obs", "grid", "--variant", "recon", "--size", "S", "--steps", "3",
                 "--anchors", "episode", "--history", "1", "--horizon", "16", "--n-train", "6", "--n-val", "3",
                 "--log-every", "3", "--eval-every", "3", "--threads", "1", "--quiet", "--lambda-lat", "0",
                 "--world-kwargs", '{"switch_visible": false, "switches_enabled": false}', "--out", str(out)])
    assert code == 0
    cfgj = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfgj["train"]["K"] == 16 and cfgj["model"]["K"] == 16 and cfgj["train"]["anchors"] == "episode"
    assert cfgj["data"]["world_params"]["switches_enabled"] is False
    code = main(["--world", "pendulum", "--obs", "angle_cs", "--variant", "jepa-sg", "--steps", "2",
                 "--n-train", "6", "--n-val", "3", "--horizon", "20", "--log-every", "2", "--eval-every", "2",
                 "--threads", "1", "--quiet", "--out", str(tmp_path / "sg")])
    assert code == 0
    assert json.loads((tmp_path / "sg" / "config.json").read_text(encoding="utf-8"))["model"]["variant"] == "jepa-sg"


# ------------------------------------------------------------------ CPU benchmark script (logic only, no timing)
def _bench_module():
    spec = importlib.util.spec_from_file_location("bench_baselines", ROOT / "scripts" / "bench_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_quiet_machine_protocol():
    bench = _bench_module()
    t, sleeps = [0.0], []

    def fake(loads, heavies):
        it_l, it_h = iter(loads), iter(heavies)

        def sleep(s):
            sleeps.append(s)
            t[0] += s
        return dict(load_fn=lambda: next(it_l), heavy_fn=lambda: next(it_h), sleep_fn=sleep, clock=lambda: t[0],
                    log=lambda *a: None)

    py = [{"pid": 7, "cpu": 2.0, "cmd": "python -m baselines.train", "python": True}]
    other = [{"pid": 8, "cpu": 1.0, "cmd": "node", "python": False}]
    r = bench.wait_for_quiet_machine(**fake([0.3, 0.3], [[]]))                         # quiet at once
    assert r["quiet"] and r["waited_sec"] == 0 and r["polls"] == 0 and sleeps == []
    r = bench.wait_for_quiet_machine(**fake([3.0] + [3.0, 2.5, 1.2], [py, py, []]))    # heavy python, then gone
    assert r["quiet"] and r["polls"] == 2 and sleeps == [60.0, 60.0] and r["waited_sec"] == 120
    assert r["load_start"] == 3.0 and r["load_end"] == 1.2 and r["heavy_seen"][0]["pid"] == 7
    sleeps.clear(); t[0] = 0.0
    r = bench.wait_for_quiet_machine(**fake([1.0] + [1.0] * 40, [other] * 40))         # non-python, low load: go
    assert r["quiet"] and r["polls"] == 0
    sleeps.clear(); t[0] = 0.0
    r = bench.wait_for_quiet_machine(max_wait=1800, **fake([4.0] * 50, [py] * 50))     # never quiet: give up
    assert not r["quiet"] and r["waited_sec"] >= 1800 and len(sleeps) == 30
    hp = bench.heavy_processes(interval=0.05)
    assert isinstance(hp, list) and all(h["pid"] != __import__("os").getpid() for h in hp)


def test_bench_plan_and_report(tmp_path):
    bench = _bench_module()
    groups = bench.plan_groups()
    n = lambda cell: sum(len(g["runs"]) * g["seeds"] for g in groups if g["cell"] == cell)
    assert n("6") == 90                                                                  # cells/6/prereg.md section 7
    assert sum(len(g["runs"]) * 3 for g in groups if g["name"].startswith("angle 主实验")) == 84   # 7 x 4 x 3
    assert sum(len(g["runs"]) * 3 for g in groups if g["name"].startswith("像素主实验")) == 36
    keys = list(bench.configs(list(bench.BENCHES)))
    fake = lambda k, base: {"sec_per_step": base * (1 + bench.ALL_SIZES.index(k[2])) ** 2
                            * (20 if k[0] == "pixels64" else 1), "sec_mean": 0.0, "n_steps": 3, "load_before": 0.2,
                            "load_mean": 0.5, "load_max": 0.9, "other_running_mean": 0.0}
    best4 = {k: fake(k, 0.01) for k in keys}
    best1 = {k: fake(k, 0.02) for k in keys}
    par = {k: {**fake(k, 0.025), "sec_per_step_jobs": [0.025] * 4, "n_jobs": 4} for k in bench.PARALLEL}
    rows = bench.budget(best4, best1, par, 20000)
    assert sum(r["n"] for r in rows) == n("0") + n("6")
    loc = [r for r in rows if r["local"]]
    assert loc and sum(r["machine_h"] for r in loc) <= bench.FLAG_HOURS + 1e-9
    assert all(r["job_max_h"] <= bench.FLAG_HOURS for r in loc)
    f, src = bench.parallel_factor(par, best1, "angle_raw", "XL")                       # borrowed from angle_cs XL
    assert f == pytest.approx(1.25) and src == "同族最近尺寸"
    info = {k: {"params": 1000, "params_recon": 1000, "params_jepa": 800, "params_encoder": 400,
                "params_transition": 200, "params_decoder": 400, "width": 16, "channels": 0} for k in keys}
    flops = {k: {"train_flops_per_sample": 1e6} for k in keys}
    meta = {"date": "2026-11-04", "cpu": "test", "ncpu": 4, "mem_gb": 15.0, "python": "3", "torch": "2", "numpy": "2",
            "batch": 128, "max_load": 1.5, "poll": 60.0, "max_wait_quiet": 1800.0, "threads": 4,
            "quiet_start": {"quiet": True, "waited_sec": 0.0, "load_start": 0.2, "load_end": 0.2, "heavy_seen": []},
            "raw": bench._raw_dict({"4threads_pass1": best4, "1threads_pass1": best1, "parallel4x1": par}),
            "benches": list(bench.BENCHES), "notes": "测试备注", "total_sec": 60.0}
    out = tmp_path / "bench.md"
    bench.write_report(out, best4, best1, par, info, flops, meta, list(bench.BENCHES), 1)
    txt = out.read_text(encoding="utf-8")
    for s in ("## 1.", "## 2.", "## 3.", "## 6.", "6.3 建议", "toggle_ep63", "jepa-sg", "测试备注", "安静机器协议"):
        assert s in txt, s
    out2 = tmp_path / "again.md"                                                         # re-render from the report
    assert bench.render_from_json([out], out2) == 0
    assert out2.read_text(encoding="utf-8") == txt
    # merge with a supplementary partial run, then re-render the merged report from itself
    k = ("pixels64", "recon", "S")
    supp = {k: {**best4[k], "sec_per_step": best4[k]["sec_per_step"] * 0.5}}
    meta2 = {**meta, "raw": bench._raw_dict({"4threads_pass1": supp}), "passes": 1, "notes": ""}
    (tmp_path / "supp.json").write_text(json.dumps(bench.payload(meta2, {k: info[k]}, {k: flops[k]})),
                                        encoding="utf-8")
    merged = tmp_path / "merged.md"
    assert bench.render_from_json([out, tmp_path / "supp.json"], merged) == 0
    mtxt = merged.read_text(encoding="utf-8")
    assert "合并 2 次运行" in mtxt and "补测" in mtxt and f"| {best4[k]['sec_per_step'] * 0.5:.4f} |" in mtxt
    again = tmp_path / "merged_again.md"
    assert bench.render_from_json([merged], again) == 0 and again.read_text(encoding="utf-8") == mtxt
    # best_of prefers an undisturbed measurement; worst_of is the conservative (slowest undisturbed) one
    fast_dist = {"sec_per_step": 1.0, "other_running_mean": 1.0}
    clean = [{"sec_per_step": 1.1, "other_running_mean": 0.0}, {"sec_per_step": 1.3, "other_running_mean": 0.1}]
    passes = [{k: fast_dist}, {k: clean[0]}, {k: clean[1]}]
    assert bench.best_of(passes)[k] is clean[0] and bench.worst_of(passes)[k] is clean[1]
    assert bench.best_of([{k: fast_dist}])[k] is fast_dist                               # all disturbed: flagged


# ------------------------------------------------------------------ checkpoints / determinism
def test_save_load_roundtrip(tmp_path):
    spec, sp = tiny_data("pendulum", {}, "angle_raw")
    out = tmp_path / "run"
    res = run(tiny_cfg(obs="angle_raw", steps=20, eval_every=10, out=str(out), standardize=True), data=(spec, sp))
    for f in ("config.json", "model.pt", "metrics.json"):
        assert (out / f).exists()
    cfgj = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfgj["params"]["params"] == count_params(res["model"]) and cfgj["train"]["seed"] == 0
    metj = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metj["steps_done"] == 20 and len(metj["curves"]["val"]) == 2 and "rollout" in metj["final"]
    model, nz, ck = load(out)
    assert ck["final"] and ck["steps_done"] == 20
    sd0, sd1 = res["model"].state_dict(), model.state_dict()
    assert sd0.keys() == sd1.keys() and all(torch.equal(sd0[k].cpu(), sd1[k]) for k in sd0)
    assert np.allclose(nz.mean, res["normalizer"].mean) and np.allclose(nz.std, res["normalizer"].std)
    ds = WindowDataset(sp["val"], 2, 8, spec["action"], nz)
    b = ds.get_batch(np.arange(16))
    res["model"].eval()
    with torch.no_grad():
        assert torch.equal(res["model"].loss(b["frames"])[0], model.loss(b["frames"])[0])
    with pytest.raises(FileExistsError):
        run(tiny_cfg(obs="angle_raw", steps=1, out=str(out)), data=(spec, sp))


def test_encode_untrained_checkpoint(tmp_path):
    """encode() on an UNTRAINED model (no trained latents are produced before pre-registration)."""
    spec, sp = tiny_data("pendulum", {}, "pixels64", n_train=5, n_val=0, horizon=12)
    torch.manual_seed(3)
    m = build_wm(spec, "jepa", None, width=8, channels=2)
    save_checkpoint(tmp_path / "model.pt", m, None, {}, 0, final=False)
    rng = torch.random.get_rng_state()
    model, nz, _ = load(tmp_path)
    assert torch.equal(torch.random.get_rng_state(), rng)                               # loading is RNG-neutral
    assert nz is None and not model.training
    tr = sp["train"]
    z, t = encode(model, tr, nz, batch_size=7)
    assert z.shape == (5, 12, 8) and z.dtype == np.float32 and np.array_equal(t, np.arange(1, 13))
    x = frames_to_tensor(tr.obs[2, 4:6][None], "cpu")
    with torch.no_grad():
        assert np.allclose(z[2, 4], model.encode(x)[0].numpy(), atol=1e-5)               # z_t = E(o_{t-1}, o_t)


def test_cpu_determinism():
    spec, sp = tiny_data("pendulum", {}, "angle_cs")
    runs = [run(tiny_cfg(variant="jepa", steps=25, seed=s), data=(spec, sp)) for s in (7, 7, 8)]
    a, b, c = (r["model"].state_dict() for r in runs)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [r["loss"] for r in runs[0]["metrics"]["curves"]["train"]] == \
           [r["loss"] for r in runs[1]["metrics"]["curves"]["train"]]
    assert not all(torch.equal(a[k], c[k]) for k in a)


@pytest.mark.parametrize("world,kw,mode,over", [
    ("pendulum", {}, "pixels64", dict(variant="recon", width=8, channels=2)),
    ("pendulum", {}, "pixels64", dict(variant="jepa-sg", width=8, channels=2)),
    ("toggle_grid", {"switch_visible": False}, "grid", dict(variant="recon", anchors="episode", history=1, horizon=16)),
])
def test_cpu_determinism_conv_and_episode_models(world, kw, mode, over):
    """Bitwise reproducible on CPU also for the conv stacks (channels-last, as train.py runs them on CPU) and for the
    discrete-action whole-episode windows of cell 6."""
    spec, sp = tiny_data(world, kw, mode, n_train=8, n_val=4, horizon=16)
    cfg = dict(world=world, obs=mode, world_kwargs=kw, steps=8, batch=16, log_every=4, eval_every=8, **over)
    a, b = (run(tiny_cfg(**cfg), data=(spec, sp)) for _ in range(2))
    sa, sb = a["model"].state_dict(), b["model"].state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    untimed = lambda rows: [{k: v for k, v in r.items() if k not in ("sec_per_step", "elapsed")} for r in rows]
    assert untimed(a["metrics"]["curves"]["train"]) == untimed(b["metrics"]["curves"]["train"])
    assert a["metrics"]["curves"]["val"] == b["metrics"]["curves"]["val"]
    assert a["metrics"]["final"] == b["metrics"]["final"]
    assert a["config"]["env"]["channels_last"] is (mode == "pixels64")


def test_state_obs_and_circular_loss_guards():
    """The ground-truth state is never a baseline's input by accident, and the circular loss only sees angles."""
    with pytest.raises(ValueError):
        TrainConfig(obs="state")                                     # pendulum (theta, omega): an oracle
    assert TrainConfig(obs="state", allow_state_obs=True).obs == "state"
    with pytest.raises(ValueError):
        main(["--obs", "state", "--steps", "1", "--quiet"])
    with pytest.raises(ValueError):
        TrainConfig(obs="angle_cs", loss="circular")                 # cos / sin are not angles
    assert TrainConfig(obs="angle_raw", loss="circular").loss == "circular"


def test_cli(tmp_path):
    out = tmp_path / "cli"
    code = main(["--world", "wright_fisher", "--obs", "freq", "--variant", "recon", "--size", "S", "--steps", "5",
                 "--n-train", "8", "--n-val", "4", "--horizon", "40", "--log-every", "5", "--eval-every", "5",
                 "--threads", "1", "--quiet", "--world-kwargs", '{"schedule": "mixed_only"}', "--out", str(out)])
    assert code == 0
    cfgj = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfgj["train"]["world_kwargs"] == {"schedule": "mixed_only"}
    assert cfgj["data"]["world_params"]["schedule"] == "mixed_only"


# ------------------------------------------------------------------ ledger
def test_ledger_append_on_temp_copy(tmp_path):
    real = ROOT / "results" / "ledger.md"
    before = real.read_bytes()
    tmp = tmp_path / "ledger.md"
    shutil.copy(real, tmp)
    n0 = len(ledger.read_rows(tmp))
    row = ledger.append_entry("2026-11-03", 0, "pendulum", "angle_cs", "recon", "M", 499965, 1,
                              {"nmse": 0.0123456, "rollout": {"8": {"mse": 1e-3}}, "gate": True, "x": None},
                              "test-only", "pipes | and\nnewlines", path=tmp)
    assert row.count("\\|") == 1 and "\n" not in row
    ledger.append_entry("2026-11-04", 6, "toggle_grid", "grid", "jepa", "S", 100025, 0, {}, "test-only", path=tmp)
    rows = ledger.read_rows(tmp)
    assert len(rows) == n0 + 2
    assert rows[-2]["metrics"] == "nmse=0.01235; rollout.8.mse=0.001; gate=true; x=NA"
    assert rows[-2]["notes"] == "pipes | and newlines" and rows[-1]["cell"] == "6"
    lines = tmp.read_text(encoding="utf-8").splitlines()
    table = [l for l in lines if l.startswith("|")]
    assert all(len(ledger._split_row(l)) == len(ledger.COLUMNS) for l in table)
    assert table[0] == "| " + " | ".join(ledger.COLUMNS) + " |"
    # rows go right after the table, even when text follows it
    tmp.write_text(tmp.read_text(encoding="utf-8") + "\n后记：表格之后的文字\n", encoding="utf-8")
    ledger.append_entry("2026-11-05", 0, "pendulum", "pixels64", "jepa", "XL", 4023444, 2, {"a": 1}, "test-only",
                        path=tmp)
    lines = tmp.read_text(encoding="utf-8").splitlines()
    i = max(k for k, l in enumerate(lines) if l.startswith("|"))
    assert "pixels64" in lines[i] and lines[-1] == "后记：表格之后的文字"
    assert len(ledger.read_rows(tmp)) == n0 + 3
    with pytest.raises(ValueError):
        ledger.append_entry("3/11/2026", 0, "p", "o", "v", "S", 1, 0, {}, "x", path=tmp)
    with pytest.raises(ValueError):
        ledger.append_entry("2026-11-05", 0, "p", "o", "v", "S", 1, 0, {}, "  ", path=tmp)
    assert real.read_bytes() == before                                                    # real ledger untouched
    assert not (tmp_path / "ledger.md.lock").exists()
