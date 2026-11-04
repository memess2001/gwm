"""Pendulum world (cells 0, 1, 4, 5, 9, 15).

Dynamics   theta'' = -omega0^2 sin(theta) - 2*gamma*theta' + u(t)     (2*gamma damping convention)
State      (theta, omega); theta wrapped to [-pi, pi), the unwrapped angle is kept in ``info``.
Energy     E = omega^2/2 - omega0^2 cos(theta);  E_min = -omega0^2 (rest at the bottom),
           separatrix E_sep = +omega0^2;  dE/dt = -2*gamma*omega^2 + u*omega.
Divergence d(theta')/d(theta) + d(omega')/d(omega) = -2*gamma, uniform over phase space.

Integrators. Conservative case (gamma = 0, no torque): velocity Verlet (kick-drift-kick) with ``n_sub``
substeps per observation step: symplectic, bounded O(h^2) energy error, no secular drift (``"yoshida4"`` is
an optional 4th-order symplectic composition). Otherwise classical RK4 with ``n_sub`` substeps; the torque
is held constant over each observation interval (zero-order hold) and the recorded float32 torque is exactly
the one that was simulated. Units: time in units of the observation clock t = k * dt; ``hold`` (random policy:
a new torque value every ``hold`` steps) is counted in observation steps, i.e. hold * dt time units, so changing
dt at fixed hold changes the torque statistics in physical time.

Observation modes
    state      (theta, omega)               DEBUG ONLY - never feed this to a world model
    angle_cs   (cos theta, sin theta)
    angle_raw  theta in [-pi, pi)           1-dim; the wrap discontinuity at +-pi is intentional (float32 range
                                            [float32(-pi), float32(pi)), i.e. pi itself is reported as -pi)
    pixels64   64x64 grayscale uint8        see ``render``
"angle observation" in cells.yaml / HANDOFF is ambiguous: angle_cs embeds S^1, angle_raw is a cut chart of it
(Batson 2021: a 1-d encoding of S^1 must break somewhere). Experiments must name the mode they use. No mode
reveals omega in a single frame (velocity needs >= 2 frames); "state" is the only mode that leaks it.

info (analysis only): energy [N, T+1]; theta_unwrapped [N, T+1]; is_rotation [N] = INITIAL energy above the
separatrix (E_0 > omega0^2). Under damping or torque the motion type can change along the trajectory, so use
info["energy"][:, t] > omega0^2 for the per-step label.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Trajectories, World

PI = np.pi
DEFAULTS: dict[str, Any] = dict(
    omega0=1.0, gamma=0.0, dt=0.1, n_sub=20, integrator="auto",      # auto | verlet | yoshida4 | rk4
    init="box", omega_max=2.5, E_lo=None, E_hi=None,                  # init: "box" or explicit states [N, 2]
    policy="none", u_max=1.0, hold=10,                                # policy: none | random (piecewise-constant torque)
    pixel_noise=0.0, radius=22.0, rod_sigma=0.8, bob_sigma=2.5,       # renderer (64x64)
)
SIZE = 64


def wrap(theta):
    """Wrap angles to [-pi, pi)."""
    w = np.mod(np.asarray(theta, dtype=np.float64) + PI, 2 * PI) - PI
    return np.where(w >= PI, w - 2 * PI, w)


def energy(state, omega0: float = 1.0):
    """E = omega^2/2 - omega0^2 cos(theta) for states [..., 2]."""
    s = np.asarray(state, dtype=np.float64)
    return 0.5 * s[..., 1] ** 2 - omega0 ** 2 * np.cos(s[..., 0])


def render(theta, radius: float = 22.0, rod_sigma: float = 0.8, bob_sigma: float = 2.5, noise: float = 0.0,
           rng: Optional[np.random.Generator] = None, quantize: bool = True, chunk: int = 128) -> np.ndarray:
    """Render angles [...] to frames [..., 64, 64] (uint8, or float32 in [0, 1] if ``quantize=False``).

    Pivot at the image centre ((64-1)/2, (64-1)/2); theta = 0 points DOWN, theta = +pi/2 points right:
    bob centre = pivot + radius * (sin theta, cos theta) in (column, row) pixel coordinates. Rod = segment
    pivot->bob with Gaussian cross-profile exp(-d^2 / 2 rod_sigma^2) (d = distance of the pixel centre to the
    segment); bob = Gaussian blob of width bob_sigma; combined by the soft union 1 - (1-rod)(1-bob), so the
    background is 0 and the peak is 1. Every pixel is a C^1 function of theta (no aliasing jumps, periodic,
    so no jump at +-pi); the bob position makes the map injective. Vectorised and chunked (``chunk`` frames
    at a time) so the float temporaries stay bounded. Optional additive Gaussian pixel noise (std ``noise`` in
    [0, 1] intensity units, needs ``rng``) is applied before clipping/quantisation.
    """
    th = np.asarray(theta, dtype=np.float64)
    flat = th.reshape(-1)
    c = (SIZE - 1) / 2.0
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dx = (xx.reshape(-1) - c).astype(np.float32)
    dy = (yy.reshape(-1) - c).astype(np.float32)
    r2 = dx * dx + dy * dy
    a_rod, a_bob, r = np.float32(0.5 / rod_sigma ** 2), np.float32(0.5 / bob_sigma ** 2), np.float32(radius)
    out = np.empty((flat.size, SIZE * SIZE), np.uint8 if quantize else np.float32)
    for i in range(0, flat.size, chunk):
        s = np.sin(flat[i:i + chunk]).astype(np.float32)[:, None]
        co = np.cos(flat[i:i + chunk]).astype(np.float32)[:, None]
        proj = s * dx + co * dy                         # coordinate along the rod direction (sin, cos)
        t = np.clip(proj, 0.0, r)                       # closest point on the segment
        rod = np.exp(-(r2 - 2 * t * proj + t * t) * a_rod)
        bob = np.exp(-(r2 - 2 * r * proj + r * r) * a_bob)
        img = 1.0 - (1.0 - rod) * (1.0 - bob)
        if noise > 0:
            if rng is None:
                raise ValueError("pixel noise needs an rng")
            img += np.float32(noise) * rng.standard_normal(img.shape, dtype=np.float32)
        img = np.clip(img, 0.0, 1.0)
        out[i:i + chunk] = np.rint(img * 255.0) if quantize else img
    return out.reshape(th.shape + (SIZE, SIZE))


def _region_betti(E_lo: float, E_hi: float, W: float, w2: float) -> tuple[int, int, int]:
    """(b0, b1, b1_wrapping) of R = {(theta, omega) in S^1 x R : |omega| <= W, E_lo <= E <= E_hi}.

    Derivation. Fix theta and put c = cos(theta). The omega-slice of R is {l(c) <= |omega| <= u(c)} with
    u = min(W, sqrt(2(E_hi + w2 c))) and l = sqrt(2 max(E_lo + w2 c, 0)). Hence
      * the slice is non-empty  iff  c_hi <= c <= c_W   with c_hi = -E_hi/w2, c_W = (W^2/2 - E_lo)/w2;
      * it is ONE interval through omega = 0 iff c <= c_lo = -E_lo/w2 (zone A1), else TWO intervals,
        omega > 0 and omega < 0 (zone A2, c_lo < c <= c_W), which merge at omega = 0 where c = c_lo.
    Always c_hi <= c_lo < c_W. c = -1 is theta = pi, c = +1 is theta = 0, so A1 lies towards theta = pi and
    A2 towards theta = 0. Contracting every fibre interval to a point, R is homotopy equivalent to a graph:
    one strand over A1, two strands (upper/lower) over A2, glued where c = c_lo. Reading off the cases:
      A2 empty (c_lo >= 1, i.e. no active lower energy cut):
          A1 = whole circle (E_hi >= E_sep)   -> annulus around the cylinder        [1, 1] (wraps)
          A1 = arc around theta = 0           -> disk                               [1, 0]
      c_lo < -1 (E_lo > E_sep, rotations only):
          c_W >= 1  -> two annuli (omega > 0, omega < 0), each wrapping             [2, 2]
          c_W < 1   -> the box cuts both bands at theta = 0: two disks              [2, 0]
      c_lo = -1 (E_lo = E_sep exactly): two bands touching at the saddle (pi, 0)
          c_W >= 1  -> wedge of two wrapping circles                                [1, 2]
          c_W < 1   -> star-shaped tree                                             [1, 0]
      -1 < c_lo < 1 (E_min < E_lo < E_sep): a hole {E < E_lo} around the origin
          c_W >= 1, E_hi >= E_sep -> annulus with a hole (pair of pants)            [1, 2] (one wraps)
          c_W >= 1, E_hi <  E_sep -> libration ring, contractible in the cylinder   [1, 1] (does NOT wrap)
          c_W <  1, E_hi >= E_sep -> ring cut at theta = 0 by the box, joined at pi  [1, 0]
          c_W <  1, E_hi <  E_sep -> ring cut at theta = 0 and at pi                [2, 0]
    Empty regions give [0, 0]. ``b1_wrapping`` counts components whose H1 maps onto H1(S^1 x R) (loops that
    go around the cylinder); b1 - b1_wrapping loops are contractible in the cylinder.
    """
    if E_lo > E_hi:
        return 0, 0, 0
    c_hi, c_lo, c_W = -E_hi / w2, -E_lo / w2, (0.5 * W * W - E_lo) / w2
    if c_lo >= 1:
        return (0, 0, 0) if c_hi > 1 else (1, 1, 1) if c_hi <= -1 else (1, 0, 0)
    if c_lo < -1:
        return (2, 2, 2) if c_W >= 1 else (2, 0, 0) if c_W >= -1 else (0, 0, 0)
    if c_lo == -1:
        return (1, 2, 1) if c_W >= 1 else (1, 0, 0)
    if c_W >= 1:
        return (1, 2, 1) if c_hi <= -1 else (1, 1, 0)
    return (1, 0, 0) if c_hi <= -1 else (2, 0, 0)


_Y1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_Y0 = 1.0 - 2.0 * _Y1


def _verlet(th, om, w2, h, n):
    a = -w2 * np.sin(th)
    for _ in range(n):
        om = om + 0.5 * h * a
        th = th + h * om
        a = -w2 * np.sin(th)
        om = om + 0.5 * h * a
    return th, om


def _rk4(th, om, w2, g2, u, h, n):
    def f(t, o):
        return o, -w2 * np.sin(t) - g2 * o + u
    for _ in range(n):
        a1, b1 = f(th, om)
        a2, b2 = f(th + 0.5 * h * a1, om + 0.5 * h * b1)
        a3, b3 = f(th + 0.5 * h * a2, om + 0.5 * h * b2)
        a4, b4 = f(th + h * a3, om + h * b3)
        th = th + h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        om = om + h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
    return th, om


class Pendulum(World):
    name = "pendulum"
    obs_modes = ("state", "angle_cs", "angle_raw", "pixels64")

    def __init__(self, **params):
        self.cfg = dict(DEFAULTS)
        self.cfg = self._c(**params)

    # ---------------------------------------------------------------- config
    def _c(self, **over) -> dict:
        bad = set(over) - set(DEFAULTS)
        if bad:
            raise TypeError(f"unknown pendulum parameters: {sorted(bad)}")
        c = {**self.cfg, **over}
        if c["omega0"] <= 0 or c["gamma"] < 0 or c["dt"] <= 0 or int(c["n_sub"]) < 1 or c["omega_max"] <= 0:
            raise ValueError("need omega0 > 0, gamma >= 0, dt > 0, n_sub >= 1, omega_max > 0")
        if c["policy"] not in ("none", "random"):
            raise ValueError("policy must be 'none' or 'random'")
        if c["integrator"] not in ("auto", "verlet", "yoshida4", "rk4"):
            raise ValueError("integrator must be auto | verlet | yoshida4 | rk4")
        lo, hi = self._window(c)
        if lo > hi:
            raise ValueError("empty energy window: E_lo > E_hi")
        return c

    @staticmethod
    def _window(c):
        return (-np.inf if c["E_lo"] is None else float(c["E_lo"]), np.inf if c["E_hi"] is None else float(c["E_hi"]))

    @staticmethod
    def _conservative(c) -> bool:
        return c["gamma"] == 0 and c["policy"] == "none"

    def _integrator(self, c, actuated: bool) -> str:
        name = c["integrator"]
        if name == "auto":
            name = "verlet" if (c["gamma"] == 0 and not actuated) else "rk4"
        if name != "rk4" and (c["gamma"] != 0 or actuated):
            raise ValueError("symplectic integrators are only used for gamma = 0 without torque")
        return name

    # ---------------------------------------------------------------- dynamics
    def vector_field(self, state, u=0.0, **over):
        """d/dt (theta, omega) for states [..., 2] and torque u (broadcastable)."""
        c = self._c(**over)
        s = np.asarray(state, dtype=np.float64)
        return np.stack([s[..., 1], -c["omega0"] ** 2 * np.sin(s[..., 0]) - 2 * c["gamma"] * s[..., 1] + u], -1)

    def _simulate(self, th0, om0, actions, horizon, c):
        """Unwrapped theta and omega, both [N, T+1]."""
        n = int(c["n_sub"])
        w2, g2, h = c["omega0"] ** 2, 2.0 * c["gamma"], c["dt"] / n
        integ = self._integrator(c, actions is not None)
        th = np.empty((len(th0), horizon + 1))
        om = np.empty_like(th)
        th[:, 0], om[:, 0] = th0, om0
        t, o = np.array(th0, dtype=np.float64), np.array(om0, dtype=np.float64)
        for k in range(horizon):
            if integ == "verlet":
                t, o = _verlet(t, o, w2, h, n)
            elif integ == "yoshida4":
                for _ in range(n):
                    for w in (_Y1, _Y0, _Y1):
                        t, o = _verlet(t, o, w2, w * h, 1)
            else:
                u = 0.0 if actions is None else actions[:, k, 0].astype(np.float64)
                t, o = _rk4(t, o, w2, g2, u, h, n)
            th[:, k + 1], om[:, k + 1] = t, o
        return th, om

    def rollout(self, state0, horizon: int, actions=None, **over) -> np.ndarray:
        """Pure dynamics: states [N, T+1, 2] (theta wrapped) from states0 [N, 2] and optional torques [N, T, 1]."""
        c = self._c(**over)
        s0 = np.atleast_2d(np.asarray(state0, dtype=np.float64))
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32).reshape(len(s0), horizon, 1)
        th, om = self._simulate(s0[:, 0], s0[:, 1], actions, horizon, c)
        return np.stack([wrap(th), om], -1)

    # ---------------------------------------------------------------- sampling
    def _init(self, rng, n, c):
        if not isinstance(c["init"], str):
            s = np.asarray(c["init"], dtype=np.float64).reshape(-1, 2)
            if len(s) != n:
                raise ValueError(f"explicit init has {len(s)} states, n_traj = {n}")
            return wrap(s[:, 0]), s[:, 1].copy()
        if c["init"] != "box":
            raise ValueError("init must be 'box' or an array of states [N, 2]")
        W, w2 = c["omega_max"], c["omega0"] ** 2
        lo, hi = self._window(c)
        if max(lo, -w2) > min(hi, 0.5 * W * W + w2):
            raise ValueError("energy window does not intersect the sampling box")
        th, om = np.empty(0), np.empty(0)
        for _ in range(100_000):
            m = max(4 * (n - len(th)), 1024)
            t, o = rng.uniform(-PI, PI, m), rng.uniform(-W, W, m)
            e = 0.5 * o * o - w2 * np.cos(t)
            ok = (e >= lo) & (e <= hi)
            th, om = np.concatenate([th, t[ok]]), np.concatenate([om, o[ok]])
            if len(th) >= n:
                return th[:n], om[:n]
        raise RuntimeError("rejection sampling failed: energy window has (almost) zero area inside the box")

    def _actions(self, rng, n, horizon, c):
        if c["policy"] == "none":
            return None
        hold = int(c["hold"])
        nb = -(-horizon // hold)
        u = rng.uniform(-c["u_max"], c["u_max"], (n, nb))
        return np.repeat(u, hold, axis=1)[:, :horizon, None].astype(np.float32)

    def sample(self, seed: int, horizon: int, n_traj: int, obs_mode: str, **kwargs) -> Trajectories:
        c = self._c(**kwargs)
        rng = np.random.default_rng(seed)
        th0, om0 = self._init(rng, n_traj, c)
        actions = self._actions(rng, n_traj, horizon, c)
        th, om = self._simulate(th0, om0, actions, horizon, c)
        state = np.stack([wrap(th), om], -1)
        obs = self._observe(state, obs_mode, rng, c)
        e = energy(state, c["omega0"])
        info = {"energy": e, "theta_unwrapped": th, "is_rotation": e[:, 0] > c["omega0"] ** 2}
        params = {k: (v if k != "init" or isinstance(v, str) else "explicit") for k, v in c.items()}
        meta = dict(world=self.name, seed=seed, horizon=horizon, n_traj=n_traj, obs_mode=obs_mode, params=params)
        return Trajectories(obs=obs, state=state, actions=actions, t=np.arange(horizon + 1) * c["dt"], info=info, meta=meta)

    # ---------------------------------------------------------------- observation
    def _observe(self, state, obs_mode, rng, c):
        s = np.asarray(state, dtype=np.float64)
        if obs_mode == "state":
            return s.astype(np.float32)
        if obs_mode == "angle_cs":
            return np.stack([np.cos(s[..., 0]), np.sin(s[..., 0])], -1).astype(np.float32)
        if obs_mode == "angle_raw":
            a = wrap(s[..., :1]).astype(np.float32)
            a[a >= np.float32(PI)] = -np.float32(PI)    # float32(pi - eps) can round up to float32(pi) > pi
            return a
        if obs_mode == "pixels64":
            noise = c["pixel_noise"] if rng is not None else 0.0
            return render(s[..., 0], c["radius"], c["rod_sigma"], c["bob_sigma"], noise, rng)
        raise ValueError(f"unknown obs_mode {obs_mode!r}; choose from {self.obs_modes}")

    def observe(self, state, obs_mode, rng=None):
        """Observations of states [..., 2]; pixel noise is only added when an rng is given."""
        return self._observe(state, obs_mode, rng, self.cfg)

    def spec(self, obs_mode: str, **over) -> dict:
        c = self._c(**over)
        shapes = {"state": ((2,), "float32"), "angle_cs": ((2,), "float32"), "angle_raw": ((1,), "float32"),
                  "pixels64": ((SIZE, SIZE), "uint8")}
        if obs_mode not in shapes:
            raise ValueError(f"unknown obs_mode {obs_mode!r}")
        act = {"type": "none"} if c["policy"] == "none" else {"type": "continuous", "dim": 1,
                                                               "low": -c["u_max"], "high": c["u_max"]}
        return {"obs_shape": shapes[obs_mode][0], "obs_dtype": shapes[obs_mode][1], "action": act}

    # ---------------------------------------------------------------- ground truth
    def ground_truth(self, **cfg) -> dict[str, Any]:
        """Analytic truths for the configuration ``self.cfg`` updated with ``cfg``.

        Region topology (Betti numbers [b0, b1] as subsets of the cylinder S^1 x R, see ``_region_betti``):
          * ``init_region_betti``: support of the initial states, R0 = box ∩ energy window,
            box = S^1 x [-omega_max, omega_max]. Exact for i.i.d. state samples (horizon 0).
          * ``region_betti``: the flow-invariant hull of R0, i.e. the energy band E_a <= E <= E_b with
            E_a = max(E_lo, -omega0^2) and E_b = min(E_hi, omega_max^2/2 + omega0^2). The box limits E
            differently at different theta: its maximal energy omega_max^2/2 + omega0^2 is only reached at
            theta = pi, so orbits with E > omega_max^2/2 - omega0^2 start inside the box only near theta = pi
            but then leave it (|omega| up to sqrt(2(E_b + omega0^2)) at theta = 0). Conservative
            trajectories sweep these full orbits, so ``region_betti`` is the topology of the set of visited
            states once horizon*dt is long enough for every orbit through R0 to be traced round (rotations:
            one revolution, a few time units; librations: phases are already spread over the box). Defined
            only for the conservative case (gamma = 0, no torque); None otherwise.
        The two coincide in the three standard regimes with the defaults (full box -> annulus [1, 1];
        libration-only E_hi < E_sep -> disk [1, 0]; rotation-only E_lo > E_sep -> two annuli [2, 2]), but
        not in general: a lower cut E_min < E_lo < E_sep punches a hole ([1, 1] libration ring that does NOT
        wrap the cylinder, or [1, 2] with rotations), and for E_lo > omega_max^2/2 - omega0^2 the box cuts
        the bands at theta = 0 (init [2, 0] vs hull [2, 2]). ``*_b1_wrapping`` counts loops that go around
        the cylinder; only those certify the S^1 factor.
        Conditioning: the Betti numbers are exact for the CLOSED set (at a threshold they equal the limit of
        slightly larger regions), but features shrink to zero size near the thresholds E_lo -> -omega0^2 (hole),
        E -> E_sep (gap at theta = pi / neck at the saddle) and omega_max^2/2 - E_lo -> +-omega0^2 (box cut at
        theta = 0); e.g. an E_lo typed as -omega0^2 with a different float rounding opens a hole of radius ~1e-8.
        Vietoris-Rips persistence on the true embedding (cos, sin, omega) sees the metric gap, not the topology:
        the disk E in [-1, 0.8] (omega0 = 1) already has a spurious H1 bar of ~0.5 (0.8 for E_hi = 0.9) against
        ~1.47 for the annulus. Calibrate instruments on configurations far from these thresholds.
        """
        c = self._c(**cfg)
        w0, g = c["omega0"], c["gamma"]
        w2, W = w0 * w0, c["omega_max"]
        cons, box = self._conservative(c), isinstance(c["init"], str) and c["init"] == "box"
        lo, hi = self._window(c)
        E_a, E_b = max(lo, -w2), min(hi, 0.5 * W * W + w2)
        init = _region_betti(lo, hi, W, w2) if box else None
        hull = _region_betti(E_a, E_b, np.inf, w2) if (box and cons) else None
        integ = self._integrator(c, c["policy"] != "none")
        return dict(
            state_space="S1 x R", state_dim=2, phase_space_betti=[1, 1],
            region_betti=None if hull is None else list(hull[:2]),
            region_b1_wrapping=None if hull is None else hull[2],
            init_region_betti=None if init is None else list(init[:2]),
            init_region_b1_wrapping=None if init is None else init[2],
            energy_band=[E_a, E_b] if box else None,
            energy_fn=lambda state: energy(state, w0),
            conserved=["energy"] if cons else [],
            energy_balance="dE/dt = -2*gamma*omega^2 + u*omega",
            divergence=-2.0 * g,
            symplectic=bool(g == 0),          # torque u(t) keeps the flow Hamiltonian (area-preserving)
            separatrix_energy=w2, energy_min=-w2,
            dt=c["dt"], vector_field=lambda state, u=0.0: self.vector_field(state, u, **cfg),
            integrator=f"{integ}, n_sub={int(c['n_sub'])}", actuated=c["policy"] != "none",
            render=dict(pivot=((SIZE - 1) / 2,) * 2, radius=c["radius"], theta0="down", injective=True),
        )
