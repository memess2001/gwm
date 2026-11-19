"""Wright-Fisher / replicator population world (cells 2, 7, 9).

State x in the simplex Delta^{K-1} (type frequencies, K >= 3, default 3), observed every dt = 0.1. Time is cut
into segments; within a segment one mechanism acts with constant parameters. Default: infinite population =
deterministic ODE. Optional finite-population drift: x_{t+1} ~ Multinomial(pop_size, Phi_dt(x_t)) / pop_size
(Phi = ODE flow over one observation interval); optional observation sampling noise (obs mode "counts").

  mechanism           dx/dt                                 flow over dt                 alpha   Hodge (Fisher metric)
  selection           x * (f - x.f)                         exact: x ∝ x0 exp(t f)       +1      gradient, V = f.x
  migration           mu (m - x)                            exact: m + (x0 - m) e^{-mu t} -1     gradient, V = mu sum_i m_i log x_i
  mutation            u x Q   (Q rate matrix, rows sum 0)   exact: x0 expm(u t Q)        -1 / None   gradient iff Q parent-independent
  mixed               selection + migration                 RK4, n_sub substeps          None    gradient, V = f.x + mu sum m log x
  mutation_selection  selection + mutation                  RK4                          None    gradient iff Q parent-independent
  rps                 x_i ((Ax)_i - x.Ax), zero-sum RPS A   RK4                          None    rotational; H = sum x*_i log x_i conserved
                      (default A; other payoffs ``rps_A``:                                       exact criteria in ``replicator_hodge``)

Conventions (Amari): alpha = +1 <-> e-connection = straight lines in the natural parameters theta_i = log(x_i/x_K);
alpha = -1 <-> m-connection = straight lines in the mixture / expectation parameters eta = x. Selection is an
e-geodesic (theta(t) = theta0 + t (f_i - f_K)); migration toward m (and uniform mutation, = migration toward
1/K) is an m-geodesic. General (parent-dependent) mutation and mixed fields are no alpha-geodesic for generic
parameters (label None). Koide & van de Venn (2026, arXiv 2609.08870) use the OPPOSITE sign (their alpha = +1 is
straight in expectation coordinates); never mix the two silently.

Hodge character in the Shahshahani (= Fisher information) metric g_ij = delta_ij / x_i on {sum_i v_i = 0}: a field
v is a gradient flow iff its metric dual 1-form omega = sum_i (v_i / x_i) dx_i is closed on the simplex. Selection:
omega = d(f.x) (Fisher's fundamental theorem). Migration: v_i/x_i = mu (m_i/x_i - 1), so omega = d(mu sum m_i log x_i)
- migration is an exact gradient flow too (V = -mu KL(m || x) + const), and so is every sum of selection, migration
and parent-independent mutation (Q_ji = q_i for all j != i; uniform mutation is the case q_i = 1/K). Parent-
dependent mutation (generic Q, e.g. "cyclic") has a rotational part ("mixed"). The zero-sum RPS field
(A antisymmetric with zero row sums) is divergence-free for the Shahshahani volume (div_g v = 1'Ax / 2 = 0),
conserves H = sum_i x*_i log x_i and has closed orbits around x*: "rotational". For a general payoff ``rps_A``
the label is computed exactly by ``replicator_hodge`` (e.g. a symmetric A is a gradient flow of x.Ax / 2, and
adding a constant to a column of A does not change the field).

RK4 mechanisms (mixed, mutation_selection, rps): each field is tangent to the hyperplane s = sum_i x_i = 1, but the
hyperplane need not be attracting. sum_i v_i = -r (s - 1) with r = x.f + mu (mixed), x.f (mutation_selection),
x.Ax (rps), so where r < 0 the round-off in s grows like exp(-r t): at rate 0.9, 1e-16 reaches 1e-6 after 25 time
units (250 steps of dt = 0.1). The flow on the simplex does not change when a constant is added to f or to A, but r
does. Each RK4 substep is therefore followed by the projection x -> clip(x, 0) / sum (the point of the simplex with
the same ratios x_i / x_j). In exact arithmetic RK4 keeps s = 1 (and x >= 0 while the step dt / n_sub resolves the
rates), so the projection removes round-off only: the accuracy is that of RK4 with step dt / n_sub, and every state
of an RK4 segment has sum 1 to a few ulp at any horizon.

Finite population (``pop_size`` = N): one multinomial resampling per observation step, i.e. ``dt`` is the
generation time. The drift covariance per unit time is (diag x - x x') / (N dt), so changing dt at fixed N changes
the stochastic dynamics (halving dt doubles the drift per unit time), unlike the deterministic case where dt is
only the observation interval. Types can go extinct: then x_i = 0, ``theta`` / ``logratio`` are +-inf or NaN.

Random segment parameters (documented defaults): f ~ f_scale * N(0, I_K), mu ~ U[mu_range] = U[0.2, 1],
m ~ Dirichlet(1_K), u ~ U[u_range] = U[0.2, 1]; x0 ~ Dirichlet(1_K) unless ``x0`` is given.
Schedules: an explicit list of (mechanism, duration, params) - durations must be multiples of dt, missing params
are drawn at random per trajectory, the list must cover the horizon - or a preset: "alternating"
(selection / migration segments of ``seg_steps`` steps, alternating, random first mechanism), "selection_only",
"migration_only", "mixed_only", "mutation_selection" (selection + mutation acting together, the classical
mutation-selection equation, new f per segment), "rps" (one RPS segment). ``seg_steps`` may be (lo, hi) for random
lengths.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm

from .base import Trajectories, World

MECHS = ("selection", "migration", "mutation", "mixed", "mutation_selection", "rps")
HAS_SEL, HAS_MIG, HAS_MUT = ("selection", "mixed", "mutation_selection"), ("migration", "mixed"), ("mutation", "mutation_selection")
PRESETS = {"alternating": None, "selection_only": "selection", "migration_only": "migration", "mixed_only": "mixed",
           "mutation_selection": "mutation_selection", "rps": "rps"}
RPS_A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
DEFAULTS: dict[str, Any] = dict(K=3, dt=0.1, n_sub=20, schedule="alternating", seg_steps=20, x0=None, pop_size=None,
                                n_obs=1000, mutation="uniform", rps_A=None, f_scale=1.0, mu_range=(0.2, 1.0),
                                u_range=(0.2, 1.0))
CONVENTION = ("Amari: alpha=+1 <-> e-connection, straight lines in natural parameters theta_i = log(x_i/x_K); "
              "alpha=-1 <-> m-connection, straight lines in mixture/expectation parameters eta = x. "
              "Koide & van de Venn (2026, arXiv 2609.08870) use the opposite sign; never mix them silently.")


def theta_coords(x):
    """Natural (e-affine) coordinates theta_i = log(x_i / x_K), i < K."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x[..., :-1]) - np.log(x[..., -1:])


def eta_coords(x):
    """Mixture / expectation (m-affine) coordinates eta = x (all K components; eta_K = 1 - sum of the others)."""
    return np.asarray(x, dtype=np.float64)


def theta_to_x(theta):
    z = np.concatenate([np.asarray(theta, dtype=np.float64), np.zeros(np.shape(theta)[:-1] + (1,))], -1)
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def _normalize(p):
    p = np.clip(p, 0.0, None)
    return p / p.sum(-1, keepdims=True)


def _field(x, mech, p, Q, A):
    v = np.zeros_like(x)
    if mech in HAS_SEL:
        v += x * (p["f"] - (x * p["f"]).sum(-1, keepdims=True))
    if mech in HAS_MIG:
        v += np.asarray(p["mu"])[..., None] * (p["m"] - x)
    if mech in HAS_MUT:
        v += np.asarray(p["u"])[..., None] * (x @ Q)
    if mech == "rps":
        ax = x @ A.T
        v += x * (ax - (x * ax).sum(-1, keepdims=True))
    return v


def _parent_independent(Q):
    """q if Q_ji = q_i for all j != i (mutation target independent of the parent), else None."""
    K = len(Q)
    off = ~np.eye(K, dtype=bool)
    q = np.array([Q[(i + 1) % K, i] for i in range(K)])
    return q if np.allclose(np.where(off, Q, 0.0), np.where(off, q[None, :], 0.0)) else None


def replicator_hodge(A) -> dict:
    """Exact Hodge character of the replicator field v_i = x_i ((Ax)_i - x.Ax) in the Shahshahani metric.

    Write A = S + N (symmetric + antisymmetric parts). On the simplex the metric-dual 1-form is
    omega = sum_i (v_i / x_i) dx_i = sum_i (Ax)_i dx_i (because sum_i dx_i = 0); sum_i (Sx)_i dx_i = d(x.Sx / 2)
    is exact and d omega = -2 sum_{i<j} N_ij dx_i ^ dx_j, which vanishes on the tangent hyperplane {sum w = 0}
    iff N = u 1' - 1 u' (then sum_i (Nx)_i dx_i = d(u.x)). Hence
      * gradient   <=> N_ij = u_i - u_j with u = N[:, K-1];  potential V = x.Sx / 2 + u.x  (dx/dt = grad_g V);
      * div_g v = 1'Ax / 2 - (K/2 + 1) x.Ax + sum_i A_ii x_i  (Shahshahani volume (prod_i x_i)^(-1/2)), which
        vanishes identically on the simplex iff (K + 2) S = 1 l' + l 1' with l = A'1 / 2 + diag(A); the faces are
        invariant, so a divergence-free field has no gradient component: "rotational";
      * both: v = 0 (labelled "gradient", V constant); neither: "mixed".
    For antisymmetric A this reduces to: rotational iff the row sums vanish (zero-sum RPS); gradient iff
    A = u 1' - 1 u' (a disguised selection field). Separately, H = sum_i x*_i log x_i is a constant of motion iff
    A - 1 diag(A)' is antisymmetric (A is zero-sum up to column constants, which do not change the field) and an
    interior equilibrium x* exists ("conservative"); for K = 3 the orbits are then closed around x*. A conservative
    field need not be "rotational" for the Shahshahani volume (antisymmetric A with non-zero row sums: "mixed").
    """
    A = np.asarray(A, dtype=np.float64)
    K, one = len(A), np.ones(len(A))
    S, N = 0.5 * (A + A.T), 0.5 * (A - A.T)
    tol = 1e-10 * max(1.0, np.abs(A).max())
    u = N[:, -1].copy()
    grad = bool(np.abs(N - (u[:, None] - u[None, :])).max() <= tol)
    ell = 0.5 * A.T @ one + np.diag(A)
    rot = bool(np.abs((K + 2) * S - np.outer(one, ell) - np.outer(ell, one)).max() <= tol)
    B = A - np.outer(one, np.diag(A))
    return {"label": "gradient" if grad else "rotational" if rot else "mixed", "gradient": grad,
            "divergence_free": rot, "zero_sum_up_to_columns": bool(np.abs(B + B.T).max() <= tol),
            "S": S, "u": u if grad else None}


class WrightFisher(World):
    name = "wright_fisher"
    obs_modes = ("freq", "counts", "logratio")

    def __init__(self, **params):
        self.cfg = dict(DEFAULTS)
        self.cfg = self._c(**params)

    # ---------------------------------------------------------------- config
    def _c(self, **over) -> dict:
        bad = set(over) - set(DEFAULTS)
        if bad:
            raise TypeError(f"unknown wright_fisher parameters: {sorted(bad)}")
        c = {**self.cfg, **over}
        if int(c["K"]) < 3 or c["dt"] <= 0 or int(c["n_sub"]) < 1:
            raise ValueError("need K >= 3 (for K = 2 every curve is a geodesic of every connection), dt > 0, n_sub >= 1")
        if isinstance(c["schedule"], str) and c["schedule"] not in PRESETS:
            raise ValueError(f"unknown schedule preset {c['schedule']!r}; presets: {sorted(PRESETS)}")
        self._Q(c), self._A(c)
        return c

    @staticmethod
    def _Q(c):
        K, m = int(c["K"]), c["mutation"]
        if isinstance(m, str):
            if m == "uniform":
                return np.ones((K, K)) / K - np.eye(K)
            if m == "cyclic":
                return np.roll(np.eye(K), 1, axis=1) - np.eye(K)
            raise ValueError("mutation must be 'uniform', 'cyclic' or a rate matrix [K, K]")
        Q = np.asarray(m, dtype=np.float64)
        if Q.shape != (K, K) or (Q[~np.eye(K, dtype=bool)] < 0).any() or not np.allclose(Q.sum(1), 0):
            raise ValueError("mutation matrix must be a rate matrix: off-diagonal >= 0, rows sum to 0")
        return Q

    @staticmethod
    def _A(c):
        if c["rps_A"] is None:
            return RPS_A if int(c["K"]) == 3 else None
        A = np.asarray(c["rps_A"], dtype=np.float64)
        if A.shape != (int(c["K"]),) * 2:
            raise ValueError("rps_A must be [K, K]")
        return A

    @staticmethod
    def _x_star(A):
        """Interior equilibrium of the replicator with payoff A (A x* = c 1, sum x* = 1), or None."""
        if A is None:
            return None
        K = len(A)
        M = np.block([[A, -np.ones((K, 1))], [np.ones((1, K)), np.zeros((1, 1))]])
        try:
            xs = np.linalg.solve(M, np.r_[np.zeros(K), 1.0])[:K]
        except np.linalg.LinAlgError:
            return None
        return xs if (xs > 0).all() else None

    def labels(self, **over) -> dict:
        """mechanism -> (alpha label, Hodge character in the Fisher/Shahshahani metric)."""
        c = self._c(**over)
        pim, A = _parent_independent(self._Q(c)) is not None, self._A(c)
        rps = "undefined" if A is None else replicator_hodge(A)["label"]   # undefined: K != 3 and no rps_A
        return {"selection": (+1, "gradient"), "migration": (-1, "gradient"),
                "mutation": (-1 if pim else None, "gradient" if pim else "mixed"),
                "mixed": (None, "gradient"), "mutation_selection": (None, "gradient" if pim else "mixed"),
                "rps": (None, rps)}

    # ---------------------------------------------------------------- dynamics
    def vector_field(self, x, mechanism, f=None, mu=None, m=None, u=None, **over):
        """dx/dt of ``mechanism`` at x [..., K] with the given segment parameters."""
        c = self._c(**over)
        p = dict(f=None if f is None else np.asarray(f, float), mu=mu, m=None if m is None else np.asarray(m, float), u=u)
        return _field(np.asarray(x, dtype=np.float64), mechanism, p, self._Q(c), self._A(c))

    def potential(self, x, mechanism, f=None, mu=None, m=None, u=None, **over):
        """Fisher/Shahshahani potential V with dx/dt = grad_g V (so dV/dt = sum_i v_i^2 / x_i); NaN if none exists."""
        c = self._c(**over)
        x = np.asarray(x, dtype=np.float64)
        if self.labels(**over)[mechanism][1] != "gradient":
            return np.full(x.shape[:-1], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            lx, V = np.log(x), np.zeros(x.shape[:-1])
            if mechanism in HAS_SEL:
                V = V + (x * np.asarray(f, float)).sum(-1)
            if mechanism in HAS_MIG:
                V = V + np.asarray(mu) * (np.asarray(m, float) * lx).sum(-1)
            if mechanism in HAS_MUT:
                V = V + np.asarray(u) * (_parent_independent(self._Q(c)) * lx).sum(-1)
        if mechanism == "rps":
            V = V + self._rps_potential(x, self._A(c))
        return V

    @staticmethod
    def _rps_potential(x, A):
        """V = x.Sx / 2 + u.x of a gradient-type payoff A (see ``replicator_hodge``)."""
        h = replicator_hodge(A)
        return 0.5 * np.einsum("...i,ij,...j->...", x, h["S"], x) + x @ h["u"]

    def _flow(self, x, mech, p, c, Q, A):
        """Exact (selection, migration, mutation) or RK4 flow of one observation interval dt (x on the simplex)."""
        dt = c["dt"]
        if mech == "selection":
            y = x * np.exp(dt * (p["f"] - p["f"].max(-1, keepdims=True)))
            return y / y.sum(-1, keepdims=True)
        if mech == "migration":
            return p["m"] + (x - p["m"]) * np.exp(-p["mu"] * dt)[..., None]
        if mech == "mutation":
            return np.einsum("ni,nij->nj", x, expm(dt * p["u"][:, None, None] * Q[None]))
        n = int(c["n_sub"])
        h = dt / n
        for _ in range(n):
            k1 = _field(x, mech, p, Q, A)
            k2 = _field(x + 0.5 * h * k1, mech, p, Q, A)
            k3 = _field(x + 0.5 * h * k2, mech, p, Q, A)
            k4 = _field(x + h * k3, mech, p, Q, A)
            # back onto the simplex after every substep: round-off in sum(x) is unstable where r < 0 (module docstring)
            x = _normalize(x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
        return x

    def _draw(self, rng, mech, given, c):
        K, p = int(c["K"]), {}
        if mech in HAS_SEL:
            p["f"] = np.asarray(given["f"], float) if "f" in given else c["f_scale"] * rng.standard_normal(K)
        if mech in HAS_MIG:
            p["mu"] = float(given["mu"]) if "mu" in given else float(rng.uniform(*c["mu_range"]))
            p["m"] = np.asarray(given["m"], float) if "m" in given else rng.dirichlet(np.ones(K))
        if mech in HAS_MUT:
            p["u"] = float(given["u"]) if "u" in given else float(rng.uniform(*c["u_range"]))
        if set(given) - set(p):
            raise ValueError(f"parameters {sorted(set(given) - set(p))} do not apply to mechanism {mech!r}")
        return p

    def _segments(self, rng, T, c):
        """[(mechanism, n_steps, params)] covering at least T steps for one trajectory."""
        sch = c["schedule"]
        if isinstance(sch, str):
            if sch == "rps":
                return [("rps", T, {})]
            L, segs, tot = c["seg_steps"], [], 0
            first = int(rng.integers(2)) if sch == "alternating" else 0
            while tot < T:
                n = int(L) if np.isscalar(L) else int(rng.integers(L[0], L[1] + 1))
                mech = ("selection", "migration")[(first + len(segs)) % 2] if sch == "alternating" else PRESETS[sch]
                segs.append((mech, n, self._draw(rng, mech, {}, c)))
                tot += n
            return segs
        segs, tot = [], 0
        for item in sch:
            mech, dur, given = item[0], float(item[1]), (dict(item[2]) if len(item) > 2 and item[2] else {})
            n = int(round(dur / c["dt"]))
            if mech not in MECHS:
                raise ValueError(f"unknown mechanism {mech!r}; mechanisms: {MECHS}")
            if n < 1 or abs(n * c["dt"] - dur) > 1e-9 * max(1.0, dur):
                raise ValueError("segment durations must be positive multiples of dt")
            segs.append((mech, n, self._draw(rng, mech, given, c)))
            tot += n
        if tot < T:
            raise ValueError(f"explicit schedule covers {tot} steps < horizon {T}")
        return segs

    # ---------------------------------------------------------------- sampling
    def sample(self, seed: int, horizon: int, n_traj: int, obs_mode: str, **kwargs) -> Trajectories:
        c = self._c(**kwargs)
        rng, K, T, N = np.random.default_rng(seed), int(c["K"]), horizon, n_traj
        Q, A = self._Q(c), self._A(c)
        x0 =rng.dirichlet(np.ones(K), N) if c["x0"] is None else np.broadcast_to(np.asarray(c["x0"], float), (N, K)).copy()
        code, seg = np.zeros((N, T), np.int64), np.zeros((N, T), np.int64)
        F, M = np.full((N, T, K), np.nan), np.full((N, T, K), np.nan)
        MU, U = np.full((N, T), np.nan), np.full((N, T), np.nan)
        for i in range(N):
            t = 0
            for j, (mech, n, p) in enumerate(self._segments(rng, T, c)):
                sl = slice(t, min(t + n, T))
                code[i, sl], seg[i, sl] = MECHS.index(mech), j
                for arr, key in ((F, "f"), (M, "m"), (MU, "mu"), (U, "u")):
                    if key in p:
                        arr[i, sl] = p[key]
                t += n
                if t >= T:
                    break
        if A is None and (code == MECHS.index("rps")).any():
            raise ValueError("rps needs K = 3 or an explicit rps_A")
        X = np.empty((N, T + 1, K))
        X[:, 0] = x0
        for t in range(T):
            nxt = np.empty((N, K))
            for k in np.unique(code[:, t]):
                idx = code[:, t] == k
                p = dict(f=F[idx, t], mu=MU[idx, t], m=M[idx, t], u=U[idx, t])
                nxt[idx] = self._flow(X[idx, t], MECHS[k], p, c, Q, A)
            if c["pop_size"]:
                nxt = rng.multinomial(int(c["pop_size"]), _normalize(nxt)) / int(c["pop_size"])
            X[:, t + 1] = nxt
        info = self._info(X, code, seg, F, MU, M, U, c)
        meta = dict(world=self.name, seed=seed, horizon=horizon, n_traj=n_traj, obs_mode=obs_mode, params=dict(c))
        return Trajectories(obs=self._observe(X, obs_mode, rng, c), state=X, actions=None,
                            t=np.arange(T + 1) * c["dt"], info=info, meta=meta)

    def _info(self, X, code, seg, F, MU, M, U, c):
        T = code.shape[1]
        lab = self.labels(**{k: c[k] for k in ("K", "mutation", "rps_A")})
        # segment in force at x_t (x_T: the last step's); the extra column = "no mechanism" (horizon 0)
        pad = lambda a: np.concatenate([np.nan_to_num(a), np.zeros((len(a), 1) + a.shape[2:])], 1)
        tt = np.minimum(np.arange(T + 1), T - 1) if T else np.zeros(1, np.int64)
        ct = np.concatenate([code, np.full((len(code), 1), len(MECHS))], 1)[:, tt]
        Ft, MUt, Mt, Ut = (pad(a)[:, tt] for a in (F, MU, M, U))
        isin = lambda names: np.isin(ct, [MECHS.index(m) for m in names])
        q = _parent_independent(self._Q(c))
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.log(X)
            fx = (Ft * X).sum(-1)
            V = np.where(isin(HAS_SEL), fx, 0.0)
            V = V + np.where(isin(HAS_MIG), MUt * (Mt * lx).sum(-1), 0.0)
            if q is not None:
                V = V + np.where(isin(HAS_MUT), Ut * (q * lx).sum(-1), 0.0)
            if lab["rps"][1] == "gradient":
                V = V + np.where(isin(("rps",)), self._rps_potential(X, self._A(c)), 0.0)
            xs = self._x_star(self._A(c))
            H = (xs * lx).sum(-1) if xs is not None else np.full(X.shape[:-1], np.nan)
        grad = np.array([lab[m][1] == "gradient" for m in MECHS] + [False])[ct]
        return dict(
            mechanism=np.array(MECHS)[code], segment_id=seg,
            alpha=np.array([np.nan if lab[m][0] is None else float(lab[m][0]) for m in MECHS])[code],
            hodge=np.array([lab[m][1] for m in MECHS])[code],
            theta=theta_coords(X), eta=X.copy(),
            fitness=F, mu=MU, pool=M, mut_rate=U,
            mean_fitness=np.where(isin(HAS_SEL), fx, np.nan),   # f.x of the segment in force at x_t
            potential=np.where(grad, V, np.nan),                # Fisher potential of the segment in force at x_t
            rps_invariant=H,     # sum x*_i log x_i; conserved only in rps segments of a conservative payoff
        )

    def _observe(self, X, mode, rng, c):
        if mode == "freq":
            return X.astype(np.float32)
        if mode == "logratio":
            return theta_coords(X).astype(np.float32)
        if mode == "counts":
            if rng is None:
                raise ValueError("obs mode 'counts' samples individuals: pass an rng")
            n, K = int(c["n_obs"]), X.shape[-1]
            return (rng.multinomial(n, _normalize(X.reshape(-1, K))) / n).reshape(X.shape).astype(np.float32)
        raise ValueError(f"unknown obs_mode {mode!r}; choose from {self.obs_modes}")

    def observe(self, state, obs_mode, rng=None):
        return self._observe(np.asarray(state, dtype=np.float64), obs_mode, rng, self.cfg)

    def spec(self, obs_mode: str, **over) -> dict:
        K = int(self._c(**over)["K"])
        shapes = {"freq": (K,), "counts": (K,), "logratio": (K - 1,)}
        if obs_mode not in shapes:
            raise ValueError(f"unknown obs_mode {obs_mode!r}")
        return {"obs_shape": shapes[obs_mode], "obs_dtype": "float32", "action": {"type": "none"}}

    # ---------------------------------------------------------------- ground truth
    def ground_truth(self, **cfg) -> dict[str, Any]:
        c = self._c(**cfg)
        lab, Q, A = self.labels(**cfg), self._Q(c), self._A(c)
        xs = self._x_star(A)
        rh = None if A is None else replicator_hodge(A)
        cons = rh is not None and rh["zero_sum_up_to_columns"] and xs is not None    # H is a constant of motion
        orbits = (None if A is None else "H not conserved: orbits are not closed in general" if not cons
                  else "closed around x* (level sets of H)" if int(c["K"]) == 3
                  else "on level sets of H (closed only for K = 3)")
        rps_pot = {"gradient": "x.Sx / 2 + u.x (S = sym(A), u from replicator_hodge(A))",
                   "rotational": "none (rotational)", "mixed": "none (gradient and rotational parts)",
                   "undefined": "none (no payoff matrix: K != 3 and rps_A not given)"}[lab["rps"][1]]
        return dict(
            state_space=f"simplex Delta^{int(c['K']) - 1}", state_dim=int(c["K"]), convention=CONVENTION,
            metric="Shahshahani = Fisher information metric g_ij = delta_ij / x_i on the tangent space sum v = 0",
            coords=dict(theta=theta_coords, eta=eta_coords, theta_to_x=theta_to_x),
            alpha={m: lab[m][0] for m in MECHS}, hodge={m: lab[m][1] for m in MECHS},
            potential={"selection": "f.x (mean fitness)", "migration": "mu sum_i m_i log x_i",
                       "mixed": "f.x + mu sum_i m_i log x_i",
                       "mutation": "u sum_i q_i log x_i if Q parent-independent (Q_ji = q_i), else none",
                       "mutation_selection": "f.x + u sum_i q_i log x_i if Q parent-independent, else none",
                       "rps": rps_pot},
            potential_fn=lambda x, mechanism, **p: self.potential(x, mechanism, **p, **cfg),
            vector_field=lambda x, mechanism, **p: self.vector_field(x, mechanism, **p, **cfg),
            mutation_matrix=Q, mutation_parent_independent=_parent_independent(Q) is not None,
            rps=dict(A=A, x_star=xs, conservative=bool(cons), orbits=orbits, hodge=lab["rps"][1],
                     divergence_free=None if rh is None else rh["divergence_free"],
                     invariant=(lambda x: (xs * np.log(np.asarray(x, float))).sum(-1)) if cons else None),
            schedule=c["schedule"], dt=c["dt"], finite_population=c["pop_size"],
            generation_time=c["dt"] if c["pop_size"] else None,
            drift=(None if not c["pop_size"] else
                   "one Multinomial(pop_size) resampling per step: covariance per unit time (diag x - x x') / (pop_size dt)"),
            per_step_info=("mechanism", "alpha", "hodge", "segment_id", "theta", "eta", "fitness", "mu", "pool",
                           "mut_rate", "mean_fitness", "potential", "rps_invariant"),
            notes=("alpha / Hodge labels describe the deterministic field of the segment; with finite pop_size the "
                   "realised path is noisy and types can go extinct (theta / logratio then +-inf or NaN). "
                   "alpha=None means 'no alpha-geodesic for generic parameters'."),
        )
