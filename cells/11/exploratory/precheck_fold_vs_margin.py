"""Cell 11 pre-prereg check (exact zoo pendulum only, no learning, no runs/).

World: zoo damped pendulum with a constant torque u (theta'' = -w0^2 sin theta - 2 gamma theta' + u), the textbook
"tilted washboard" (Strogatz, Nonlinear Dynamics and Chaos, sec. 8.5). It has the two kinds of transition a
singularity-map planner would care about:
  * a fold (saddle-node) of the locked equilibrium at u = w0^2: local, a Thom catastrophe of V = -w0^2 cos theta - u theta;
  * a homoclinic bifurcation at u_c(gamma) < w0^2 where the running periodic orbit is born: global, not a singularity
    of any potential, and it sets the lower edge of the hysteresis loop (Melnikov, small gamma: u_c ~ (4/pi) 2 gamma w0).
Part A: on the equilibrium branch, the singularity set is exactly the zero set of det J (implicit function theorem);
        the "stability margin" max Re(lambda) is flat at -gamma until the last 1e-4 of the branch, so the same-source
        Jacobian baseline must be det J / sigma_min / |lambda|_min, not Re(lambda). d theta*/du = 1/det J (cell 21's
        sensitivity). Normal-form (square-root) knowledge enters only as an extrapolation exponent on the same margin.
Part B: hysteresis edge u_c(gamma) by rollouts of the zoo simulator; neither det J nor the catastrophe set sees it.
Part C: a planning consequence at gamma = 0.1, u = 0.5: the locked equilibrium is far from its fold, yet a kicked
        pendulum stays running for ever; only a rollout (or a basin computation) says so."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm_repo")
from zoo.pendulum import Pendulum

W0 = 1.0


def jac_fd(world, th, u, gamma, eps=1e-6):
    """Jacobian of the zoo vector field by central differences (what one would do with a learned f_hat)."""
    J = np.zeros((2, 2))
    s = np.array([th, 0.0])
    for j in range(2):
        e = np.zeros(2); e[j] = eps
        J[:, j] = (world.vector_field(s + e, u=u, gamma=gamma) - world.vector_field(s - e, u=u, gamma=gamma)) / (2 * eps)
    return J


def newton_branch(world, us, gamma):
    """Continuation of the locked equilibrium with Newton on f(theta, 0) = 0 from the previous solution."""
    th, out = 0.0, []
    for u in us:
        ok = False
        for _ in range(50):
            f = world.vector_field(np.array([th, 0.0]), u=u, gamma=gamma)[1]
            d = jac_fd(world, th, u, gamma)[1, 0]
            if abs(d) < 1e-12:
                break
            th -= f / d
            if abs(f) < 1e-13:
                ok = True
                break
        out.append((u, th if ok else np.nan))
        if not ok:
            break
    return out


def part_a(gamma=0.1):
    w = Pendulum(gamma=gamma)
    print(f"## A. locked branch, gamma = {gamma}: singularity set = zero set of det J")
    print("   u        theta*    det J     sigma_min  |lam|_min  maxRe(lam)  dtheta*/du  1/detJ")
    for u, th in newton_branch(w, [0.0, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0, 1.001], gamma):
        if np.isnan(th):
            print(f"   {u:<8g} Newton fails: no equilibrium (past the fold)")
            continue
        J = jac_fd(w, th, u, gamma)
        lam = np.linalg.eigvals(J)
        sens = (np.arcsin(min(u + 1e-7, 1.0)) - np.arcsin(u - 1e-7)) / 2e-7 if u < 1 else np.inf
        print(f"   {u:<8g} {th:8.5f}  {np.linalg.det(J):8.5f}  {np.linalg.svd(J, compute_uv=False)[-1]:8.5f}   "
              f"{np.abs(lam).min():8.5f}   {lam.real.max():9.5f}   {sens:9.3f}  {1 / np.linalg.det(J):9.3f}")
    # fold location extrapolated from subcritical samples u in [0.6, 0.8] (what a model trained below the fold gives)
    us = np.linspace(0.6, 0.8, 5)
    det = np.sqrt(W0 ** 4 - us ** 2)
    lin = np.polyfit(us, det, 1); sq = np.polyfit(us, det ** 2, 1)
    print(f"   extrapolated fold from u in [0.6, 0.8]: det J linear -> {-lin[1] / lin[0]:.3f};"
          f" (det J)^2 linear (fold normal-form exponent) -> {-sq[1] / sq[0]:.3f}; max Re(lam) -> never crosses 0; truth 1.000")


def running(world, us, gamma, T=3000):
    """True where a pendulum started fast stays on the running orbit under constant torque u (the zoo simulator)."""
    s0 = np.stack([np.zeros_like(us), np.maximum(us / (2 * gamma), 0) + 3.0], -1)
    act = np.broadcast_to(us[:, None, None], (len(us), T, 1))
    traj = world.rollout(s0, T, actions=act, gamma=gamma, n_sub=10)
    return traj[:, -1000:, 1].sum(1) * world.cfg["dt"] > 2 * np.pi   # at least one full turn in the last 100 time units


def part_b():
    print("\n## B. lower edge of the hysteresis loop (homoclinic, global) vs the fold (local), zoo rollouts")
    print("   gamma   u_c (rollout)   Melnikov 8 gamma/pi   fold u   width of bistable band")
    w = Pendulum()
    for gamma in (0.05, 0.1, 0.2, 0.3, 0.45, 0.55, 0.6, 0.65, 0.8):
        us = np.linspace(0.005, 0.999, 200)
        r = running(w, us, gamma)
        if not r.any():
            print(f"   {gamma:<6g}  none below 1    {8 * gamma / np.pi:8.3f}            1.000    0 (no bistability)")
            continue
        assert r[np.argmax(r):].all(), "running set should be an interval [u_c, 1)"
        uc = us[np.argmax(r)]
        print(f"   {gamma:<6g}  {uc:8.3f}        {8 * gamma / np.pi:8.3f}            1.000    {max(0.0, 1 - uc):.3f}")


def part_c(gamma=0.1, u=0.5):
    print(f"\n## C. planning consequence, gamma = {gamma}, constant u = {u}")
    w = Pendulum(gamma=gamma)
    th = np.arcsin(u)
    print(f"   locked equilibrium theta* = {th:.3f}, det J = {np.linalg.det(jac_fd(w, th, u, gamma)):.3f}"
          " (far from the fold: map and margin both say 'safe')")
    starts = np.array([[th, 0.0], [th, 1.0], [th, 2.0], [th, 3.0]])
    T = 3000
    tr = w.rollout(starts, T, actions=np.full((4, T, 1), u))
    for s, x in zip(starts, tr):
        print(f"   start omega = {s[1]:.1f}: mean omega over the last 100 time units = {x[-1000:, 1].mean():6.3f}"
              f" -> {'RUNNING' if x[-1000:, 1].sum() * w.cfg['dt'] > 2 * np.pi else 'locked'}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
