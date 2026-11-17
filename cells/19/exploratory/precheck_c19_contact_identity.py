"""Cell 19 precheck (2026-11-16): ground truth / symbolic only, no model is trained.

Contact Hamiltonian equations (Bravetti 2017; contact form dS - p dq), n degrees of freedom:
    q' = dH/dp,   p' = -dH/dq - p dH/dS,   S' = p . dH/dp - H.
1. When is the observable (q, p) dynamics closed (independent of the unobserved S)?  Claim: iff H = H0(q, p) + g S
   with a constant g. Then p' = -dH0/dq - g p: conformally symplectic, i.e. "HNN + scalar linear momentum
   damping". Its (q, p) divergence is -n g, uniform in phase space.
2. The zoo damped pendulum is exactly this case (H0 = p^2/2 - w0^2 cos q, g = 2 gamma), compared numerically with
   zoo.Pendulum.vector_field. In 1 dof every force a(q) is conservative, so the same field is also
   "second-order model q'' = a(q) - g q'" (the Gruver-style control).
3. Gauge: H0 -> H0 + c leaves (q, p) unchanged and shifts S' by -c; S(0) is free. So "S = entropy" and
   "dH/dS = dissipation rate" are fixed only up to this gauge, and dH/dS = g is the scalar friction coefficient.
4. Where contact (S-independent) is misspecified (worlds not in the zoo): the trace of the Jacobian at a fixed point
   is coordinate-free, and a conformally symplectic flow has trace -n g at every fixed point. Double pendulum with
   joint viscous friction (-c q'), and Van der Pol: the traces differ between fixed points or invariant sets, so no
   chart or symplectic form makes them contact. A port-Hamiltonian model R grad H with R = diag(0, c) represents
   joint friction exactly.
"""
import sys
import numpy as np
import sympy as sp

sys.path.insert(0, "/home/user/gwm_repo")
from zoo.pendulum import Pendulum

q, p, S, g, c, w0 = sp.symbols("q p S g c omega0", real=True)

def contact_field(H, qs, ps, S):
    dq = [sp.diff(H, pi) for pi in ps]
    dp = [-sp.diff(H, qi) - pi * sp.diff(H, S) for qi, pi in zip(qs, ps)]
    dS = sum(pi * sp.diff(H, pi) for pi in ps) - H
    return dq, dp, dS

# -------------------------------------------------------------------------------------------- 1
print("1. S-dependence of the (q, p) part of the contact field")
Hgen = sp.Function("H")(q, p, S)
dq, dp, _ = contact_field(Hgen, [q], [p], S)
print("   d/dS q' =", sp.simplify(sp.diff(dq[0], S)))
print("   d/dS p' =", sp.simplify(sp.diff(dp[0], S)))
A = sp.Function("A")(q, p); B = sp.Function("B")(q, S)
dq, dp, _ = contact_field(A + B, [q], [p], S)
e = sp.expand(sp.diff(dp[0], S))
print("   with H = A(q,p) + B(q,S):  d/dS p' =", e, " -> must vanish for every p: B_SS = 0 and B_qS = 0 -> B = g S + C(q)")
for name, H in (("H0 + g S", sp.Function("H0")(q, p) + g * S), ("H0 + g S^2/2", sp.Function("H0")(q, p) + g * S**2 / 2),
                ("H0 + S*k(q)", sp.Function("H0")(q, p) + S * sp.Function("k")(q))):
    dq, dp, _ = contact_field(H, [q], [p], S)
    print(f"   {name:13s}: d/dS q' = {sp.simplify(sp.diff(dq[0], S))},  d/dS p' = {sp.simplify(sp.diff(dp[0], S))},"
          f"  div_(q,p) = {sp.simplify(sp.diff(dq[0], q) + sp.diff(dp[0], p))}")

# n = 2 version of the same statement
q1, q2, p1, p2 = sp.symbols("q1 q2 p1 p2", real=True)
H2 = sp.Function("H0")(q1, q2, p1, p2) + g * S
dq, dp, _ = contact_field(H2, [q1, q2], [p1, p2], S)
div2 = sum(sp.diff(a, x) for a, x in zip(dq + dp, [q1, q2, p1, p2]))
print("   n = 2, H = H0 + g S: div_(q,p) =", sp.simplify(div2), " (uniform)")

# -------------------------------------------------------------------------------------------- 2
print("\n2. zoo damped pendulum vs contact field with H = p^2/2 - w0^2 cos q + 2 gamma S")
rng = np.random.default_rng(0)
for gamma, om0 in ((0.05, 1.0), (0.3, 1.7)):
    H = p**2 / 2 - w0**2 * sp.cos(q) + g * S
    dq, dp, _ = contact_field(H, [q], [p], S)
    f = sp.lambdify((q, p, S, g, w0), [dq[0], dp[0]], "numpy")
    st = np.c_[rng.uniform(-np.pi, np.pi, 1000), rng.uniform(-3, 3, 1000)]
    Svals = rng.normal(0, 10, 1000)                                    # any S: the (q, p) field ignores it
    mine = np.stack(f(st[:, 0], st[:, 1], Svals, 2 * gamma, om0), -1)
    zoo_v = Pendulum(gamma=gamma, omega0=om0).vector_field(st)
    print(f"   gamma={gamma}, omega0={om0}: max |contact - zoo| = {np.abs(mine - zoo_v).max():.2e};"
          f" zoo divergence {Pendulum(gamma=gamma, omega0=om0).ground_truth()['divergence']:+.3f}, contact -g = {-2 * gamma:+.3f}")
print("   energy: d/dt H0 along the contact flow =",
      sp.simplify(sp.diff(p**2 / 2 - w0**2 * sp.cos(q), q) * dq[0] + sp.diff(p**2 / 2 - w0**2 * sp.cos(q), p) * dp[0]),
      " (zoo: dE/dt = -2 gamma omega^2)")

# -------------------------------------------------------------------------------------------- 3
print("\n3. gauge of S")
H0 = sp.Function("H0")(q, p); k = sp.symbols("k", real=True)
f1 = contact_field(H0 + g * S, [q], [p], S); f2 = contact_field(H0 + k + g * S, [q], [p], S)
print("   (q', p') change under H0 -> H0 + k:", [sp.simplify(a - b) for a, b in zip(f1[0] + f1[1], f2[0] + f2[1])],
      "; S' changes by", sp.simplify(f2[2] - f1[2]))

# -------------------------------------------------------------------------------------------- 4
print("\n4. fixed-point traces (coordinate-free) where contact is misspecified")
# double pendulum, unit masses and lengths, absolute angles th1, th2; joint viscous friction -c * (relative rates)
th1, th2, w1, w2 = sp.symbols("th1 th2 w1 w2", real=True)
G = sp.Integer(1)
Mq = sp.Matrix([[2, sp.cos(th1 - th2)], [sp.cos(th1 - th2), 1]])
Vq = -2 * G * sp.cos(th1) - G * sp.cos(th2)
P = sp.Matrix([p1, p2])
qd = Mq.inv() * P
T = (P.T * Mq.inv() * P)[0] / 2
Hd = sp.simplify(T + Vq)
def field(damp):
    dq_ = [sp.diff(Hd, p1), sp.diff(Hd, p2)]
    force = sp.Matrix([-sp.diff(Hd, th1), -sp.diff(Hd, th2)])
    if damp == "joint":       # friction torques -c*(w1) at the pivot and -c*(w2 - w1) at the elbow
        Dm = sp.Matrix([[2, -1], [-1, 1]]); force = force - c * Dm * qd
    elif damp == "mass":      # mass-proportional (conformally symplectic): -c * p
        force = force - c * P
    return sp.Matrix(dq_ + list(force))
X = sp.Matrix([th1, th2, p1, p2])
for damp in ("joint", "mass"):
    F = field(damp)
    tr = sp.simplify(F.jacobian(X).trace())
    vals = {fp: sp.nsimplify(sp.simplify(tr.subs({th1: a, th2: b, p1: 0, p2: 0, c: 1})))
            for fp, (a, b) in {"down-down": (0, 0), "down-up": (0, sp.pi), "up-down": (sp.pi, 0), "up-up": (sp.pi, sp.pi)}.items()}
    print(f"   double pendulum, {damp:5s} damping: tr(Jacobian) at fixed points (c = 1): {vals}")
# Van der Pol: trace at the origin vs the mean divergence on the limit cycle
mu = 1.0
from scipy.integrate import solve_ivp
sol = solve_ivp(lambda t, y: [y[1], mu * (1 - y[0]**2) * y[1] - y[0]], (0, 200), [2.0, 0.0], max_step=0.01, rtol=1e-9)
tt = sol.t > 100
print(f"   Van der Pol (mu = {mu}): trace at the origin {mu:+.3f}; time-mean divergence on the limit cycle "
      f"{np.mean(mu * (1 - sol.y[0][tt]**2)):+.3f} (the two must be equal for a conformally symplectic flow)")
