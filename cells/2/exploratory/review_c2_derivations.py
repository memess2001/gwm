"""Independent review of cells/2/thought_experiment_2026-11-16.md, claims (a), (b), (c) -- symbolic part.

Re-derives, does not import c2_truth.py.  Convention: Ito SDE dx = b dt + sqrt(2D) dW, generator L = b.grad + D:Hess,
WF diffusion D = (diag x - x x') / (2 Ne)  (Ne = N for the note's per-generation time; Ne = N * dt for zoo/wright_fisher.py,
whose per-unit-time covariance is (diag x - x x') / (N dt) = 2 D).
Charts used: affine (x_1..x_{K-1}), e-flat theta_i = log(x_i/x_K), and the NON-flat chart s_i = sqrt(x_i).
"""
import sympy as sp

def header(t):
    print("\n" + "=" * 8 + " " + t)

# ---------------------------------------------------------------- (a) D^{-1} = 2Ne * Shahshahani on the tangent space
header("(a) covariance = inverse Shahshahani metric (K-1 affine chart)")
Ne = sp.symbols("N_e", positive=True)
for K in (3, 4):
    z = sp.symbols(f"x1:{K}", positive=True)                     # x_1..x_{K-1}
    xK = 1 - sum(z)
    D = sp.Matrix(K - 1, K - 1, lambda i, j: (z[i] * (1 if i == j else 0) - z[i] * z[j]) / (2 * Ne))
    g = sp.Matrix(K - 1, K - 1, lambda i, j: (1 / z[i] if i == j else 0) + 1 / xK)   # Shahshahani / Fisher metric
    print(f"K={K}: simplify(D * g) == I/(2Ne):", sp.simplify(D * g - sp.eye(K - 1) / (2 * Ne)) == sp.zeros(K - 1))
    print(f"      det g = 1/prod x_i:", sp.simplify(g.det() - 1 / (sp.prod(z) * xK)) == 0)

# ---------------------------------------------------------------- (b) div D is a D-gradient -- in which charts?
header("(b) Ito correction div D")
def divD(Dm, q):
    n = len(q)
    return sp.Matrix([sum(sp.diff(Dm[i, j], q[j]) for j in range(n)) for i in range(n)])

def curl2(F, q):     # exterior derivative of the 1-form F_1 dq_1 + F_2 dq_2 (K = 3)
    return sp.simplify(sp.diff(F[0], q[1]) - sp.diff(F[1], q[0]))

for K in (3, 4):
    z = sp.symbols(f"x1:{K}", positive=True)
    xs = list(z) + [1 - sum(z)]
    D = sp.Matrix(K - 1, K - 1, lambda i, j: (z[i] * (1 if i == j else 0) - z[i] * z[j]) / (2 * Ne))
    dv = divD(D, z)
    print(f"K={K} affine chart: div D =", [sp.factor(e) for e in dv])
    psi = sum(sp.log(e) for e in xs)                               # candidate potential sum_i log x_i
    gradpsi = sp.Matrix([sp.diff(psi, zi) for zi in z])
    print(f"      div D - D grad(sum log x_i) == 0:", sp.simplify(dv - D * gradpsi) == sp.zeros(K - 1, 1))

# e-flat chart theta (K = 3)
th = sp.symbols("t1:3", real=True)
Z = 1 + sp.exp(th[0]) + sp.exp(th[1])
xt = [sp.exp(th[0]) / Z, sp.exp(th[1]) / Z, 1 / Z]
Jx = sp.Matrix(2, 2, lambda i, j: sp.diff(xt[i], th[j]))           # dx/dtheta
Dx = lambda X: sp.Matrix(2, 2, lambda i, j: (X[i] * (1 if i == j else 0) - X[i] * X[j]) / (2 * Ne))
Dth = sp.simplify(Jx.inv() * Dx(xt) * Jx.inv().T)
F = sp.simplify(Dth.inv() * divD(Dth, th))
print("theta chart (e-flat): curl of D^{-1} div D =", curl2(F, th), " -> div D is a D-gradient here too")

# NON-flat chart s_i = sqrt(x_i) (K = 3)
s = sp.symbols("s1:3", positive=True)
xsq = [s[0] ** 2, s[1] ** 2, 1 - s[0] ** 2 - s[1] ** 2]
Jxs = sp.Matrix(2, 2, lambda i, j: sp.diff(xsq[i], s[j]))          # dx/ds
Ds = sp.simplify(Jxs.inv() * Dx(xsq) * Jxs.inv().T)
Fs = sp.simplify(Ds.inv() * divD(Ds, s))
c = curl2(Fs, s)
print("sqrt chart (sphere chart, not affine): curl of D^{-1} div D =", sp.factor(c))
# a generic nonlinear chart: y1 = x1 + x2^2, y2 = x2
y = sp.symbols("y1 y2", real=True)
z1_, z2_ = sp.symbols("x1 x2", positive=True)
Dxy = Dx([z1_, z2_, 1 - z1_ - z2_])
Jyx = sp.Matrix([[1, 2 * z2_], [0, 1]])
Dy = (Jyx * Dxy * Jyx.T).subs({z1_: y[0] - y[1] ** 2, z2_: y[1]})
cy = sp.factor(curl2(sp.simplify(Dy.inv() * divD(Dy, y)), y))
print("generic shear chart y1 = x1 + x2^2: curl of D^{-1} div D =", cy)
print("   -> nonzero: div D is a D-gradient in the affine, theta and sqrt charts, but NOT in a generic chart;")
print("      the chart-free statement is about b - div D (next block), so decompose b - div D unless the chart is affine/theta.")

# chart-independence of f_irr = b - div D - D grad log rho (generic b, generic rho), x-chart vs sqrt chart
header("(c) f_irr = b - div D - D grad log rho is a chart-independent vector field (x chart vs sqrt chart)")
z1, z2 = sp.symbols("x1 x2", positive=True)
b1, b2, r = sp.Function("b1")(z1, z2), sp.Function("b2")(z1, z2), sp.Function("rho")(z1, z2)
Dxx = Dx([z1, z2, 1 - z1 - z2])
bx = sp.Matrix([b1, b2])
firr_x = bx - divD(Dxx, (z1, z2)) - Dxx * sp.Matrix([sp.diff(sp.log(r), z1), sp.diff(sp.log(r), z2)])
sub = {z1: s[0] ** 2, z2: s[1] ** 2}
# Ito drift in s: b_s = (ds/dx) b + sum_jk d2 s/dx_j dx_k D_jk   (generator D:Hess)
sfun = [sp.sqrt(z1), sp.sqrt(z2)]
J_sx = sp.Matrix(2, 2, lambda i, j: sp.diff(sfun[i], (z1, z2)[j]))
ito = sp.Matrix([sum(sp.diff(sfun[i], (z1, z2)[j], (z1, z2)[k]) * Dxx[j, k] for j in range(2) for k in range(2)) for i in range(2)])
bs = (J_sx * bx + ito).subs(sub)
rho_s = r.subs(sub) * abs(Jxs.det())                                   # density w.r.t. ds
firr_s = bs - divD(Ds, s) - Ds * sp.Matrix([sp.diff(sp.log(rho_s), si) for si in s])
diff = sp.simplify(firr_s - (J_sx * firr_x).subs(sub))
print("pushforward(f_irr in x chart) - f_irr in s chart == 0:", diff == sp.zeros(2, 1))

# ---------------------------------------------------------------- (c) Wright formula & boundary flux
header("(c) PIM mutation + selection: Wright's density gives J = 0; boundary normal flux; cyclic has curl")
u, Ne_ = sp.symbols("u", positive=True), Ne
q = sp.symbols("q1:4", positive=True)
f = sp.symbols("f1:4", real=True)
x = [z1, z2, 1 - z1 - z2]
def drift(xv, Q, u, f):
    xf = sum(xv[k] * f[k] for k in range(3))
    sel = [xv[i] * (f[i] - xf) for i in range(3)]
    mut = [u * sum(xv[j] * Q[j][i] for j in range(3)) for i in range(3)]
    return sp.Matrix([sel[i] + mut[i] for i in range(2)])
qs = sum(q)
Q_pim = [[(q[i] if i != j else 0) for i in range(3)] for j in range(3)]
for j in range(3):
    Q_pim[j][j] = -sum(Q_pim[j][i] for i in range(3) if i != j)
b = drift(x, Q_pim, u, f)
rho = sp.prod([x[i] ** (2 * Ne_ * u * q[i] - 1) for i in range(3)]) * sp.exp(2 * Ne_ * sum(f[i] * x[i] for i in range(3)))
Jcur = b * rho - sp.Matrix([sum(sp.diff(Dxx[i, j] * rho, (z1, z2)[j]) for j in range(2)) for i in range(2)])
print("PIM: J = b rho - div(D rho) with Wright's rho == 0:", sp.simplify(sp.expand(Jcur / rho)) == sp.zeros(2, 1))
Q_cyc = [[-1, 1, 0], [0, -1, 1], [1, 0, -1]]                          # zoo 'cyclic': x Q = (x3 - x1, x1 - x2, x2 - x3)
bc = drift(x, Q_cyc, u, f)
Fc = Dxx.inv() * (bc - divD(Dxx, (z1, z2)))
cc = sp.factor(curl2(Fc, (z1, z2)))
print("cyclic: curl of D^{-1}(b - div D) =", cc, " (nonzero for u > 0 -> no potential -> J != 0, EP > 0)")
Fp = Dxx.inv() * (b - divD(Dxx, (z1, z2)))
print("PIM:    curl of D^{-1}(b - div D) =", sp.simplify(curl2(Fp, (z1, z2))))

# the absolute Fisher norm of b under rho: integrand rho * b' D^{-1} b near the face x1 = 0
header("(c) integrability near a face: rho * |b|^2_{D^{-1}} ~ x1^(alpha-2), alpha = 2 Ne u q_1")
e = sp.symbols("epsilon", positive=True)
a = sp.symbols("alpha", positive=True)
print("int_0^1 x^(alpha-2) dx =", sp.integrate(e ** (a - 2), (e, 0, 1), conds="piecewise"))
print("-> |b|_{rho,Fisher} (and the gradient part) are infinite when 2 Ne u q_i <= 1; f_irr can still be L2.")
print("   cyclic mutation: inflow into type i on its face is u x_{i-1} -> 0 at a vertex, so alpha(y) < 1 near")
print("   every vertex for every Ne, u: the rho-weighted Fisher norm of b is always infinite (ratio readouts break).")

# ---------------------------------------------------------------- non-Fisher noise (steelman, Section 3)
header("non-Fisher noise: fluctuating selection (fitness noise covariance S per unit time) on top of WF drift")
# x = softmax(Y), dY = f dt + dW_f, Cov(dW_f) = S dt.  Ito drift of x gets + P (diag(S)/2 - S x) (replicator with
# symmetric payoff -> Shahshahani gradient), and the covariance becomes P/Ne + P S P.
sig = sp.symbols("sigma", positive=True)
S = sp.diag(sig ** 2, 0, 0)                                           # only type 1 has a fluctuating fitness
X3 = sp.Matrix(x)
P3 = sp.diag(*x) - X3 * X3.T
nd = P3 * (sp.Matrix([S[i, i] for i in range(3)]) / 2 - S * X3)      # noise-induced drift (3-vector)
# check it against Ito's formula on softmax:
Y = sp.symbols("y1:4", real=True)
sm = [sp.exp(Y[i]) / sum(sp.exp(Y[k]) for k in range(3)) for i in range(3)]
ito3 = [sp.Rational(1, 2) * sum(sp.diff(sm[i], Y[j], Y[k]) * S[j, k] for j in range(3) for k in range(3)) for i in range(3)]
chk = [sp.simplify(ito3[i] - nd[i].subs({z1: sm[0], z2: sm[1]})) for i in range(3)]
print("noise-induced drift = P(diag S / 2 - S x)  (Ito on softmax):", all(c_ == 0 for c_ in chk))
b_env = b + sp.Matrix([nd[0], nd[1]])
G = sp.Matrix(2, 2, lambda i, j: (1 / x[i] if i == j else 0) + 1 / x[2])
print("b_env is a Fisher (Shahshahani) gradient: curl(G b_env) =", sp.simplify(curl2(G * b_env, (z1, z2))))
D_env = Dxx + sp.Matrix(2, 2, lambda i, j: (P3 * S * P3)[i, j] / 2)
F_env = D_env.inv() * (b_env - divD(D_env, (z1, z2)))
num = {u: sp.Rational(1, 2), Ne_: 10, sig: 1, q[0]: sp.Rational(1, 3), q[1]: sp.Rational(1, 3), q[2]: sp.Rational(1, 3),
       f[0]: sp.Rational(4, 5), f[1]: -sp.Rational(3, 10), f[2]: -sp.Rational(1, 2)}
cf = curl2(F_env.subs(num), (z1, z2))
vals = [float(cf.subs({z1: p1, z2: p2})) for p1, p2 in ((0.2, 0.3), (0.5, 0.2), (0.3, 0.6))]
print("D_env^{-1}(b_env - div D_env) curl at 3 points (uniform mutation + selection + type-1 fitness noise):", vals)
print("-> nonzero: the process is irreversible (EP > 0) although every mechanism in b is a Fisher gradient.")
