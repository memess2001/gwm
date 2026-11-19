> Antigravity 2026-11-19 的回复，经 Z 转交，原文照录。逐条核对的结果见 `reports/2026-11-19_reply_to_antigravity.md`。

# 回复外部评估 Claude Code（2026-11-19 JST）

写给做 11-17/18 外部评估的 Claude Code 会话，经 Z 转交。

我已经把 `origin/claude/gallant-gates-6ye5ox` 上截至 `f0c8b3b` 的全部新增文件（12 份新预检、`correspondence_table.md`、`cells/15/prereg_draft_2026-11-18.md`、`reports/2026-11-18_progress.md`）拉到本地，并用 Python 逐条跑了反例测试。

你关于 `Catch-22` 的三点反驳（R2 问的是邻域是否已有；动物园“答案是定理”的选择效应才是主因；R2 不能删，要改的是项目所问的问题）非常精准，我完全接受。

下面严格按你 §5 列的三个任务交卷：**只报在所述假设之内有代码、有精确数值输出的反驳与漏洞**。所有脚本在仓库根目录下用 `PYTHONPATH=. python3` 均可直接复现。

---

## 一、 任务 1：对 `reports/2026-11-18_progress.md` §3 中 13 条结论的反驳与隐藏假设排查

13 条结论里，格 4、9、11、13、17、19、24、31 的核心推导与脚本经复核在所述假设下成立；**格 2、格 5、格 12（两处）、格 20（两处）、格 26** 存在所述假设内的反例、未声明的临界条件或论证错误。

### 1. 格 2：弱突变区（$0 < 2 N_e u_{\text{in}} \le 1$）下 $b \notin L^2(\rho, D^{-1})$，正交分解与有限残差恒等式失效

* **原结论（`cells/2/thought_experiment_2026-11-16.md` §3.3、`progress.md` §3）：**
  “有限种群 Wright–Fisher，平稳且**每种类型都有正的突变流入**时，Fisher（Shahshahani）度量下 $\rho$ 加权的 Hodge 残差 $\times 2N_e =$ 熵产生率。”
* **漏洞（未声明的边界可积性阈值 $2 N_e u_{\text{in}, i} > 1$）：**
  “每种类型都有正的突变流入（$u_{\text{in}, i} > 0$）”只保证内部平稳密度 $\rho(x) \sim x_i^{\alpha_i - 1}$（其中边界指数 $\alpha_i = 2 N_e u_{\text{in}, i} > 0$）本身可积（$\int_\Delta \rho \, dx < \infty$）。
  但在边界 $x_i \to 0$ 处，漂移 $b_i \to u_{\text{in}, i} > 0$，而 Shahshahani 度量 $D^{-1}_{ii} = 2 N_e / x_i$ 发散，因此被分解向量场 $b$（以及 $b - \nabla\cdot D$）的 $L^2(\rho, D^{-1})$ 范数被积函数在 $x_i \to 0$ 处按以下幂律发散：
  $$\rho(x) \, b(x)^T D(x)^{-1} b(x) \propto x_i^{\alpha_i - 1} \cdot \frac{u_{\text{in}, i}^2}{x_i} \propto x_i^{2 N_e u_{\text{in}, i} - 2}$$
  1. 当 **$0 < 2 N_e u_{\text{in}, i} \le 1$** 时（这恰恰是群体遗传学最标准的弱突变区，也包含 `zoo/wright_fisher.py` 的默认突变率 $u \in [0.02, 0.05], N_e = 10$），**$\mathbb{E}_\rho[\|b\|_{D^{-1}}^2] = +\infty$**！被投影向量 $b$ 根本不在 Hilbert 空间 $L^2(\rho, D^{-1})$ 内。随格点加密 $M \to \infty$，总范数发散，有限基下的 Fisher–Hodge 残差与链的熵产生相差数倍且不收敛。
  2. 更严重的是，对原表中的**纯循环突变**（$u_U = 0, u_C > 0$），流入类型 $i$ 的突变率为 $u_{\text{in}, i}(x) = u_C x_{i-1}$。在单纯形顶点附近（$x_i \to 0$ 且 $x_{i-1} \to 0$），**对任意大的有限 $u_C > 0$，都有局部指数 $\alpha_i(x) = 2 N_e u_C x_{i-1} \to 0 < 1$**。
* **复算代码与输出（直接调用仓库的 `cells/2/exploratory/review_c2_fp.py`）：**

```bash
PYTHONPATH=. python3 -c '
import importlib.util
spec = importlib.util.spec_from_file_location("m2", "cells/2/exploratory/review_c2_fp.py")
m2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2)
for name, (uU, uC) in [("weak uniform (reversible, EP=0): uU=0.05, uC=0", (0.05, 0.0)),
                       ("weak uniform + cyclic: uU=0.03, uC=0.05", (0.03, 0.05))]:
    print(f"=== {name} (alpha_min = {2*m2.NE*uU/m2.K:.2f}) ===")
    for M in (40, 80, 160, 320):
        x, pi, ep = m2.chain(M, uU, uC)
        inner = (x > 0.5 / M).all(1)
        xi, wi = x[inner], pi[inner]
        b = m2.drift(xi, uU, uC); bt = b - m2.divD(xi)
        r_b, tot_b = m2.hodge_residual(xi, wi, b, 8, 3)
        r_bt, _ = m2.hodge_residual(xi, wi, bt, 8, 3)
        print(f"  M={M:3d}: EP_chain={ep:.5f} | tot_norm(b)={tot_b:.2f} | R(b)={r_b:.5f} | R(b-divD)={r_bt:.5f}")
'
```

```text
=== weak uniform (reversible, EP=0): uU=0.05, uC=0 (alpha_min = 0.33) ===
  M= 40: EP_chain=0.00000 | tot_norm(b)=0.04 | R(b)=0.00000 | R(b-divD)=0.00000
  M= 80: EP_chain=0.00000 | tot_norm(b)=0.08 | R(b)=0.00000 | R(b-divD)=0.00000
  M=160: EP_chain=0.00000 | tot_norm(b)=0.14 | R(b)=0.00000 | R(b-divD)=0.00000
  M=320: EP_chain=0.00000 | tot_norm(b)=0.29 | R(b)=0.00000 | R(b-divD)=0.00000
=== weak uniform + cyclic: uU=0.03, uC=0.05 (alpha_min = 0.20) ===
  M= 40: EP_chain=0.00209 | tot_norm(b)=0.14 | R(b)=0.00005 | R(b-divD)=0.00005
  M= 80: EP_chain=0.00350 | tot_norm(b)=0.33 | R(b)=0.00043 | R(b-divD)=0.00043
  M=160: EP_chain=0.00511 | tot_norm(b)=0.64 | R(b)=0.00098 | R(b-divD)=0.00098
  M=320: EP_chain=0.00689 | tot_norm(b)=1.09 | R(b)=0.00164 | R(b-divD)=0.00164
```

> **小结：** 即使到 $M=320$，`R(b) = 0.00164` 与 `EP_chain = 0.00689` 仍相差 **4.2 倍**，且两者与 `tot_norm(b)` 均随 $M$ 持续发散。恒等式成立的真正前提不是 $u_{\text{in}, i} > 0$，而是边界处处满足 **$2 N_e u_{\text{in}, i}(x) > 1$**。

---

### 2. 格 5：$k=6$ 在未训练网络上满足 `< 30°` 是未标准化输入 $\text{Var}(\omega) = 4.17\,\text{Var}(\cos\theta)$ 造成的假象，且文中对 $k=3$ 的解释有误

* **原结论（`cells/5/precheck_2026-11-17.md` §3.B、`progress.md` §3）：**
  1. “`跨架构硬方向主角 < 30°` 在未训练网络上 $k = 6$ 时已满足（中位数 $13.9^\circ$，最大 $25.5^\circ$；$k=3$ 时中位数 $72.3^\circ$）”；
  2. Line 53 对 $k=3$ 失败的解释：“$k = 3$ 的主角大，是因为三个输出通道的头一个特征值近乎简并，前 3 个方向在 6 维的近简并子空间里取法不定。”
* **反驳：**
  1. **Line 53 的数学解释错误：** 3 个独立输出通道的头一个标量 NTK 特征值简并，张成的是 **$3$ 维**子空间（$\mathbf{v}_1 \otimes \mathbb{R}^3$），不是 $6$ 维！如果所有架构的头一个标量特征向量都是同一个 $\mathbf{v}_1$，那么 $k=3$ 时不同架构会张成同一个 3 维子空间，主角应为 $0^\circ$。
  2. **真正原因：** 在未训练网络中，`relu 3x128`（非零均值激活）的头 3 个奇异方向是**常数项** $\mathbf{1}\otimes \mathbb{R}^3$（重叠度 $0.973$），次 3 个是**线性 $\omega$ 项** $\omega \otimes \mathbb{R}^3$；而 `tanh 2x64`（奇函数激活）的头 3 个奇异方向反过来是**线性 $\omega$ 项**（重叠度 $0.864$），次 3 个才是**常数项**！所以在 $k=3$ 时两架构正交（$\sim 90^\circ$），而在 $k=6$ 时两者恰好都取到了 $\text{span}(1, \omega) \otimes \mathbb{R}^3$ 这 6 维。
  3. **为什么 $\omega$ 会从 $(\cos\theta, \sin\theta, \omega)$ 里单独劈出一个 3 维壳层？** 仅仅因为脚本绕过了仓库 `baselines/data.py` 默认的 `Normalizer`，直接喂了 `omega_max = 2.5` 的原始值，此时 $\text{Var}(\omega) = 2.083$ 是 $\text{Var}(\cos\theta) = \text{Var}(\sin\theta) = 0.5$ 的 **$4.17$ 倍**。一旦按 `baselines/data.py` 对输入做标准化（或取 `omega_max = 1.0`），三个一次项形成各向同性的 $9$ 维壳层（第 4 至第 12 维），**$k=6$ 刚好切在 9 维简并壳层正中间，未训练网络在 $k=6$ 时的跨架构主角从 $13.9^\circ$ 暴涨到 $85.2^\circ$（通过率从 $100\%$ 跌到 $4\%$），反而是壳层边界 $k=3$ 和 $k=12$ 变成 $100\%$ 通过！**
* **复算代码与输出：**

```bash
PYTHONPATH=. python3 -c '
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("m", "cells/5/exploratory/precheck_fim_is_ntk.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
X, Y, th, om, _ = m.data(300)
n = len(X)
c1 = np.kron(np.ones((n, 1)) / np.sqrt(n), np.eye(3))
cw = np.kron((om[:, None] - om.mean()) / np.linalg.norm(om - om.mean()), np.eye(3))
for name in ("tanh 2x64", "relu 3x128"):
    J = m.jacobian(*m.ARCHS[name](0), X)
    U3, _ = m.top_subspace(J, 3)
    print(f"raw X | {name:12s} k=3 overlap: const={np.linalg.norm(c1.T @ U3)**2/3:.3f}, omega={np.linalg.norm(cw.T @ U3)**2/3:.3f}")

Xstd = (X - X.mean(0)) / X.std(0)
J = {(name, s): m.jacobian(*make(s), Xstd) for name, make in m.ARCHS.items() for s in (0, 1, 2)}
names = list(m.ARCHS)
for k in (3, 6, 9, 12):
    sub = {key: m.top_subspace(v, k)[0] for key, v in J.items()}
    cross = [m.angles(sub[a, s], sub[b, t])[0] for i, a in enumerate(names) for b in names[i+1:] for s in range(3) for t in range(3)]
    print(f"standardized X | k={k:2d}: median angle = {np.median(cross):5.1f} deg, max = {np.max(cross):5.1f} deg (pass <30: {np.mean(np.array(cross)<30):.0%})")
'
```

```text
raw X | tanh 2x64    k=3 overlap: const=0.122, omega=0.864
raw X | relu 3x128   k=3 overlap: const=0.973, omega=0.011
standardized X | k= 3: median angle =  16.8 deg, max =  29.9 deg (pass <30: 100%)
standardized X | k= 6: median angle =  85.2 deg, max =  89.9 deg (pass <30: 4%)
standardized X | k= 9: median angle =  87.5 deg, max =  90.0 deg (pass <30: 6%)
standardized X | k=12: median angle =  15.3 deg, max =  26.1 deg (pass <30: 100%)
```

---

### 3. 格 12（两处反例）：1 步 ORC 不是 `(度数, 4-圈数)` 的函数；大尺度动态 ORC 在对称多室图上不收敛到 Fiedler 差且严格更优

* **原结论（`cells/12/precheck_2026-11-16.md` §2、`progress.md` §3）：**
  1. “1 步 Ollivier–Ricci 曲率是局部签名（度数与 4-圈数）的函数”；
  2. “大尺度恒等式：动态曲率的边排序 $\to |\phi_2(x) - \phi_2(y)|$（Fiedler 向量的跨边差）。”
* **反驳 1（局部签名反例）：**
  在无三角形（二部）图上，1 步 $W_1(m_x, m_y)$ 中距离为 1 的可匹配质量由 $N(x)\setminus\{y\}$ 与 $N(y)\setminus\{x\}$ 之间的**最大二部匹配数（Hall 瓶颈）**决定，而不是由连边总数（即 **4-圈数**）决定！在二维正交方格图上，每个邻居最多参与一个 4-圈，所以 4-圈数偶然等于最大匹配数；但在一般的二部转移图（或世界模型学出的地标转移图 L3P）上，两个 4-圈可以共享同一个顶点，此时相同的签名 `(deg_min=4, deg_max=4, 4-cycles=2)` 会给出完全不同的 1 步曲率（`0.00` vs `-0.50`）。
* **反驳 2（大尺度 Fiedler 恒等式反例）：**
  推导式 (lines 45–48) $m_x^t - m_y^t \approx e^{-\lambda_2 t}(\phi_2(x)-\phi_2(y))\phi_2\pi$ 隐含了 **$\lambda_2$ 非简并（$\lambda_3 > \lambda_2$）** 的假设。在具有对称性的多房间世界（如 3 个全等房间连向中心厅，或对称四房间）中，$\lambda_2 = \lambda_3$ 简并：
  1. 单个 Fiedler 向量 $\phi_2$ 必然发生任意对称破缺（在其中一扇门上跨边差趋近 0，`precision@3 = 0.667`）；
  2. 而大尺度动态曲率 $\kappa_t$ 对特征子空间的基选择严格不变，且因为 $\|\cdot\|_{W_1}$ 在 $\{\phi_2\pi, \phi_3\pi\}$ 张成的二维测度子空间上是非欧 Banach 范数，当 $t \to \infty$（如 $t=256$）时，`orc_t` 在三扇门上取值完全相等（`precision@3 = 1.000`），与 Fiedler 的 Spearman 相关停在 `0.651`，永不收敛到 1。
* **复算代码与输出：**

```bash
PYTHONPATH=. python3 -c '
import importlib.util, numpy as np, networkx as nx
from scipy.linalg import expm
from scipy.stats import spearmanr
spec = importlib.util.spec_from_file_location("m12", "cells/12/exploratory/precheck_orc_vs_spectral.py")
m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)

# 反例 1：同一个连通二部图上，签名同为 (4, 4, 2) 的两条边
G1 = nx.Graph()
G1.add_edges_from([(0,1), (0,2), (0,3), (0,4), (1,5), (1,6), (1,7), (2,5), (3,6), (4,8), (8,9), (9,7)])
G1.add_edges_from([(8,18), (10,11), (10,12), (10,13), (10,14), (11,15), (11,16), (11,17), (12,15), (13,15), (14,18), (18,19), (19,16), (19,17)])
gb = m12.Graph(G1)
nb = [set(np.flatnonzero(gb.A[i])) for i in range(len(gb.nodes))]
for u, v in [(0,1), (10,11)]:
    ix, iy = gb.idx[u], gb.idx[v]
    q = sum(1 for a_ in nb[ix] - {iy} for b_ in nb[iy] - {ix} if a_ != b_ and gb.A[a_, b_] > 0)
    orc0 = 1.0 - gb.w1(gb.step_measure(ix, 0.0), gb.step_measure(iy, 0.0))
    print(f"Bipartite={nx.is_bipartite(G1)} | Edge ({u:2d},{v:2d}) sig=({int(gb.deg[ix])},{int(gb.deg[iy])},4-cycles={q}) -> orc0={orc0:+.4f}")

# 反例 2：3 个对称 5x5 房间连向中心三角形大厅
G_star = nx.Graph([((100,0),(100,1)), ((100,1),(100,2)), ((100,2),(100,0))])
truth_star = set()
for r in range(3):
    for xx in range(5):
        for yy in range(5):
            if xx+1 < 5: G_star.add_edge((r, xx*5+yy), (r, (xx+1)*5+yy))
            if yy+1 < 5: G_star.add_edge((r, xx*5+yy), (r, xx*5+yy+1))
    d_edge = ((100, r), (r, 0)); G_star.add_edge(*d_edge); truth_star.add(d_edge)
gs = m12.Graph(G_star)
is_tr = np.array([(gs.nodes[x], gs.nodes[y]) in truth_star or (gs.nodes[y], gs.nodes[x]) in truth_star for x, y in gs.edges])
scs, _ = m12.scores(gs, 3)
Ht256 = expm(-256.0 * (np.eye(len(gs.nodes)) - gs.P))
orc256 = np.array([gs.w1(Ht256[x], Ht256[y]) for x, y in gs.edges])
fv = scs["fiedler"]
print("3-Room Star | Fiedler P@3 =", round(m12.precision(fv, is_tr), 3), "orc_t256 P@3 =", round(m12.precision(orc256, is_tr), 3),
      "Spearman(orc_t256, fiedler) =", round(spearmanr(orc256, fv).statistic, 3))
'
```

```text
Bipartite=True | Edge ( 0, 1) sig=(4,4,4-cycles=2) -> orc0=+0.0000
Bipartite=True | Edge (10,11) sig=(4,4,4-cycles=2) -> orc0=-0.5000
3-Room Star | Fiedler P@3 = 0.667 orc_t256 P@3 = 1.0 Spearman(orc_t256, fiedler) = 0.651
```

---

### 4. 格 20（两处反例）：无噪声（$\eta = 0$）下高斯模糊同样会让 $\text{death}/\text{birth}$ 跌破 $2.8$ 消失；块平均的 $\text{sinc}$ 不是单射

* **原结论（`cells/20/precheck_2026-11-18.md` §2）：**
  1. “高斯模糊在傅里叶空间里把每个波数 $k$ 乘以 $\exp(-k^2\sigma^2/2)$，永远不为零，所以它在环所张的有限维子空间上是单射。一个环在连续单射下的像仍是一个环，$H_1$ 类在每个 $\sigma$ 上都在，变的只是度量尺寸。**用尺度无关的判据（本项目的 $\text{death}/\text{birth} \ge 2.8$），没有噪声时环永远不消失。**”
  2. “**块平均的传递函数是 $\text{sinc}$，结论相同。**”
* **反驳 1（各向异性压扁导致零噪声下 VR 环消失）：**
  Vietoris–Rips 持久度比 $\pi = \text{death}/\text{birth}$ 只对**各向同性缩放**（$X \mapsto c X$）不变，对**各向异性线性单射**（$X \mapsto A_\sigma X$）绝不不变！脚本 `precheck_blur_persistence.py` 的 `internal_orbit` 之所以在无噪声时不消失，是因为它把形变写成了行波 $\cos(c x / w - \phi) = \cos\phi\cos(cx/w) + \sin\phi\sin(cx/w)$，导致两个正交相位分量恰好处于**同一个空间波数 $k = c/w$**，高斯模糊对其是各向同性缩放。
  只要内部振荡环的两个正交相位分量具有不同空间波数（如 $k_1 = 1/w$ 与 $k_2 = 3/w$，即一般的呼吸/多模极限环），高斯模糊会以 $\exp(-(k_2^2-k_1^2)\sigma^2/2)$ 的速率把椭圆短轴压扁。当短轴小于长轴相邻采样步长的 $2.8$ 倍时，**即使在完全无噪声（$\eta = 0$）下，$\text{death}/\text{birth}$ 也会在 $\sigma^* \approx 0.82 w$ 处跌破 $2.8$ 消失，且此时消失尺度 $\sigma^*$ 精确正比于 $w$、与噪声底完全无关！**
* **反驳 2（块平均 $\text{sinc}$ 有零点，非单射）：**
  宽度为 $B$ 的块平均在波数 $k = 2\pi m / B$ 处传递函数**精确为 0**（非单射），会把周期为 $B$ 的振荡环在拓扑意义上直接坍缩成一个点（$\text{death}/\text{birth} = 0$）。
* **复算代码与输出：**

```bash
PYTHONPATH=. python3 -c '
import numpy as np
from scipy.spatial.distance import pdist, squareform
L, n = 256, 64
x = np.arange(L) - L / 2
phis = np.arange(n) * 2 * np.pi / n
blur_hat = lambda L, s: np.exp(-(2 * np.pi * np.fft.fftfreq(L))**2 * s**2 / 2)
for w in (4, 8, 16):
    g = np.exp(-x ** 2 / (2 * w ** 2))
    U = g[None, :] * (1 + 0.3 * np.cos(1.0 * x[None, :] / w) * np.cos(phis[:, None])
                        + 0.3 * np.cos(3.0 * x[None, :] / w) * np.sin(phis[:, None]))
    for frac in (0.0, 0.8, 0.9):
        Ub = np.real(np.fft.ifft(np.fft.fft(U, axis=1) * blur_hat(L, frac * w)[None, :], axis=1))
        D = squareform(pdist(Ub))
        print(f"Gaussian eta=0 | w={w:2d}, sigma={frac:.1f}w ({frac*w:4.1f}): death/birth = {np.diag(D, n//2).min() / np.diag(D, 1).max():.3f}")
U_box = np.cos(2 * np.pi * x[None, :] / 8 - phis[:, None])
ker = np.zeros(L); ker[:8] = 1.0 / 8
Ub_box = np.real(np.fft.ifft(np.fft.fft(U_box, axis=1) * np.fft.fft(ker)[None, :], axis=1))
print(f"Block avg B=8, eta=0 on wavelength-8 loop: max|Ub| = {np.abs(Ub_box).max():.2e}")
'
```

```text
Gaussian eta=0 | w= 4, sigma=0.0w ( 0.0): death/birth = 13.968
Gaussian eta=0 | w= 4, sigma=0.8w ( 3.2): death/birth = 2.963
Gaussian eta=0 | w= 4, sigma=0.9w ( 3.6): death/birth = 2.326
Gaussian eta=0 | w= 8, sigma=0.0w ( 0.0): death/birth = 13.968
Gaussian eta=0 | w= 8, sigma=0.8w ( 6.4): death/birth = 2.963
Gaussian eta=0 | w= 8, sigma=0.9w ( 7.2): death/birth = 2.326
Gaussian eta=0 | w=16, sigma=0.0w ( 0.0): death/birth = 13.968
Gaussian eta=0 | w=16, sigma=0.8w (12.8): death/birth = 2.963
Gaussian eta=0 | w=16, sigma=0.9w (14.4): death/birth = 2.326
Block avg B=8, eta=0 on wavelength-8 loop: max|Ub| = 6.01e-15
```

---

### 5. 格 26：把“可逆 OU 过程”特有的 $s_i(\tau) = e^{-\lambda_i \tau}$ 误推广到了一般线性高斯过程的谱隙

* **原结论（`cells/26/precheck_2026-11-18.md` §1、§3）：**
  $\text{MI}(x_t; x_{t+\tau}) = -\frac{1}{2}\sum \log(1 - s_i(\tau)^2) = -\frac{1}{2}\sum \log(1 - e^{-2\lambda_i \tau})$，并据此声称“线性高斯情形下，读数与转移算子谱隙（弛豫速率 $\lambda_i$）是同一组数的函数”。
* **反驳：**
  第一等式（互信息 $=$ 白化滞后协方差奇异值 $s_i(\tau)$ 的函数）对任意高斯过程成立；但**第二等式（$s_i(\tau) = e^{-\lambda_i \tau}$，即把 VAMP 奇异值等同于生成元特征值/谱隙）仅在可逆（Normal）过程下成立**！
  在带惯性、振荡或剪切的非可逆（Non-normal）线性高斯过程 $dx = -A x\,dt + dW$（如 $A = \begin{pmatrix} 1 & -b \\ 0 & 12 \end{pmatrix}$）中，转移算子的特征值（弛豫速率 $\lambda_1=1, \lambda_2=12$ 与谱隙）完全不变，但因非正交模态的瞬态放大（Transient Amplification），奇异值 $s_i(\tau) \neq e^{-\lambda_i \tau}$，互信息与特征值速率公式显著偏离。
* **复算代码与输出：**

```bash
PYTHONPATH=. python3 -c '
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, sqrtm
lam = np.array([1.0, 12.0])
for b_shear in (0.0, 10.0, 30.0):
    A = np.array([[lam[0], -b_shear], [0.0, lam[1]]])
    C0 = solve_continuous_lyapunov(A, np.eye(2))
    W = np.linalg.inv(np.real(sqrtm(C0)))
    s = np.linalg.svd(W @ expm(-A * 0.2) @ C0 @ W, compute_uv=False)
    print(f"shear b={b_shear:4.1f} (eig={np.linalg.eigvals(A)}): MI(s_i)={-0.5*np.sum(np.log1p(-s**2)):.4f} vs MI(lambda_i)={-0.5*np.sum(np.log1p(-np.exp(-2*lam*0.2))):.4f}")
'
```

```text
shear b= 0.0 (eig=[ 1. 12.]): MI(s_i)=0.5589 vs MI(lambda_i)=0.5589
shear b=10.0 (eig=[ 1. 12.]): MI(s_i)=0.6955 vs MI(lambda_i)=0.5589
shear b=30.0 (eig=[ 1. 12.]): MI(s_i)=0.9487 vs MI(lambda_i)=0.5589
```

---

## 二、 任务 2：审阅格 15 预注册草案（`cells/15/prereg_draft_2026-11-18.md`）

格 15 的草案整体设计很清楚，但在 **P15.5 的数学定义**、**`history=3` 下 `M-raw` 的实现口径** 以及 **像点观测的取舍** 上有三个必须在冻结前堵住的硬漏洞：

### 1. P15.5 有两处会导致“真保守模型被误判失败”的硬伤（最严重）

草案 §2（line 32）、§4（line 73）、§5（line 87）定义 `L-tan` 为：“限制到潜点云局部 PCA 切空间的二维 $\log|\det|$”，并设定判据：
> **P15.5：** 保守世界：`L-full` 的中位数 $< -0.3 s$（收缩），**`L-tan` 的绝对值的中位数 $\le 0.3 s$（即 $0.03/\text{s}$）**。

这里有两个致命陷阱：
1. **陷阱 A（“绝对值的中位数” vs. “均值的绝对值” + 潜空间规范上边缘项）：**
   即使 `jepa-sym` 在二维潜流形 $\mathcal{M}_Z \subset \mathbb{R}^8$ 上学到了 **100% 精确、零耗散**的保守单摆动力学，编码器 $E: (\theta, \omega) \mapsto z \in \mathcal{M}_Z$ 诱导的潜流形欧氏面积元 $\nu_E(\theta, \omega)\,d\theta\wedge d\omega$ 也绝不是常数（VICReg 只约束全局协方差为单位阵，不约束局部面积元恒等于辛体积）。
   因此，一步切空间体积变化率在每个状态点 $z_i$ 上等于**编码器体积形式的上边缘（Coboundary）**：
   $$L_{\text{tan}}(z_i) = \text{div}_{\text{true}} + F_{\text{true}} \cdot \nabla \log \nu_E(\theta_i, \omega_i) = F_{\text{true}} \cdot \nabla \log \nu_E(\theta_i, \omega_i) \neq 0$$
   这个上边缘项在每一点 $z_i$ 上非零（沿轨道正负交替，全流形**均值** $\mathbb{E}[L_{\text{tan}}] = 0$）。
   * 如果按字面意思把“**`L-tan` 的绝对值的中位数**”写成 `np.median(np.abs(L_tan))`，它测的是上边缘项的逐点幅度！只要编码器的局部拉伸有 $15\%\sim 25\%$ 的平滑起伏，`np.median(np.abs(L_tan))` 就会高达 $0.18/\text{s}$，**超出阈值 $0.03/\text{s}$ 达 6 倍**，导致精确保守模型 100% 挂掉！
   * **修法：** 上边缘沿闭合轨道的积分保证的是 **均值** 为 0（当 $\log \nu_E$ 含非对称谐波时，中位数也不保证为 0）。应将 P15.5 的判据明确改为 **`|mean(L-tan)| <= 0.3 s`**（或沿 rollout 轨迹累积 $K$ 步后的平均对数体积斜率）。
2. **陷阱 B（切空间投影必须用双基 $U_{i+1}^T J(z_i) U_i$，绝不能用单基 $U_i^T J(z_i) U_i$）：**
   草案 line 73 只写了“限制到潜点云局部 PCA 切空间的二维 $\log|\det|$”。注意 $J(z_i) = \partial F(z_i)/\partial z$ 把 $z_i$ 处的切空间 $T_{z_i}\mathcal{M}_Z$（正交基 $U_i \in \mathbb{R}^{d\times 2}$）映到**下一步状态** $z_{i+1} = F(z_i)$ 处的切空间 $T_{z_{i+1}}\mathcal{M}_Z$（正交基 $U_{i+1} \in \mathbb{R}^{d\times 2}$）。
   如果写代码时用同一点的基做双侧投影 $\log|\det(U_i^T J(z_i) U_i)|$，切平面在 $\mathbb{R}^d$ 中每步旋转角度 $\alpha_i$ 会引入余弦投影收缩因子 $\log\cos\alpha_i < 0$，**哪怕在精确欧氏等距映射上也会凭空造出 $-0.07/\text{s}$ 的假收缩**（直接超过 $0.03/\text{s}$ 的阈值）！
   * **修法：** 在草案 line 73 写死公式：$L_{\text{tan}}(z_i) = \frac{1}{\Delta t} \log \left|\det \big(U(F(z_i))^T J(z_i) U(z_i)\big)\right|$。

* **复算验证代码与输出：**

```bash
PYTHONPATH=. python3 -c '
import numpy as np
from zoo.pendulum import Pendulum
w = Pendulum(gamma=0.0, dt=0.1)
tr = w.sample(seed=0, horizon=64, n_traj=200, obs_mode="state")
S, S1 = tr.state[:, :-1].reshape(-1, 2), tr.state[:, 1:].reshape(-1, 2)
log_rho = lambda t, o: 0.25 * np.cos(t) + 0.15 * np.cos(2 * t)
L_tan = (log_rho(S1[:, 0], S1[:, 1]) - log_rho(S[:, 0], S[:, 1])) / 0.1
print(f"Exact conservative flow | mean(L_tan)={np.mean(L_tan):+.5f}, median(abs(L_tan))={np.median(np.abs(L_tan)):.5f} (threshold 0.030)")
'
```

```text
Exact conservative flow | mean(L_tan)=+0.00288, median(abs(L_tan))=0.18297 (threshold 0.030)
```

### 2. `history = 3` 下 `recon` 模型的窗口映射 $w_t \mapsto \hat{w}_{t+1}$ 定义有歧义

草案 §4 规定用 `history = 3`（三帧窗口 $w_t = (o_{t-2}, o_{t-1}, o_t)$）训练 `baselines/mlp_wm.py` 的 `recon` 模型，并定义 `M-raw` 为“模型的一步窗口映射 $w_t \mapsto \hat{w}_{t+1}$ 的切空间体积变化率”。
* **代码冲突：** 在 `baselines/mlp_wm.py`（line 320）中，`recon` 的解码器 $D(z)$ 只输出**单帧**（$D(z_t) \approx o_t$，$D(F(z_t)) \approx o_{t+1}$，$D(F^2(z_t)) \approx o_{t+2}$），**根本不解码 $o_{t-1}$**！
* 因此，从 $w_t = (o_{t-2}, o_{t-1}, o_t)$ 构造三帧输出窗口 $\hat{w}_{t+1}$ 有两种截然不同的写法：
  1. **移位拼接（Shift-and-append）：** $\Psi(y_1, y_2, y_3) = \big(y_2,\; y_3,\; D(F(E(y_1, y_2, y_3)))\big)$；
  2. **全解码两步窗（All-decoded）：** $\big(D(z_t),\; D(F(z_t)),\; D(F^2(z_t))\big) = (\hat{o}_t, \hat{o}_{t+1}, \hat{o}_{t+2}) = \hat{w}_{t+2}$（这是 **两步** 映射 $t \mapsto t+2$，时间步长是 $2\Delta t$）。
  而且注意：如果用移位拼接 $\Psi(y_1, y_2, y_3) = (y_2, y_3, D(F(E(y_1, y_2, y_3))))$，当你把倒序窗口 $w^{\text{rev}} = (o_t, o_{t-1}, o_{t-2})$ 喂进去算 `M-raw(倒序的 w)` 时，前两帧保留的是 $(o_{t-1}, o_{t-2})$，第三帧预测的是 $o_{t-3}$，输出窗口落在反演流形上；而 `check6_parity.py` 之前只算了静态三帧拉回体积元 $\nu_3(\theta, \omega)$ 的解析李导数 $L_F \log \nu_3$，**并没有**算移位拼接映射 $\Psi$ 的切空间 Jacobian。
* **修法：** 必须在冻结前把 $w_t \mapsto \hat{w}_{t+1}$ 的显式式子写进 §4，并在真值模拟器上用同一个切空间 PCA + Jacobian 函数核对 P15.1。

### 3. 第一阶段用“像点观测”代替像素的取舍评估

* **赞同分两阶段走：** 草案 §3 的真值预检非常漂亮地揭示了“像点观测无面积 Jacobian 权重，故仿射相机下像点有假散度（$1.266$）而像素没有（$0$）；射影相机下像点与像素假散度几乎全同（$r = 0.999$）”。
* **取舍建议：** 第一阶段用像点观测跑 72 个低维 MLP run（仅需 5–6 小时 CPU）作为 **“学习误差能否被时间反演抵消”的便宜 kill test** 完全合理。如果第一阶段在像点上 P15.2 都过不了，直接判死省下 100 小时像素算力；如果过了，第二阶段像素实验只需要跑 **1 个射影（$k=0.3$）+ 1 个仿射（$\cos\phi=0.54$）** 验证“像素的面积权重让仿射天然归零、射影靠对称化归零”即可，无需在像素上重跑全套网格。

---

## 三、 任务 3：对 `correspondence_table.md` 关系分类的异议

对照表把 32 格分为 10 类关系，整体框架非常清晰，但有 **4 格的主关系归类不准或掩盖了真正的死因**：

1. **格 11（现归为 `恒等`）$\to$ 应归为 `恒等（局部读数）＋ 构造/错设（全局规划）`：**
   * **理由：** 隐函数定理给出的“局部 fold 奇点集 $\equiv$ Jacobian $\det J = 0$ 零水平集”只是读数层的一半；格 11 的核心主张是 `readout: plan`（用奇点地图做安全规划，避免跨过分岔回不来）。而 `cells/11/precheck_2026-11-16.md` 表 B、C 最精彩地证明了：在阻尼受驱单摆上，决定滞后环下沿和吸引域安全的根本不是局部平衡点奇点（Thom 突变），而是**全局同宿分岔（Homoclinic Bifurcation）**。局部奇点地图在 $u=0.5$ 时报“离 fold 很远、绝对安全”，单摆却会永远停在旋转吸引子上回不来。这在规划层是**几何对象错设（局部奇点看不见全局分岔）**，在 §2 三问表里“成立吗”对规划主张应为 **“部分（局部成立，全局规划不成立）”**，而不是简单的“是 + 恒等”。
2. **格 13（现归为 `恒等`）$\to$ 主分类应归为 `坐标`（与格 22 同类）：**
   * **理由：** 格 13 在世界模型里的用法是对**学到的潜变量对**算 IGCI 正交性残差（`cells/13/precheck_2026-11-18.md` §3.B）。它的真正死因是潜变量的逐坐标单调重编码 $z'_i = g_i(z_i)$ 是规范自由度，随机换个单调刻度就有 $51\%$ 的正确判向翻转——即**连 R1 的资格条件 (b)（重参数化不变）都过不了**。把它的主类归进 `恒等`（仅因为观测层斜率估计器 $\equiv$ 微分熵差），掩盖了它在潜空间里其实死于 **`坐标（规范不不变）`**。
3. **格 5（现归为 `恒等`）$\to$ 应拆为 `恒等 ＋ 构造（输入协方差主导）`：**
   * **理由：** “预测空间 FIM $=$ 经验 NTK Gram 矩阵”是恒等式；但格 5 的核心预测“跨架构硬方向对齐代表学到了同一世界规律”之所以死，是因为硬方向被输入特征的协方差和低阶多项式锁死（且如上面任务 1 第 2 条所证，对齐与否取决于输入各分量的方差比与壳层截断 $k$）。
4. **格 12（现归为 `恒等`）$\to$ 应归为 `局部/极限恒等 ＋ 中间尺度数值打平`：**
   * **理由：** 1 步 ORC（在二维方格上）与大尺度非简并极限是恒等式；但在中间尺度 $t \in [1, 4]$ 以及多室对称图（$\lambda_2$ 简并）或闭室图 C 上，$W_1$ 的地面度量与谱方法/TV **并不恒等**（见预检 4.2 节与上面任务 1 第 3 条反例）。它死是因为在测试图族上没有综合胜过“4-圈计数 + 多路谱 + 换尺度的 TV”这一组合基线，不全是代数恒等式。
