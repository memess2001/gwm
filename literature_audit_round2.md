# 文献审计第二轮（2026-11-03）

**本轮做了什么。** 五路并行，分组为：格 10 | 格 7+9 | 格 6+12+2 | 格 3+15+总体项 U | 格 0+1+4+8。目标有两个：一是补跑第一轮因 WebSearch 预算耗尽而没执行的搜索词，二是核对影响决策的引用（是否存在、作者、年份、venue，以及被引用的那句话是否真在文中）。合计约 150 次 WebSearch，另加 HF papers 索引和全文抽取核对。

**原始数据。** 结构化的 agent 输出逐条保存在 `results/lit_round2_raw.json`。本文件是人工整理，结论已经写进 `cells.yaml` v0.4 的 `collision / nearest / must / notes` 字段。

**等级记号同第一轮。** A = 直接撞车；B = 近邻；C = 远邻；+/− 表示偏向。

---

## 总表

| 格 | 第一轮 | 第二轮 | 一句话 |
|---|---|---|---|
| 2 自然梯度流 | B+ | **A−（命题）/ B+（方法）** | Legacci et al. ICML 2024 已在 Shahshahani 度量下给出 replicator 的势 / 调和分解，原 kill test 测的是定理。改为"Fisher vs 欧氏"的度量归因测试 |
| 6 联络 / 和乐 | B | B | 无新撞车。操纵变量改为"数据是否覆盖可区分环路"。Delliaux 当不了零和乐对照，WLA 可以 |
| 7 两种直线 | B− | B−（但已被思想实验杀） | 没有人从观测动力学读出 α；连续 α-计被数值检查杀掉（见 `cells/7/`） |
| 9 信息时间 | B | B | 加惊奇时钟，三元对照改为四元；TOTR 被 HANDOFF 过度解读；确定性世界里必死 |
| 10 RG | B | **B+** | 交集仍空，但漏掉了最近邻 Tao–Tao–You–Zhang（Physica D 2025） |
| 12 离散曲率 | A− | A− | 截至今日 Castro 组没有正式版；Liang 2026 的谱瓶颈子目标是必须打平的基线 |
| 3 WFR | A/B | A/B，**thought_killed** | WFR 分解只依赖 ρ_t，"Lenia = 纯改权"在密度层不可辨识；δ-谱近似只是 ∂tρ 的频谱滤波 |
| 15 Liouville | B | **B+** | LeHami（LeCun 组，2026-10）、PH-Dreamer 已在测潜相体积，读法相反、都不声明体积形式；耗散版已被预告 |
| 0 地基 | — | — | 前提被部分测过，但都是目测；持续同调定量测量和"粘合"测试零命中 |
| 1 测地线 | B+ | B+；改写版约 B− | 解码器拉回对墙要么平坦，要么被数据支撑混杂；Cauchy–Green 版本空 |
| 4 拓扑 | B+ | **A−** | CHyLL 做完了混合系统版闭环；GD-VAE 表 3 的读法需更正；单摆当不了错配试验场 |
| 8 拟度量 | A− | **A，lit_killed** | NEEP（PRL 2020）、LENS（2025）、FiRL（ICML 2026） |
| U | — | — | 命名仍空；无伞状纲领；AWM 核实，一作在浙大 |

---

## 格 10：尺度即时间（RG）。B → B+

**结论。** "RSMI 型判据 + 时间演化 + 学到的规则沿尺度流动并有不动点 + 空间扩展动力系统（CA / RD / Lenia）"这个交集截至 2026-11 仍为空。neural RG 在 2025–26 仍然全是平衡态：RGFlow (MLST 2026)、Alao–Huang–Lu (NeurIPS-W 2026)、Shukla–Lokhov 2025、percolation 2026 都是。
但交集周边明显变密，第一轮漏了下面这些。

**新近邻：**

| 论文 | 做了什么 | 对本格的含义 |
|---|---|---|
| **Tao, Tao, You, Zhang**, *Data driven modeling for self-similar dynamics*, Physica D 2025（arXiv 2310.08282） | 学粗粒化映射，加一个跨尺度共享的动力学核（作者称之为 RG 不动点），用一致性 MSE 判定；覆盖全部 256 条 ECA（找回 21 条自相似规则），以及扩散、Vicsek、Ising | **最近邻**，等于在不动点处把 Israeli–Goldenfeld 学习化了。作者与 NIS+ 同组，文中已把 EI 列为未来方向，**被抢先的风险高**。本格剩下的差异是：不共享的 F_k、F_k 离开不动点的轨迹、RSMI 判据、连续 CA |
| Friston et al., *From pixels to planning: scale-free active inference*, Front. Netw. Physiol. 2025（arXiv 2407.20292） | renormalising generative models（RGM）：空间加时间重整化，MI 驱动的结构学习，各层动力学同形只变参数，用于规划 | 占住了"重整化世界模型"这个口号，**本格要改名** |
| Yasuda & Bischoff, *Predictor-Driven Diffusion*, 2026（arXiv 2604.00327） | 尺度 λ 与物理时间是两根独立轴 | (k, t) 双轴本身不再是新点 |
| Bertoni & Renes, J. Phys. A 2022 | 要让粗粒后耦合最少，应保持短程 MI；RSMI 只是它的松弛 | 判据对比必须加入短程 MI |
| Edlund & Nilsson Jacobi, J. Stat. Phys. 2010 | 自相似 ECA 的 RG 不动点在所有方向上不稳定 | "收敛到不动点"这个预测与定理冲突，应改为"找到不动点并测出不稳定方向" |
| Alao, Huang, Lu, NeurIPS-W 2026 | gauge-fixed neural RG：学到的块映射沿 RG 流排列 | Iso 2018 负面结果的正面对照，可作 Ising 校准的复现目标 |
| Chan, Lenia 原文 §3.1.1 | R > 12 时斑图对缩放近似不变 | Lenia 上的"不动点"是构造使然，只能当有已知答案的校准 |
| 其他 | Zhao–Fogler–You RGFlow（MLST 2026，最小 MI 原则，方向与 RSMI 相反）；Shukla–Lokhov 2025（静态场论耦合沿尺度的流）；ROMA 2025（图上的学习 RG + 每层动力学，含 Kuramoto）；SAMPO 2025（next-scale 世界模型）；DySIB 2026（过去–未来对称 IB，可借作时间方向估计器）；Meshulam 2018；Hoel 2025 CE 2.0；L. Chen 2025 | 均为远邻或工具 |

**引用核对：**
- RSMI-NE 中"远离平衡需加时间 buffer、留作未来工作"已确认，出现在 PRE 2021 正文的 chipping 例子之后和结语，不在摘要里。
- NIS+：作者 6 人（Mingzhe Yang, Zhipeng Wang, Kaiwei Liu, Yingqi Rong, Bing Yuan, Jiang Zhang），出处 NSR 12(1) nwae279。GoL 实验只有一级粗粒；Multistory NIS+ 只出现在 arXiv v3。第一轮说它"不提 RG"有误：它引了 Eberhardt & Lee 2022，并处理过重整化与时间演化的可交换性。
- Bayesian Renormalization 的作者应补上 Stapleton。
- Lenggenhager PRX 2020 关于无序系统的措辞应为"抑制"，而不是"不制造"。
- 以下均已确认：Israeli–Goldenfeld 2004/06、Di Sante PRL 2022、Iso PRE 2018、Cotler–Rezchikov PRD 2023、Berman–Klinger (Entropy 2024)、Barnett–Seth PRE 2023。
- LED 是否发表于 NMI 2022 无法确认。

---

## 格 7：两种直线 / α-计。B− 不变；已被思想实验杀

**结论。** 18 次针对性搜索后，没有找到任何工作把观测轨迹分段判为 e-直或 m-直，或者从观测分布轨迹读出 α。
"从数据估 α"只出现在静态或横截面场景：Choi et al. 2013 学 α-积分的 α；Tsagris et al. 2011 选单纯形幂变换的 α；Burlaga–Viñas 2005 从太阳风序列估 Tsallis q。
本格的"连续 α-计"已在 2026-11-03 被数值检查杀掉（`cells/7/alpha_check.txt`），所以这些文献只用于把记录写完整。

**会改实验设计的技术发现：**
- α-Flow（Cheng et al. 2025）App. B：单纯形上的 α-测地线 = 归一化 α-插值再做一次时间重参数化。这从另一方向证实了"秩-2 / 幂均值"刻画（作为曲线的像），所以只能做参数化无关的检验。
- Koide & van de Venn 2026：测地线分类索引与 Amari 相反（m-直线 / Brownian bridge 是 α_K = 1）；闭式 α_K = 2D/(γR²) − 1 依赖制备尺度 R。
- 在高斯子族上，α-混合会离开高斯族，秩-2 检验只认得出 e 端，所以"与 Koide 闭式一致"这条预测不可用。
- Hofrichter–Jost–Tran 2017 专著与 Tran 2012 博士论文里都找不到"选择 = e、迁移 = m"的表述，应写成自证引理：e 侧引 Harper 2009，m 侧引 Toussaint 2004。
- 必要基线：Mathieson & McVean 2013（Genetics）的 (s, m) 联合似然。

**新近邻：** q-Paths（Masrani et al. UAI 2021，幂均值退火路径，α 是设计超参）；α-Flow 2025；Tsagris 2011；Choi 2013；Legendre 动力学（Fong et al. 2025）；Wolfer–Watanabe 2021（可逆 Markov 链的 e/m 结构，不撞车）；Amari 1982（曲指数族的嵌入曲率：α-计 ≈ 使 α-嵌入曲率最小的 α）；Obata–Hara–Endo 1992 与 Nakamura 1993（理论根源）。

**引用核对：**
- Fujiwara 1993 确认，Thm 3.1 为 η̇ = −(η − η_q)，第一轮写作 −η。
- 以下确认：Omiya–Komaki 2025、Koide–van de Venn 2026（记号警告正确）、Amari 2007（经转述）、Nakamura 1993（仅书目）。

---

## 格 9：时间即弧长。B 不变

**结论。** 用 extended 模式做了直接撞车检查，HF 索引也查了，都没有命中，即没有人把 Fisher 弧长或信息长度当时间坐标来学习环境动力学。
新增三条近邻线：
1. 设计路径的信息时间调度：Stancevic et al. 2025 熵时间、Goshtasbpour et al. 2023 恒速率 AIS、Barzegar et al. PRE 2024；
2. 事件时间世界模型：VPR（ICLR 2022）、THICK（ICLR 2024）、EAWM（ICLR 2026），以及 Roseboom et al. 2019（Nat. Commun.）用累积欧氏表征变化做的主观时钟；
3. information rate 已在真实数据上当诊断量：Nicholson & Kim 2016（音乐）、Choong–Kim–He 2025（EEG）。

**改实验设计：** 三元对照改为四元，加入惊奇或熵时钟。原因是 KL(q_{t+1}‖q_t) ≈ ½Γ²dt²，审稿人会把 Fisher 时间读成"累积的平方根贝叶斯惊奇"。
另外，2026-11-03 的思想实验显示，确定性世界里 Fisher 弧长 ≡ 加权欧氏弧长（`cells/thought_experiments_2026-11-03.md` §3）。

**引用核对：**
- information length 应归属 Nicholson & Kim 2015 / Kim et al. 2016，不是 PRE 2017。
- TOTR（Standridge, Livescu, Cizmas 2026，arXiv 2603.16583）的"降 1–2 个数量级"是相对另外两种时间重参数化方法而言，主要指 τ-MSE。它没有和墙钟比，也没有和 Fisher 比，所以 **HANDOFF §7 第 4 条"欧氏弧长就能带来大部分增益"是过度解读**。
- Zakharov et al. 2020 是阈值化的惊奇时钟（KL[后验‖转移先验] > ε = 5），不是信念变化时钟。
- Sumaya-Martinez 2026 的完整标题按 arXiv 页面多出 "...Physical Distinguishability"。
- 以下确认：Tenkès 2017（原句确认）、Kim–Hollerbach PRE 95 062107、Mao PNAS 2024（按测地线进度重索引的说法在正文里）。

---

## 格 6：联络 / 和乐。B 不变

**结论。** 用四组精确短语（parallel transport / holonomy 与 world model / latent dynamics 搭配）加 HF 语义检索，都没有找到"学到的联络或和乐被用来探测隐状态"的世界模型工作。HCDP 附录 G.2 自己把连续理论划出了范围，可以直接当空位证据。

**改实验设计：**
- HCDP 的"被动不可辨识"在正文 Thm 4.2，其构造依赖采样策略漏掉一个可区分动作。所以操纵变量应是"数据是否覆盖可区分环路"，而不是"被动 vs 主动"，否则小网格上的随机游走可能直接成功，结果难以解读。
- Delliaux 2025 的转移 z' = z ⊕ Δ(z, a) 依赖状态，全文没有 abelian 或 commutative 字样，所以当不了"和乐必为零"的对照。WLA（Hayashi / Koyama 2025）才在 Assumption 1 里明文假设阿贝尔群。
- ‖T_loop − I‖ 在共轭变换下会变，需要换成规范不变的读数：观测层探针差异，或 Sevetlidis & Pavlidis（ICLR 2026）的 gauge-invariant representation holonomy。

**新近邻 / 根源：** Sevetlidis & Pavlidis ICLR 2026（工具）；Rivest–Schapire 1993（用 homing sequence 代替 reset）；Holmes–Isbell ICML 2006；Sinitsyn–Nemenman 2007（随机化学动力学中的 Berry 相，未打开）；Davis 2026（Zenodo，质量存疑）。

**引用核对：**
- HCDP、Blattner 2025、WLA、Dehdarirad 2026（交换子是 Phase 2 强加的建模假设，torsion 只是类比）、Wang & Wang NeurIPS-W 2025 均确认。
- US Patent 12651010 属于 Atombeam Technologies，发明人 Brian Galvin，2026-06-09 授权。

---

## 格 12：离散曲率。A− 不变

**结论。** 截至 2026-11-03，没有找到 Saheb Pasand, Castro, Bashivan 那篇 workshop 论文的正式版。

**需要注意：**
- 第一轮写的"增益小"与 Mila 页面摘要的口径相反（"substantially improves，优于 random 与 count-based"），OpenReview PDF 被 challenge 页拦截，**引用前要人工复核**。
- 他们用的 successor representation 本身就是学到的多步转移模型，所以"WM 预测曲率而非经验计数"这个差异点偏弱。
- 有向转移核直接用 Ollivier 2007 的 Markov 链曲率即可，不必依赖 Wiesler 2026。
- 必须打平的基线：Liang et al. ICML 2026（Laplacian 谱瓶颈子目标，AntMaze 96.5%，带理论保证）。另加 Bar–Talmon–Meir ICML 2020、G2QDR（Zhang, Wang, Chang, Precup，TMLR 2026）。

**引用核对：** 以下均确认：Fei ICML 2026 VPS（批评语气温和）、Liang ICML 2026、Micaletto–Nigrelli NeurIPS-W 2026、Wiesler 2026（学位论文，以综述为主）。

---

## 格 2：下降即规律。B+ → A−（命题）/ B+（方法）

**结论：**
- **命题层。** Legacci, Mertikopoulos, Pradelski（ICML 2024 spotlight）已在 Shahshahani 度量下给出博弈动力学的势 / 调和分解，并证明调和类回归，摘要还明说欧氏 Helmholtz 与指数权重动力学的几何不相容。Zhang…Wang（JCP 2012）发表了"演化 = 适应度梯度 + curl 通量"。所以 replicator 世界上的三条原预测已是定理，wright_fisher 的原 kill test 杀不死这个想法。
- **方法层。** MPL（Sohail…Barton，Nat. Biotech. 2021）是"WF 协方差加权拟合线性 F"的闭式解，等于 natural deviation 在线性 F 时的特例；Wasserstein Residuals（NeurIPS 2026）是 W 度量版。Fisher 残差加旋度作为世界模型诊断仍然是空的。

**有两个来源独立给出的数学发现：**
- 迁移和均匀突变都是 Shahshahani 精确梯度流。zoo agent 推导并数值核对（势为 −μ·KL(m‖x)），R3 也独立核对过。
- 由此，原预测"带突变的 replicator 落在两者之间"对均匀突变是错的；循环突变的旋度随 μ 线性增长（μ = 0.05 时为 0.59，μ = 0.1 时为 1.17）。
- 对一般对称收益 A = [[0,1,0],[1,0,0],[0,0,0]]，欧氏旋度为 0.069，Shahshahani 旋度约 1e-12，这就是新 kill test 的依据。

**引用核对：**
- Conjugate Natural Selection 的第一作者是 **Raab**（不是 de Alfaro）；natural deviation 定义在正文 Def. 8。
- Velde 只是 arXiv 预印本，没有 Nat. Commun. 记录，很可能是和 ddHodge 混了。
- ddHodge 是 Nat. Commun. 16:11342（2025-12-29）。
- Wasserstein Residuals 是 NeurIPS 2026 poster。
- HHD-GP 原文的措辞是"对称先验使模型可辨识"，第一轮写的"需要先验"偏强。
- Jin Wang 2015 的 Adv. Phys. 综述只在搜索摘要里见到，未核实。

---

## 格 3：搬运还是改权（WFR）。A/B 不变，但核心主张被思想实验杀

**文献层**
- 2026 年下半年，WFR 轨迹推断的方法层没有新环节。只找到两篇：projected WFR flow 用于 Hawkes 背景强度，BranchSBM（ICLR 2026）对应"分裂"类过程。
- 试验场层仍空。Lenia/CA + OT 零命中，查过的来源有：两次 web、短语追查、HF 索引、ALIFE 2026 程序册全文 grep。
- 唯一的 CA + Monge–Kantorovich 命中是 CAMs（arXiv 2503.15086），属远邻：那里格子状态是测度，MK 距离只用来给配置空间定拓扑。
- Flow-Lenia 两版全文都没有 transport / Wasserstein / Monge / Kantorovich 字样。某次 WebSearch 的自动摘要称 Flow-Lenia "受 OT 启发"，经全文核对是假的。

**被杀的理由**（R4 推论，我已复核线性化部分）
- 任何基于密度的 WFR 分解只依赖密度曲线 ρ_t。一个刚性平移的 Lenia glider，机制上是逐点改权；一个同形同速的 Flow-Lenia glider，机制上是搬运。两者密度曲线相同，δ-谱也就相同。所以"Lenia = 纯改权"是机制真值，从密度上不可辨识。
- 在近均匀密度下，最小作用解为 v = −∇φ、g = −φ/δ²，波数 k 上的搬运占比 = k²δ²/(1 + k²δ²)。也就是说 ratio(δ) 大致只是 ∂tρ 的空间频谱滤波，与机制无关。
- 能从 ρ_t 辨识的只有质量收支，这是平凡量。

**复活条件**（满足其一）
- δ-谱打败"净质量变化率 + ∂tρ 功率谱"这个平凡基线；
- 或者改成用学到的世界模型做离轨干预来归因，这相当于一个新格。

**引用核对**
- UPFI（Zhang, Maddu, Qiu, Chardès；NeurIPS 2025；arXiv 2505.13197）：不可辨识只出现在全文第 2 节（Cor. 2.2，OU 加二次 fitness）。WFR 能量只是作者选的"自然的正则之一"，唯一性只在线性-二次情形证明。第一轮写的"承认非线性情形未解决"在原文里找不到，属过度转述。
- WFR 截断距离已核原文：Chizat et al. 2015 Cor. 4.1 为 |Δx| < πδ；LMS 式 (1.19) 为 π/2。
- 以下确认：WFR-FM（ICLR 2026）、WFR-MFM、USB、VGFM、DeepRUOT（ICLR 2025 oral）、TIGON（NMI 2024，经 NSF PAR 页核实）、Flow-Lenia（Artificial Life 31(2):228–248）。

---

## 格 15：Liouville 散度。B → B+

**新近邻**（第一轮之后出现或漏检；HF 索引一篇都没收录）

| 论文 | 做了什么 | 含义 |
|---|---|---|
| **LeHami**（Yalovetzky, Balestriero, LeCun，arXiv 2610.15267，2026-10-11） | JEPA 预测器换成动作条件 Hamiltonian 流（leapfrog）。用 (1/n)Σlog\|det J\| 作"相体积变化"诊断，点名 Liouville。在保守的 64×64 视觉单摆上，无约束 LeWM 出现**虚假收缩**。原文："we leave dissipative modeling to future work" | "测潜相体积"已由高曝光组发表，耗散版已预告，**有时效风险**。虚假收缩是读耗散之前必须先控制的混淆 |
| PH-Dreamer（Luan & Shi，arXiv 2605.18303） | port-Hamiltonian RSSM，用 PCA 对数相体积，**把体积降低 4–8% 当作优点** | 与 LeHami 读法相反，两者都不声明体积形式 → 领域缺标定 |
| H-JEPA（Zoabi, Ali, Wolf，arXiv 2609.33497）、ELWM（2608.09876） | 耗散 pH 动力学进入 JEPA / 潜世界模型 | 若用 pH 结构做对照，以它们为基线 |
| Wang，arXiv 2606.24945 | certified horizon：认证**解码后的物理不变量**，而不是潜 Hamiltonian | 支持"散度在解码后的物理坐标里算" |
| Das & Manikandan，Commun. Phys. 2026；Boffi & Vanden-Eijnden 2024 | 随机侧沿轨迹定位熵产生 / 局部 EPR | 加上 Zhu 2025，随机侧的"耗散地图"已有三篇 |

**审计要改的两处**
- Wang 2606.13092 已经用"Liouville anchor ∑λ = −N"校验估计器，所以"三估计器互校"不是新颖点。
- "2026 年 Hamiltonian 世界模型都不碰相体积"已过时。

**仍然空的**：在耗散世界上、带解析真值、声明了体积形式的局部 div_μ f 地图。另有一个机会是"潜体积读数的标定 benchmark"，见 `cells.yaml` 格 15 的 notes 与 `cells/15/exploratory/`。

**引用核对**
- 以下确认：Zhu 2025（散度公式在 App. B.2，不在方法节）、GENERIC-FNO（现为 v9）、HaM-World（现为 NeurIPS 2026 poster）、DreamSAC（CVPR 2026；CVPR 页列 7 位作者，arXiv 列 6 位）、LaWM、Cui & Ma（立场文）、Gruver ICLR 2022、Kim 2606.28751。
- DreamSAC 的"|ΔH| 当好奇心奖励""自报高耗散失效"两点，摘要里都看不到，仍未核实。

---

## 总体项 U

**AWM（Pan, Tan, Yuan, Yao；arXiv 2610.40503，v2 2026-10-30）**
- 主要数字全部核实。
- 第一作者在浙大，HANDOFF 写作"清华姚期智 / 袁洋组"，属小出入。
- 作者自己说明证明只在学到的模型这一前提下成立；不可能性证明依赖"颜色计数守恒"定理。
- 5 天内没有学术跟进。同赛道已有其他程序化世界模型：OPINE-World、Rodionov 2026，均未核实。

**WorldBench、GeoWorld**：核实无误。

**命名**："information-geometric world model(s)"和"invariant-structured world model(s)"在 2026-11-03 精确短语检索仍为零命中。Zenodo 上一篇未经同行评议的 GDL-for-world-models 评论又占用了一次"geometric world models"。

**伞状纲领**：2026-09 以来没有出现带可测几何不变量的纲领。近似物：
- World Model Interpretability（arXiv 2610.25296）：Ground / Structure / Verify / Act 四阶段议程，不涉及几何；
- Four Principles for Physically Interpretable World Models（2025，第一轮漏检）；
- Wang 的 certified world models 两篇；
- Karim & Chlon 2026：动作参数化改写后，潜世界模型检索退化 2.6–13.4 倍，说明"不变性要认证而不是假设"。

---

## 格 0：公共前提（地基格，不评撞车等级）

**结论：前提已被部分测过，但都是目测。**
- Ivashkov, Balestriero, Schölkopf 2026（arXiv 2606.20104）：Reacher 上的"flat 2-torus"只有 PCA 投影加图注，没有任何拓扑量化。SIGReg 基线的关节角线性探针 R² 为 0.999，反而高于 SMWM 的 0.946。
- DySIB 2026：R² 潜空间里的环形是目测比对，而且是十个种子里挑最好的一个。
- Bao 2026（arXiv 2608.23526）：在冻结的 DreamerV3 单摆模型里做无标签搜索，找到跨种子一致的类能量近守恒量。把潜状态投影回初始水平集后，rollout 误差下降。
- 两个问题零命中，这是本格的新部分：
  - WM 潜空间持续同调的定量跨种子测量；
  - 原始 wrapped 角度在无解码器训练下是否被粘成圆。像素观测本身已是 S¹ 的连续像，测不出粘合。

**方法学警示（进入 `cells/0/prereg.md`）**

| 论文 | 结论 | 对设计的要求 |
|---|---|---|
| Liao & Cao 2026（arXiv 2607.03728） | 能量线性可解码（R² ≈ 1），但 donor-swap 干预几乎不改变预测，即因果惰性 | 可解码 ≠ 被使用，需要第二、三级判据 |
| Internò 2026（arXiv 2602.12218） | 高容量探针或微调会破坏潜物理结构 | 探针用冻结的低容量线性读出 |
| Nicollier 2026（arXiv 2609.21656） | JEPA 正则器的目标几何（如 SIGReg 的高斯目标）就是拓扑先验 | 正则器类型要当实验因子 |
| Tan 2026（arXiv 2607.27017） | 预测时程决定潜空间里编码哪些物理量 | 预测时程 K 要当实验因子 |
| Kvalheim & Sontag 2025（arXiv 2511.04807） | 拓扑障碍只在 ℓ = m 的闭流形上出现，且只体现在最坏误差；S¹×R 可嵌入 R² | 单摆上出现 H1 不说明需要几何约束 |

**引用核对**：Batson et al. 2021，作者应补全为 Batson, Haaf, Kahn, Roberts，JHEP 04 (2021) 280。

---

## 格 1：曲率即规律。原命题 B+；改写版约为 B−

**结论：我提出的新顾虑成立，而且问题更大。**
- 固定 σ 的解码器拉回对墙是平坦的。
- 方差感知的解码器拉回（Latent Space Oddity 一类）在无数据区会膨胀。墙内没有数据，所以墙会"发亮"，但这是数据支撑造成的混杂，不是动力学。
- 动力学版本，即预测分布的 Fisher 拉回 = k 步 Cauchy–Green / FTLE，在学到的动作条件 WM 上没有人做过：做了 5 次检索（含 1 次 extended），外加 2 次 predictive Fisher 检索，均零命中。
- FTLE 只在以下对象上算过：neural ODE 分类器（Wöhrer & Kuehn 2026）、神经 flow-map 代理（Sahoo & Berger 2022）、储备池（Sisodia & Jalan 2026）。
- 风险：学到的平滑模型会抹平接触处的不连续（Suh et al. ICML 2022 的类比），所以 FTLE 脊要对照模拟器真值校准。

**改写版必带四个对照**
- (a) 固定 σ 解码器拉回，作阴性对照；
- (b) 数据支撑图；
- (c) 模拟器真值 FTLE；
- (d) 不变性预测：C_k 对潜空间重参数化不变，所以 Temporal Straightening 训练前后的障碍图应一致。这是一条可证伪的预测。

**引用核对**
- Friedl ICLR 2025 与 2026 续作的细节属实，两篇都不分析学到度量的曲率。
- Temporal Straightening 已是 **ICML 2026**，7 位作者，LeCun 第 6、末位作者 Mengye Ren。
- Koide 原话属实。
- Arvanitidis AISTATS 2022 的原文是"a wide range of decoder distributions"，不是"任意"。

---

## 格 4：形状即规律。B+ → A−

**新近邻**
- GD-VAE 表 3 的正确读法：
  - 同维数下，带流形投影的 AE 在 t = 1 s 时，误差比无投影低约 16 倍（柱面）和 7.8 倍（环面）；短时程反而输。
  - GD-VAE 本身在两张表里都不是最优。
  - 大胜出现在欧氏潜空间本来就能嵌入该流形的维数，所以更像流形约束的收益，不一定是拓扑本身。
- **CHyLL**（Teng, Liu, Song, Sreenath，arXiv 2512.10117）在混合系统上完成了"发现粘合 → 学流 → ripser 验 Betti 数 → 长程胜出"的闭环。
- atlas 法（Floryan & Graham，NMI 2022）提供了一条不需要发现拓扑的替代路线。
- Kvalheim & Sontag 2025：拓扑障碍只体现在最坏误差。
- persistent cup-length（Ivshina et al. 2025）：仅靠 Betti 数不能唯一确定流形。
- van Hulst 2026 是静态 VAE，主类是 cs.LG（math.AT 只是交叉）。
- Esmaeili 2023 应补上作者 Zimmermann。

---

## 格 8：拟度量 / Finsler。A− → A，lit_killed

**新近邻**
- **NEEP**（Kim, Bae, Lee, Jeong，PRL 125, 140604, 2020）：学出反对称两点函数，最优解就是逐跃迁随机熵产生，与解析值 R² ≈ 0.99。
- **LENS**（Ji, Raj, Eysenbach, Reddy 2025）：前向 / 反向分类器的 logit 被参数化为"势差项 B + 反对称环流项 A"，正好是本格预注册里的分解；代码改自 JaxGCRL。
- **FiRL**（Hossain & Roy，ICML 2026）：已把 Finsler 代价和满足三角不等式的不对称路径代价放进 RL。

**剩下的问题**：规划用的最短时间 / 最小代价拟度量，其不对称部分是否追踪熵产生。对 −log p 型距离，这由涨落定理直接推出，所以新颖性只能来自不是对数似然比的拟度量。

**物理陷阱**
- 带 FDT 噪声的阻尼单摆在稳态时熵产生为 0；
- 角速度是奇宇称，时间反转必须翻转 ω；
- 确定性阻尼没有有限的随机熵产生。

**引用更正**
- Rahaman 已发表为 ICLR 2020。
- Vesessook & Yang 的 TRA 随耗散非单调，在中等耗散处达到峰值。
- Myers NeurIPS 2025 应补上作者 Zheng。
- TD-JEPA 的头部是 MRN 形式，不保证三角不等式。
- Seif, Hafezi, Jarzynski 是否发表于 Nature Physics 2021 仍无法确认。
