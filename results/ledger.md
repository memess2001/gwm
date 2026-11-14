# 实验流水账（ledger）

规则：每次跑完一个配置追加一行；只记录预注册里列出的量；判定（verdict）只能是 prereg 里定义过的类别。
日期用实际运行日期。`params` 为实际参数量。

| date | cell | world | obs | variant | size | params | seed | metrics | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|

## 预注册冻结记录（格 0）

- 2026-11-12 JST · 格 0 · prereg v2 · commit ca9c154。sha256：

```
5c8fdaa719569589ba7f3d95475c31bdc936ec2748845474f10f356f5e12361d  cells/0/prereg.md
fb94ae6cea460e0a2bc61f79c891572482d84811a1feea76947175a11607fbdd  analysis/seam.py
94af7c0276eee11ebda7a5ae46fb0dba23457029e3af7f93f6f2fa44090e8b63  cells/0/c0lib.py
53fff637cd01e651d376df0973427b4bfcd32927d6c08c5648d59edd54263c6b  cells/0/c0_score.py
18e27ac743b983e320f73aefe35504a748c7d7b7613271f51b5faf0d94842f85  cells/0/c0_aggregate.py
```

- 2026-11-12 JST · 格 0 · prereg v2 · commit ca9c154：审稿不确认（`cells/0/review_v2.md`），未运行任何东西，由 v2.1 取代。
- 2026-11-12 JST · 格 0 · prereg v2.1 · commit cdebf56。sha256（含所调用的仪器文件）：

```
532ec03b3f95d8951131e205a4ba305c63d07ac04475d81b038b3437da5e7616  cells/0/prereg.md
3836f67e14071db17fd0d0a1c03e0bf4df95192e76176d5283e469d97a5d179f  analysis/seam.py
22d4a6edd789521c0a448331b6eabb37a0131b73023dd1dd59d67859188ef433  cells/0/c0lib.py
43a4c7c2fce3db9c99e166a5828dedf7b92736d7c40d6ad1ac38def5b68a0784  cells/0/c0_score.py
be445c2eec335cda5334a37eef2141bf1335492d16f0bde3eec6e04ffad0ad70  cells/0/c0_aggregate.py
3645d838cbdd86341f6a7278495dc5e64573357996d529905c1bcd81cd8238cc  analysis/topology.py
3da571aa1c9428747f5ae42dd396b862468ce044d90ce9fac4920dc4f6ccc675  analysis/smoothness.py
2a29a0cfae9d96e264f3519d6201d668ed672ef9d790bb89065bf884e849fb17  analysis/gates.py
eb44263da6ab50ea985e11cfc58cbfaa7665f94cb22a0117e2e561b3ef4ab300  analysis/controls.py
```
- 2026-11-12 JST · 格 0 · prereg v2.1（cdebf56）：第二轮审稿不确认（`cells/0/review_v2_1.md`），未运行，由 v2.2 取代。
- 2026-11-12 JST · 格 0 · prereg v2.2 · commit 08702ee。sha256：

```
18dce12e6490a60f1394dac07856ebf85b096d4c1f2963eeffc939e293c4686c  cells/0/prereg.md
3836f67e14071db17fd0d0a1c03e0bf4df95192e76176d5283e469d97a5d179f  analysis/seam.py
22d4a6edd789521c0a448331b6eabb37a0131b73023dd1dd59d67859188ef433  cells/0/c0lib.py
edd3fc75cb76144962c4337f3b1c2a93fe6229dba6dfb4a3effa110308b831da  cells/0/c0_score.py
c45f95c7b686cb045967fbb97979779070bc02fc0ecf41f6c2d05e496df14eaa  cells/0/c0_aggregate.py
3645d838cbdd86341f6a7278495dc5e64573357996d529905c1bcd81cd8238cc  analysis/topology.py
3da571aa1c9428747f5ae42dd396b862468ce044d90ce9fac4920dc4f6ccc675  analysis/smoothness.py
2a29a0cfae9d96e264f3519d6201d668ed672ef9d790bb89065bf884e849fb17  analysis/gates.py
eb44263da6ab50ea985e11cfc58cbfaa7665f94cb22a0117e2e561b3ef4ab300  analysis/controls.py
```
- 2026-11-12 JST · 格 0 · prereg v2.2 的审稿确认：`cells/0/review_v2_2.md`（结论：确认；预先授权第 3 条）。v2.1、v2.2 都是冻结后的修订，按 Z 2026-11-12 的决定仍需 Z 逐次确认后才能开跑。
- 2026-11-14 JST · 格 0 · Z 接受 prereg v2.2（含两处点名的改动：像素仪器失败只停像素部分；收敛判据用 val 损失），允许开跑阶段 1，在云端跑。决定记录在 `cell6/v1` 分支 `cells.yaml` 的 `decisions`。
