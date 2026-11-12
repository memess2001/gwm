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
