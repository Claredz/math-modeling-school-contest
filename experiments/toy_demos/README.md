# 模型选型 toy demos

本目录的五组实验全部是为 Stage 3 模型与算法选型构造的**合成小例**。它们不读取赛题
场景、不复用历史正式结果，也不产生 Q1–Q4 的正式结论。JSON 只写入本目录的
`results/`；项目根目录的正式 `results/` 不受影响。

设计承诺统一为：每个 toy 使用人工可判定的 synthetic 代理问题；在适用时保留
题面结构或量纲，但不得声称其为正式题面实例。所有产物均满足
`synthetic=true`、`formal_result=false`。

| 问题 | 抽象程度 | 保留的结构 | 未保留/不得外推 |
|---|---|---|---|
| Q1 连续优化 | 归一化二维无量纲代理 | 多峰候选、局部精修、解析上界、独立 verifier | 题面坐标、秒数、正式动力学 |
| Q2 约束生成 | 无量纲单位圆几何代理 | 有限主问题漏点、连续 separation、见证补充 | 正式时空 oracle 的收敛与误差界 |
| Q2 联合原型 | 人工 time unit 与三角核 | 离散—连续联合、局部失败、上下界与 unresolved | 正式烟幕量纲、全局可解性 |
| Q3 多目标 | 三 UAV 离散 benefit/risk 代理 | 词典序、ε-约束、精确有限 Pareto 对照 | 正式覆盖能力、能耗与连续前沿 |
| Q4 调度 | 人工 slot/value 任务包 | 因果信息、离线上界、rolling 与贪心比较 | 正式威胁价值、时间和资源参数 |

## 运行与复核

从仓库根目录执行：

```powershell
python -m experiments.toy_demos.run_all
python -m experiments.toy_demos.run_all --check
python -m pytest tests/toy_demos -q
```

运行器使用固定种子 `20260731`，为每个模块写一份 JSON。`--check` 会重新运行全部
实验，并比较除真实墙钟计时 `runtime_s` 外的所有确定性字段。容差按语义分层：
局部求解器坐标/时刻和中间 master 值允许绝对误差 \(10^{-4}\)，求解器派生的
objective/lower-bound/gap 允许 \(10^{-5}\)，其他数值（包括 violation/certificate）
只允许 \(10^{-12}\)；相对误差统一为 \(10^{-12}\)。布尔值、整数、字符串和结构
精确比较。文件缺失、JSON 无效、超出容差的确定性字段过期或目录中出现未声明
JSON 时均返回非零退出码。可用 `--seed` 和 `--output-dir` 显式覆盖默认值。

## 五类比较

- Q1：同一已知上界的二维连续合成目标上比较五条优化路线。
- Q2 约束生成：展示有限初始网格漏约束、分离 oracle 补点和独立复核。
- Q2 联合原型：比较候选生成加连续精修与分离-oracle 路线，并保留全局界和
  `unresolved` 状态。
- Q3：用精确枚举得到审核用 Pareto 前沿，比较 epsilon-constraint 选择与多种子
  DEAP NSGA-II 的覆盖率、精度和稳定性。
- Q4：比较离线枚举、离线 MILP、零预测滚动 MILP 和因果贪心；离线值仅是后见
  上界，在线方法不能看到未来批次。

Q1 五路最终候选都通过 toy verifier，但只有 Multi-start、Sobol 和 SHGO 原生
成功；DE 达迭代上限后依赖局部 polish，PSO 无原生收敛证书。因此 toy 只支持
多路线框架，不支持指定 DE 为正式主算法。

Q2 联合原型的候选组合路线局部失败，constraint-generation 路线达到迭代上限；
两条路线均为 `unresolved`，全局 gap 约 1.176。单独的约束生成小例只证明连续
separation 必要，不证明正式 oracle 已可用。

Q4 的 offline/rolling/greedy 分别为 19/13/18。当前 rolling 是
zero-forecast、whole-package-commitment 弱基线，整包立即不可撤销，且没有终端
价值、资源保留或机会成本；它不能作为正式主算法证据，因果贪心必须保留。

## 保证边界

`ToyRunRecord.converged` 只表示对应合成算法按其自身终止与验证规则收敛；失败原因、
未收敛和未解析状态会原样保留。toy demo 不能证明正式赛题的几何假设、物理参数、
算法全局最优性或最终数值答案。正式采用何种模型仍须结合文献证据、题意事实与
Stage 3 决策矩阵。
