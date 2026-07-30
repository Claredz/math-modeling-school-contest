# 模型选型 toy demos

本目录的五组实验全部是为 Stage 3 模型与算法选型构造的**合成小例**。它们不读取赛题
场景、不复用历史正式结果，也不产生 Q1–Q4 的正式结论。JSON 只写入本目录的
`results/`；项目根目录的正式 `results/` 不受影响。

## 运行与复核

从仓库根目录执行：

```powershell
python -m experiments.toy_demos.run_all
python -m experiments.toy_demos.run_all --check
python -m pytest tests/toy_demos -q
```

运行器使用固定种子 `20260731`，为每个模块写一份 JSON。`--check` 会重新运行全部
实验，并比较除真实墙钟计时 `runtime_s` 外的所有确定性字段；文件缺失、JSON 无效、
确定性字段过期或目录中出现未声明 JSON 时均返回非零退出码。可用 `--seed` 和
`--output-dir` 显式覆盖默认值。

## 五类比较

- Q1：同一已知上界的二维连续合成目标上比较五条优化路线。
- Q2 约束生成：展示有限初始网格漏约束、分离 oracle 补点和独立复核。
- Q2 联合原型：比较候选生成加连续精修与分离-oracle 路线，并保留全局界和
  `unresolved` 状态。
- Q3：用精确枚举得到审核用 Pareto 前沿，比较 epsilon-constraint 选择与多种子
  DEAP NSGA-II 的覆盖率、精度和稳定性。
- Q4：比较离线枚举、离线 MILP、零预测滚动 MILP 和因果贪心；离线值仅是后见
  上界，在线方法不能看到未来批次。

## 保证边界

`ToyRunRecord.converged` 只表示对应合成算法按其自身终止与验证规则收敛；失败原因、
未收敛和未解析状态会原样保留。toy demo 不能证明正式赛题的几何假设、物理参数、
算法全局最优性或最终数值答案。正式采用何种模型仍须结合文献证据、题意事实与
Stage 3 决策矩阵。
