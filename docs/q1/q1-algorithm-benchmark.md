# Q1 算法 benchmark

在相同变量边界、随机种子 20260731、评估预算和独立 verifier 下比较：

1. Multi-start SLSQP；
2. Sobol + SLSQP；
3. SHGO；
4. Differential Evolution + SLSQP。

`results/q1_rebuild/q1_results.json` 保存每个代表场景的方法原生状态、评估次数、
运行时间和认证状态。原生成功不等于认证可行；最终选择只按 Q1 词典序认证结果，
不宣称连续非凸问题的全局精确最优。
