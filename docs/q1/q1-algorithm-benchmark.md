# Q1 算法 benchmark

在相同变量边界、随机种子 20260731、评估预算和独立 verifier 下比较：

1. Multi-start SLSQP；
2. Sobol + SLSQP；
3. SHGO；
4. Differential Evolution + SLSQP。

Sobol 路线先用与最终选择完全相同的候选排名函数筛选最佳已评估样本，再交给
SLSQP；构造失败、非有限值和无效候选不能成为 warm start。最终选择也使用同一排名，
并以起爆/中心时刻作确定性平局规则。

修复后四路线的认证覆盖总时长平均仍为约 10.3761 s；Sobol 的平均最大裸露从旧
实现约 8.9100 s 降为约 8.5751 s，但 SHGO 的约 6.9509 s 仍最低，因此当前有限预算
下的 winner 仍为 `shgo`。结果写入 `results/q1_rebuild/q1_algorithm_benchmark.csv`；
原生成功不等于认证可行，也不宣称连续非凸问题的全局精确最优。
