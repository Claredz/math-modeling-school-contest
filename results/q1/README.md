# Q1 惯性纯追踪计算结果

> 正式层采用一阶航向惯性纯追踪；瞬时纯追踪只作为消融对照。

单个固定烟幕完整覆盖的解析上限为 `10.376135 s`。表中方案已从运动舰艇实时位置起飞，并通过 28 m/s 航段、98 m 投弹惯性段和 12 km 作战半径检查。

| 方位 | 初距/m | 中值参数探测时长/s | 最佳完整覆盖/s | 裸露/s | 严格状态 | 参数敏感 | 最坏参数场景 |
|---|---:|---:|---:|---:|---|---|---|
| front | 8000 | 24.167709 | 10.376135 | 13.791574 | certified_infeasible | 否 | `q1_q3_front_d8000_k0p5_w5` |
| front | 10000 | 24.167709 | 10.376135 | 13.791574 | certified_infeasible | 否 | `q1_q3_front_d10000_k0p5_w5` |
| front | 12000 | 24.167709 | 10.376135 | 13.791574 | certified_infeasible | 否 | `q1_q3_front_d12000_k0p5_w5` |
| front | 15000 | 24.167709 | 10.376135 | 13.791574 | certified_infeasible | 否 | `q1_q3_front_d15000_k0p5_w5` |
| rear | 8000 | 25.361043 | 10.376135 | 14.984908 | certified_infeasible | 否 | `q1_q3_rear_d8000_k0p5_w5` |
| rear | 10000 | 25.361043 | 10.376135 | 14.984908 | certified_infeasible | 否 | `q1_q3_rear_d10000_k0p5_w5` |
| rear | 12000 | 25.361043 | 10.376135 | 14.984908 | certified_infeasible | 否 | `q1_q3_rear_d12000_k0p5_w5` |
| rear | 15000 | 25.361043 | 10.376135 | 14.984908 | certified_infeasible | 否 | `q1_q3_rear_d15000_k0p5_w5` |
| side | 8000 | 24.766938 | 10.376135 | 14.390803 | certified_infeasible | 是 | `q1_q3_side_d8000_k0p5_w5` |
| side | 10000 | 24.770269 | 10.376135 | 14.394134 | certified_infeasible | 是 | `q1_q3_side_d10000_k0p5_w5` |
| side | 12000 | 24.772971 | 10.376135 | 14.396836 | certified_infeasible | 是 | `q1_q3_side_d12000_k0p5_w5` |
| side | 15000 | 24.776259 | 10.376135 | 14.400124 | certified_infeasible | 是 | `q1_q3_side_d15000_k0p5_w5` |
| oblique | 8000 | 25.187503 | 10.376135 | 14.811369 | certified_infeasible | 是 | `q1_q3_oblique_d8000_k0p5_w5` |
| oblique | 10000 | 25.189181 | 10.376135 | 14.813047 | certified_infeasible | 是 | `q1_q3_oblique_d10000_k0p5_w5` |
| oblique | 12000 | 25.190533 | 10.376135 | 14.814398 | certified_infeasible | 是 | `q1_q3_oblique_d12000_k0p5_w5` |
| oblique | 15000 | 25.192165 | 10.376135 | 14.816030 | certified_infeasible | 是 | `q1_q3_oblique_d15000_k0p5_w5` |

## 结论

- 共汇总 16 组基础几何；其中中值参数下 16 组严格目标被认证为不可行。
- 不可行并不等于无效：结果仍按词典序给出最大完整覆盖时长、最大连续裸露、最小裕度与飞行距离。
- `parameter_sensitive` 由 9 组惯性参数的探测/覆盖结果范围判定；它不把中值参数误写成题面事实。
- 逐场景哈希、命中/探测事件、解析证书、候选路径与消融结果见 [q1_sweep_results.json](q1_sweep_results.json)。
