# Q2 联合覆盖验证

联合 verifier 对每个检测时间组件执行连续时间细分。在每个时间单元使用舰船速度界扩大目标圆盘、使用烟幕衰减速度收缩烟幕圆盘，再调用空间覆盖证书。闭合的覆盖单元写入 `certified_covered_intervals`；精确见证点只有在同一固定点能保守地覆盖整个时间单元时，才写入 `certified_exposed_intervals`。否则继续细分，达到时间容差仍无法闭合的单元保留为 `unresolved`。

输出同时给出：

- `coverage_lower_s` 与 `coverage_upper_s`；
- 总裸露时间的上下界；
- `maximum_continuous_exposure_s`（认证裸露下界）及显式的 `maximum_exposure_lower_s` / `maximum_exposure_upper_s`；
- `joint_gain_s = joint_coverage_lower_s - best_single_smoke_coverage_lower_s`；
- 最佳单烟幕基线的 `best_single_smoke_candidate_id`；
- 最坏时间见证和未决时间单元。

若路径超出 12 km 作战半径，直接返回 `certified_infeasible`。因此 Q2 结果中的 `certified_feasible` 只表示当前 verifier 已关闭的时间—空间包络，不表示全局最优；`unresolved` 不被隐藏或提升为可行。

未决区间不被静默并入覆盖或裸露；最大连续裸露不再使用“总量减覆盖下界”的粗上界。四个代表场景的结果仍为当前有界搜索下的最好已知多弹方案；`q1_covered_duration_s` 只作为跨问参考，不参与 `joint_gain_s` 的正式定义。
