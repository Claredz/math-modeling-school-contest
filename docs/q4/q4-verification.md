# Q4 验证

验证器检查资源容量、决策时刻不早于任务揭示时刻、选中任务来自认证包，并显式保留无法安排的任务。在线 rolling/greedy 的可选任务必须同时满足已 reveal、资源可用和 `certified is True`；未认证或未闭合任务进入 `unresolved_task_ids`。因此 `total_value == certified_value`，不把未认证包的收益写入任何价值字段。hindsight 同样只优化已认证候选，未认证/未揭示任务保留为 unresolved；它只作为离线上界，`causal=false`，绝不冒充在线策略。
