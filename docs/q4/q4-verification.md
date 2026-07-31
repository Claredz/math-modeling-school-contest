# Q4 验证

验证器检查资源容量、决策时刻不早于任务揭示时刻、选中任务来自认证包，并显式保留无法安排的任务。输出 `certified_feasible` 或 `unresolved`，不把未认证包的收益写入认证价值。hindsight 只作为离线上界，`causal=false`，绝不冒充在线策略。
