# Stage 6 敏感性分析

本轮以一个正式 Q1 代表场景为锚点，生成五类局部证据：瞬时纯追踪与一阶惯性响应率、PN 反事实的参数来源边界、失锁—重捕获参数 \((\tau_T,\tau_L,T_R)\)、有界烟幕漂移、响应时间和 UAV 作战半径。PN 因缺少导航比/过载来源而明确延期，不虚构数值；失锁参数行明确标为 `experimental_counterfactual`，不改变正式假设登记。

输出位于 `results/sensitivity_rebuild/`，图位于 `figures/sensitivity_rebuild/sensitivity_summary.png`。这些扫描用于识别高敏感假设和适用范围，不构成全参数鲁棒最优性证明。
