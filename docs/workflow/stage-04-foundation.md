# Stage 4 Foundation 冻结记录

> 完成时间：2026-07-31T09:30:21+08:00  
> Gate B：`approved`（人工）  
> Gate C：`approved_fast_track`  
> Stage 5：已开始，但尚未完成

## 1. 假设、符号与术语

正式 7 条核心假设见
[`assumption_register.md`](../modeling/assumption_register.md)，统一模型与符号合同见
[`model_contract_v0.1.md`](../modeling/model_contract_v0.1.md)。本阶段冻结 14 个
唯一符号，全部给出单位、类型和范围；术语包括瞬时纯追踪（IPP）、半无限规划
（SIP）、约束生成（CG）、连续分离（CS）和独立验证器（IV）。

## 2. 时间与事件语义

- `command → drop：固定 2 s`；
- `drop → burst：固定 3.5 s`；
- UAV 在 2 s 响应期间正常飞行；
- 禁止负时间预指令；
- 同机相邻 drop 至少间隔 1 s；
- 起爆后烟幕中心固定，半径按题面分段函数演化。

## 3. solver / verifier 分离

solver 只负责候选生成、连续精修和 warm start。verifier 独立重建事件轴、路径约束、
连续探测集和烟幕并集覆盖，只能返回：

- `certified_feasible`；
- `certified_infeasible`；
- `unresolved`。

optimizer 的 native success 不能替代认证；任何未闭合区间保留为 unresolved。

## 4. 证据与产物隔离

| 层级 | 位置/标签 | 可支持结论 |
|---|---|---|
| 正式 Q1 | `results/q1_rebuild/` | 正式 4×4 基准矩阵内的认证结果 |
| 正式 Q2 | `results/q2_rebuild/` | 当前最好已知多弹方案及三值认证 |
| 失锁反事实 | `experimental_counterfactual`、`formal_baseline=false` | 参数敏感性，不外推为题面事实 |
| synthetic toy | `experiments/toy_demos/`、`synthetic=true` | 算法链与失败传播，不是正式结果 |
| 旧 Q1 | `results/q1/` | 历史模型结果，不覆盖 |

## 5. 一致性预检与 L1

- Stage 2：变量、四问目标层级和跨问复用链一致；
- Stage 3：瞬时纯追踪、固定中心、分段直线和 Q1/Q2 算法框架与人工决定一致；
- 高严重度问题：无；
- L1：五维均 9，`pass_early`；
- 因无高严重度问题，自动批准 Gate C 并进入 Stage 5。
