# PR #5 根因审计

## 1. 版本与处置状态

- PR：[#5 实现 Q2 单机多烟幕联合时空覆盖](https://github.com/Claredz/math-modeling-school-contest/pull/5)
- 状态：Draft、已关闭、未合并。
- base：`main@44fd43b83df2c91bf14f40f9a1cb73882694753f`。
- head：`agent/q2-multi-smoke@9e6a46f7edbe4179369b1cb80117351a39c70e9e`。
- 云端 `main` 仍为 `44fd43b`；Q2 源码、结果和计划不在 main。
- [废弃说明](https://github.com/Claredz/math-modeling-school-contest/pull/5#issuecomment-5133151640)
  已明确禁止合并和继承。

本审计只读取 GitHub diff、PR 元数据和远端对象；没有 checkout、merge、rebase、
cherry-pick 或复制 PR #5 的生产代码/数值结果。

## 2. 提交链

| commit | 内容 |
|---|---|
| `a74c57f` | Q2 设计 |
| `da68e70` | 实施计划 |
| `0260fc9` | 联合烟幕空间缺口 |
| `43c3834` | 连续时间联合覆盖验证 |
| `c0275bb` | Q2 模型与单机有序路径 |
| `31802f2` | Q2 候选库 |
| `f390be2` | 候选组合枚举与排序 |
| `a7daf91` | 144+16 场景结果 |
| `9e6a46f` | 结果可复现性 |

## 3. 计划与实现差距

| 承诺 | 实际 | 根因 |
|---|---|---|
| 文献与算法选型 | 12 个变更文件中无文献综述、证据图或 Stage 0–3 记录 | 工程先行，模型选择滞后 |
| 覆盖时空原始空间 | 默认最多 3 个 Q1 基候选，纵向偏移仅 0，横向仅 -50/0/50 m | 过早离散化 |
| 候选附近低维连续精修 | 生产链只有生成→枚举→选优，没有优化器调用 | 设计承诺未进入实施任务 |
| 真正 branch-and-bound | 函数先全枚举，再修改方法标签 | 同一算法重命名，不是独立交叉验证 |
| 最大裸露用保守上界 | 只对确定裸露区间取最大值 | unresolved 未纳入风险上界 |
| 结果保存不确定信息 | 持久化证书无 unresolved 字段 | 无法从结果文件复核风险 |

连续精修审计的调用链是：

```text
generate_q2_candidate_library
  → enumerate_q2_combinations
  → select_best_q2_plan
```

PR 的生产源、运行脚本和测试中没有 `scipy.optimize`、`minimize`、
`differential_evolution` 或独立 refinement API。PR 自身“已知限制”也承认没有
加入连续变量精修。

## 4. 候选空间为何不代表 Q2

PR head 的硬编码限制为：

- `maximum_center_time_bins=3`；
- `longitudinal_offsets_m=(0.0,)`；
- `lateral_offsets_m=(-50.0,0.0,50.0)`；
- 时间和基础结构继承 Q1 `generate_q1_candidates`。

代码证据：

- [`q2.py:374-450`](https://github.com/Claredz/math-modeling-school-contest/blob/9e6a46f7edbe4179369b1cb80117351a39c70e9e/src/smoke_defense/q2.py#L374-L450)
- [`q2.py:395-413`](https://github.com/Claredz/math-modeling-school-contest/blob/9e6a46f7edbe4179369b1cb80117351a39c70e9e/src/smoke_defense/q2.py#L395-L413)

只读聚合结果显示，144/144 个正式场景都只有 9 个候选，每场枚举 129 个组合：

\[
129=\binom91+\binom92+\binom93.
\]

因此“精确枚举”只对固定 9 候选组合库成立。它不能排除：

- 不在三个横向偏移上的更优起爆中心；
- 连续移动投放/起爆时刻后的改进；
- 单弹 Q1 结构之外、但多烟幕联合可行的布局；
- 不同 UAV 路径/响应/烟幕解释下的方案。

## 5. unresolved 风险缺陷

验证器内部同时收集 `exposed_cells` 和 `unresolved_cells`：

- [`verification.py:273-275`](https://github.com/Claredz/math-modeling-school-contest/blob/9e6a46f7edbe4179369b1cb80117351a39c70e9e/src/smoke_defense/verification.py#L273-L275)
- [`verification.py:378-390`](https://github.com/Claredz/math-modeling-school-contest/blob/9e6a46f7edbe4179369b1cb80117351a39c70e9e/src/smoke_defense/verification.py#L378-L390)

但最大连续裸露只在确定 `exposed_intervals` 中取最大值：

- [`verification.py:447-478`](https://github.com/Claredz/math-modeling-school-contest/blob/9e6a46f7edbe4179369b1cb80117351a39c70e9e/src/smoke_defense/verification.py#L447-L478)

这会低估风险上界，因为 unresolved 可能本身裸露，也可能连接两个确定裸露区间。
Q2 持久化转换又完全丢弃 unresolved：

- [`q2.py:220-233`](https://github.com/Claredz/math-modeling-school-contest/blob/9e6a46f7edbe4179369b1cb80117351a39c70e9e/src/smoke_defense/q2.py#L220-L233)
- [`q2.py:679-714`](https://github.com/Claredz/math-modeling-school-contest/blob/9e6a46f7edbe4179369b1cb80117351a39c70e9e/src/smoke_defense/q2.py#L679-L714)

结果中的 `indeterminate_count=0` 只表示最终选中方案的全局状态计数，不能证明所有
候选没有 unresolved，也不能证明最大连续裸露是保守上界。

正确输出至少需要：

```text
exposed_duration_lower_bound_s
exposed_duration_upper_bound_s
maximum_exposed_interval_lower_bound_s
maximum_exposed_interval_upper_bound_s
unresolved_intervals
```

## 6. 数值结论失效范围

PR 报告的以下数字全部失去当前结论地位：

- 0/144 全覆盖；
- 一弹 36 场、三弹 108 场；
- 最大连续裸露改善 0–5.188958 s；
- 空间联合新增覆盖 0.000148–0.001955 s；
- UAV 航程范围。

原因不是“代码不能复现”，而是：

1. 结果只描述固定 9 候选网格；
2. 没有连续精修、网格加密或边界扩展；
3. unresolved 未进入最大裸露上界；
4. branch-and-bound 不是独立算法；
5. 排序使用可能低估的最大裸露量；
6. 119 个测试只证明内部实现一致，不证明候选代表性和连续最优性。

可重复得到同一个离散结果，不等于该结果对原数学问题有效。

## 7. 可独立复审的公共思想

以下只可作为重新推导和 toy demo 的材料，不可直接继承代码或结论：

- 单烟幕解析缺口和圆盘数据结构；
- 内外正多边形联合覆盖证书与精确反例点；
- 半径偏移的联合缺口二分；
- 起爆/保持结束/失效事件切分；
- “单枚均不完整、并集完整”的人工反例；
- 固定速度有序投弹路径几何构造。

复审要求：

- 从 main 的公共层独立实现最小 toy；
- 补数学证明、退化几何、随机反例和高精度参考；
- 同时输出确定裸露下界及把 unresolved 视为裸露的风险上界；
- 公共几何通过不代表 Q2 候选、优化器和结果通过。

## 8. 禁止继承

禁止 merge、rebase、cherry-pick 或复制：

- PR #5 的 9 个提交和全部 Q2 源码/脚本/正式结果；
- 两份 Q2 计划作为新流程的权威依据；
- 固定 9 候选及 3×3 网格；
- “129 组合等于 Q2 精确求解”；
- 当前最大裸露统计和排序；
- 名义 branch-and-bound 及其等价性测试；
- 0/144、用弹分布、改善秒数和毫秒级联合收益；
- 用“119 tests passed”替代模型有效性的表述；
- 用 Q1 候选结构限制 Q2 连续原始域。

最终 verdict：PR #5 是“内部可复现、但因过早离散化和证据链缺失而不能外推到原
问题”的失败案例。公共几何思想可在本轮 toy demo 中独立重建；候选、优化、风险
统计和全部结果不得继承。
