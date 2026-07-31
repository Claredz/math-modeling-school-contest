# Stage 8 论文声明—证据审计

审计对象：`agent/stage8-paper-finalization`。论文直接使用 `cumcmthesis` 类文件写作；未生成完整 Markdown 再转换。Stage 8 未修改 `src/`、`scripts/`、`results/` 或根目录 `figures/` 子目录中的算法、科学结果和冻结图表。

## 基线与来源

- PR #8 merge commit：`c85c8b11a93f6829bc6ff66a76a7180c15c02e91`。
- 结果 JSON 的冻结生成基线：`9dea2ef0ee22278479b74a4016b0c0ce6f782e65`。
- Q1--Q4 与灵敏度文件的 CSV/JSON 均在本分支从合并后的 `main` 读取。
- LaTeX 类文件：项目级 `.agents/skills/mathmodel-skill/templates/latex/cumcm/cumcmthesis/cumcmthesis.cls` 的副本 `paper/cumcmthesis.cls`；两者 SHA-256 在构建审计中核验一致。
- 分章节结构参考 [EmpyreanHYR/CUMCM-Latex-template](https://github.com/EmpyreanHYR/CUMCM-Latex-template) 的 README：摘要、问题重述、问题分析、假设与符号、模型建立与求解、误差/灵敏度、模型评价、参考文献、附录。

## 论文声明—证据对照表

| 论文声明 | 结果文件 | 代码/算法 | 图表 | 证据等级 | 允许措辞 |
|---|---|---|---|---|---|
| Q1 比较四种算法 | `results/q1_rebuild/q1_algorithm_benchmark.csv/json` | `src/smoke_defense/q1_rebuild.py` 的 `Q1_METHODS`、`benchmark_q1_methods` | `paper/figures/q1_coverage_timeline.png` | A+B | “固定种子与预算下的四算法 benchmark” |
| Q1 SHGO 平均最大暴露约 6.95 s | Q1 benchmark CSV 按 `method=shgo` 聚合 | `q1_candidate_rank` 与独立 `verify_q1_candidate` | Q1 timeline | A | “本次有限预算 benchmark 中较小” |
| Q1 正式矩阵为 4×4 且 16 行认证不可行 | `q1_results.csv/json` | `run_q1_rebuild.py` formal scenarios | Q1 margin/trajectory | A+B | “16 个正式场景均为 `certified_infeasible`” |
| Q1 独立 verifier 重新检查路径和覆盖 | Q1 JSON 的 `solver_verifier_separated=true` | `verify_q1_candidate`、`certify_single_smoke_continuous_coverage` | Q1 margin | B+C | “由独立连续验证器复核” |
| Q2 使用有限事件候选和 SLSQP 精修 | `q2_results.csv/json` 的候选/状态字段 | `_candidate_burst_times`、`solve_q2_candidates`、`minimize(method=SLSQP)` | Q2 timeline | B+C | “有限候选上的局部精修” |
| Q2 联合覆盖下界为 17.2634–17.2710 s | `q2_results.csv/json` | `certify_joint_coverage` | Q2 coverage timeline | A+C | “四个代表场景的联合覆盖下界” |
| Q2 联合增益为 6.8896–6.8971 s | `best_single_smoke_coverage_lower_s` 与 `joint_gain_s` | `joint_gain_s = joint - best_single` | Q2 joint gain | A+C | “相对最佳单烟幕下界的联合增益” |
| Q2 最大连续暴露被单独认证 | `maximum_continuous_exposure_s`、`unresolved_intervals` | `merge_certified_intervals`、时间单元分类 | Q2 timeline | A+B+C | “认证暴露区间的最大连续长度；未解析不默认为暴露” |
| Q3 固定三架 UAV、每架一枚 | `q3_results.json` 的 `main_interpretation` 与 CSV 计数 | `construct_q3_plan` | Q3 schedule | A+B+C | “三架 UAV、每架恰好一枚的主解释” |
| Q3 路径半径和成对冲突证书通过 | `pairwise_conflict_ok=true`、路径字段 | `certify_operation_radius`、`certify_pairwise_separation` | Q3 bounds/schedule | A+B+C | “按代码安全距离参数的证书通过” |
| Q3 使用带 ε 的词典序筛选 | `lexicographic_rank`、`epsilon_constraints` | `q3_verification_rank` | Q3 schedule | A+B | “带显式数值容差的成功优先词典序” |
| Q4 使用因果 rolling/greedy | `q4_results.csv/json` 的 `causal=true` | `schedule_causal_rolling`、`schedule_causal_greedy` | Q4 resource cases | A+B+C | “认证任务包上的因果策略比较” |
| Q4 hindsight 是离线上界 | `offline_hindsight_upper_bound=true` | `schedule_offline_hindsight` | Q4 bars | A+B+C | “离线 hindsight 子集上界，不是在线策略” |
| Q4 容量 4/2/1 的三种值为 46.4/33.0/17.2 | `q4_results.csv/json` | `run_q4_rebuild.py` | Q4 resource cases | A+C | “当前有限任务包下三种值相同” |
| 灵敏度为五类局部证据 | `sensitivity_results.csv/json` | `run_sensitivity_rebuild.py` | `sensitivity_summary.png` | A+B+C | “局部敏感性，不是全局鲁棒性证明” |

## CSV/JSON/图表一致性核验

1. Q1 benchmark CSV 与 JSON 的四方法、4 场景、24 次预算和聚合均一致；正式 Q1 CSV 与 JSON 的 16 行状态均为 `certified_infeasible`。
2. Q2/Q3 CSV 中的覆盖、联合增益、最大连续暴露与 JSON 场景字段一致；Q2 的联合增益均以最佳单烟幕下界为基线。
3. Q4 CSV 的 rolling/greedy/hindsight 数值与 JSON 的 `resource_cases` 一致；CSV 的 `unresolved` 状态与 JSON 的未解析任务列表一致，非矛盾。
4. 灵敏度 CSV 的 44 行与 JSON 的 `row_count=44` 一致；丢失导引条目属于 JSON 明确标记的实验性反事实。
5. 论文图片均为根目录 `figures/q1_rebuild/`、`figures/q2_rebuild/`、`figures/q3_rebuild/`、`figures/q4_rebuild/`、`figures/sensitivity_rebuild/` 的当前冻结基线副本；未在 Stage 8 重新运行绘图脚本。由于结果 JSON 的 `git_sha` 是生成基线而非当前 Stage 8 HEAD，正文不声称图片由当前 HEAD 重新生成。

6. `q1_coverage_timeline.png` 没有绘制绿色覆盖区间，而 Q1 CSV/JSON 明确给出覆盖时长约 $10.3761\,\mathrm{s}$；正文不从该图反推“没有覆盖”，Q1 数值以 CSV/JSON 为准。
7. `q2_joint_gain.png` 的纵轴标签对上下界区分不充分；正文联合覆盖上下界以 Q2 CSV/JSON 字段为准，图只作为相对趋势展示。

## 最终构建与图片指纹

- `paper/main.pdf`：22 页，844315 bytes；正文公式环境 23 个，嵌入图片 10 张，BibTeX 条目 15 个；第一页为摘要，无目录、图目录和表目录；文本抽取无替换字符。
- `supporting_materials/AI工具使用详情.pdf`：3 页，92506 bytes；文本抽取无替换字符。
- 构建命令：`powershell -NoProfile -ExecutionPolicy Bypass -File .\build_paper.ps1`；AI 报告使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\build_ai_report.ps1`。

图片副本 SHA-256 如下（与 `paper/figures/README.md` 的来源约定对应）：

| 文件 | SHA-256 |
|---|---|
| `q1_counterfactual_comparison.png` | `8B5FDA68525E3603667ECC099E8507B69E38727647DC98708A3F6403E80943E3` |
| `q1_coverage_timeline.png` | `307CAA58F12C41569D952948EE620D65A1266AE4F04179849E8D7B9FB2465124` |
| `q1_margin_curve.png` | `FCC715AC458AE5C1C7370948C6D96DF40A3C78B404760166A78FE1773309313D` |
| `q1_trajectories.png` | `2FA6AAA271F6C7E4B3F8D08EE4D581172D4BD17421E0E2097B3873C6134D3BE2` |
| `q2_coverage_timeline.png` | `9BCB740F6816A7280B3021B5162BD757AB0F021304E755694A8AED5D849737E8` |
| `q2_joint_gain.png` | `CA0846CBFE52281BFC9265B35EB1C031BF32DAB26349A94F23512A7DDBCCB061` |
| `q3_coverage_bounds.png` | `70A7F8EA68AECD69C16EEF754E524CFCF3F940BE1A6EC725A670E84F53BD843A` |
| `q3_schedule.png` | `1C843B80BB7586C636BA39F3B3A5AC8B3B7EC86252FAB9E08B8CC53051DBF198` |
| `q4_resource_cases.png` | `1486EA4DC65852440A2BDBC44D025DADA3C9F01833BFB5A3AFDA2A4EB99F3B3C` |
| `sensitivity_summary.png` | `F5218E611B9136CBF91509DF197FB60100252CE86B4C43FAE4BC6B0221C61833` |

## 禁止措辞清单

下列说法与实际代码或证据不符，论文不得使用：

- “完整 SIP 求解”“完整 Pareto 前沿”；
- “滚动 MILP”“在线最优策略”；
- “连续全局最优”“全局最优解已证明”；
- “Q2/Q3 已完全消除暴露”或把覆盖下界直接叫做联合增益；
- “Q3 现实飞行安全已充分证明”（当前安全距离参数为 $0$）；
- “Q4 三种策略在一般任务流上等价”；
- 把 `native_success` 写成物理认证可行；
- 把未解析时间单元或未揭示任务默认为暴露/已知。

## Stage 8 审计结论

未发现已提交 CSV、JSON 之间的明确科学矛盾；发现两处排版/可视化表达风险：Q1 时间线缺少覆盖色带，Q2 联合增益图的纵轴标签未充分区分上下界。两处均保留冻结图，不修改科学图表，在正文和本审计表中声明以 CSV/JSON 为准。另一个需要持续保留的语义差异是：Q1--Q3 的正式状态多为认证不可行，但仍有可量化的覆盖下界；Q4 的值相同但状态仍为未解析。
