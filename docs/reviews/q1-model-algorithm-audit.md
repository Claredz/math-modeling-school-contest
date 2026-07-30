# Q1 模型、算法、代码与结论审计

## 1. 审计结论

当前 Q1 具备两类有价值资产：

1. 在一组明确旧假设下，`10.376135 s` 固定单烟幕几何上界和 `5.5 s`
   最早烟幕形成下界可以作为严格解析证书；
2. 场景哈希、事件集合、连续时间单烟幕验证、路径半径证书和结果序列化具有良好
   工程基础。

但当前 Q1 **没有实现计划承诺的决策变量连续优化**。正式入口只构造有限个
“烟幕中心位于舰船轨迹、若干解析事件时刻、最早/最晚起飞”的候选，再在候选库中
按词典序取最大。现有最佳方案只能称“旧模型下有限候选库最优”；不得称原始连续
空间的局部最优或全局最优。

本审计不重写 Q1、不运行正式结果脚本，也不生成新正式数值。只运行了既有的 26 个
Q1/证书/验证单元测试，结果为 `26 passed`。

## 2. 真实调用图

```text
scripts/run_q1.py:29-45
  └─ solve_all_q1_sweeps()
      └─ solve_q1_guidance_sweep()            q1.py:371-440
          ├─ solve_q1_scenario() × 9 formal   q1.py:219-314
          │   ├─ integrate_inertial_missile()
          │   ├─ build_detection_set()
          │   ├─ certify_single_smoke_duration()
          │   ├─ certify_earliest_smoke_availability()
          │   ├─ generate_q1_candidates()     candidates.py:502-648
          │   │   ├─ _candidate_center_times()
          │   │   ├─ _candidate_takeoff_times()
          │   │   ├─ build_shipborne_release_path()
          │   │   ├─ certify_operation_radius()
          │   │   └─ evaluate_smoke_against_detection()
          │   │       └─ certify_single_smoke_continuous_coverage()
          │   └─ max(candidates, candidate_rank_key)
          ├─ solve_q1_scenario() × 1 ablation
          └─ cross-validate reference candidate across 9 models
  ├─ write_q1_sweep_result()
  └─ write_q1_markdown_summary()
```

关键证据：

- `scripts/run_q1.py:38` 直接进入 `solve_all_q1_sweeps`，没有优化器参数或回调。
- `q1.py:275-289` 生成候选后直接调用内置 `max`。
- `candidates.py:530-545` 枚举有限中心时刻和有限起飞时刻。
- `candidates.py:535` 强制起爆中心为 `ship.position(center_time_s)`。
- `candidate_rank_key`（`candidates.py:69-81`）依次比较严格状态、覆盖时长、
  最大裸露、最小裕度和航程，词典序实现本身清楚。

## 3. 连续优化是否真实调用

### 3.1 计划承诺

旧设计 `docs/plans/2026-07-29-b-problem-modeling-design.md:127` 写明候选后使用
`differential_evolution` 或 SLSQP 连续精修；
`docs/plans/2026-07-29-b-problem-execution-plan.md:432` 再次要求
`scipy.optimize.differential_evolution`。

### 3.2 实际实现

`src/smoke_defense/candidates.py` 唯一的 SciPy 导入是
`scipy.optimize.brentq`（第 9 行），用于：

- 解两段定速路径长度方程（第 143–149 行）；
- 求最晚可达起飞时刻边界（第 443–449 行）。

`events.py` 的 `minimize_scalar` 用于检测事件区间内的标量裕度极值，不优化投放
决策。Q1 入口及候选模块没有调用 `minimize`、SLSQP、`trust-constr`、
`differential_evolution`、PSO、Sobol 或确定性全局优化器。

独立动态探针在一个正式场景中观察到 6 个最终候选、5 个中心时刻和至多 2 个
起飞时刻；SciPy 调用计数为候选路径 `brentq=11`、事件查根 `brentq=6`、
切根搜索 `minimize_scalar=30`。这些调用都服务于解析构造和事件定位，不是对
Q1 原始决策向量的连续优化。

Verdict：**计划—实现不一致，连续精修未实施。**

## 4. 两个解析证书

### 4.1 `10.376135 s` 单烟幕上界

在固定烟幕中心 \(\boldsymbol c\)、舰船以速度 \(v_s\) 直线运动、舰船等效圆盘
半径 \(r_s\)、烟幕最大半径 \(R_{\max}\) 的模型下，完整覆盖要求：

\[
\|\boldsymbol s(t)-\boldsymbol c\|+r_s\le R_{\max}.
\]

舰船中心穿过半径 \(R_{\max}-r_s\) 的可覆盖圆，最长弦为
\(2(R_{\max}-r_s)\)，故

\[
T_{\max}^{(1)}
\le \frac{2(R_{\max}-r_s)}{v_s}
=\frac{2(120-80)}{7.71}
=10.3761348898\rm\,s.
\]

代码位于 `src/smoke_defense/certificates/q1.py:27-56`，核心公式在第 39–44
行；测试 `tests/certificates/test_q1.py:11-16` 验证长于上界的单个探测分量
被认证为不可行。

证据等级：**严格解析上界**，但只在以下前提同时成立时有效：

- 舰船等效区域采用 80 m 圆盘；
- 烟幕最大半径 120 m；
- 烟幕中心固定；
- 舰船匀速直线且速度 7.71 m/s；
- 单枚烟幕独立覆盖；
- 需要连续覆盖某一个探测连通分量。

它不依赖候选生成器，也不证明短于该值的窗口可行。烟幕风漂、中心跟随、其他遮蔽
判据或多烟幕联合覆盖会使该证书不再直接适用。

### 4.2 `5.5 s` 最早烟幕形成证书

旧解释把 2 s 响应延时定义为指令到释放的最短时间，再加 3.5 s 固定起爆延时：

\[
t_e\ge t_{\rm cmd}+2+3.5=t_{\rm cmd}+5.5\rm\,s.
\]

代码位于 `certificates/q1.py:74-115`，公式在第 84–88 行。若探测集合与
\([t_{\rm cmd},t_{\rm cmd}+5.5)\) 相交，并且无预部署烟幕、严格要求全窗口覆盖，
代码在第 100–109 行返回 `certified_infeasible`。测试位于
`tests/certificates/test_q1.py:29-47`。

证据等级：**条件性严格因果下界**。它依赖：

- 2 s 的旧响应语义正确；
- 指令最早时刻确为当前设置的 0 s；
- 不允许预部署烟幕；
- 探测在烟幕形成前已经开始；
- 严格目标要求所有探测时刻无裸露。

UAV 实际飞往投放点可能让最早时刻晚于 5.5 s，因此 5.5 s 是乐观下界，不是所有
场景的精确最早起爆时刻。若响应语义、时间原点或预部署规则改变，必须重算。

## 5. 模型逐项审计

| 项目 | 当前实现 | 审计结果 | 处理 |
|---|---|---|---|
| 导弹模型 | 一阶惯性纯追踪 + 9 组参数 | 结构有邻近文献支持，参数无题面标定 | 降级为 Stage 3 候选；与瞬时 PP、PN 比较 |
| 初始航向 | 指向舰船中心 | 题面未给 | 场景/候选解释 |
| 探测窗口 | 距离 + 实际航向 FOV，事件化检测 | 接口和多分量集合可保留；结果依赖制导/光轴解释 | 保留通用引擎，重算模型相关结果 |
| 命中事件 | `hit_radius_m=80`（`q1.py:205,214`） | 80 m 是等效被探测半径，不是题面命中/毁伤半径 | 不得保留为物理事实；需重定义终止条件 |
| 烟幕中心 | 起爆后固定 | 最小参数候选，不是题面明示 | 候选；比较风漂/有界漂移 |
| 覆盖判据 | 80 m 圆盘被单烟幕圆盘包含 | 与 P7 的几何主判据高度一致 | 作为主候选保留；视线/轮廓作反事实 |
| 连续验证 | 事件分段 + Lipschitz 上界，三值状态 | 单烟幕代码结构严谨，能保留 `indeterminate` | 保留通用思想和测试；模型变更后复核界 |
| UAV 起飞 | 从运动舰船中心起飞 | “舰载”支持来源，但题面未唯一规定点/时刻 | 候选场景语义 |
| UAV 路径 | 28 m/s 一/两段直线、瞬时转弯 | P3 说受机动性能约束；数值未知不等于无约束 | 理想化下界候选 |
| 弹道 | 继承 UAV 速度 3.5 s | “惯性飞行”的一个解释 | 候选，不能称题面事实 |
| 12 km | 相对运动舰船全时段 | 题面基准点不清 | 候选语义 |
| 词典序 | 状态→时长→最大裸露→裕度→航程 | 与 P7 优先级大体一致 | 保留结构，目标定义待 Stage 3 |
| 候选生成 | 中心锁在舰船轨迹；有限事件时刻 | 明显收缩原始连续域 | 仅 warm start/基线 |
| 参数扫描 | 4 方位 × 4 距离 × 9 惯性参数 | 全部是团队场景而非题面实例 | 可保留为敏感性框架，不保留正式数值 |

## 6. 连续验证器的证据边界

`verification.py:45-136` 对单烟幕缺口函数使用：

- 探测分量端点及起爆、保持结束、失效等事件切分；
- 正缺口点作为严格不可行见证；
- 舰速与半径变化率构造 Lipschitz 上界；
- 上界跨零且区间小于时间容差时返回 `indeterminate`。

该三值结构值得保留。它认证的是“给定固定中心圆烟幕、给定舰船轨迹和给定探测
集合”的连续覆盖，不认证：

- 制导律或光轴模型正确；
- 候选是局部/全局最优；
- 失锁后导弹一定不能命中；
- 多烟幕并集的连续时空覆盖；
- UAV 真实机动可执行性。

## 7. 测试和结果文件

Fresh 目标测试：

```text
python -m pytest tests/certificates/test_q1.py tests/test_candidates.py \
  tests/test_q1.py tests/test_verification.py -q
26 passed
```

测试覆盖解析证书、候选排序、舰载路径、参数透传、连续验证和结果溯源。这些测试
证明代码按旧合同运行，不证明旧合同由题面或文献唯一支持。

`results/q1/q1_sweep_results.json` 是历史结果：

- 文件内部记录运行 SHA `15b50878...`；
- Git 历史显示结果文件最后提交在 `48bd674...`；
- 当前研究分支已经把旧合同解冻；
- 当前文件仍写 `model_contract_version=v0.2` 和
  `assumption_register_version=v0.3`。

因此该 JSON 可以用作旧实现复现样本和回归输入，不能作为当前 Gate B 后的正式
模型结果。本轮不重跑、不覆盖它。

## 8. 结论分级

### 8.1 可保留为严格解析结论

- `10.376135 s` 是固定中心、单烟幕、圆盘完整覆盖模型下的上界；
- `5.5 s` 是旧响应语义、无预部署、指令从 0 s 开始时的最早形成下界；
- 若某个连续探测分量严格长于前者，则该模型下单烟幕全分量覆盖不可行；
- 若探测在最早烟幕形成前发生，则该方案类别的严格全窗口目标不可行。

### 8.2 可保留为认证结论

- 对给定候选、给定旧模型和给定容差，单烟幕连续验证器返回的
  `certified_feasible / certified_infeasible / indeterminate`；
- 对给定分段直线路径和旧 12 km 语义的路径半径证书。

这些认证均须连同模型、场景、容差和前提书写。

### 8.3 必须降级

- “最佳投放方案”→“旧模型下有限候选库最佳方案”；
- “连续精修后的解”→不存在，计划未实现；
- “唯一最优”→未证明；候选结构和对称性都可能产生并列解；
- “全局最优”→未证明；
- 9 组惯性参数下的表格→团队场景敏感性结果，不是题面正式答案；
- 80 m 命中事件→无题面依据，相关探测终止时间需重算。

### 8.4 需要未来正式阶段重算

- 所有 Q1 数值投放坐标、投放/起爆时刻和航程；
- 所有依赖旧导弹、命中半径、UAV 起飞、弹道和烟幕中心解释的探测窗口；
- 正式最佳已知方案和任何最优性 gap；
- 若 Stage 3 更换模型，全部 144 个旧正式场景和 16 个消融结果。

## 9. 可保留与待替换模块

可保留并继续测试的通用工程资产：

- 常数单一事实源（但每个字段来源需重标）；
- Pydantic 场景、schema、哈希与结果溯源；
- 角度、闭区间、事件根、时间线和序列化工具；
- ODE 积分接口及不同制导适配器结构；
- 三值认证状态和 `unresolved/indeterminate` 语义；
- 单烟幕缺口、连续时间 Lipschitz 认证的框架；
- 分段路径的几何/作战半径验证器框架；
- 词典序比较器的结构。

必须降级或替换的模型相关资产：

- `FROZEN_ASSUMPTION_IDS` 与“frozen contract”断言；
- 正式一阶惯性纯追踪和无来源 9 组参数；
- `hit_radius_m=effective_radius_m`；
- 舰船中心起飞、固定 28 m/s 无悬停/瞬时转弯；
- 固定烟幕中心和完整速度继承弹道；
- 有限中心时刻/起飞时刻候选作为唯一搜索域；
- 历史 Q1 数值结果的正式结论地位。

## 10. Gate B 建议

Q1 Stage 3 toy demo 应使用人工可判定的低维代理问题，在相同函数评估预算下比较：

- multi-start SLSQP；
- DE + SLSQP；
- PSO + SLSQP/SQP；
- Sobol + `trust-constr`；
- `shgo` 或另一低维确定性全局对照。

所有候选必须通过同一独立 verifier。推荐算法只能根据 toy 的真实收敛、可行率、
耗时和人工正确性选择；本审计不预先冻结正式 Q1 算法。
