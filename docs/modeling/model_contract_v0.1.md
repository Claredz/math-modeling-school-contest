# B题统一模型契约 v0.2

> **已解冻待审计（2026-07-31）：** 本文件仅作为旧契约快照。当前研究状态已
> 回退到 Stage 2/3，尚无有效的 Stage 4 模型契约；文中的“冻结”“当前有效”和
> “不得绕过”等措辞不再授权正式实现。未来是否恢复其中条款，取决于文献、
> Q1 审计和 toy demo。
>
> 状态：2026-07-30 惯性纯追踪与舰载起飞修订后冻结；当前有效契约
> 文件名保留 `v0.1` 仅为兼容既有链接，正文版本和效力以本页为准。
> 事实来源：[problem_facts.md](problem_facts.md)
> 假设来源：[assumption_register.md](assumption_register.md)
> 修订设计：[惯性纯追踪与舰载起飞模型修订设计](../plans/2026-07-30-inertial-pursuit-shipborne-launch-design.md)
> 目的：固定四问共享的数学对象、时间语义、输入输出和成功判据。实现不得绕过本契约另写一套判定。

## 1. 契约原则

1. 题面事实、解释、场景假设和高级扩展分层存储。
2. 四问共享同一套状态方程、烟幕半径函数、探测集合和联合覆盖判据。
3. 优化器只产生候选，独立验证器决定候选是否合法。
4. 任何 \(100\%\) 防御结论必须对应全称条件，而不是仅比较持续时间或离散样本平均值。
5. 解析不可行性判定优先于数值优化。

## 2. 坐标、时间和单位

### 2.1 坐标系

基准坐标按已批准的 A-001 采用：

\[
\boldsymbol s(0)=\boldsymbol 0,\qquad
\boldsymbol e_s=(1,0)^\mathsf T.
\]

这是对任意原始场景做刚体变换后的规范坐标，不改变距离、夹角、圆盘包含关系和可达性。

### 2.2 时间

统一时间原点 \(t=0\) 为任务决策开始时刻（A-018），不自动等于导弹进入探测区时刻。每枚弹 \(j\) 必须区分：

- \(t^c_j\)：投弹指令时刻；
- \(t^d_j\)：实际投弹/释放时刻；
- \(t^e_j\)：起爆时刻；
- \(t^h_j=t^e_j+18\)：恒定半径结束时刻；
- \(t^f_j=t^e_j+23\)：完全失效时刻。

按已批准的 A-005、A-006：

\[
t^d_j\ge t^c_j+2,\qquad
t^e_j=t^d_j+3.5.
\]

### 2.3 单位

- 长度：m；
- 时间：s；
- 速度：m/s；
- 角度：计算层使用 rad，展示层可转为 degree；
- 能耗未知时：以飞行距离 m 和等待时间 s 分项报告，不伪造焦耳。

## 3. 索引与共享符号

| 符号 | 含义 | 单位 | 类型 |
|---|---|---:|---|
| \(i\in\mathcal I\) | 导弹索引 | 无量纲 | 索引 |
| \(u\in\mathcal U\) | 无人机索引 | 无量纲 | 索引 |
| \(j\in\mathcal J\) | 烟幕弹/烟幕索引 | 无量纲 | 索引 |
| \(\boldsymbol s(t)\) | 舰船中心位置 | m | 状态 |
| \(S(t)\) | 舰船完整等效被探测区域 | m²区域 | 派生集合 |
| \(\boldsymbol m_i(t)\) | 第 \(i\) 枚导弹位置 | m | 状态 |
| \(\psi_i(t)\) | 导弹实际速度航向角 | rad | 状态 |
| \(\lambda_i(t)\) | 导弹指向舰船的视线角 | rad | 派生量 |
| \(\alpha_i(t)\) | 舰船相对导弹探测光轴偏角 | rad | 派生量 |
| \(\boldsymbol u_u(t)\) | 第 \(u\) 架无人机位置 | m | 状态 |
| \(\boldsymbol p^d_j\) | 第 \(j\) 枚弹投放点 | m | 决策变量 |
| \(\boldsymbol c_j(t)\) | 第 \(j\) 个烟幕中心 | m | 状态 |
| \(R_j(t)\) | 第 \(j\) 个烟幕半径 | m | 状态 |
| \(C_j(t)\) | 第 \(j\) 个烟幕区域 | m²区域 | 派生集合 |
| \(\mathcal D_i\) | 第 \(i\) 枚导弹有效探测时刻集合 | s的集合 | 派生集合 |
| \(g_{\mathcal J}(t)\) | 联合覆盖缺口函数 | m | 验证指标 |

## 4. 舰船对象

题面给出匀速直线运动：

\[
\boldsymbol s(t)=\boldsymbol s_0+v_s t\boldsymbol e_s,
\qquad
v_s=7.71.
\]

舰船完整等效区域定义为闭圆盘：

\[
S(t)=B(\boldsymbol s(t),r_s)
=\{\boldsymbol x:\|\boldsymbol x-\boldsymbol s(t)\|\le r_s\},
\qquad r_s=80.
\]

## 5. 导弹对象

### 5.1 正式惯性纯追踪模型

惯性纯追踪是 A-002，不是题面直接给出的制导律。对每枚导弹使用状态
\((m_{i,x},m_{i,y},\psi_i)\)：

\[
\dot{\boldsymbol m}_i(t)
=v_{m,i}
\begin{bmatrix}
\cos\psi_i(t)\\
\sin\psi_i(t)
\end{bmatrix},
\]

\[
\dot\psi_i(t)
=
\operatorname{clip}
\left(
k_i\operatorname{wrap}_{(-\pi,\pi]}
\bigl(\lambda_i(t)-\psi_i(t)\bigr),
-\omega_{i,\max},
\omega_{i,\max}
\right).
\]

其中 \(k_i>0\) 是航向响应系数，\(\omega_{i,\max}>0\) 是最大转弯角速度。标准单威胁场景取 \(v_{m,i}=320\)，问题四允许按 A-012 在场景中覆盖。

按 A-021，正式场景扫描：

\[
k\in\{0.5,1,2\}\ {\rm s^{-1}},
\qquad
\omega_{\max}\in\{5^\circ,10^\circ,20^\circ\}/{\rm s}.
\]

按 A-022，导弹出现时刻的航向为：

\[
\psi_i(t_i^{\rm app})=\lambda_i(t_i^{\rm app}).
\]

命中事件定义为：

\[
\|\boldsymbol m_i(t)-\boldsymbol s(t)\|\le r_s.
\]

进入该闭圆盘的首时刻记为 \(t^{\rm hit}_i\)。若出现时已经满足命中条件，应直接记录命中事件，不计算零相对向量下未定义的视线角。

### 5.2 视线和视场

\[
\lambda_i(t)=\operatorname{atan2}
\bigl(s_y(t)-m_{i,y}(t),s_x(t)-m_{i,x}(t)\bigr).
\]

按 A-003，导弹光轴与实际速度方向 \(\psi_i(t)\) 一致，因此：

\[
\alpha_i(t)=\operatorname{wrap}_{(-\pi,\pi]}
\bigl(\lambda_i(t)-\psi_i(t)\bigr).
\]

有效视场条件是 \(|\alpha_i(t)|\le15^\circ\)，不是 \(|\lambda_i(t)|\le15^\circ\)。惯性转弯过程中 \(\alpha_i(t)\) 一般不为零，因此距离入口、视场入口、视场出口和命中都必须作为独立事件求解。探测集合允许由多个连续分量组成。

### 5.3 瞬时纯追踪消融

原瞬时纯追踪仅作为极限对照：

\[
\dot{\boldsymbol m}_i(t)
=v_{m,i}
\frac{\boldsymbol s(t)-\boldsymbol m_i(t)}
{\|\boldsymbol s(t)-\boldsymbol m_i(t)\|}.
\]

该模型必须标记 `model_layer: ablation`，不得进入正式方案优化或 9 组惯性参数范围统计。

### 5.4 可选比例导引扩展

只有在 X-001 获批后才启用。建议使用闭合速度和航向状态：

\[
\dot\psi_i=
\frac{\operatorname{clip}
\left(N_iV_{c,i}\dot\lambda_i,-a_{i,\max},a_{i,\max}\right)}
{v_{m,i}},
\]

\[
\dot{\boldsymbol m}_i
=v_{m,i}(\cos\psi_i,\sin\psi_i)^\mathsf T.
\]

其中 \(N_i,a_{i,\max}\) 必须来自场景假设；不得把 \(N=4\) 或 \(10g\) 写成题面事实。

## 6. 无人机与可达域

### 6.1 起飞等待与连续路径约束

基准模型使用 A-009、A-019。令 \(\tau_u\) 为第 \(u\) 架无人机起飞时刻。起飞前 UAV 留在舰上并随舰移动：

\[
\boldsymbol u_u(t)=\boldsymbol s(t),
\qquad t<\tau_u,
\]

\[
\boldsymbol u_u(\tau_u)=\boldsymbol s(\tau_u).
\]

场景不得提供 UAV 自由初始坐标或非零发射偏置。起飞后的空中轨迹满足：

\[
\boldsymbol u_u:[\tau_u,T]\rightarrow\mathbb R^2
\quad\text{连续且分段可微},
\]

\[
\|\dot{\boldsymbol u}_u(t)\|=28,
\qquad t\in(\tau_u,T)
\quad\text{（所有飞行航段）}.
\]

无人机可在起飞前等待，但一旦起飞，基准模型不允许通过令
\(\|\dot{\boldsymbol u}_u\|=0\) 在空中悬停。舰上等待不计入飞行航程；飞行能耗继续按 A-014 以实际路径长度代理。

### 6.2 无转弯约束的前向可达集

无人机在时刻 \(\tau\) 的发射位置唯一确定为 \(\boldsymbol s(\tau)\)。允许起飞前等待且起飞后速度固定时，时刻 \(t\) 的可达集写成：

\[
\mathcal R_u(t)
=\bigcup_{0\le\tau\le t}
\mathcal R_u(t\mid \tau_u=\tau,\,
\boldsymbol u_u(\tau)=\boldsymbol s(\tau)).
\]

必须对起飞时刻 \(\tau\) 取并集，不能把发射点视为舰船初始位置或场景中的固定基地。

无论使用哪种解析外逼近，最终都必须显式构造一条起飞后全程速度为28 m/s、无空中悬停的连续分段直线路径。

### 6.3 作战半径

按修改后批准的 A-010 和可执行约束 A-020，无人机起飞后的任意时刻必须满足：

\[
\|\boldsymbol u_u(t)-\boldsymbol s(t)\|\le12000,
\qquad t\in[\tau_u,T].
\]

这是相对运动舰船实时位置的约束，不是相对舰船初始位置、固定基地或累计航程的约束。累计飞行距离只作为A-014的能耗代理另行报告。

### 6.4 多次投弹和安全距离

同一无人机相邻实际投弹时刻满足：

\[
t^d_{j_{k+1}}-t^d_{j_k}\ge1.
\]

单机投弹数不超过 3；Q3 中每机严格等于或不超过 1 枚，以题目具体要求为准。

仅当两架无人机都已起飞时检查多机安全约束：

\[
\|\boldsymbol u_{u}(t)-\boldsymbol u_{v}(t)\|
\ge d_{\rm safe},
\quad u\ne v,
\]

其中 \(d_{\rm safe}\) 是场景参数 A-013。若两机同时从同一舰船位置起飞，则起飞瞬间间距为零，方案不可行；仍在舰上的 UAV 不参加空中安全距离检查。

## 7. 投弹、起爆和烟幕

### 7.1 起爆位置

采用 A-006 时：

\[
\boldsymbol p^d_j=\boldsymbol u_{a(j)}(t^d_j),
\]

\[
\boldsymbol c_j(t^e_j)
=\boldsymbol p^d_j+3.5\dot{\boldsymbol u}_{a(j)}(t^d_j),
\]

其中 \(a(j)\) 表示携带第 \(j\) 枚弹的无人机。

这要求投弹点、投弹时刻和投弹瞬间航向共同决定起爆点，不能只优化起爆点后忽略反演可达性。

### 7.2 烟幕中心

固定中心基准 A-007：

\[
\boldsymbol c_j(t)=\boldsymbol c_j(t^e_j),
\qquad t\ge t^e_j.
\]

风漂扩展 X-004：

\[
\boldsymbol c_j(t)
=\boldsymbol c_j(t^e_j)
+\boldsymbol v_{w,j}(t-t^e_j).
\]

### 7.3 烟幕半径与区域

令 \(\tau_j=t-t^e_j\)，则：

\[
R_j(t)=
\begin{cases}
0,&\tau_j<0,\\
120,&0\le\tau_j\le18,\\
120-24(\tau_j-18),&18<\tau_j\le23,\\
0,&\tau_j>23.
\end{cases}
\]

\[
C_j(t)=B(\boldsymbol c_j(t),R_j(t)).
\]

## 8. 探测集合

第 \(i\) 枚导弹在无遮挡条件下的几何探测集合定义为：

\[
\mathcal D_i=
\left\{
t\in[0,t_i^{\rm hit}]:
\|\boldsymbol m_i(t)-\boldsymbol s(t)\|\le8000,\quad
|\alpha_i(t)|\le15^\circ
\right\}.
\]

由于 \(\alpha_i(t)\) 一般不为零，\(\mathcal D_i\) 可写成若干闭区间分量的并：

\[
\mathcal D_i=\bigcup_{\ell=1}^{L_i}[a_{i,\ell},b_{i,\ell}].
\]

求解器不得假设 \(L_i=1\)，也不得复用瞬时纯追踪下的固定探测时长界。

本定义采用 A-004 和 A-017：

- 不擅自加入锁定延迟；
- 基准威胁轨迹先独立计算；
- 遮蔽用于判断探测是否被阻断。

如果未来采用 X-003，\(\mathcal D_i\) 将升级为依赖离散状态的混合系统输出，不能继续直接使用上述静态集合。

## 9. 联合覆盖与防御成功

### 9.1 精确集合判据

时刻 \(t\) 的严格遮蔽条件为：

\[
S(t)\subseteq\bigcup_{j\in\mathcal J}C_j(t).
\]

定义联合覆盖缺口函数：

\[
g_{\mathcal J}(t)=
\max_{\boldsymbol x\in S(t)}
\min_{j\in\mathcal J}
\left(\|\boldsymbol x-\boldsymbol c_j(t)\|-R_j(t)\right).
\]

其中 \(\mathcal J(t)\) 只包含已经起爆且尚未失效的烟幕；若
\(\mathcal J(t)=\varnothing\)，约定 \(g_{\varnothing}(t)=+\infty\)。

则：

\[
S(t)\subseteq\bigcup_jC_j(t)
\iff g_{\mathcal J}(t)\le0.
\]

这一标量函数是 Q1–Q4 的统一遮蔽接口。对于单烟幕：

\[
g_{\{j\}}(t)
=\|\boldsymbol s(t)-\boldsymbol c_j(t)\|+80-R_j(t),
\]

正好退化为圆盘完全包含条件。

### 9.2 保守充分条件

\[
\exists j:\quad
\|\boldsymbol s(t)-\boldsymbol c_j(t)\|+80\le R_j(t)
\]

只是“某一烟幕独立完整覆盖”的充分条件。Q1中它与精确条件等价；Q2、Q3中不能将它误当成联合覆盖的必要条件。

### 9.3 单威胁成功

\[
\operatorname{success}_i
\iff
\forall t\in\mathcal D_i,\quad g_{\mathcal J}(t)\le0.
\]

总裸露时长可作为次级指标：

\[
T^{\rm exposed}_i
=\mu\{t\in\mathcal D_i:g_{\mathcal J}(t)>0\},
\]

但 \(T^{\rm exposed}_i=0\) 不能在未经连续时间认证时替代全称条件。

### 9.4 多威胁成功

每枚导弹分别使用自身 \(\mathcal D_i\)，但烟幕集合可以共享：

\[
\operatorname{success}_i
\iff
\forall t\in\mathcal D_i,\quad
S(t)\subseteq\bigcup_{j\in\mathcal J_i(t)}C_j(t).
\]

全局目标按照题面优先级采用词典序，而非来源不明的加权和。

## 10. 数值验证契约

### 10.1 空间联合覆盖

推荐采用“有界误差的多边形内外逼近”计算 \(g_{\mathcal J}(t)\) 的符号：

1. 用舰船圆盘的外接正多边形 \(S^{\rm out}_K\) 外逼近 \(S\)；
2. 用每个烟幕圆盘的内接正多边形 \(C^{\rm in}_{j,K}\) 内逼近 \(C_j\)；
3. 若 \(S^{\rm out}_K\subseteq\bigcup_jC^{\rm in}_{j,K}\)，认证为覆盖；
4. 用舰船内逼近和烟幕外逼近认证反例；
5. 其余情况增加 \(K\)，仍不能判定则返回 `indeterminate`，禁止返回伪成功。

单烟幕直接使用解析判据，不走多边形近似。

### 10.2 连续时间认证

在时间网格上检查还不够。固定烟幕中心时，\(g_{\mathcal J}\) 的保守时间 Lipschitz 常数可取：

\[
L_g\le v_s+\max_j
\left(\|\dot{\boldsymbol c}_j\|+|\dot R_j|\right).
\]

固定中心、半径恒定时 \(L_g\le7.71\)；衰减阶段可取 \(L_g\le31.71\)。若相邻采样间隔为 \(\Delta t\)，只有在端点/中点的负裕度足以覆盖 \(L_g\Delta t/2\) 和空间近似误差时，才能认证整段时间覆盖。

由于题面把起爆建模为半径从0瞬时跳到120 m，\(g_{\mathcal J}(t)\) 在起爆事件处不连续。连续时间认证必须先按所有 \(t^e_j,t^h_j,t^f_j\)、距离入口、视场入口/出口、探测分量端点和命中事件切分时间轴，再仅在每个连续模式内部使用 Lipschitz 界；事件时刻本身单独按闭区间端点规则检查。

### 10.3 结果状态

每个方案必须返回三值状态：

- `certified_feasible`；
- `certified_infeasible`；
- `indeterminate_at_tolerance`。

不得把数值优化器的“success”字段等同于物理可行。

## 11. 四问接口

### 11.1 Q1输出：`SingleSmokeCandidate`

至少包含：

- 场景ID和假设ID集合；
- `guidance_model`、`model_layer`、\(k\) 和 \(\omega_{\max}\)；
- \(t^c,t^d,t^e,\boldsymbol p^d,\boldsymbol c_e\)；
- 无人机连续路径或路径节点；
- 覆盖起止区间；
- \(g(t)\) 的最大值、最小值及认证误差；
- 最大裸露区间、总裸露时长；
- 可达性状态和完整防御可行性状态；
- 若不可行，解析证书或数值反例。

### 11.2 Q2输出：`MultiSmokePlan`

复用 Q1 候选并增加：

- 单机有序路径；
- 投弹顺序和间隔；
- 每时刻参与联合覆盖的烟幕集合；
- 联合 \(g_{\mathcal J}(t)\) 认证；
- 独立覆盖与联合覆盖的对比；
- 用弹量、航程、鲁棒裕度。

### 11.3 Q3输出：`CooperativePlan`

增加：

- 三架无人机路径和最小动态间距；
- 每机一枚的分配；
- 单点失效后的最坏覆盖指标；
- 词典序或 \(\varepsilon\)-约束求解记录；
- 若使用NSGA-II，必须附穷举/多起点基准和随机种子。

### 11.4 Q4输入：`DefensePackage`

Q1–Q3向Q4提供经过独立验证的任务包：

- 适用威胁场景和探测窗口；
- 所需无人机、弹药、时间窗；
- 起止位置和任务持续时间；
- 认证覆盖率、最大空档、鲁棒裕度；
- 航程和单点失效指标；
- 不确定性假设；
- 9 组惯性参数下的中值、范围、最不利组合和参数敏感标记。

Q4不得直接使用未经空间联合覆盖验证的一维时间区间。

## 12. 契约版本与过时内容

v0.2 在 v0.1 的联合覆盖、词典序和解析优先原则上增加两项实质修订：

1. Q1–Q4 的正式导弹模型改为带一阶航向惯性和最大转弯率的纯追踪；
2. Q1–Q4 的 UAV 均可在舰上等待，但只能从实际起飞时刻的舰船位置起飞。

以下 v0.1 内容已经过时：

- 瞬时纯追踪作为正式基准；
- 视场偏角恒为零；
- 探测窗口必为单个连续区间；
- UAV 发射位置由场景指定；
- 非零甲板偏置作为可用扩展；
- 由瞬时纯追踪直接推得的 24.1677–25.3610 s 正式探测窗口界。

当前实现必须以本契约和
[惯性模型实施计划](../plans/2026-07-30-inertial-pursuit-shipborne-launch-implementation-plan.md)
为唯一入口。
