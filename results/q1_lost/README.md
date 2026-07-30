# Q1 失锁耦合改进结果

运行命令：

```powershell
$env:PYTHONPATH='src'
python scripts/run_q1_lost.py
```

当前可行性示例采用侧向 12 km 来袭、初始航向误差 10°、最大转弯率 5°/s、失锁转弯惯性时间 5 s、重捕获确认时间 1 s。

| 指标 | 数值 |
|---|---:|
| 起飞时刻 | 0.0 s |
| 投放时刻 | 13.0 s |
| 起爆时刻 | 16.5 s |
| 投放坐标 | (29.215, 0) m |
| 起爆坐标 | (127.215, 0) m |
| 烟幕失锁时刻 | 约 16.52 s |
| 最小舰弹距离 | 约 1810.42 m |
| 150 s 内命中 | 否 |
| 永久失锁逃逸证书 | 是 |
| 14–19 s、0.5 s 网格唯一最优 | 是 |

结构化事件和可追溯字段见 [q1_lost_result.json](q1_lost_result.json)。这里的“唯一”只针对声明的离散搜索网格；连续时间全局唯一性尚未证明。

## 评委自定义参数入口

按“距离 + 方位角”运行：

```powershell
$env:PYTHONPATH='src'
python scripts/evaluate_q1_lost_custom.py `
  --distance 12000 `
  --direction-deg 90 `
  --initial-heading-error-deg 10 `
  --lost-turn-decay-time-s 5
```

也可直接指定世界坐标：

```powershell
python scripts/evaluate_q1_lost_custom.py --missile-x 0 --missile-y 12000
```

程序自动粗搜并精搜起爆时刻，输出 `evaluation.json`、`trajectory.png` 和 `timeline.png`。方位角定义为场景出现时从舰船指向导弹位置的极角，0°、90°、180°分别对应 (+x,+y,-x)。

论文可行域图运行：

```powershell
python scripts/generate_q1_lost_paper_figures.py
```

输出位于 `figures/q1_lost/`。
