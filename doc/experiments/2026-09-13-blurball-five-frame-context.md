# 双向五帧视觉是否提供独立于物理先验的帮助

日期：2026-09-13。状态：窗口与GT诊断完成，训练/评价入口及真实batch已验证；正式运行状态见下方追加记录。

协议：[五帧上下文v1](../protocols/blurball-five-frame-context-v1.md)。前文：[物理与多帧取舍](../research/2026-09-13-ball-physics-feasibility.md)。本轮按用户允许未来帧的指令，从因果基线扩展到双向视觉定位，不先加入新物理模块。

## 当前判断

训练标签表明，目标前后的真实观测能提供比单纯过去外推更强的中心位置约束。但更长窗口不必更好，三帧双向线性估计已经是不可忽略的简单解释。此处GT拟合不使用目标位置，不是自动检测结果；正式视觉实验仍须自己证明前后证据能够被模型利用。

本轮采用五帧双向与五帧因果的配对开发实验，在同参数、同训练/验证目标与同轮次预算下检验输入时间分配。旧三帧不重训、不改原有事实；本轮也不称五帧已经是最佳长度。

## 合法窗口与实际覆盖

新增`scripts/analyze_blurball_context.py`只读既有RGB缓存的metadata及确认过的内部边界。新`temporal_windows`按原始match/rally/frame索引，不按标签删输入，不跨缺帧或片段，不解码。目标池为原缓存的末帧目标，三帧合法基线为38,854个训练目标、14,192个验证目标。

| 输入 | 自然训练目标 | 自然验证目标 |
|---|---:|---:|
| 因果3帧，t−2:t | 38,854 | 14,192 |
| 双向3帧，t−1:t+1 | 38,541 | 14,122 |
| 因果5帧，t−4:t | 38,222 | 14,052 |
| 双向5帧，t−2:t+2 | 38,222 | 14,052 |
| 双向7帧，t−3:t+3 | 37,590 | 13,912 |
| 双向9帧，t−4:t+4 | 36,958 | 13,772 |

五帧两臂自然数量相等，身份却不同：因此主训练/评价采用两臂交集，**37,590 train / 13,912 val**。所有六种上下文共有36,958 train / 13,772 val。训练入口独立构造并核对得到同一数量和顺序；以下GT比较也采用各自明确的共同群体，不直接比较不同分母。

共同五帧训练目标中当前V1为32,134个。因果五帧的其余四帧均非V1有411例，双向五帧为23例；至少一个支撑V1的增加说明前后可见性具有潜在互补性，不等于这些历史位置已能被模型正确定位。V1也不保证像素特征可读。

## 排除目标位置的GT拟合

对每个窗口，以真实PTS为时间，仅使用非目标时刻的GT二维中点拟合一次或二次多项式，在目标时刻求值。目标坐标只用于最后误差计算。因果条件是外推，双向条件是内插，但接触点附近仍可能不满足单一平滑模型。

五帧两臂比较进一步限定联合所有支撑及目标均V1，共25,702个训练目标；这只是诊断子集，正式训练不据此筛除样本。

| 五帧条件 | 模型阶数 | 中位误差px | 误差<4px | 误差<16px | 长拖影组<4px |
|---|---:|---:|---:|---:|---:|
| 因果 | 一次 | 7.914 | 18.82% | 76.50% | 23.72% |
| 因果 | 二次 | 4.497 | 46.56% | 79.57% | 27.89% |
| 双向 | 一次 | 4.007 | 49.90% | 89.82% | 45.47% |
| 双向 | 二次 | 1.680 | **76.22%** | **97.12%** | **63.09%** |

长拖影为目标半长度l>10px，在这一共同群体内3,116例。原图坐标、标签中点语义保持不变；表中不是球速度或三维动力学估计。

为比较不同长度，另在全部六种上下文共同且所有相关位置均V1的23,506个训练目标上计算：

| 输入与拟合 | 中位误差px | 误差<4px | 误差<16px |
|---|---:|---:|---:|
| 因果3帧，一次 | 3.944 | 50.68% | 87.83% |
| 双向3帧，一次 | 1.973 | **76.23%** | 95.56% |
| 因果5帧，二次 | 4.540 | 46.28% | 79.47% |
| 双向5帧，二次 | 1.685 | **76.16%** | 97.19% |
| 双向7帧，二次 | 2.049 | 70.34% | 96.04% |
| 双向9帧，二次 | 2.789 | 61.35% | 93.54% |

未来观测有明确的几何潜力，但增加更远的点可能带入接触、曲率变化或噪声。这里不能把二次拟合的退步直接解释为神经网络必然退步，也不能声称双向五帧优于双向三帧。五帧是接下来视觉实验的一个有限候选；三帧双向仍是后续更低计算量的必要竞争解释。

这也修订了上一轮优先级：过去三帧恒速规则失败，不足以否定用目标之后的真实观测。GT轨迹在假定身份已知时更准，并不解决错误候选、对应可靠性或可见性判断；不能用本表作为物理模块的自动定位成绩。

## 实现与相关验证

`build_dino_model`新增实际帧数，默认仍为3。五帧模型共享相同前缀，960通道输入读出，实际参数1,279,169。训练窗口准备统一返回当前标签及target slot；双向为2，因果为4。五帧CLI限定真实history与cross_address，防止其他输入组合误标同一协议。

固定局部读出入口也同步支持五帧：去除特征重组的固定3倍通道，复用训练共同窗口和目标位置，按实际帧数构造模型。五帧评价batch沿用训练配置4，旧三帧路径保持原值。局部半径、温度和q不变。

针对性测试先捕获缺少窗口构造、GT目标进入拟合的风险、五帧reshape错误与协议参数误用，再通过实际行为检查。已验证缺帧/rally/内部边界隔离、不可见输入保留、目标位置排除、两臂共同身份及默认三帧行为；真实DINO小输入的forward/backward及批内去重与直接forward一致。相关已有prefetch、resume和空间交互测试复用，不扩大为数据包扫描。

真实512×288、batch4、两种五帧输入各执行一个GPU优化步：均输出147,457类，所有模型参数梯度有限，使用同一组当前目标；各自峰值allocated为4,406.12 MiB。两臂初始loss为9.53133/9.53439，**只是通路检查，不是性能结果**。GPU同时有其他项目任务，单步时间不作吞吐比较。

## 预算、产物与运行

每臂12完整epoch，seed0、batch4，设置与判断边界见锁定协议。先运行center5，随后运行causal5；顺序执行以控制显存。不缩减每轮batch数，不改动旧训练记录。

```bash
conda activate zshihyc
python scripts/analyze_blurball_context.py
python scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/five_frame_context/center5_seed0 \
  --window center5 --interaction cross_address --temporal-input history \
  --epochs 12 --batch-size 4 --seed 0
python scripts/analyze_blurball_readout.py outputs/blurball/five_frame_context/center5_seed0
python scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/five_frame_context/causal5_seed0 \
  --window causal5 --interaction cross_address --temporal-input history \
  --epochs 12 --batch-size 4 --seed 0
python scripts/analyze_blurball_readout.py outputs/blurball/five_frame_context/causal5_seed0
```

诊断产物为`outputs/blurball/context_feasibility/summary.json`与`gt_context_predictions.npz`，保存共同身份及拟合位置，避免改变阈值时重做拟合。真实单批证据为`outputs/blurball/five_frame_context/preflight.json`，一次性脚本也在对应输出目录。正式运行将保存实际代码版本、命令、config、history、best/last与预测。

模型效果尚未得出。完成两臂共同验证之后，再根据位置、检测、逐比赛及救回/破坏决定物理路线，不先填预期增益。


## 正式启动记录

实施提交`e2ba179`后，已于2026-09-13 14:36（Asia/Shanghai）启动顺序任务。独立进程运行`outputs/blurball/five_frame_context/run_pair.sh`，启动记录在同目录`launch.json`，主进程PID3936244；首臂center5子进程PID3936247。实际配置确认输入`t−2:t+2`、target_slot2、12轮、batch4及共同37,590/13,912目标。

首次观察时进程正在全量epoch0验证，随后进入训练；causal5尚在顺序任务后续，未声称已完成。各臂`train.log`与`history.jsonl`记录实际进展，退出状态保存在`exit_status.txt`。两臂完成后才作正式比较，单批检查及GT诊断不替代训练结果。


随后已观察到center5完成epoch0全量验证并进入epoch1，首个训练进度记录为1,600/37,590目标、累计loss8.29160。该值只说明训练正常推进，不是验证定位收益。

两臂完成后的比较入口已准备：`python outputs/blurball/five_frame_context/compare.py`。它只读保存预测，核对共同身份/标签、实际完成轮次及主要训练配置；分别比较argmax和fixed-local，报告原位置与正确输出的救回/破坏、q变化和逐比赛/拖影分组。旧cross只抽取共同目标作参考。新增计数已用CPU手算样例验证；正式比较尚未运行，不自动将任何GT拟合率写入模型成绩。
