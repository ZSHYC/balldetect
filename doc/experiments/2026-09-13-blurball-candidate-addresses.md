# 多个匹配地址有没有独立定位增量

日期：2026-09-13。状态：两条件各30轮完成，best复算通过。协议：[候选对应地址v1](../protocols/blurball-candidate-addresses-v1.md)。前序：[集合监督与软对应](2026-09-13-blurball-candidate-set-supervision.md)。

## 结论

**本轮不采用“前16匹配分数及二维地址直接拼接MLP”作为新motion分支。** 同分数、同容量、同初始化条件下，addressed相对scores_only的验证TP4增加2帧，但全部可见帧4px正确数减少33，16px正确数减少41，且match21相对cross仍退步。没有满足运行前要求的多指标与比赛条件。

全局池化按定义丢弃匹配地址，但这尚未被证明是它落后同址特征的原因；本轮补充地址也没有验证这一因果解释。

这不是“对应地址没有信息”的证明。固定第30轮，addressed训练可监督集合内的正确位置比scores_only多22帧；两者都优于原cross。被否定的是这个冻结特征、固定候选、rank排序标量拼接和固定优化条件的具体配方。不能从一次负结果直接跳到“必须加Transformer”或“必须微调大骨干”。

scores_only比旧同址残差增加了匹配分数，也改变首层维度，不能解释成纯容量增益。其完整验证PCK4/F1/PCK16略优于旧同址，但match19的4px正确数比cross少2；保留为竞争参考，不宣布为跨比赛稳健的新最佳方法。

## 实验到底改变了什么

原cross best3的当前K16位置、peak logits、q及三帧视觉输入固定。旧stage0 query与同址历史缓存复用。每个候选对t−1、t−2各取原生72×128网格的前16个cosine和对应地址；并列优先较低格号。16个原生格可以靠在同一峰附近，不等于16个不同目标或不同运动模式。

两新组都读取288维当前/同址描述符和96维匹配标量，MLP为384→32→1，12,353参数；只有描述符乘sqrt(96)。scores_only把位移列置零，addressed保留绑定分数的归一化二维位移。末层零初始化，残差加原候选logit。近远帧顺序、格中心与原图坐标变换见协议。

训练和验证仍为38,854/14,192个合法目标，比赛划分00–17/18–21；每轮遍历全部训练目标，31,887个可监督目标计算集合loss，所有实际batch均有监督。seed0、batch256、AdamW lr3e−4/weight_decay .01，每组30轮。以TP4、raw4、最早epoch依次选优，含epoch0。最终22–25未读取。

## 完整验证结果

全部可见目标12,896。PCK是固定原图欧氏距离严格小于4/16px；F1使用项目本地逐帧定位检测规则，不是作者另一套宽松指标。q固定，发出条件q≥0.5。

| 条件 | best epoch | PCK4 | PCK16 | F1@4 | TP4 |
|---|---:|---:|---:|---:|---:|
| 原cross | 3 | 82.0797% | 85.8716% | 82.2349% | 9,876 |
| 旧同址残差，集合监督 | 5 | 82.6923% | 86.9882% | 82.4930% | 9,907 |
| scores_only | 16 | **82.8707%** | **87.2364%** | 82.5097% | 9,909 |
| addressed | 5 | 82.6148% | 86.9184% | **82.5263%** | **9,911** |

仅按TP/F1看，addressed略好；但raw4是候选位置本身的质量，两者不能混为一个结论。由于q和发出集合完全相同，addressed相对scores_only的raw4净变化−33，可以精确分解为已发出部分+2、被q拒绝部分−35。这不意味着应事后更换q阈值，也没有据此重新校准q。

| 从前者改为后者 | 4px救回 | 4px破坏 | raw4净变化 | TP4净变化 |
|---|---:|---:|---:|---:|
| cross→scores_only | 151 | 49 | +102 | +33 |
| cross→addressed | 104 | 35 | +69 | +35 |
| 旧同址→scores_only | 61 | 38 | +23 | +2 |
| 旧同址→addressed | 19 | 29 | −10 | +4 |
| scores_only→addressed | 39 | 72 | −33 | +2 |

按各自best比较包含选优过程，不是同一优化步的纯输入差值。为检查结论是否仅来自best16对best5，额外复用已记录的30轮指标作同epoch比较：addressed的PCK4与PCK16在**30/30轮都低于scores_only**；TP4则18轮较高、1轮相同、11轮较低。不重训、不重新挑选checkpoint。这支持精位置退步不是仅由不同best轮次造成，但仍不能唯一归因于信息不足、优化或分布差异。

### 比赛与高速子集

以下均为4px正确帧的净变化，独立来源仍只有四场，不能把相邻帧当成独立重复实验。

| 群体 | cross→scores_only | cross→addressed | scores_only→addressed |
|---|---:|---:|---:|
| match18 | +42 | +40 | −2 |
| match19 | −2 | +2 | +4 |
| match20 | +36 | +35 | −1 |
| match21 | +26 | −8 | −34 |
| d1≥16px，n=3,675 | +26 | +23 | −3 |
| d2≥16px，n=7,986 | +72 | +50 | −22 |

退步主要集中在match21；大位移组也未出现地址优势。因此不能用全体TP多2帧证明高速球motion贡献。

固定match21历史群体保持原成员：原same在16px内而cross不在的202帧，scores_only救回34帧@4px、47帧@16px；addressed只救回6/12。原cross在16px内而same不在的74帧，scores_only破坏5/6帧@4/16px，addressed破坏6/7。这些群体只作解释，没有参与选优。

## 搜索阶段确实保留了多少正确地址

这里用GT选择“当前K16中离当前球最近的位置”作为诊断query，且要求当前/历史都V1。没有把GT输入任何提取或推理路径。条件覆盖只看当前候选已在4px内的行；联合覆盖使用所有当前/历史都V1的行，当前候选漏球也算失败。原生R16指历史GT所在格在前16名，16px覆盖指至少一个格中心在历史球16px内，两者含义不同。

| 划分/间隔 | 双帧V1数 | 当前覆盖4px数 | 条件原生R16 | 条件历史16px覆盖 | 联合当前4px+历史16px |
|---|---:|---:|---:|---:|---:|
| train Δ1 | 31,748 | 30,747 | 80.22% | 88.50% | 85.71% |
| train Δ2 | 31,072 | 30,072 | 71.40% | 79.54% | 76.98% |
| val Δ1 | 12,680 | 11,296 | 90.49% | 96.64% | 86.09% |
| val Δ2 | 12,547 | 11,176 | 84.13% | 91.01% | 81.06% |

前序原生top1在相同验证条件下的历史16px命中约79.76%/65.59%；保留前16确实增加了正确历史位置的可达性。但高覆盖仍不等于自动选对球：它以正确当前query为条件，不解决哪个当前候选是球；16px也不是4px精定位；分数和地址不直接编码每个历史匹配点的球身份。当前目标仍受固定K16覆盖上限限制。

训练和验证的对应条件覆盖差异明显，且这里验证更高。不能笼统写成“验证对应更差导致退步”；应按实际位移、模糊与比赛背景区分。这个差异也提醒我们不要用总体单一对应成功率代替自动定位机制。

### 运行后的探索性错误候选竞争

进一步只读现有缓存，限定当前K16可达4px、当前/历史均V1，但cross原top1误差≥16px。比较GT最近的正确当前query与错误top1 query的历史匹配。此诊断在训练结果之后增加，不参与checkpoint或超参选择；不是固定202/74群体的重新定义。

| 群体/间隔 | n | 正确query前16覆盖历史16px | 错误query前16覆盖历史16px | 正确query最高cosine胜过错误query | q发出数 |
|---|---:|---:|---:|---:|---:|
| train Δ1 | 248 | 66.13% | 21.77% | 50.81% | 177 |
| train Δ2 | 251 | 44.22% | 28.69% | 51.00% | 180 |
| val Δ1 | 679 | 89.40% | 27.98% | 20.77% | 253 |
| val Δ2 | 672 | 76.19% | 28.87% | 21.43% | 250 |
| val match21 Δ1 | 185 | 89.73% | 25.95% | 11.35% | 11 |
| val match21 Δ2 | 181 | 77.35% | 29.28% | 11.60% | 11 |

这比全体96.64%/91.01%的条件覆盖更贴近待修复的错误：正确query的搜索覆盖有所下降，但不少历史球位置仍在集合里。与此同时，在验证远距离错误群体中，正确query的最高cosine只约21%能赢过错误query，match21约11%；更强匹配分数不能直接解释为更像球。错误query的前16也有约28%覆盖到历史球，因此“能找到历史球”本身仍不是当前query一定正确的充分条件。

该统计未逐帧验证错误最高对应属于背景线条、球员还是其他物体，不能直接把所有输掉的匹配都命名为静态背景；目前证明的是**错误当前位置的匹配分数竞争**。

训练难例只有约250个，而验证约670个，两者正确query覆盖与cosine胜率的结构也不同。原cross在训练集拟合后产生的候选，不能被默认认为覆盖了验证中的同类排序错误；这提示检查候选监督的错误分布，不证明必须用某一种重采样或out-of-fold训练修复。match21群体绝大多数又被固定q拒绝，解释了raw定位收益为何很少转成TP。后续机制研究需同时看真实当前候选、历史对应和发出决策，而不能只优化其中一个条件成功率。

## 固定last30的训练拟合

| 条件 | 全部训练V1 PCK4 | 可监督31,887帧correct4 | 可监督PCK4 |
|---|---:|---:|---:|
| 原cross | 93.9367% | 31,001 | 97.2214% |
| 旧同址残差 | 94.4761% | 31,179 | 97.7797% |
| scores_only | 94.5185% | 31,193 | 97.8236% |
| addressed | 94.5852% | 31,215 | 97.8926% |

addressed末轮训练loss约0.12774，scores_only约0.13749；训练精位置也略好。这只是单seed拟合差异，不是地址具有稳定收益的阳性证据。与旧soft监督实验不同，本轮不存在“loss下降但训练精定位普遍恶化”的共同故障。现有证据符合训练拟合改善而验证精位置未增益，但尚不能区分具体的优化偏置、比赛统计差异或rank读出局限。

## 工程成本、文件与复现

设备RTX 5070 Ti Laptop GPU，冻结源输入512×288三帧，stage0网格72×128，float32 cosine与MLP、原候选float64坐标。新提取以8目标为batch，train/val分别39,486/14,332个唯一源帧，每帧各一次。

- 一次提取：train 146.35秒、val 48.89秒，合计195.24秒；包含源RGB缓存读取、stage0、匹配与写盘。峰值分配显存515.31 MiB。
- 缓存：scores float32、cells int16、目标ID，共163,382,448 bytes；不保存全cost或每个匹配的96维key。旧约1.63GB外观池化缓存继续复用。
- scores_only训练/逐轮验证/末轮与best输出共21.20秒；addressed 19.96秒。两者峰值分配显存各1,284.92 MiB，含加载的特征。缓存装载时间另外保存在各config。
- 这些不是端到端在线速度，GPU当时也有其他项目占用；不以两组约1秒耗时差解释结构性能。

实现提交`8eb8e90`。提取manifest在启动时记录前一版本`77de460`和明确改动文件，这些改动随后收入`8eb8e90`；两组训练记录`8eb8e90`。没有新增数据内容哈希。

产物位于`outputs/blurball/spatial_interaction/candidate_addresses/`：`matches/manifest.json`与两split数组，`scores_only/`和`addressed/`配置、best/last、30轮日志、完整train/val预测及last训练选择，另有`comparison.json`、`coverage.json`、`same_epoch_comparison.json`、`smoke.json`、`competition.json`。小型派生分析脚本随输出保存，正式训练与提取入口在scripts，数据/模型/缓存不提交。

实际命令以仓库根为工作目录，Python为Conda zshihyc：

```bash
python scripts/export_blurball_candidate_features.py --representation top_matches --output outputs/blurball/spatial_interaction/candidate_addresses/matches
python scripts/train_blurball_candidate_residual.py --features outputs/blurball/spatial_interaction/candidate_residual/features --matches-cache outputs/blurball/spatial_interaction/candidate_addresses/matches --condition scores_only --loss set --output outputs/blurball/spatial_interaction/candidate_addresses/scores_only
python scripts/train_blurball_candidate_residual.py --features outputs/blurball/spatial_interaction/candidate_residual/features --matches-cache outputs/blurball/spatial_interaction/candidate_addresses/matches --condition addressed --loss set --output outputs/blurball/spatial_interaction/candidate_addresses/addressed
python outputs/blurball/spatial_interaction/candidate_addresses/compare.py
python outputs/blurball/spatial_interaction/candidate_addresses/coverage.py
python outputs/blurball/spatial_interaction/candidate_addresses/competition.py
```

地址排序/绑定、非方网格坐标/near-far、标量独立缩放三项测试通过；既有五项读出测试通过。真实256目标batch含233监督，两组初始loss同为0.08961794，末层零初始化保留原top1，梯度有限。提取首个验证batch最大cosine与旧缓存差0、地址完全一致；完整运行核对源帧不重复、目标身份与q一致、每轮完整目标以及best复现。纯指标比较复用保存结果，没有再次forward。

## 下一步判断

停止本地址拼接配方，不扫宽度、温度、候选数或训练轮数。旧同址残差继续作为强的廉价对照；scores_only保留为有匹配分数的竞争参考，两者都不能被包装成新运动贡献。

本轮进一步确认了错误当前query的分数竞争，以及训练/验证难例结构差异；尚未确认这些错误匹配的具体视觉身份。下一步结合有限失败画面与已有匹配缓存，区分当前空间证据不足、历史目标身份缺失及候选监督偏置，再决定哪个机制值得实现。绑定历史点外观、任务条件匹配监督或训练难例处理仍是待检验选项；不因为“地址仍失败”直接继续堆模块。
