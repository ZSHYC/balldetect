# 全量微调后，真实历史是否仍有定位增量？

日期：2026-09-11。状态：正式30epoch、保存结果核对与固定配对分析均已完成。
协议：[Tennis全量时序输入对照v1](../protocols/tennis-full-temporal-control-v1.md)。依据：[HRNet/DINO完整比较](2026-09-11-full-hrnet-dino-comparison.md)。

## 要区分的解释

已完成的DINO真实三帧系统包含当前图像、历史内容和读出能力的共同作用。本轮从相同官方前缀与相同随机新头重新训练重复当前帧`[t,t,t]`，与已有`[t−2,t−1,t]`比较；不继承训练后的历史模型。

同一12,167/1,863个训练/验证目标、原标签和窗口合法性，seed0、batch8、30epoch、float32、无AMP或增强。模型、两个学习率、AdamW、交叉熵、输出格和选优完全沿用DINO历史组。条件和停止规则以锁定协议为准。

### 相同槽位参数化不等于独立单帧模型

本项保持同一读出头，但不能把它说成“有效容量、优化行为也完全一致”的单帧对照。现有GroupNorm按帧分组且没有仿射参数；当三个槽位都是同一个特征F时，归一化后仍是相同特征G。第一个1×1卷积因此满足：

```text
W0 G + W1 G + W2 G + b = (W0 + W1 + W2) G + b
```

absence线性投影也可按三个槽位求和。其后的GELU、3×3卷积和PixelShuffle保持原样，故重复输入的头可在代数上合并为单帧头；名义20,197个头参数对应一个7,525参数的合并表示。这不是对整个前缀模型参数量的比较，也没有在正式运行中合并网络。

进一步，重复组的三个输入投影块获得相同梯度。AdamW的逐元素初始矩为零且各块配置相同时，三个块的自适应更新方向相同，求和后的更新量为单块的三倍，而衰减仍是`(1−lr×weight_decay)`乘以权重和。因此它并不等同于“直接构建192通道单帧头并沿用同一学习率”的优化过程。这里只说明参数化的边界，不据此改学习率或追加单帧训练。

该推导已用现有SpatialProbe、CPU float32、2×192×36×64合成特征作一次检查：合并前后最大logit差4.47e−7，两个输入投影的跨槽位梯度差均为0。使用一个位置类别和一个absence类别产生非平凡梯度；没有读取真实数据、运行backbone或执行优化器步。命令及[结果](../../outputs/full_heatmap/repeat_head_factorization.json)：

```bash
python outputs/full_heatmap/check_repeat_head_factorization.py
```

所以本项估计的是**固定参数化、初始化及训练规则下，真实历史输入相对重复当前帧的可实现收益**。它仍是必要的输入内容对照，但不证明匹配机制成立、历史信息的理论上限，或所有独立单帧模型都已被充分优化。训练和已锁定判定规则保持不变。

训练期间完成的[空间支撑与真实窗口几何补充](2026-09-11-search-support.md)进一步限定解释：同格融合仍有较大的局部图像支撑和GroupNorm全图统计依赖；当前VC1且历史有VC1的1,603个验证目标中，仅18个未被旧两档局部范围的任一合格分支覆盖，已有DINO的124/130个@16错误却发生在覆盖组。该探索性分析不改变本协议；它说明几何未覆盖不足以先验解释多数错误，并不证明有效对应已建立。

## 实现与数值检查改变了什么决定

最终实现只在加载合法三帧窗口后，将模型使用的三个索引替换为该窗口当前帧索引。后续训练仍执行原来的`BackboneProbe.forward`，三个位置槽位、归一化和参数初始化不变。验证沿用既有的批内唯一帧提取及原顺序恢复，不建立跨更新特征缓存。源metadata保留真实历史，事后历史分组仍依据源metadata。

曾检查只执行一次当前帧prefix、再复制特征的训练优化。CPU输出/梯度检查通过，但在真实CUDA batch8中未通过预定逐元素梯度容差：输出logits、位置及存在分数均完全一致，62个prefix参数张量中25个梯度未通过`rtol=1e−4, atol=2e−6`。最大绝对梯度差0.0048828，各参数张量的最大相对L2差约3.60e−4；head梯度最大绝对差约5.96e−8。

仅作为诊断关闭cuDNN TF32后，prefix最大绝对差降至3.24e−5、最大相对L2差约2.73e−6，仍有一个张量未满足原逐元素容差。该观察支持batch形状与数值计算的解释，不说明数学表达式不等价，也没有证明这种差异会造成性能下降。没有放宽容差或只给新组改变精度设置；删除该训练优化，保留与历史组相同的显式三帧计算路径。

两个诊断均从官方前缀和seed0新头开始，使用真实训练batch，含7个VC1和1个VC0；不是最终验证性能。可复查的脚本与结果留在outputs：

```bash
python outputs/full_heatmap/dino_repeat_current_gate.py
DIAG_DISABLE_TF32=1 python outputs/full_heatmap/dino_repeat_current_gate.py
```

结果分别为[dino_repeat_current_gate.json](../../outputs/full_heatmap/dino_repeat_current_gate.json)与[关闭TF32的诊断](../../outputs/full_heatmap/dino_repeat_current_gate_no_tf32.json)，`passed`均为false，表示被拒绝的优化。最终训练不调用该优化，也不设置上述诊断变量。两组所称float32指张量精度及无AMP，不宣称禁用了cuDNN默认TF32。

## 已执行的入口验证

最终CPU输入测试3项通过，覆盖两模型不同输入/解码、真实窗口去重顺序，以及重复当前帧只使用当前内容。smoke复用20帧小缓存，训练显式重复三帧，验证每个当前帧只提取一次特征。

```bash
python -m unittest discover -s tests -p 'test_full_model_inputs.py' -v
python scripts/train_tennis_heatmap.py --model dino --temporal-input repeat_current --rgb-cache outputs/full_heatmap/smoke_rgb --output outputs/full_heatmap/dino_repeat_current_smoke_seed0 --batch-size 8 --epochs 1 --seed 0
```

smoke实际loss=10.1295166，保存epoch0/1并按预定同分规则选epoch0；训练/验证各8行预测的身份、标签和重算指标均一致。与原DINO smoke的epoch0模型逐张量比较，全部初始参数和buffer完全一致，排除了本次修改意外改变初始化的情形。计时1.166秒、峰值已分配5,261.24MiB；它只证明运行链路，不是能力或速度结论。

smoke与CUDA诊断发生在09d7058上的本次未提交修改，不能把旧HEAD单独当作这些修改的完整代码版本。正式运行在最终实现提交后启动，实际版本写入该run的config。环境为Conda zshihyc，没有安装依赖；测试、smoke均正常退出。

## 正式运行与结果

已从4781357启动以下命令，config与epoch0记录已落盘：

```bash
python scripts/train_tennis_heatmap.py --model dino --temporal-input repeat_current --rgb-cache data/cache/tennis/rgb_512x288_all_h2 --output outputs/full_heatmap/dino_repeat_current_seed0 --batch-size 8 --epochs 30 --seed 0
```

输出使用新目录；原历史组位于outputs/full_heatmap/dino_prefix_seed0。复用同一RGB缓存，无新增解码。重复组实际只需14,030个不同当前帧；源缓存仍包含14,160帧，保留真实历史及已有实验用途。已核对正式config中的代码版本、输入槽位、目标数、参数量及训练设置；日志位于outputs/full_heatmap/dino_repeat_current.log。

正式运行已正常退出，按既定F1@16、其次F1@8、同分取先出现者，选中重复组epoch1；真实历史组此前选中epoch7。31条epoch0–30记录、有限训练loss、检查点epoch、训练/验证目标身份与标签，以及保存指标复算全部通过。保留全部30epoch结果，没有因为中途验证下降改变训练、选择规则或精度设置。

以下汇总已执行成功；基准为重复当前帧、挑战者为真实历史，因此正的净救回表示历史增量。现有visibility脚本仅增加实际比较对象及输出文件参数，默认调用仍复现原HRNet/DINO汇总，不重新计算模型特征。

```bash
python outputs/full_heatmap/summarize_saved_result.py outputs/full_heatmap/dino_repeat_current_seed0
python scripts/compare_predictions.py --baseline outputs/full_heatmap/dino_repeat_current_seed0/val_predictions.csv --challenger outputs/full_heatmap/dino_prefix_seed0/val_predictions.csv --output outputs/full_heatmap/dino_repeat_vs_history_seed0.json
python outputs/full_heatmap/summarize_visibility_predictions.py --runs dino_repeat_current_seed0 dino_prefix_seed0 --output dino_repeat_vs_history_visibility.json
```

### 整体结果：真实历史有可实现增量

| 验证指标 | 重复当前帧，epoch1 | 真实历史，epoch7 | 历史减重复 |
|---|---:|---:|---:|
| PCK@8 | 72.2795% | 81.4433% | +9.1638 pp |
| PCK@16 | 81.3860% | 88.0298% | +6.6438 pp |
| PCK@32 | 82.8751% | 89.8053% | +6.9301 pp |
| 检测F1@16 | 78.7476% | 86.2557% | +7.5081 pp |
| 检测@16 TP／FP／FN | 1421／442／325 | 1525／265／221 | +104／−177／−104 |
| 存在判断TP／FP／FN | 1746／117／0 | 1693／97／53 | −53／−20／+53 |

原图8px内的配对为救回231、破坏71，净增160；16px内为救回151、破坏35，净增116；32px内为救回151、破坏30，净增121。位置指标不含拒绝阈值，故@16位置净增116和检测TP净增104并不矛盾：历史组有12个位置正确但被拒绝的目标。重复组在所有1,863个验证目标上均输出，117个VC0全部误报；历史组误报仍多，不能把整体提升说成absence已经解决。

| clip | 有位置目标数 | @8救回／破坏 | @16救回／破坏 |
|---|---:|---:|---:|
| 1 | 124 | 26／4 | 16／3 |
| 2 | 200 | 22／5 | 10／6 |
| 3 | 44 | 5／1 | 4／0 |
| 4 | 870 | 98／31 | 48／7 |
| 5 | 186 | 11／13 | 11／9 |
| 6 | 83 | 10／4 | 5／2 |
| 7 | 124 | 30／8 | 29／4 |
| 8 | 33 | 5／1 | 6／0 |
| 9 | 82 | 24／4 | 22／4 |

@16在九个clip均有正净增；@8在八个clip为正，Clip5净减2。整体不是只由Clip4贡献，但这些clip仍来自同一开发比赛，不能当九场独立比赛证明泛化。

### 固定困难与无球历史分组

当前VC2且历史含VC1的64例，PCK@16从24/64（37.50%）提高到34/64（53.125%），救回12、破坏2，净增10；PCK@8从21/64提高到29/64，救回11、破坏3，净增8。@16的clip贡献为：Clip1 +1、Clip2 0、Clip3 0、Clip4 +5、Clip5 +1、Clip6 0、Clip7 +2、Clip9 +1；该组在Clip8没有目标。因而本次困难组的正增量并非只发生在一个clip，但仍有30/64未达到@16，其中28例两组都错。

当前VC0且历史含VC1的25例，误报从25降至22；历史不含VC1的另92例，误报从92降至75。历史输入未增加这两个固定组的误报数量，但带可见历史的无球目标仍有22/25误报，不支持“可靠地区分历史证据与当前存在”。这些分组中的历史标签仅供事后分析，未输入网络。

### 耗时与解释边界

重复组总耗时9,909.045秒，峰值已分配5,269.84MiB；历史组为10,287.650秒、5,265.46MiB。两者均显式三槽位训练，数据准备复用同一缓存；这里记录各次真实运行，不能把一次耗时差解释为稳定部署加速。重复组所选epoch1的训练PCK@8/@16为86.05%/93.80%，历史epoch7为98.86%/99.69%；选优时刻和优化轨迹不同，重复输入的参数化边界仍按前文保留，不宣称所有独立单帧模型已经充分优化。

**锁定判定已通过**：真实历史提高F1@16且没有降低PCK@8，固定困难组也有净救回。因此，不触发协议中的“因时序输入无增量而暂停”条件。但这不自动批准更大搜索半径或新cost；此前几何覆盖、局部cost负结果和细定位限制仍成立。可成立的结论仅是：这个固定DINO系统在当前单seed、单验证比赛中有效利用了真实历史的额外信息。它尚未区分跨位置对应、短时变化、额外外观线索及优化路径，未建立新motion机制或论文贡献。

完整产物为[结果与资源记录](../../outputs/full_heatmap/dino_repeat_current_seed0/results.json)、[选优及错误分解核对](../../outputs/full_heatmap/dino_repeat_current_seed0/error_summary.json)、[逐clip配对](../../outputs/full_heatmap/dino_repeat_vs_history_seed0.json)与[固定visibility条件配对](../../outputs/full_heatmap/dino_repeat_vs_history_visibility.json)。同期完成的[位置集中度诊断](2026-09-11-readout-concentration.md)独立回答读出分数问题，没有改变这里的预测或判定。

主要分析完成后，又复用旧几何作[双端VC1的位移分组](2026-09-11-search-support.md#输入控制完成后的几何配对2026-09-11补充)：Δ1/R2已覆盖跨格组的@8净增为11.251pp，同格仅0.737pp；Δ2/R4分别为9.333pp和3.546pp。它支持保留跨位置时间信息这一研究因素，但不将描述性位移关联当成correspondence的因果证明，也不改变主要判定。
