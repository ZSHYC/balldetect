# 可见中心端点辅助：关系排序能否转化为自动定位

日期：2026-09-11起。状态：三组正式30epoch、保存结果复核与统一关系诊断均完成；本辅助配方未通过预定自动定位收益条件，停止该配方的扩展。

执行依据为[端点辅助协议v1](../protocols/tennis-endpoint-auxiliary-v1.md)。比较无辅助full-history、relation与全局appearance-query辅助。主定位器、三真帧输入和推理输出保持既有定义；辅助只在训练时读已有VC1中心标签。下面分别保存实施smoke与正式结果，不混用两类证据。

## 为什么选这项有限干预

[真实历史控制](2026-09-11-full-temporal-control.md)已确认定位增量，[几何配对](2026-09-11-search-support.md)显示跨native格群体有较大净救回。但主CE没有直接指定历史端点关系，当前数据还无法识别模型内部依赖的是对应、外观补充、局部变化还是背景/历史先验。缺少显式监督不是已证实的失败根因。

本项只改变训练期辅助query：relation从当前GT格取实例特征；appearance用共享192维向量。两者以同一当前GT格定义ROI，使用相同历史候选与正确offset。相比全图空间辅助头，这避免额外改变类别数；它仍未匹配query梯度方向与训练全程优化轨迹，不能声称完全隔离了描述子比较。

[DINO-Tracker、LoFTR与点跟踪近邻](../literature/2026-09-11-point-relation-supervision.md)已经覆盖预训练特征对应适配、候选分布监督与query条件点定位的一般形式。因此本轮是决定后续结构的诊断，不作为新的对应算法贡献。推理仍自动预测当前位置；GT-query排名必须单独报告，不能代替自动检测。

## 实现与检查证据

新增逻辑位于[correspondence.py](../../src/ballmotion/correspondence.py)：只读取当前GT query及两档历史候选，复用当次prefix特征与逐帧空间均值；不用全图unfold，不增加dense cost缓存。训练入口仍为[train_tennis_heatmap.py](../../scripts/train_tennis_heatmap.py)，添加实际对照选择与辅助系数，默认无辅助路径保持原训练规则。w独立初始化，不改变主模型随机序列，单独保存训练状态。

[合成检查](../../tests/test_endpoint_auxiliary.py)验证手工构造的历史−当前offset、越界候选的softmax排除、VC1筛选、空pair零梯度、按非空Δ等权，以及appearance只向历史特征/w回传梯度。该检查在端点函数尚未实现时失败，实现后通过；初次以tests包形式调用失败后改为项目既有直接文件入口。既有三项输入/去重检查与一项前缀梯度检查也通过。

[真实batch校准入口](../../scripts/check_tennis_endpoint_auxiliary.py)运行命令：

```bash
conda activate zshihyc
python scripts/check_tennis_endpoint_auxiliary.py \
  --output outputs/full_heatmap/endpoint_auxiliary_calibration_seed0.json
```

实测使用game1/Clip1/0002–0009八个训练目标，Δ1/Δ2均有8个有效pair；选择依据仅是固定原训练顺序，不看验证结果。70个共同模型参数/buffer与原DINO epoch0逐张量完全相同，新forward拆分后的主logit最大差0。两臂均从此初始化完成一次AdamW更新，所有参数梯度与更新后值有限，prefix首参数最大变化约1.00043e−5；w也获得非零梯度。

| 起始量 | 实测 |
|---|---:|
| 主CE | 10.14242363 |
| relation辅助CE | 1.78868842 |
| appearance辅助CE | 3.67850304 |
| main的prefix梯度L2 | 175.58224487 |
| relation的prefix梯度L2 | 462.32669067 |
| appearance的prefix梯度L2 | 477.25027466 |
| 固定relation系数 | 0.1 |
| 校准appearance系数 | 0.09687300672690803 |
| 加权relation/main起始梯度范数比 | 0.26331062 |

检查段耗时2.6645秒、峰值5,862.17MiB；包括额外forward、三种梯度测量及两次smoke更新，不是正式训练或部署成本。原始记录在[校准JSON](../../outputs/full_heatmap/endpoint_auxiliary_calibration_seed0.json)。执行时HEAD为8fcc96d并带本次未提交实现；随后仅把日志标量转换改成显式detach以去除PyTorch提示，未重复校准或改变数值。本次实现提交后，正式训练以新提交版本单独记录。

## 正式运行与预定解释

两组均用seed0、30epoch、batch8，主checkpoint只按既定F1@16/F1@8规则选择。relation从29bb383运行；appearance随后从5e43b4b按下列固定命令启动，两版本间训练实现没有变化。两组的目标数、输入槽位、辅助系数、epoch预算及完整epoch0验证指标均已核对；appearance额外192参数的学习率也与协议一致。

```bash
python scripts/train_tennis_heatmap.py --model dino \
  --rgb-cache data/cache/tennis/rgb_512x288_all_h2 \
  --output outputs/full_heatmap/dino_relation_aux_seed0 \
  --auxiliary relation --epochs 30 --batch-size 8 --seed 0

python scripts/train_tennis_heatmap.py --model dino \
  --rgb-cache data/cache/tennis/rgb_512x288_all_h2 \
  --output outputs/full_heatmap/dino_appearance_aux_seed0 \
  --auxiliary appearance --auxiliary-weight 0.09687300672690803 \
  --epochs 30 --batch-size 8 --seed 0
```

relation的[日志](../../outputs/full_heatmap/dino_relation_aux_seed0.log)与[配置](../../outputs/full_heatmap/dino_relation_aux_seed0/config.json)、appearance的[日志](../../outputs/full_heatmap/dino_appearance_aux_seed0.log)与[配置](../../outputs/full_heatmap/dino_appearance_aux_seed0/config.json)分别保存。初始指标一致验证了正式入口没有改变共同模型；它不预示训练后的结果。

结束后复用保存预测核对选优、目标身份、PCK/F1，并按固定clip/visibility/位移组比较。三个主任务选优checkpoint另外测同域relation-query exact R@1和NLL：同样的局部候选、同样双端VC1与范围条件，按Δ及跨格组报告。只有relation同时胜过两种对照的预定关系读数和自动PCK@8/F1@16，才支持继续检验结构化融合；指标不一致时按协议保留竞争解释，不扫描更多温度、半径或系数。

最终测试集未使用，没有新增依赖、人工标注或数据解码。

## 三组正式结果

relation与appearance的30epoch均完成；31条epoch记录、主/辅助损失有限、F1选优、checkpoint epoch、训练/验证目标身份及标签、保存预测的指标复算均通过。relation选epoch11，appearance选epoch7，不改选其它epoch。

| 指标 | 无辅助history，epoch7 | appearance，epoch7 | relation，epoch11 |
|---|---:|---:|---:|
| 验证PCK@8 | 81.4433% | 80.0687% | 80.5842% |
| 验证PCK@16 | 88.0298% | 87.2852% | 86.2543% |
| 验证F1@16 | 86.2557% | 85.4407% | 84.3198% |
| 检测@16 TP/FP/FN | 1525/265/221 | 1517/288/229 | 1503/316/243 |
| 存在TP/FP/FN | 1693/97/53 | 1704/101/42 | 1718/101/28 |
| 正式总时长 | 10,287.65秒 | 10,090.45秒 | 10,890.28秒 |
| 正式峰值显存 | 5,265.46MiB | 5,316.09MiB | 5,316.09MiB |

relation的PCK@8下降0.8591个百分点、F1@16下降1.9359个百分点，已经没有通过协议中“优于无辅助基线”的必要条件。训练PCK@8为99.5664%，不能用训练拟合较好替代验证任务收益；不同最优epoch也不独立证明差异来自过拟合或优化。

appearance相对无辅助的PCK@8下降1.3746个百分点、F1@16下降0.8149个百分点。relation相对appearance的PCK@8高0.5155个百分点，但F1@16低1.1209个百分点。因此两种辅助均未取得原定位器之上的联合收益；不能只选relation较高的PCK或appearance较高的F1叙述成功。耗时来自各次真实运行，不能把运行间的差异单独当作辅助算子的理论开销。

## 统一关系诊断入口

[probe_full_endpoint_relations.py](../../scripts/probe_full_endpoint_relations.py)对完成训练且有results.json的run读取原主任务best.pt，不读取或按关系指标另选checkpoint。三个模型统一使用当前实例query；appearance的训练向量w不参与该诊断。批内唯一帧编码及inverse恢复与现有预测路径一致，无额外解码或持久化dense特征。

输出各run的endpoint_relations.csv/json，保存原帧身份、Δ、端点native格、offset类别、exact、NLL、same-cell与有效候选数；按Δ、同/跨格及clip统计，macro对两Δ等权，不按pair总数加权。脚本会核对已完成几何给出的1518=407+1111和1491=141+1350；范围外30/35的旧统计不冒充本脚本新读数。

[一项合成检查](../../tests/test_probe_full_endpoint_relations.py)已通过，抓取批内pair身份恢复错误、两Δ不等数量时宏平均误加权、空分组及手推NLL不符。它只证明CPU聚合逻辑；实际GPU运行及数值另记。示例：

```bash
python scripts/probe_full_endpoint_relations.py --run outputs/full_heatmap/dino_prefix_seed0
python scripts/probe_full_endpoint_relations.py --run outputs/full_heatmap/dino_relation_aux_seed0
```

### 实际GPU读数：排序改善，自动定位未改善

上述两次诊断均在5e43b4b正常退出，合法pair计数与既有几何完全一致：Δ1共1518，Δ2共1491。两者分别耗时10.0485秒、8.2799秒，峰值均501.87MiB；计时含RGB索引/H2D、前缀、局部关系、逐pair CPU读数与分组，不含加载模型/元信息和最终写盘。结果见[无辅助JSON](../../outputs/full_heatmap/dino_prefix_seed0/endpoint_relations.json)与[relation JSON](../../outputs/full_heatmap/dino_relation_aux_seed0/endpoint_relations.json)，逐pair CSV随各run保存。

| 共同GT-query条件 | pair数 | 无辅助 exact R@1 | relation exact R@1 | 无辅助NLL | relation NLL |
|---|---:|---:|---:|---:|---:|
| Δ1/R2，全部范围内 | 1518 | 61.9895% | 76.6140% | 1.142307 | 0.760188 |
| Δ1/R2，同格 | 407 | 78.6241% | 85.5037% | 0.681226 | 0.485971 |
| Δ1/R2，跨格 | 1111 | 55.8956% | 73.3573% | 1.311218 | 0.860644 |
| Δ2/R4，全部范围内 | 1491 | 53.8565% | 70.3555% | 1.543916 | 1.046462 |
| Δ2/R4，同格 | 141 | 70.9220% | 85.1064% | 0.985975 | 0.537853 |
| Δ2/R4，跨格 | 1350 | 52.0741% | 68.8148% | 1.602190 | 1.099583 |

按两Δ等权，范围内总体R@1由57.9230%升至73.4847%，NLL由1.343111降至0.903325；跨格R@1由53.9848%升至71.0861%，NLL由1.456704降至0.980113。这些集合包含同一目标的不同Δ，不能当作3009个独立视频样本。Δ1跨格九个clip的R@1全提高；Δ2八个提高，Clip7由44/86降至43/86。两Δ各clip的跨格NLL均降低。因此改善不限于同格或单一clip，但仍只来自一个开发比赛，不能使用统计显著性表述。

### 同一自动定位任务的配对退步

[保存预测配对](../../outputs/full_heatmap/dino_baseline_vs_relation_aux_seed0.json)中，@8救回97、破坏112，净减15；@16救回55、破坏86，净减31。@8九个clip的净变为−6、−2、+2、−9、+6、−1、−5、0、0；@16为−2、0、+1、−26、+4、−1、−5、−2、0。Clip4贡献较多退步，但并非唯一退步片段。

[固定双VC1几何配对](../../outputs/full_heatmap/dino_baseline_vs_relation_aux_motion_groups.json)进一步排除“只在未受辅助监督的困难visibility退步”的解释：Δ1范围内跨格组@8净减6/1111、@16净减19/1111；Δ2对应净减8/1350、20/1350。对应排序提高和自动位置退步可以同时出现在同一预定可见跨格群体。两个Δ仍然不是独立干预，不能据此确定哪一个历史分支导致退步。

[visibility上下文配对](../../outputs/full_heatmap/dino_baseline_vs_relation_aux_visibility.json)也保留正反变化：当前VC2且历史有VC1的64例，@8从29到31、@16从34到37（@16救回5、破坏2），并非所有困难群体都变差；但VC0且历史有VC1的25例误报由22到23。整体正确但被拒绝的位置从12降到3，错误且被输出的位置从168升到215，VC0误报从97升到101。PCK退步不依赖存在阈值，不能把全部下降解释为置信度校准；也不能只强调少量困难目标救回。

### 当前可作出的研究判断

当前候选已改善了给定正确当前格、双端VC1且历史中心在范围内时的descriptor排序，但没有通过自动定位收益的必要条件。辅助读数变好不说明推理时已自动找到正确query，更不说明几像素的细位置、absence或背景竞争获得改善。硬native格监督、特征更新与原读出的共同优化、当前细节保留，都仍是竞争解释；这些结果没有把其中任何一个定位成根因。

[已确认的空间支撑](2026-09-11-search-support.md)还说明该descriptor包含较大范围的视觉上下文；GT中心处取特征不等于只取球像素。排序改善可以利用球或周围上下文的可区分变化，当前读数不能证明已经恢复微小球本身的纯外观信息或物理对应。

因此本轮不进入transport或增加融合模块，也不把“主loss没有显式对应约束”写成已证实的错误。appearance完成后的比较如下；不能因为relation在排序上胜过appearance，就事后放宽原协议中同时改善自动指标的门控。

## appearance完成后的完整机制判断

appearance的共同GT-query诊断正常退出，仍为同样的1518/1491个pair；在c020c2d执行，耗时9.4181秒、峰值501.87MiB。输出见[appearance关系JSON](../../outputs/full_heatmap/dino_appearance_aux_seed0/endpoint_relations.json)。该读数使用其当前实例descriptor，与其他两组完全相同，未把训练用w当作推理query。

| 两Δ等权宏平均 | 无辅助 | appearance | relation |
|---|---:|---:|---:|
| 范围内exact R@1 | 57.9230% | 66.4045% | 73.4847% |
| 范围内target NLL | 1.343111 | 1.057163 | 0.903325 |
| 跨格exact R@1 | 53.9848% | 62.8608% | 71.0861% |
| 跨格target NLL | 1.456704 | 1.167672 | 0.980113 |

appearance的Δ1/Δ2范围内R@1分别为69.6970%/63.1120%，NLL为0.903378/1.210948；跨格R@1为64.5365%/61.1852%，NLL为1.059915/1.275429。同格R@1为83.7838%/81.5603%，NLL为0.476073/0.593571。relation相对appearance的跨格R@1在Δ1八个clip提高、Clip7持平，在Δ2八个提高、Clip9持平；跨格NLL两Δ各clip均更低。Δ1同格NLL却略差于appearance，不能声称所有分组均改善。

这里得到两个同时成立的结果。第一，**额外端点外观监督本身就能改善实例query的匹配排序**：不能把无辅助到relation的全部排序增量归给跨帧实例比较。第二，在这组初始化、监督尺度和选优条件下，relation还比appearance多约7.08个百分点的总体R@1、8.23个百分点的跨格R@1；它提供了额外的条件排序改善，但同时带着当前query梯度、优化轨迹与不同选优epoch的差别。单seed、单开发比赛不支持统计显著性或唯一机制归因。

[无辅助→appearance位置配对](../../outputs/full_heatmap/dino_baseline_vs_appearance_aux_seed0.json)中，@8救回65/破坏89、@16救回37/破坏50；[appearance→relation](../../outputs/full_heatmap/dino_appearance_vs_relation_aux_seed0.json)中，@8救回100/破坏91，@16救回48/破坏66。[appearance的固定几何组](../../outputs/full_heatmap/dino_baseline_vs_appearance_aux_motion_groups.json)同样在范围内跨格退步：Δ1 @8净减14/1111，Δ2净减15/1350，不是只有relation才产生任务负迁移。

[appearance的visibility上下文](../../outputs/full_heatmap/dino_baseline_vs_appearance_aux_visibility.json)保留了困难群体的局部正结果：VC2且历史有VC1的64例@8仍为29，@16由34升至36（救回2、破坏0）；VC0且历史有VC1的25例误报为23，仍高于原来的22。整体错误且输出的位置为187、正确但拒绝为7、VC0误报101。局部正结果不能替代整体与预定跨格条件的任务退步。

**本轮决定：停止该训练期端点辅助配方，不扫温度、系数或半径，不据此加入transport。** 原主定位器仍是后续比较基线；两个辅助checkpoint和诊断结果保留为有用负证据，不删除。这个结论不否定真实历史增量，也不否定所有motion representation；它否定的是“把这组端点排序辅助加在现有定位器上即可提高自动定位”的具体假设。

下一问题应直接针对自动视觉证据到位置输出的路径。当前浅层空间细节是否被削弱，是值得单独审查的候选；尚未证明它就是本次负迁移的原因。若后续采用新的读出实验，要重新明确其有限假设和对照，不把它记成本轮门控通过后的融合扩展。
