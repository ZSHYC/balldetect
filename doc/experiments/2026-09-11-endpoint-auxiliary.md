# 可见中心端点辅助：关系排序能否转化为自动定位

日期：2026-09-11。状态：relation正式30epoch完成并通过保存结果复核；appearance尚未启动；统一关系诊断入口已完成CPU验证，真实GPU读数待运行。

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

两组均用seed0、30epoch、batch8，主checkpoint只按既定F1@16/F1@8规则选择。relation已从29bb383启动，配置中的训练/验证目标12,167/1,863、输入槽位、辅助系数及30epoch预算均符合协议，完整epoch0验证指标与无辅助基线完全相同，首200batch损失有限。appearance按下列固定命令随后串行运行：

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

relation完整stdout/stderr保存在[正式日志](../../outputs/full_heatmap/dino_relation_aux_seed0.log)，配置见[config.json](../../outputs/full_heatmap/dino_relation_aux_seed0/config.json)。初始指标一致验证了正式入口没有改变共同模型；它不预示训练后的结果。

结束后复用保存预测核对选优、目标身份、PCK/F1，并按固定clip/visibility/位移组比较。三个主任务选优checkpoint另外测同域relation-query exact R@1和NLL：同样的局部候选、同样双端VC1与范围条件，按Δ及跨格组报告。只有relation同时胜过两种对照的预定关系读数和自动PCK@8/F1@16，才支持继续检验结构化融合；指标不一致时按协议保留竞争解释，不扫描更多温度、半径或系数。

最终测试集未使用，没有新增依赖、人工标注或数据解码。

## relation正式结果

30epoch已正常退出；31条epoch记录、主/辅助损失有限、F1选优、checkpoint epoch、训练/验证目标身份及标签、保存预测的指标复算均通过。选中epoch11，不改选其它epoch。

| 指标 | 无辅助history，epoch7 | relation，epoch11 |
|---|---:|---:|
| 验证PCK@8 | 81.4433% | 80.5842% |
| 验证PCK@16 | 88.0298% | 86.2543% |
| 验证F1@16 | 86.2557% | 84.3198% |
| 检测@16 TP/FP/FN | 1525/265/221 | 1503/316/243 |
| 存在TP/FP/FN | 1693/97/53 | 1718/101/28 |
| 正式总时长 | 10,287.65秒 | 10,890.28秒 |
| 正式峰值显存 | 5,265.46MiB | 5,316.09MiB |

relation的PCK@8下降0.8591个百分点、F1@16下降1.9359个百分点，已经没有通过协议中“优于无辅助基线”的必要条件。训练PCK@8为99.5664%，不能用训练拟合较好替代验证任务收益；不同最优epoch也不独立证明差异来自过拟合或优化。

这仍没有回答局部relation究竟是否学好。保留appearance对照，可以区分这种失败是否也出现在同标签的共享外观辅助中；不据此反复调温度或权重。下一步只补齐已计划的同域排序与配对错误，再作完整结论。

## 统一关系诊断入口

[probe_full_endpoint_relations.py](../../scripts/probe_full_endpoint_relations.py)对完成训练且有results.json的run读取原主任务best.pt，不读取或按关系指标另选checkpoint。三个模型统一使用当前实例query；appearance的训练向量w不参与该诊断。批内唯一帧编码及inverse恢复与现有预测路径一致，无额外解码或持久化dense特征。

输出各run的endpoint_relations.csv/json，保存原帧身份、Δ、端点native格、offset类别、exact、NLL、same-cell与有效候选数；按Δ、同/跨格及clip统计，macro对两Δ等权，不按pair总数加权。脚本会核对已完成几何给出的1518=407+1111和1491=141+1350；范围外30/35的旧统计不冒充本脚本新读数。

[一项合成检查](../../tests/test_probe_full_endpoint_relations.py)已通过，抓取批内pair身份恢复错误、两Δ不等数量时宏平均误加权、空分组及手推NLL不符。它只证明CPU聚合逻辑；实际GPU运行及数值另记。示例：

```bash
python scripts/probe_full_endpoint_relations.py --run outputs/full_heatmap/dino_prefix_seed0
python scripts/probe_full_endpoint_relations.py --run outputs/full_heatmap/dino_relation_aux_seed0
```
