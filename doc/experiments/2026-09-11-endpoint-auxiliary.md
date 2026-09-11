# 可见中心端点辅助：关系排序能否转化为自动定位

日期：2026-09-11。状态：协议已锁定，CPU与真实batch检查通过；正式30epoch尚未启动。

执行依据为[端点辅助协议v1](../protocols/tennis-endpoint-auxiliary-v1.md)。比较已完成的无辅助full-history、relation与全局appearance-query辅助。主定位器、三真帧输入和推理输出保持既有定义；辅助只在训练时读已有VC1中心标签。正式结果尚未产生，不将以下smoke写成性能证据。

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

两组均用seed0、30epoch、batch8，主checkpoint只按既定F1@16/F1@8规则选择。计划命令如下，实际启动/完成后更新状态：

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

结束后复用保存预测核对选优、目标身份、PCK/F1，并按固定clip/visibility/位移组比较。三个主任务选优checkpoint另外测同域relation-query exact R@1和NLL：同样的局部候选、同样双端VC1与范围条件，按Δ及跨格组报告。只有relation同时胜过两种对照的预定关系读数和自动PCK@8/F1@16，才支持继续检验结构化融合；指标不一致时按协议保留竞争解释，不扫描更多温度、半径或系数。

最终测试集未使用，没有新增依赖、人工标注或数据解码。
