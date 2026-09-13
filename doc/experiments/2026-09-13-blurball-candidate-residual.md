# 可学习候选读出：软对应未获增量，共同位置监督需先修订

日期：2026-09-13。状态：三条件各30轮已完成，保存best复算及全量配对完成。
协议：[冻结候选残差读出v1](../protocols/blurball-candidate-residual-v1.md)。

## 结果与决定

**当前固定软对应配方没有通过预定判据。** current和correspondence的验证最佳均为epoch0零残差，保留原cross；stationary的best1略有总体增量，但match19/21下降，不能作为一致收益。

这次不再由阴性立即推断缺少另一种motion模块。共同的异常是：三组训练loss下降，连训练集严格4px定位也明显下降。它排除了“仅在验证出现的泛化失败”作为充分解释，提示先处理软标签目标、受限残差表示及原定位分数之间的关系。还没有证明它们中哪个是唯一原因。

本轮实际增加了一个可训练、保留当前unary的读出对照；不是以前“丢弃外观、只选最高cosine”的弱规则。它仍是冻结候选选择实验，不是端到端motion网络，也不含新位置生成。对应核心已有SELSA等直接先例，见协议中的一手来源与区别。

## 三条件及监督

固定cross best3、K16局部坐标和q、真实`t−2:t→t`、stage0描述符。三个残差MLP均为288→32→1、GELU，末层weight/bias清零，原peak logit差始终作为加法基底。新增参数各9,281。

| 条件 | 新增残差的当前输入 | 新增残差的历史输入 |
|---|---|---|
| current | 当前候选query | 两向量置零 |
| stationary | 同一query | 历史同坐标双线性特征 |
| correspondence | 同一query | query对历史全图cosine/0.1 softmax后的加权unit特征 |

current仅关闭**新增残差**的历史向量，原cross分数和候选仍来自三帧，不能叫整个系统的单帧模型。其名义参数量相同，但历史输入权重没有有效数据梯度；correspondence与stationary是更直接的同输入维度聚合方式对照。

每epoch全部38,854训练目标按相同seed0批序、batch256遍历。实际33,002个训练V1中，31,887个K16含<4px候选，只有这些位置可达样本计算候选soft-target CE；1,115个V1候选漏球和5,852个V0不产生位置loss。三组每轮可监督数均31,887，没有全无监督而跳过的batch。所有14,192验证目标、12,896个V1仍进入完整评价，验证K16可达4px数11,432，没有借GT子集过滤自动预测。

软标签是固定`exp(−distance²/32)`在K16归一化；只用于训练。AdamW lr3e−4、weight_decay .01、30轮，无学习率扫描。选优TP4→raw正确4→最早，epoch0参与；保持q使该TP排序等价于本地F1@4排序。三个best均从实际保存模型复算出相同验证指标。

## 完整验证结果

| 条件 | 实际完成轮数 | 最佳epoch | PCK@4 | PCK@16 | 本地F1@4 | TP4 |
|---|---:|---:|---:|---:|---:|---:|
| 原cross | 来源best3 | — | 82.0797% | 85.8716% | 82.2349% | 9,876 |
| current | 30 | 0 | 82.0797% | 85.8716% | 82.2349% | 9,876 |
| stationary | 30 | 1 | 82.2658% | 87.0425% | 82.2932% | 9,883 |
| correspondence | 30 | 0 | 82.0797% | 85.8716% | 82.2349% | 9,876 |

epoch0是新增残差为零、精确保持原候选的合法起点；不是把实际30轮隐藏或少算。两个条件训练后未获得更优验证选择，按运行前规则保留起点。

stationary相对cross在4px救141、破117，净+24；16px救277、破126，净+151。TP只净+7，不把raw正确但q拒绝的位置混成检测成功。其总体增量如下分布：

| 群体 | V1数 | 4px救 / 破 | 4px净变化 |
|---|---:|---:|---:|
| match18 | 3,987 | 42 / 29 | +13 |
| match19 | 1,616 | 8 / 22 | −14 |
| match20 | 3,221 | 82 / 15 | +67 |
| match21 | 4,072 | 9 / 51 | −42 |
| d1≥16px | 3,675 | 45 / 17 | +28 |
| d2≥16px | 7,986 | 93 / 39 | +54 |

固定match21远错202例救7@4、13@16，而原cross救回的74例又破坏14@4/16。没有给它升级为新的默认定位器。不同组有共享帧，不能相加当独立收益；这也只是seed0开发证据。

correspondence的选定输出与cross逐点相同，所以它既没有相对current增益，也没有超过stationary。不能靠忽略epoch0、另选某个非最优epoch宣称motion贡献。

## 训练loss下降并没有保住训练位置

完整日志中三组均随训练降低soft CE，但验证后期持续弱于起点。为区分“训练拟合改善但泛化差”与“共同训练目标未保住精定位”，固定读取每组**第30轮last**，仅从缓存重算全部训练候选读出。没有重选模型，也没有重新提取特征；验证第30轮直接复用已有日志。

| 固定epoch30 | 训练soft CE | 训练PCK@4 | 训练4px救 / 破 | 验证PCK@4 |
|---|---:|---:|---:|---:|
| 原cross起点 | 2.81320 | 93.9367% | — | 82.0797% |
| current | 2.24250 | 89.4006% | 122 / 1,619 | 76.3337% |
| stationary | 2.14505 | 89.2764% | 138 / 1,676 | 76.4501% |
| correspondence | 2.17043 | 89.3158% | 117 / 1,642 | 77.1557% |

这里soft CE仅在31,887个可监督V1上计算，训练PCK使用全部33,002个V1；每个原本4px正确帧必然属于可监督集合。故破坏不是由未监督漏候选帧单独造成。表中CE为固定last模型重新评价，不是epoch30内不同batch权重更新期间累计的平均loss。

训练可监督集合中，原top1正确31,001、错误886。分组CE如下：

| 原top1组 | 初始CE | current last | stationary last | correspondence last |
|---|---:|---:|---:|---:|
| 已正确4px，n=31,001 | 2.75374 | 2.17540 | 2.08244 | 2.10227 |
| 错误4px但K16有球，n=886 | 4.89399 | 4.59036 | 4.33577 | 4.55534 |

原已正确组贡献初始soft CE总和约95.17%。因此大量loss优化发生在原本已经完成定位的样本，而非仅修补886个可救错误。其原始候选peak logit差与按候选坐标重新生成的软标签不是同一分布，原argmax正确并不保证该soft CE小。

更具体地，可监督目标的软标签平均有**9.7846%概率质量分配给距GT≥4px的候选**，平均标签熵0.44175。优化会要求对这些不满足严格定位的候选分配正质量。与此同时，受限共享残差不能任意独立调节每帧每个候选，降低平均soft CE和保持所有正确argmax可以出现冲突。

这是下一项可区分的监督问题，不是“已经证明软标签导致全部失败”。理想无容量限制的soft标签最优排序仍偏好最近GT的候选；当前结果还混合了特征容量、优化、训练内候选分布及模型限制。需要改变有明确含义的监督目标来检验解释，而不能仅凭loss曲线断言motion表征无信息或一般过拟合。

## 实现、成本与验证

代码提交：`63ec238`。协议及[训练入口](../../scripts/train_blurball_candidate_residual.py)、[候选读出](../../src/ballmotion/candidate_readout.py)、[缓存导出](../../scripts/export_blurball_candidate_features.py)、[完整配对](../../scripts/compare_blurball_candidate_residual.py)均保存。对应工具复用同一坐标和query/key归一化函数；两类池化共用一次前缀。

| 阶段 | 唯一源帧 / 目标 | 实测耗时 | 峰值allocated显存 |
|---|---|---:|---:|
| 训练特征导出 | 39,486 / 38,854 | 130.32s | 479.86MiB |
| 验证特征导出 | 14,332 / 14,192 | 47.26s | 479.86MiB |
| current 30轮及最终预测 | 38,854训练目标/轮 | 14.56s | 971.09MiB |
| stationary 30轮及最终预测 | 同上 | 16.43s | 971.09MiB |
| correspondence 30轮及最终预测 | 同上 | 19.51s | 971.09MiB |

设备为RTX5070Ti Laptop，float32 matmul highest；缓存1,629,998,512字节（约1.63GB十进制），保存query/同址/软对应向量和目标ID。没有重复解码，也没有因三个条件各自提取一份前缀。首验证8目标的历史max cosine差0、匹配格完全一致；唯一生产编码数与合法源帧数相等。

导出计时含前缀、两种汇聚和写盘，不含公共模型构造；训练计时含训练/每轮验证/最终预测，不含缓存加载。它们是缓存实验周转时间，不是原视频端到端吞吐。尤其没有单独测量全局相关相对同址的算子增量，不声称两者等算力。

定向测试共9项通过：旧候选对应与条件汇总4项，新软池化2项，新残差/软标签3项。真实缓存16目标CUDA smoke验证零残差逐分数相等、有限梯度更新，以及保存模型/optimizer后下一步参数精确一致；该smoke的合成分类目标只用于通路验证，不算训练性能证据。三个完整run各epoch0和best均另外完成实际验证。

独立设计审查促成了current条件和可达监督mask；代码审查核对了真实near/far顺序、候选源、全目标遍历、GT仅训练使用、选优方向、best/last及恢复路径，无必要修正。Ruff F、编译和diff检查通过。没有触及用户修改的AGENTS.md。

实际命令与结果位置：

```bash
python -u scripts/export_blurball_candidate_features.py \
  --output outputs/blurball/spatial_interaction/candidate_residual/features
for task_condition in current stationary correspondence
do
  python -u scripts/train_blurball_candidate_residual.py \
    --features outputs/blurball/spatial_interaction/candidate_residual/features \
    --output "outputs/blurball/spatial_interaction/candidate_residual/$task_condition" \
    --condition "$task_condition"
done
python scripts/compare_blurball_candidate_residual.py \
  --root outputs/blurball/spatial_interaction/candidate_residual
python outputs/blurball/spatial_interaction/candidate_residual/analyze_training_fit.py
```

实际使用`/home/zshyc/miniforge3/envs/zshihyc/bin/python`。各run保存配置、epoch0–30日志、best/last、完整train/val预测和结果；共同目录保存`comparison.json`、`training_fit.json`及紧凑last训练选择。一次性训练拟合分析脚本跟随产物保存。旧候选/原始数据不动，当前约1.63GB缓存能够直接服务下一项监督对照，有明确复用价值，保留。

## 后续决策

停止本v1软标签配方，不扫温度、宽度、学习率或添加新motion模块。当前证据不足以支持部署任何新增残差，也不足以推翻对应表征路线。

下一项优先把“修正当前候选选择”写成与其可达目标一致的监督：比较是否应只鼓励概率进入已有<4px候选集合，而不要求给≥4px候选分配正质量，并允许多个都满足定位容差的候选。它应使用同一冻结缓存、三个条件和完整评价，首先检验训练定位是否真正随目标优化改善，然后才解释验证的对应增量。具体损失与对照另锁定协议；这是根据本轮共同失败修订监督问题，不以继续训练旧run或挑验证阈值代替验证。
