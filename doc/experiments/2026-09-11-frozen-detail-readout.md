# 固定现有表示后，当前帧细节能否纠正自动定位

日期：2026-09-11起。状态：协议已锁定，CPU检查、固定基线logits复现与真实batch更新检查通过；两臂正式训练尚未启动。

执行[固定细节读出协议v1](../protocols/tennis-frozen-detail-readout-v1.md)。[端点辅助三臂](2026-09-11-endpoint-auxiliary.md)已经停止；本轮单独检验现有无辅助模型的当前stage0空间增量，不把它当作已证实的失败根因或新的motion贡献。

## 设计取舍

独立架构审查支持保留已选epoch7基线，冻结全部前缀与原主头，只训练3,392参数的当前位置残差。原分辨率native与归一化后2×2池化再bilinear恢复的pooled，共用初始化、输出网格、训练样本与主CE。若共同微调前缀，浅层梯度会同时改变后续stage1，就不能回答已有表示的可读性；因此本轮不采用该更宽的问题。

末层无bias且权重初始化为零，确保epoch0等于原基线。无bias并不固定presence：位置残差仍会改变联合softmax，必须并看PCK与F1。stage0的上下文、GN统计、平滑与相位差异也限制解释；native胜出仅支持这份高空间带宽对本读出有增量，不证明仅来自球像素或已经恢复motion。

源模型为`outputs/full_heatmap/dino_prefix_seed0/best.pt`，原选epoch7。目标仍为12,167训练/1,863验证，三帧历史由固定原主logits携带，新分支只看当前t。固定logits缓存约517MB，顺序对应合法目标；它与原RGB缓存共享两臂，不建立约50GB浅层特征副本。每batch在线计算当前帧冻结stage0，因此实际训练耗时不可称部署耗时。

## 实现与有限检查

入口为[train_tennis_detail_readout.py](../../scripts/train_tennis_detail_readout.py)，复用既有模型构建、目标映射、评价和预测CSV写入；原全量训练脚本与模型接口不修改。checkpoint只存新残差及epoch，配置引用固定源模型，避免每轮另存冻结权重。

[一个CPU检查](../../tests/test_frozen_detail_readout.py)先在入口尚不存在时失败，实现后通过：当前槽位与归一化正确；两个分支零起点严格等价；共同GN后才低通；absence标量不改；固定前缀无梯度，残差末层首步有梯度，前层次步开始有梯度。实际命令：

```bash
conda activate zshihyc
python tests/test_frozen_detail_readout.py
```

固定缓存已生成，12,167训练/1,863验证目标的身份顺序一致；全部argmax位置变化0，presence概率最大差0。数组共517,258,040字节，生成及复现共58.5609秒、峰值501.86MiB；不含模型/RGB元信息加载。来源和逐目标身份见[缓存元信息](../../data/cache/tennis/dino_prefix_seed0_epoch7_logits/metadata.json)。两臂复用这一确定性基线结果，不重复创建。

随后固定训练顺序的game1/Clip1/0002–0009八目标分别完成一次真实AdamW更新。起始CE均为0.08709055，主logits与缓存严格相等；末层梯度L2 native为0.30519745、pooled为0.04631176，两臂均得到有限更新，全部原模型参数/buffer逐张量不变且无梯度。末层梯度不同是平滑改变优化输入的实际表现，不额外校准学习率来改变锁定对照；阳性不能排除优化条件参与。检查共0.4022秒、峰值335.05MiB，与缓存生成分开计时。[原始检查JSON](../../outputs/full_heatmap/frozen_detail_readout_check.json)保存条件与结果，执行于6e6a3ee加本次未提交实现。

最终测试数据未读取，没有新增依赖或原始解码。

独立只读架构复核通过：冻结、当前槽位、GN/低通次序、零末层、原logits恢复与残差独立优化均符合协议。没有发现需要改变正式训练的实现或解释问题；该复核未重复运行GPU检查。

## 正式命令与事前判断

准备检查通过并提交实现后串行运行：

```bash
python scripts/train_tennis_detail_readout.py \
  --detail native --logit-cache data/cache/tennis/dino_prefix_seed0_epoch7_logits \
  --output outputs/full_heatmap/dino_frozen_detail_native_seed0 \
  --epochs 30 --batch-size 8 --seed 0

python scripts/train_tennis_detail_readout.py \
  --detail pooled --logit-cache data/cache/tennis/dino_prefix_seed0_epoch7_logits \
  --output outputs/full_heatmap/dino_frozen_detail_pooled_seed0 \
  --epochs 30 --batch-size 8 --seed 0
```

只有native在选优checkpoint同时严格超过固定基线与pooled的PCK@8/F1@16，才支持保留该分支；逐clip和配对反向结果仍要报告。epoch0参与选优使F1不降本身没有证据价值，不能用它宣布成功。两个分支也没有消除重复使用同一开发比赛选优的乐观偏差。

若只有一般残差收益、只有低通收益或没有联合收益，按协议收窄解释并停止本配方，不追加宽度、学习率或平滑尺度扫描。正式结果尚未产生。
