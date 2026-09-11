# 固定现有表示后，当前帧细节能否纠正自动定位

日期：2026-09-11起。状态：三次native与前两次pooled已完成且通过核对，最后pooled seed2运行中。复核的F1均值条件已被数学下界排除，停止该固定残差配方的扩展；最后一臂仍完成以保留完整成对记录。

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

准备检查通过并提交实现后，两臂先后从7dcabb4按下列命令启动，完整epoch0验证指标均与固定基线相同。日志分别使用输出目录同名`.log`文件。

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

若只有一般残差收益、只有低通收益或没有联合收益，按协议收窄解释并停止本配方，不追加宽度、学习率或平滑尺度扫描。

## 两臂正式结果

两臂均正常结束30epoch。既有保存结果核对入口通过各31条epoch、有限损失、F1选优/残差checkpoint一致、train/val目标身份与标签一致、全部保存指标复算；native选epoch24、pooled选epoch17。结果见[native JSON](../../outputs/full_heatmap/dino_frozen_detail_native_seed0/results.json)与[pooled JSON](../../outputs/full_heatmap/dino_frozen_detail_pooled_seed0/results.json)。

| 指标 | 固定无辅助基线 | native，epoch24 | pooled，epoch17 |
|---|---:|---:|---:|
| 验证PCK@8 | 81.4433% | 82.6460% | 82.0160% |
| 验证PCK@16 | 88.0298% | 88.6025% | 88.6025% |
| 验证PCK@32 | 89.8053% | 90.3780% | 90.4926% |
| 验证F1@16 | 86.2557% | 86.3649% | 86.3406% |
| 检测@16 TP/FP/FN | 1525/265/221 | 1536/275/210 | 1536/276/210 |
| 存在TP/FP/FN | 1693/97/53 | 1709/102/37 | 1710/102/36 |

native训练集PCK@8为99.7789%、PCK@16为99.9575%。本轮固定读出的训练及评价共857.2261秒，峰值354.02MiB；另有共享基线缓存准备成本58.5609秒。它只计算当前浅层和残差，不能与完整模型端到端部署时间直接相比。

pooled训练集PCK@8为99.3198%、PCK@16为99.9405%，训练及评价832.6096秒、峰值354.02MiB。两臂复用缓存，未再次生成原模型logits。不同选优epoch和训练拟合程度不足以把差异唯一归为信息量或优化。

[保存预测配对](../../outputs/full_heatmap/dino_baseline_vs_frozen_detail_native_seed0.json)：@8救回41、破坏20，净增21/1746，即1.2027个百分点；@16救回20、破坏10；@32救回23、破坏13。@8各clip净变依次为−1、0、+1、+9、+8、−2、+2、0、+4，五升两降两平，不能称每个clip改善。@16净变为−2、+1、+1、+3、+2、0、+2、−1、+4。VC1/2/3的@8净增分别为17/2/2。

F1@16只提高0.1093个百分点：正确输出增11，同时错位输出从168增到173，VC0误报从97增到102，正确位置被拒绝从12减到11。PCK增加是真实位置变化，但存在判断也变了，不能只报救回不报额外误报。

[固定visibility上下文](../../outputs/full_heatmap/dino_baseline_vs_frozen_detail_native_visibility.json)中，当前VC2且历史含VC1的64例，@8由29到30、@16由34到35，均只救回1例而未破坏。VC0且历史含VC1的25例误报22→23，历史无VC1的92例误报75→79。新分支只读当前特征，以上小群体增量不构成新的历史融合或对应证据。

### pooled及两臂之间的配对

[基线到pooled](../../outputs/full_heatmap/dino_baseline_vs_frozen_detail_pooled_seed0.json)：@8救回33、破坏23，@16救回19、破坏9，@32救回22、破坏10。其[固定VC2且历史含VC1的64例](../../outputs/full_heatmap/dino_baseline_vs_frozen_detail_pooled_visibility.json)，@8由29到30、@16由34到36；VC0的两类历史组误报同native，分别为23/25与79/92。一般残差读出也能改善严格定位，不能将native相对原基线的全部增量归给较高空间带宽。

[pooled到native](../../outputs/full_heatmap/dino_frozen_detail_pooled_vs_native_seed0.json)：@8救回35、破坏24，净增11/1746，即0.6300个百分点；@16救回16、破坏16，净变0；@32救回17、破坏19，净减2。@8各clip净变为−1、−3、+1、+10、+5、−3、0、0、+2，四升三降两平；Clip4和Clip5贡献较多，不能称跨clip一致。VC1/2/3的@8净变分别为+9/0/+2，@16为+2/−2/0。

两臂PCK@16、检测TP=1536和FN=210完全相同，VC0误报也同为102。native的F1@16只高0.02427个百分点，来自净少1个可定位帧的错位输出（173对174），不是更多@16正确定位；这里描述的是汇总净差，不声称两组只在同一个样本上发生变化。

## 当前研究判断

native的PCK@8/F1@16数值均严格高于固定基线和pooled，满足原先锁定的方向门控；不事后改写通过条件。但**方向通过不等于稳定的检测优势**：相对低通的F1仅由净1个错位输出决定，PCK@16持平、PCK@32略差，多个clip方向相反。只有有限的证据支持当前stage0较高空间带宽对严格位置读出有增量，尚无motion或泛化结论。

30次额外开发集选优、固定基线先前已被同game选择及单seed限制继续适用。下一步优先固定同一基线表示，检验残差初始化与训练顺序的随机性，而不是立即联合微调前缀或增加motion模块。该复核若执行，应事先锁定seed范围及结束判断，不能追加seed直到获得正结果；其证据仍只覆盖固定表示上的读出稳定性。

## 预定的读出随机性复核

独立只读审查同意这一优先级，并确认必须称为初始化与batch顺序合并的读出优化随机性。已在[协议追加条款](../protocols/tennis-frozen-detail-readout-v1.md#seed0完成后的有限随机性复核)锁定：只增加seed1/2的两臂，共四次30epoch；源backbone和主头继续固定原model seed0 epoch7，复用同一缓存。seed0是探索结果，新增两seed是事前限定复核，不能把同一game7或同一原模型当成独立三份数据。

全部三seed结果与同seed配对均保留。native的PCK@8/F1@16描述均值须均严格胜固定基线与pooled均值，同时至少两seed各自通过原四项严格方向，才进入一项后续系统干预；不通过则结束本固定残差配方，不继续追加seed。该判断只控制下一步研究投入，不声称统计显著或跨比赛稳定。

复用上方正式命令，将`--seed`分别改为1、2，输出目录分别为`dino_frozen_detail_native_seed1`、`dino_frozen_detail_pooled_seed1`、`dino_frozen_detail_native_seed2`、`dino_frozen_detail_pooled_seed2`。运行顺序也按这个次序，GPU串行；实现和原30epoch训练条件没有改变。

四次复核已从084e4bb按上述顺序串行启动，首个native seed1的epoch0完整验证与基线相同；其余运行依次接续。未完成的run不填结果或提前用于三seed均值。

### 首个复核结果：native seed1未复现联合增量

native seed1完成30epoch并通过31条记录、有限loss、选优/checkpoint、目标身份/标签及保存指标复算。最终选epoch0，train与val的完整指标均等于固定原基线；[位置配对](../../outputs/full_heatmap/dino_baseline_vs_frozen_detail_native_seed1.json)在@8/16/32均为救回0、破坏0。这是按预定选择规则得到的未复现，不改成按PCK挑选其他epoch，也不删除这一seed。

本次训练及评价909.1248秒、峰值354.02MiB；[结果](../../outputs/full_heatmap/dino_frozen_detail_native_seed1/results.json)与原缓存准备成本分开。pooled seed1已接续运行，seed2两臂仍按预定顺序执行；完整复核判断仍未形成。

### seed1配对完成：本次低通读出取得更高联合指标

pooled seed1随后完成30epoch并通过同样的保存结果核对，选epoch25；[结果](../../outputs/full_heatmap/dino_frozen_detail_pooled_seed1/results.json)为PCK@8=81.6151%、PCK@16=87.8580%、PCK@32=89.5762%、F1@16=86.4880%。检测TP/FP/FN为1517/245/229，presence为1671/91/75。耗时1,088.7778秒、峰值354.02MiB。

相对固定基线（也即native seed1），pooled的@8仅净多3个命中，却在@16净少3、@32净少4；[配对](../../outputs/full_heatmap/dino_baseline_vs_frozen_detail_pooled_seed1.json)分别为37救回/34破坏、19/22、20/24。检测@16少8个TP，同时少20个FP，F1提高0.23237个百分点；VC0误报97→91。这是位置与拒绝共同变化后的结果，不能把F1提高说成更准确的@16位置。

以[pooled到native的同seed方向](../../outputs/full_heatmap/dino_frozen_detail_pooled_vs_native_seed1.json)报告，@8净变−3，clip净变依次为+1、−3、0、0、0、−1、+5、0、−5；两升三降四平。native seed1未通过原四项联合方向，与seed0结果不同。native seed2已经接续，pooled seed2随后运行；尚不计算完整三seed结论。

### native seed2完成及复核门槛的确定结果

native seed2完成30epoch并通过同样的选优、目标/标签与保存指标核对，选epoch27。[结果](../../outputs/full_heatmap/dino_frozen_detail_native_seed2/results.json)为PCK@8=82.3597%、PCK@16=88.4307%、PCK@32=90.6071%、F1@16=86.2690%；检测TP/FP/FN=1533/275/213，presence=1706/102/40。训练及评价1,149.5166秒、峰值354.02MiB。

[相对基线配对](../../outputs/full_heatmap/dino_baseline_vs_frozen_detail_native_seed2.json)在@8救回42、破坏26，@16为20/13，@32为28/14；@8各clip净变为+1、−1、+1、+5、+3、−1、+3、0、+5。位置增量为正，但F1只超过基线0.01334个百分点，不能把较多@8救回当作较大的检测收益。

此时最后的pooled seed2仍运行中，**实际pooled三seed均值尚未知**。不过选优包含epoch0，所以它最终F1至少为固定基线0.862556561086。已完成的三native平均F1为0.862965210155，而pooled最终平均F1的下界为：

`(0.863406408094 + 0.864880273660 + 0.862556561086) / 3 = 0.863614414280`。

native均值已经低于这个下界，因此预定“平均F1严格胜pooled”的必要条件不可能通过。这是选优规则与已完成结果给出的界，不是把未完成run填成epoch0后冒充实测均值。后续pooled seed2的改善只会提高该下界对应的最终值，不会使native反超。

据此停止本固定残差配方的系统扩展，不追加seed、联合微调细节分支或调头/学习率挽救。最后pooled seed2仍完成原30epoch，以补齐约定的成对位置与检测证据；其实际结果和完整描述均值随后记录。否定的是本配方的预定联合收益，不能扩写为浅层信息完全无用或高分辨率路线一般无效。
