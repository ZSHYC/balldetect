# 更新视觉表示能否超出继续训练读出头的收益？

日期：2026-09-10。状态：三个seed的冻结/微调共六组正式实验完成。
协议：[相同初始化下的前缀适配](../protocols/tennis-prefix-adaptation-v1.md)。

## 当前问题与控制

[局部对应实验](2026-09-10-local-cost.md)在hidden32三个seed和hidden64的控制中持续损害严格容差；当前不再增加cost结构。保留更强的hidden32无cost stack作为共同起点，比较冻结视觉前缀只继续训练head与同时更新前缀。这个实验区分训练条件，不是新motion算法或完整benchmark。

两臂从同一个预训练ConvNeXt前两个stage和`outputs/temporal_probe/stack_seed0/best.pt`出发，重置优化器、保持输入与标签相同。原头来自epoch8；把新实验epoch0纳入选优，避免两臂全部退步时还被迫报告一个训练后的checkpoint。学习率、15epoch与判别条件先锁定，不用验证结果选择它们。

## 确定性RGB缓存

`scripts/cache_tennis_rgb.py`复用已有`Images`读取和PIL bilinear RGB处理，按源metadata的唯一frames行序保存uint8，不重新筛选标签或复制时序窗口。输出位于`data/cache/tennis/rgb_512x288_step8_h2`，为5,199×3×288×512，约2.14GiB；原1,733个三帧窗口仅以索引保留。

实际生成耗时20.56秒，包含读取、resize、写入和flush；没有DINO forward。模型训练会复用这些已缩放像素，但在每个batch在线计算当前前缀特征。避免把冻结特征复用到正在更新的网络，也不为单个冻结控制额外保存约8.57GiB的float32特征。

```bash
python scripts/cache_tennis_rgb.py --source-cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output data/cache/tennis/rgb_512x288_step8_h2
```

一项标准库综合测试先以缺少实现失败，再验证4张非方形、不同RGB图片与两个重叠窗口仅产生4帧存储，CHW/uint8/颜色和metadata行序正确。相同来源可在原图已删除的测试条件下命中完成缓存；更换source传入同一output会明确报错，不静默复用错误来源。测试只处理临时图片，没有删除任何真实原图。未增加哈希或缓存框架。

## 前缀实现与实际预检

`BackboneProbe`保留一个实际使用的视觉前缀和原空间头：B×T批量执行共享前缀，再恢复T个完整特征的通道拼接。已知输入测试与显式逐帧调用一致；冻结/微调两臂初始输出相同，head都存在梯度，仅微调前缀得到非零梯度。该测试先因模块缺失失败，实现后通过。

真实预检严格加载原作者Tiny完整权重后，只取原stem、stage0、下采样和stage1。一个真实输入的输出与官方完整网络`get_intermediate_layers(..., n=[1], reshape=True, norm=False)`逐值完全相同，最大差0。真实batch8的冻结/微调初始化logits最大差同样为0。

共同微调的float32前反向使用8个真实三帧训练窗口，即24个输入图像，峰值已分配显存5267.99MiB；实际前缀参数1,235,040、读出20,197。使用真实分类目标的loss为0.47950，首个前缀参数梯度范数92.734，尚未执行optimizer step。这证明连接与梯度可运行，不代表已经获得训练收益。

一次前反向用时0.514秒，含本次初始化/共享资源条件，不能据此宣称完整epoch或端到端视频吞吐。预检保存于`outputs/adaptation_probe/preflight.json`，作用域明确为提交前实现smoke；正式实验另记录实际代码版本。batch8实测可用后，两臂都锁定8，保留共享GPU空间，不将未测试的batch16描述成已经证实可用或不可用。

## 完整入口验证

`scripts/train_tennis_adaptation.py`的完整入口以`--epochs 0`完成一次smoke，保存到`outputs/adaptation_probe/smoke_epoch0`。全部230个验证目标的身份顺序一致，位置错位0、0.5存在判断错位0；best_epoch为0，重新加载checkpoint后PCK8/16/32仍为64.84%/83.56%/85.39%，并保存1,503/230行训练/验证预测。该运行总计21.03秒、峰值880.63MiB，只含forward、指标与保存，不含参数更新；不能用它的显存代替此前反向的5267.99MiB。

入口smoke的config记录了当时父提交`43b76d2`，实现本身为其后的未提交修改；不能据这个字段声称父提交已经含有新脚本。下一个实现提交后启动正式运行，使正式config对应实际版本。独立只读审阅未发现阻断问题；语法编译通过，未把不可用的Python LSP计作验证证据。

```bash
python scripts/train_tennis_adaptation.py --rgb-cache data/cache/tennis/rgb_512x288_step8_h2 --init-run outputs/temporal_probe/stack_seed0 --output outputs/adaptation_probe/smoke_epoch0 --backbone-mode finetune --epochs 0
```

新脚本不重复打开原图、原位置CSV或最终测试数据；每个正式运行检查自己的完整epoch0起点，避免初始化或路径变化混入比较。

只有finetune相对frozen的PCK16严格提高且PCK8不下降，才用对应seed1/2起点复核。所有条件结果和epoch0仍完整保留；一个验证比赛的继续训练不能替代跨比赛或跨球种证据。

## seed0正式结果

两臂均在实现提交`b7b03a8`后运行，环境为Conda `zshihyc`、RTX 5070 Ti Laptop GPU、float32，物理batch8。共享GPU上的实际总耗时包含训练、每epoch验证、保存及最佳checkpoint重新评价，不作为纯模型吞吐。

```bash
python scripts/train_tennis_adaptation.py --rgb-cache data/cache/tennis/rgb_512x288_step8_h2 --init-run outputs/temporal_probe/stack_seed0 --output outputs/adaptation_probe/frozen_seed0 --backbone-mode frozen --epochs 15 --batch-size 8 --head-lr 0.0003 --backbone-lr 0.00001 --seed 0
python scripts/train_tennis_adaptation.py --rgb-cache data/cache/tennis/rgb_512x288_step8_h2 --init-run outputs/temporal_probe/stack_seed0 --output outputs/adaptation_probe/finetune_seed0 --backbone-mode finetune --epochs 15 --batch-size 8 --head-lr 0.0003 --backbone-lr 0.00001 --seed 0
```

| 条件 | 选中epoch | 验证PCK8 / 16 / 32 | 误差中位数 / 均值（原像素） | 总耗时 | 峰值已分配显存 |
|---|---:|---|---|---:|---:|
| frozen | 0 | 64.84% / 83.56% / 85.39% | 6.04 / 55.61 | 248.32秒 | 897.85MiB |
| finetune | 5 | 70.32% / 86.30% / 88.13% | 5.70 / 51.28 | 636.80秒 | 5278.30MiB |

epoch0的230个验证身份、位置和存在判断均重现共同起点。冻结臂最终保留epoch0，不能将继续训练中较高的PCK8单独拼接成一个不存在的最优模型。两臂训练集PCK16分别为96.63%和99.66%，明显高于验证表现；训练拟合提高不等于跨比赛问题得到解决。

逐帧比较保存在`outputs/adaptation_probe/frozen_vs_finetune_seed0.json`：8px救回18个、新错6个，净增12/219，即5.48个百分点；16px为9/3，32px为7/1，均净增6/219，即2.74个百分点。

这些净收益全部来自VC1。VC2的8个困难目标和VC3的4个遮挡位置在两臂的8/16/32px内都没有命中；微调后的两组平均误差反而分别由408.83→419.04和280.44→321.88像素。两臂均将11个无球目标中的10个报为存在，219个有位置目标均报存在。检测F1@16从81.70%增至84.38%，改善来自定位，不能写成无球拒绝或遮挡恢复改善。

完整学习曲线为[PNG](../../outputs/adaptation_probe/seed0_learning.png)与[PDF](../../outputs/adaptation_probe/seed0_learning.pdf)，包含epoch0–15，圆点标记按PCK16、再PCK8选出的同一个checkpoint。表格原始汇总为`outputs/adaptation_probe/seed0_summary.json`。

## seed0时的复核决定

seed0通过预先固定的复核条件，因此按各自对应的旧stack初始化，顺序运行`frozen_seed1 → finetune_seed1 → frozen_seed2 → finetune_seed2`；参数、训练预算与选优规则不变。这是对同一比赛上随机初始化/训练差异的复核，不增加独立比赛数量。

当前只支持一个有限判断：允许这个预训练前缀适应定位，在seed0优于只继续训练读出。它没有隔离单帧外观适配与跨帧对应改善，没有证明新的motion representation，也没有解决困难目标与存在判断。三seed结果完成后再决定这一训练条件是否进入后续控制，不根据单次正结果添加新模块。

## 三seed完成后的结果与判断

所有六组epoch0都重现对应原stack的230个目标身份、位置和0.5存在判断，错位均为0。六组都完成15epoch；没有因不利结果提前停止或更改学习率。运行实际代码版本为6400e06, b7b03a8；版本差异涉及期间的已记录提交，适配训练实现保持相同。

| seed | 条件 | best epoch | PCK8 / 16 / 32 | F1@16 | presence TP / FP / FN | 耗时（秒） |
|---:|---|---:|---|---:|---|---:|
| 0 | frozen | 0 | 64.84% / 83.56% / 85.39% | 81.70% | 219 / 10 / 0 | 248.32 |
| 0 | finetune | 5 | 70.32% / 86.30% / 88.13% | 84.38% | 219 / 10 / 0 | 636.80 |
| 1 | frozen | 3 | 65.30% / 84.02% / 85.84% | 82.70% | 216 / 10 / 3 | 237.93 |
| 1 | finetune | 2 | 67.58% / 84.02% / 86.76% | 82.46% | 211 / 9 / 8 | 644.82 |
| 2 | frozen | 2 | 66.67% / 84.47% / 87.21% | 82.96% | 217 / 10 / 2 | 260.48 |
| 2 | finetune | 2 | 69.41% / 84.02% / 86.30% | 82.25% | 216 / 10 / 3 | 648.53 |

| 三seed均值±样本标准差 | PCK8 | PCK16 | PCK32 | detection F1@16 |
|---|---:|---:|---:|---:|
| frozen | 65.60±0.95% | 84.02±0.46% | 86.15±0.95% | 82.45±0.67% |
| finetune | 69.10±1.40% | 84.78±1.32% | 87.06±0.95% | 83.03±1.17% |

这些标准差只描述同一game7上的seed差异，不是跨比赛置信区间。冻结/微调峰值已分配显存分别约897.85/5278.30MiB；总耗时不与不同阶段共享资源下的旧探针作严格速度比。

逐帧净变化：

| seed | 8px 救回/新错 | 16px 救回/新错 | 32px 救回/新错 |
|---:|---:|---:|---:|
| 0 | 18/6 | 9/3 | 7/1 |
| 1 | 13/8 | 5/5 | 6/4 |
| 2 | 14/8 | 4/5 | 3/5 |

三seed的PCK8均提高，均值增加3.50个百分点；PCK16一升、一平、一降，均值增加0.76个百分点；F1@16均值增加0.58个百分点，但seed1/2下降。不能把均值小幅提高写成三个seed均有效。seed1的finetune比frozen多5个有球漏报，seed2多1个；仅seed1少1个无球误报，其余仍各10个。

困难/遮挡没有稳定恢复：seed1的frozen在16px命中1/8 VC2、1/4 VC3，finetune为1/8、0/4；seed0和seed2均为0/8、0/4。这些是位置条件PCK，不保证相应预测通过presence阈值。

最终判断：前缀可适配性对精细定位有重复支持，但不是全面更强的定位/存在系统，也未证明correspondence改善。结束这个固定条件的适配诊断，不围绕lr、宽度、current或cost继续网格搜索。下一阶段建立[全量因果HRNet竞争系统](../protocols/tennis-full-causal-v1.md)，将数据密度、空间热图与完整训练作为必须面对的系统参照，并保留同全量DINO控制的需求。

完整三seed汇总为outputs/adaptation_probe/multiseed_summary.json，图为[PNG](../../outputs/adaptation_probe/multiseed_accuracy.png)与[PDF](../../outputs/adaptation_probe/multiseed_accuracy.pdf)。12个VC2/3全部输入图为[第一页](../../outputs/adaptation_probe/hard_inputs_page1.png)、[第二页](../../outputs/adaptation_probe/hard_inputs_page2.png)、[第三页](../../outputs/adaptation_probe/hard_inputs_page3.png)，仅从已有RGB缓存制作，显示三帧相同空间裁剪与各自标签，不重新解码或修改原标签。
