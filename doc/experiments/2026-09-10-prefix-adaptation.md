# 更新视觉表示能否超出继续训练读出头的收益？

日期：2026-09-10。状态：输入缓存、时间/梯度单测、真实前缀/显存预检与完整epoch0入口验证完成；正式两臂待启动。
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

## 正式实验待执行

`scripts/train_tennis_adaptation.py`的完整入口以`--epochs 0`完成一次smoke，保存到`outputs/adaptation_probe/smoke_epoch0`。全部230个验证目标的身份顺序一致，位置错位0、0.5存在判断错位0；best_epoch为0，重新加载checkpoint后PCK8/16/32仍为64.84%/83.56%/85.39%，并保存1,503/230行训练/验证预测。该运行总计21.03秒、峰值880.63MiB，只含forward、指标与保存，不含参数更新；不能用它的显存代替此前反向的5267.99MiB。

入口smoke的config记录了当时父提交`43b76d2`，实现本身为其后的未提交修改；不能据这个字段声称父提交已经含有新脚本。下一个实现提交后启动正式运行，使正式config对应实际版本。独立只读审阅未发现阻断问题；语法编译通过，未把不可用的Python LSP计作验证证据。

```bash
python scripts/train_tennis_adaptation.py --rgb-cache data/cache/tennis/rgb_512x288_step8_h2 --init-run outputs/temporal_probe/stack_seed0 --output outputs/adaptation_probe/smoke_epoch0 --backbone-mode finetune --epochs 0
```

之后顺序运行frozen/finetune两臂。新脚本不重复打开原图、原位置CSV或最终测试数据；每个正式运行仍会检查自己的完整epoch0起点，避免初始化或路径变化混入比较。

只有finetune相对frozen的PCK16严格提高且PCK8不下降，才用对应seed1/2起点复核。所有条件结果和epoch0仍完整保留；一个验证比赛的继续训练不能替代跨比赛或跨球种证据。
