# 相同初始化下的DINO前缀冻结与微调对照

状态：2026-09-10锁定。真实float32共同微调前反向smoke在物理batch8下通过，峰值已分配5267.99MiB；共享GPU条件下两臂统一采用batch8，不为扩大batch再增加峰值。沿用[真实因果三帧](tennis-temporal-probe-v1.md)的1,503/230个训练/验证目标，最终game8–10不参与。

## 要区分的解释

已有三帧冻结探针显著优于单帧；显式cost拼接的细定位代价在三个seed和单次hidden64控制下持续存在。本轮不添加motion结构，检验“已有读出继续训练”与“允许预训练视觉表示适应球定位”是否产生不同结果。微调有收益不等于新的motion贡献；没有收益也只限于本次前缀、读出和学习率条件。

采用现代backbone本身是研究前提。先做这个控制，可以继续保持标签、因果窗口和监督密度；[WASB本地强基线](../literature/2026-09-10-wasb-implementation.md)仍需后续建立，但其原生VC3和多输出监督条件不同，不适合作为本轮唯一直接对照。

## 共同起点与模型

两臂都从`outputs/temporal_probe/stack_seed0/best.pt`的hidden32读出和原DINOv3 ConvNeXt-Tiny预训练权重出发。hidden64的无cost stack没有成为更强对照，因此不更换初始化。不是接着原优化器状态运行：两臂都重置AdamW和相同seed。

严格加载完整作者权重后，模型只执行`downsample_layers[0] → stages[0] → downsample_layers[1] → stages[1]`，保持原层、参数及归一化，输出192×36×64。这里是预训练ConvNeXt的前两个stage，不将其参数/计算冒称完整四stage网络；不计算无用的后两个stage，不修改第三方实现。三帧共享此前缀，按原时间顺序拼接576通道，连接原hidden32、72×128输出及appearance absence头。

- `frozen`：前缀参数冻结，只继续训练原读出头。
- `finetune`：前缀与原读出头共同更新。

两臂前缀均保持eval；这个具体构造使用LayerNorm、drop_path=0，eval不会阻止finetune的梯度。每帧预处理仍为RGB、PIL bilinear 512×288、除255及原ImageNet mean/std。没有输入增强、cost、额外监督或轨迹后处理。

## 输入、优化与选优

复用原窗口metadata，将5,199个唯一帧各自解码/resize一次，保存`uint8 [N,3,288,512]`，约2.14GiB，窗口仅保存索引。训练时两臂均在线float32前缀forward；冻结臂不反传前缀，不另生成约8.57GiB的float32特征缓存。微调时绝不使用旧冻结stage1.npy作为更新表示的输入。

固定15epoch、seed0、物理batch8、head lr=3e−4、prefix lr=1e−5、AdamW weight_decay=.01。沿用H×W+1分类目标、包含VC3的位置监督和PCK8/16/32、presence/detection评价。原初始化头曾用batch16；本轮两臂共享smoke后锁定的物理batch，因此和旧运行的步数差异通过frozen继续训练臂控制。不开AMP或梯度累积，不根据验证结果选择batch。

**epoch0必须纳入选优。** 它的完整230目标位置与0.5存在判断须重现保存的stack预测；两臂初始logits一致性先经同一输入smoke检验。若接线、归一化、时间顺序或数值差异改变起点，先修正，不运行正式比较。选优仍按最大验证PCK16、再PCK8、完全相同则保留最早epoch；允许最终选择epoch0，不能在继续训练全程退步时强制挑一个较少退步的checkpoint。

## 实施验证与下一步判断

一次已知输入测试分别按帧执行前缀，与B×T批量执行后重组比较；冻结/微调初始输出应相同，冻结前缀无梯度、微调前缀有非零梯度，head两臂均可训练。真实小批量再核对前缀输出与官方完整网络stage1相同，并测共同微调的前反向峰值；如果不适配设备可用显存，两臂同步降低batch后锁定。

正式输出为`outputs/adaptation_probe/{frozen,finetune}_seed0`，保存初始评价、配置、history、最佳checkpoint、完整train/val逐帧预测、耗时和显存。初始化已在同一验证比赛选优，本轮也是开发诊断，不是独立测试泛化或完整训练基准。

只有finetune相对frozen的PCK16严格提高、PCK8不下降，才对各自对应的stack seed1/2起点补两臂复核；同时保留困难/遮挡与存在判断结果。否则不围绕prefix学习率或宽度继续网格搜索，转入共同协议下的完整强基线。无论结果如何，本轮均不声称获得了新的motion representation。
