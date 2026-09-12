# BIRD 与 STSN：任务监督的可变形时序对齐证据

**目的。** 本文只为下一次 motion-mechanism 决策补足两个近邻的机制边界；它不批准新模块，也不替代当前 BlurBall 三帧因果中点基线。结论应读作“可比较的对照是什么”，而非“已有机制已经适合高速 RGB 小球”。

**本次阅读范围与版本（2026-09-12）。**

| 方法 | 一手全文 | 版本/状态 | 作者代码的可确认状态 | 本次证据等级 |
|---|---|---|---|---|
| BIRD, *Bidirectional Temporal Information Propagation for Moving Infrared Small Target Detection* | [arXiv HTML 全文](https://arxiv.org/html/2508.15415)（方法、损失、实现、消融均读） | arXiv:2508.15415v1，2025-08-21；未见正式会议版本 | 论文页未链接代码；检得同名 [BIRD 仓库](https://github.com/xiaomingxige/BIRD)，README 自称 PyTorch implementation，但 [nets/Network.py](https://github.com/xiaomingxige/BIRD/blob/main/nets/Network.py) 仅留下“paper accept 后公开细节”的注释。仓库归属无法由论文页确认，且无论归属都不能作为可复现模型源码使用。 | 论文 A；代码为“归属未确认且发布不完整” |
| STSN, *Object Detection in Video with Spatiotemporal Sampling Networks* | [arXiv HTML 全文](https://arxiv.org/html/1803.05549) 与 [ECCV 2018 正式 PDF](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Gedas_Bertasius_Object_Detection_in_ECCV_2018_paper.pdf)（方法、实现、实验/消融均读） | arXiv v2，2018-07-24；ECCV 2018 | 论文页未链接代码；以题名和作者作 GitHub 检索未找到可确认作者实现。第三方复现不计为作者代码证据。 | 论文 A；作者源码“未确认” |

两份概要已同步链接本次补读：STSN 当前阅读深度为论文全文，作者代码未确认；旧条目的“官方摘要”仅是当时的读取范围。

## 可证实事实

### BIRD：三帧局部 DCN 嵌在双向、非因果的 clip 传播中

1. **时间范围、状态与边界。** BIRD 以一个连续 clip 为单位，同时输出该 clip 各帧的检测结果；特征先逐帧提取，反向支路令 `F_i^B` 依赖 `F_{i+1}^B`，因而带入未来；正向支路又使用反向特征及 `F_{i-1}^F`。完整模型对任意中间帧均使用未来信息，作者也明确将其称为有 future information、会产生实时等待延迟的方案。[方法 §III-A、§III-B](https://arxiv.org/html/2508.15415#S3)

   状态只在**处理 clip 内**递推：反向初始特征 `F_{t+N+1}^B` 与正向初始特征 `F_{t-1}^F` 设为零；clip 尾不足长度时复制末帧，clip 两端缺邻帧时复制边界特征。[§III-A/III-B](https://arxiv.org/html/2508.15415#S3.SS1) 论文没有跨 clip 传递状态的规则。因此它不是可直接搬到 rally 边界的持续 tracker；“复制边界帧”也是原论文的计算补齐，而不是本项目可默认采用的时间语义。

2. **offset 从何而来、能到多远。** 每个 LTMF 先把中心帧与其 `i-1,i+1` 的提取特征拼接，经 1×1 bottleneck 得 `F_i^c`；三层 AGRD 和 3×3 卷积从这个**三帧混合特征**预测 DCN offset `ΔP_i` 与 modulation `ΔM_i`，再对同一个混合特征做 modulated DCN。[§III-B1，式 (9)--(11)](https://arxiv.org/html/2508.15415#S3.SS2)

   因此 `ΔP` 不是“当前像素对某一历史帧”的显式一对一对应，也没有 cost volume、候选位移表或 no-match 变量。基本 3×3 格点是局部的，论文未给 offset 的数值上界、实际像素覆盖或高速目标的可达范围；可扩展到远时刻的来源是 GTMF 的递推特征，而非一次高分辨率全局匹配。DCN 可采样分数位置，但这不等价于已验证物理位移。

3. **监督实际约束。** 检测损失是每帧 YOLOX 的框回归、类别和 objectness；加上的 STF 是 `L1(F_i^E,F_i^b)`（正向为对应的反向特征与局部融合特征），总损失在 clip 内各帧求和。[§III-C，式 (15)--(18)](https://arxiv.org/html/2508.15415#S3.SS3) 没有 flow、位移、对应、offset、轨迹或 GT 初始化监督。STF 约束局部融合特征接近提取特征，作者动机是减少 DCN offset overflow；它不监督“采样点等于球在另一帧的中心”。

4. **实验与成本边界。** 原任务是红外小目标**框检测**（DAUB/IRDST），不是 RGB 单球中心定位。训练 5 帧、推理 8 帧、输入 544×544；表中 BIRD 为 9.39 M 参数、98.36 G FLOPs、4090D 上 55.08 FPS。[§IV-B、§IV-C3](https://arxiv.org/html/2508.15415#S4.SS2) 其 baseline 是“5 帧输入、只输出中间帧、三帧 3×3 卷积融合”；LTMF/STF/GTMF 均在此 IR 框检测协议中消融。[§IV-D](https://arxiv.org/html/2508.15415#S4.SS4) 这些数字不能转换为 DINO、288×512、三帧末帧中点的成本或精度承诺。

   可靠性方面，论文承认长 clip 大运动下递推会误差积累并可能造成过大框或额外 false detection，RDCA channel attention 仅用于缓解；没有可校准 correspondence confidence、拒绝匹配或多假设输出。[§III-B2](https://arxiv.org/html/2508.15415#S3.SS2)

### STSN：每个 reference--support pair 的直接重采样，不保存传播状态

1. **时间范围、未来与边界。** STSN 对每个 reference frame 独立构造 supporting frames；原方法形式上用 `K` 个过去和 `K` 个未来支持帧。训练取 `T=3,K=1`：reference 前、后各随机取一张 support；论文随后说第二阶段在某个邻域内随机抽样，所读段落没有给出固定的相邻间隔。推理用 `T=27,K=13`，即前后各 13 帧。推理时把逐帧 backbone 特征放进内存缓存，随后对该 reference 的所有支持帧采样；这只是计算缓存，不是跨 reference 的递归 latent state。[§4、§4.1](https://arxiv.org/html/1803.05549#S4)

   完整原协议是离线的，因为使用未来帧；视频首尾以首/末帧复制各 `K` 次补齐。[§4.1 Inference](https://arxiv.org/html/1803.05549#S4.SS1) 没有“传播状态重置”问题，但同样不能把复制帧当作本项目 rally 边界处真实运动证据。

2. **offset、时程与匹配形式。** 对每个 `(reference t, support t+k)`，先将 `f_t` 与 `f_{t+k}` 拼接，首层 offset 由这个 pair 特征预测；随后四个 3×3 deformable-convolution stages 中，中间采样特征继续产生下一层 offset，最终利用该 offset 从**原始 support feature** `f_{t+k}` 重采样出 `g_{t,t+k}`。各 support 的结果再按逐像素 cosine-similarity softmax 权重汇聚，reference 自身也是一个 support。[§4，尤其 Spatiotemporal Feature Sampling / Feature Aggregation](https://arxiv.org/html/1803.05549#S4)

   所以 STSN 的 temporal reach 是直接选择的 `k`，并非递推累积；其局部空间基础是 3×3 DCN，论文也未报告 offset 数值上界、以原图像素计的最大位移覆盖，或 high-resolution global all-to-all correspondence volume。采样位置是受检测目标驱动的隐式对齐；汇聚 softmax 是 support 相对权重，不能解释为“该 pair 物理匹配的校准置信度”或 no-match 决策。

3. **监督实际约束。** STSN 先用 ImageNet DET 的重合30类框标签预训练完整模型，将同一静态图片重复作为 support；再用 ImageNet VID 的框和类别微调下游 RPN/R-FCN 检测。采样位置由检测任务端到端优化，没有 optical-flow data/supervision，也没有 motion GT、trajectory GT、offset GT 或 GT 初始化；“无光流监督”不等于没有外部检测预训练。[§4.1](https://arxiv.org/html/1803.05549#S4.SS1) 这些事实只足以推出任务监督能让 offset 有检测效用，不能推出每个 offset 是真实对应或在 tiny ball 上可靠。其物体附近采样可视化属于定性证据，不是对应误差或拒绝率评估。[§5.3](https://arxiv.org/html/1803.05549#S5.SS3)

4. **实验与计算边界。** 证据来自 25--30 FPS、30 类、框标注的 ImageNet VID；主指标 mAP@0.5。论文给出帧数/stride 消融（更多支持帧至 27 后趋于平台，`k=2` 稍优而 `k=4` 下降），但未报告可直接换算到当前 DINO 三帧中点任务的 FLOPs、吞吐或小目标大位移分桶结果。[§5--5.2](https://arxiv.org/html/1803.05549#S5) 因此不得以 STSN 的“无光流”表述为“低成本”或“天然适合 tiny ball”。

## 推论：同一三帧体育球定位时，什么才是最小可比机制

| 原论文 | 可作为未来对照的最小机制 | 与当前 `[t-2,t-1,t] → t` 因果中点协议的关系 | 不能继承的结论 |
|---|---|---|---|
| BIRD | 可在相同 `[t-2,t-1,t]` 上研究 LTMF 式“先混合三个槽位，再预测 offset/modulation 并重采样混合特征”。这是一个因果机制改写，不能称为完整 BIRD 复现。 | 改写同时改变原中心帧与两侧邻帧的角色，移除双向传播，并须明确是否保留 STF。若要保留原中心三帧语义，则需使用未来帧并另行规定离线协议；当前不切换协议。 | 相比简单拼接增加的收益尚混有采样容量与优化差异；原模型的远程传播、JOS 速度、STF 稳定性和 IR 检测结果不能直接继承。 |
| STSN | 对 target `t`，以 `f_t` 为 reference，分别用 `f_{t-1}`、`f_{t-2}` 作两个 support，做 pair-conditioned DCN 后加上 reference branch 的加权汇聚。这保留“reference--support 直接隐式对齐”的核。 | 这限制了原方法原本的 support set，并移除了未来帧，故是**因果改写的机制对照**，不是作者的最佳报告协议；它不需传播状态，clip/rally 边界只需禁止取跨界 support。 | 不能称为 STSN 复现、不能借用其 27-frame mAP，也不能把 support 权重叫作可靠性。 |

两个机制都没有显式 global correspondence volume，也没有被论文验证为 RGB 级别几像素自动球发现。两者均无额外 motion GT 或 GT initialization；但 BIRD 仍有用于稳定特征融合的 STF 辅助目标，不能概括为“只有检测 loss”。它们是“以最终任务监督驱动的 DCN 采样”先例，足以排除“只要加 DCN 就是新的 motion representation”这种论述；尚不足以证明它们无法处理 tiny ball，原因只是论文没有测该任务。

这里还需分清**采样点数和搜索范围**：3×3只是基础采样格，不能据此断言只能看邻近3×3区域；反过来，offset未被显式限制也不保证学会大位移。多个DCN采样点经加权融合，并不自动代表多个互斥的球位移假设。若最终研究要声称对应或多假设，必须分别说明位置所指的帧、保留的候选语义及其任务收益。

## 未来实验建议（非当前批准实现）

先完成[BlurBall真实历史输入控制](../protocols/blurball-full-temporal-control-v1.md)。只有实际获益与残差支持进一步区分时间证据的空间使用，才选择一个机制干预；本次文献补读不能提前替实验选定DCN。

若随后选择采样路线，沿用共同DINO特征、三帧末帧目标、读出和训练规则。仅把新DCN块与简单拼接比较，最多测得系统增量，不能隔离“对齐”的贡献；需要在具体实现中选择能区分采样位置作用和额外容量的一项控制，而非先看到阳性再补解释。原STSN的四个1024通道DCN层及权重子网也不能被“无光流”三个字省略出计算预算。[原实现设置](https://arxiv.org/html/1803.05549#S4.SS1)

受限对照失败就停止该配方；完整BIRD、更多帧、STF或未来信息各自改变不同条件，不能作为未运行的补救证据。成功也仍需保留自动定位、误检和来源分组，不能仅以漂亮offset或GT查询下的对应指标替代任务收益。

## 未解决项

- 检得的同名 BIRD 仓库不含可审计的网络定义，且论文页未确认其归属；故未从源码确认张量实现、真实 clip loader 或论文复杂度测法。本文所有 BIRD 算法事实均以论文为准。
- 未找到 STSN 作者代码，故没有源码级的边界处理、offset 初始化或精确检测 loss 配置锚点；本文只断言论文明确写出的机制与协议。
- 两篇都未测 BlurBall 的中点标签、visibility、拖影轴、RGB 运动模糊下的像素误差或相机连续运动；这些仍只能由本项目的预注册式对照回答。
