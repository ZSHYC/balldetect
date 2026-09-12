# 从因果点跟踪重新审查球定位：TAPNext、TAPNext++ 与 Track-On2

日期：2026-09-12。本文补读的是三篇方法正文、实验与相关附录，不是仅据摘要列名。结论用于约束本项目的机制与比较对象；没有运行这些 tracker，也没有把它们的结果换算为球定位成绩。

## 最值得改变的判断

**Track-On2 已经占据了“DINOv3 + 多尺度细特征 + 全局候选分类 + top-k 重排 + 亚网格细化 + 因果记忆”的主要设计空间。TAPNext 系列则证明，显式 cost volume 不是一般点跟踪能力的必要架构。** 两者都需要给定查询点，因此不能直接替代本项目的自动逐帧球定位；但“任务不同”也不能让我们忽略它们对算子新颖性的限制。

另一个更深的反例来自训练：TAPNext++ 在原架构上改变长序列训练与数据增强，改善持续跟踪和重出现能力；Track-On2 也把训练时长、记忆容量与骨干更新分开消融。看到模型失败时，不能直接跳到“缺一个 motion module”。这些证据来自各自的任务和训练条件，仍不是本地失败原因的证明。

## 来源、版本和实际阅读范围

| 方法 | 版本与发表入口 | 本次阅读 | 公开资源边界 |
|---|---|---|---|
| Artem Zholus et al., **TAPNext: Tracking Any Point (TAP) as Next Token Prediction** | [arXiv:2504.05579v2，2025-04-14](https://arxiv.org/abs/2504.05579v2)；[ICCV 2025，9693–9703](https://openaccess.thecvf.com/content/ICCV2025/html/Zholus_TAPNext_Tracking_Any_Point_TAP_as_Next_Token_Prediction_ICCV_2025_paper.html) | arXiv 方法、训练、实验与附录 C–G；官方 PyTorch 坐标头、certainty 与公开 loss 路径 | 正式版的附录编号与 arXiv 不同，本文章节号指 arXiv；未声称逐页比对了两版。代码读取日期为本日，未运行 |
| Sebastian Jung et al., **TAPNext++: What's Next for Tracking Any Point (TAP)?** | [arXiv:2604.10582v1，2026-04-12](https://arxiv.org/abs/2604.10582v1)；[CVPR Findings 2026，8429–8438](https://openaccess.thecvf.com/content/CVPR2026F/html/Jung_TAPNext_Whats_Next_for_Tracking_Any_Point_TAP_CVPRF_2026_paper.html) | 正文及附录 A–D；[官方专页 README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapnextpp/README.md) | Findings 与主会分开称呼。公开 checkpoint 的训练组合不等于主表最终模型，见下文 |
| Görkay Aydemir, Weidi Xie, Fatma Güney, **Track-On2: Enhancing Online Point Tracking with Memory** | [arXiv:2509.19115v2，2026-03-16](https://arxiv.org/abs/2509.19115v2)；记录关联 [TPAMI DOI](https://doi.org/10.1109/TPAMI.2026.3675257) | v2 完整方法、主要实验、消融与失败分析；[作者项目页](https://kuis-ai.github.io/track_on2/) | 本文没有完成其训练源码审计；不能以“读了论文”声称本地可直接复现 |

## TAPNext：没有显式 cost volume，仍有空间关系建模

### 论文与公开代码事实

TAPNext 把图像 patch tokens 与给定点的 query/mask tokens 联合处理：每层的时间 SSM 沿 token 序列递推，空间 ViT 则在当帧图像与点 tokens 间做全注意力。输出不作为下一帧的显式坐标输入，但隐状态持续传播。主模型 patch 为 8×8；x、y 分别预测 256-bin 分布，并以截断局部期望得到连续坐标。坐标分类、Huber 回归及 visibility BCE 在训练中有不同职责。[方法与附录 D](https://arxiv.org/html/2504.05579v2#S3)

小规模消融中，patch 从 8 改为 16，DAVIS AJ 从 55.0 降至 49.7；改成纯回归坐标头降至 44.7。这是其配方内的消融，不是“所有 ViT/16 都不适合球”的证据。主模型使用大规模合成轨迹数据，包含摇摄与运动模糊；Boots 版另有真实视频自训练。[表 3、附录 E/G](https://arxiv.org/html/2504.05579v2#S4)

公开[坐标头](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapnext/tapnext_torch.py)确实分别截取 x/y 峰值附近 20 bins 再求期望；[certainty 函数](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapnext/tapnext_torch_utils.py)由两个边缘分布的乘积计算预测点邻域质量。不能把这种输出浓度自动解释为可校准的 no-match 概率。

### 本项目的推论

**第一，算子的名字不是 motion 存在性的判据。** 空间注意力可让历史状态与当前远处图像位置交互；显式 `(u, delta)` cost volume 只是实现关系的一种方式。反过来，某个注意力头看起来像 cost volume，也只是可视化证据，不能证明最终坐标主要由该头决定。

**第二，“多峰位置分布”与“多条成对位移假设”不同。** 假设正确候选是 `(a,b)` 或 `(c,d)`，两个独立边缘分布的乘积还可能给 `(a,d)`、`(c,b)` 分配质量。这个数学例子说明因子化输出不能一般地保存二维配对结构；它不是本地已经观察到的 TAPNext 失败，也不证明我们的联合二维热图必胜。若未来贡献涉及多假设，必须说清保留的是位置模式、对应关系，还是跨时间路径。

**第三，局部期望早已有先例。** 本地 [BlurBall 固定重心读出](../experiments/2026-09-12-blurball-local-readout.md)的价值是发现现成 logits 中仍可读出更准位置，不能称新的 motion 表示。TAPNext 的边缘分布、有回归辅助训练和本地二维分类头并不相同，所以它的窗口与温度也不是现成超参答案。

**第四，处理一次帧不等于计算量对空间线性。** 令每帧图像 tokens 数为 P，点数为 Q；TAPNext 的全空间注意力至少包含 `(P+Q)²` 成对项。时间 SSM 的线性递推没有消掉这个空间成本。不能只摘“无 cost volume”“常量时间状态”便承诺高分辨率轻量部署。

## TAPNext++：架构失败与训练分布失败需要分开

### 论文与发布事实

TAPNext++ 沿用 TAPNext，增加 1024 帧训练、周期平移/旋转的 roll 增强，以及有合成真值的帧内遮挡点坐标监督（权重 0.2）。它用重复静态帧揭示旧模型的长时漂移，并以重出现后的 AJ 指标补充整轨平均；这些实验主要检验状态持续性与重捕获。[方法与实验](https://arxiv.org/html/2604.10582v1#S3)

其结果并非所有数据集都同步改善；附录不含 PointOdyssey 训练的变体也保留收益，却弱于包含该训练集的版本。作者分析没有发现 LRU 特征值整体明显向 1 移动，具体稳定机制仍是解释而非定论。[附录 A/D](https://arxiv.org/html/2604.10582v1#A1)

公开 checkpoint 使用 PointOdyssey 与 Kubric-1024，**没有 DynHumans**；因此不应直接冠以主表三数据混合模型的全部分数。[官方说明](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapnextpp/README.md) 论文的 562 FPS 是 H100 上 256 queries、32 帧分块设置；逐帧版本约 193 FPS。训练用 8 张 H100 做序列并行，不是本机轻量训练预算。[表 2、实验设置](https://arxiv.org/html/2604.10582v1#S4)

### 本项目的推论

原架构经训练改变就能改善，足以否定“旧模型失败，故原函数类必然做不到”的强推断；但不能反向证明所有失败都可仅靠数据修复。论文也不是保持数据、监督与训练预算完全不变的单变量比较。

本项目当前三帧头每个窗口独立，无跨窗口隐状态。故该论文的千帧记忆问题**不是当前直接瓶颈**，不应为了阅读它就新增 SSM 或长记忆。它给我们的可复用研究方法是：把状态稳定性、可见证据、重捕获与输出读出分开。只有以后真正引入跨帧状态，才需要相应的持续性诊断。

roll 把同一对象移出一边再移入另一边，适合训练重捕获，但它改变了场景运动分布。把现有体育视频直接 wrap，不等于模拟真实连续相机平移，更不等于球曝光内 blur。当前没有新增这种增强的依据。

遮挡点监督更不能直接照搬。模拟器可以给不可见物体表面点位置，公开球数据的某些 V0/unknown 没有可回归中心。让模型预测此位置属于带额外真值的状态估计；把历史插值填进去再称“原始视觉标注”会改变研究任务。

## Track-On2：DINOv3 与候选重排的直接近邻

### 方法事实

v2 使用冻结 DINOv3 ViT-S+、ViT-Adapter/FPN 与 stride-4 融合特征。给定点位置采样出初始 query，query 与当前特征、同伴点和自身 FIFO 记忆交互；多尺度 cosine 分类生成全图候选，top-16 候选通过 deformable sampling 更新 query，再计算全图分布，最后预测粗格与 offset。候选质量辅助及 uncertainty 用于训练；最终可见决策没有直接使用 uncertainty。[§III、§IV-A](https://arxiv.org/html/2509.19115v2#S3)

主实验为 query-first，另加 20×20 support grid；训练 48 帧，推理对较长视频扩展记忆。骨干对照特意调整 ViT 输入比例以匹配 DINOv2/3 token 数；DINOv3 ViT-S 并未全面优于 v2，ViT-S+ 的收益也依数据而变。[§IV-A/E2](https://arxiv.org/html/2509.19115v2#S4)

论文明确报告重复纹理、无纹理和细结构失败；更长训练通常比单纯扩大记忆更有效，但扩展推理记忆也可能降低短视频表现。[§IV-E/F](https://arxiv.org/html/2509.19115v2#S4)

### 对本项目创新边界的直接影响

1. **高分辨率 DINO 特征、全局候选、重排、细化都已有强组合先例。** 将这些模块换名，或把 Q 减成一颗球，不足以构成 motion 表示贡献。
2. **top-k 用于形成新 query，不必意味着最终位置被硬限制在 top-k 中。** 原文随后重新计算全图相似度。评审其他候选系统时，必须沿数据流判断候选是否真的截断支持，不能见 top-k 就认定下游绝无机会救回。
3. **oracle top-k 曲线不是自动候选系统的充分证据。** 它从已知查询对象出发；本项目要先确定哪个当前或历史位置代表球。GT 初始化、省略无球状态或只统计候选已覆盖样本，都会把更难的一部分移出评价。
4. **“没用 uncertainty 做拒绝”不等于“没有拒绝输出”。** 它另有 visibility 阈值。我们可以指出其 uncertainty 语义/用法与 no-match 不同，却不能把它写成一定强制输出可见位置的反例。

## 从给定点到自动球定位，缺的究竟是什么

TAP 类任务通常跟踪给定的物体表面点。本项目中的球中心是检测语义位置，BlurBall 中点又是拖影几何标签。旋转球的可见纹理点、轮廓点和曝光积分中点未必是同一个随时间保持身份的物质点。这一差别不否定 tracker 可用于球定位，但意味着其 query、坐标损失与遮挡真值不能未经核对就直接继承。

可以把两类问题写成：

`给定点追踪：视频 + 已知(t0,p0) → 此点后续位置/可见性`

`自动球定位：合法时间窗 → 协议所定义的球位置/状态`

在前者里，把 Q 个已知 query 与 P 个图像位置匹配，匹配项可以是 `O(QP)`；在后者里还要解决 query 的产生。但这**不意味着自动定位必然要 O(P²)**：学得的类别 query、dense heatmap 或其他自动发现机制都可能避免遍历全部候选对。复杂度必须依据最终算法，不能以“需要自动发现”先证明全局搜索一定昂贵。

若以后用这些权重做有界诊断，GT 初始化仅能回答“已给准确种子，剩余追踪是否成立”；真实检测种子回答的是检测与跟踪组合。二者应分开，且都不能替代完整自动逐帧定位指标。本轮没有启动这类诊断。

## 与当前实证相接的决定

- [真实历史与重复当前帧对照](../protocols/blurball-full-temporal-control-v1.md)继续按原计划完成。这些论文不改变正在运行的 epoch、输入窗口、标签或选择规则。
- [本地冻结细节分支负结果](../experiments/2026-09-11-frozen-detail-readout.md)与 [关系辅助负结果](../experiments/2026-09-11-endpoint-auxiliary.md)继续有效；新论文的正结果不是重跑原配方的理由。
- 若以后提出显式候选对应，Track-On2 是必须解释差别的近邻；若提出无显式匹配的递推表示，TAPNext 系列是必须解释差别的近邻。比较应取决于实际候选机制，不要求现在额外训练三个大模型。
- 若历史收益主要体现在状态判断，就不能用这些 tracker 的“motion”命名把它重新包装为精确位移证据；若有明确位置增益，也仍要比较可承受的竞争解释，而非从阅读直接选定 Transformer/SSM。

本轮新增的是可追溯的研究约束，而不是已批准的新模型。最重要的收获是：**现代骨干、显式搜索、记忆、分布输出和训练时序范围各自都有强先例，论文需要解释它解决了哪种实际可测的剩余失败。**
