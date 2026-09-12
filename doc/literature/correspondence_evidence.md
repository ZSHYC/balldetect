# 通用对应与新增检索证据

核验日期：2026-09-09。本文补充主报告的对应算子、经典时序与近期新稿来源。它是本轮检索的可追溯笔记，不是全量系统综述。阅读层次分为正文/方法核验、官方摘要核验、作者资源核验；三者不能混写为“已完整复现”。

**2026-09-12 补充：** [因果点跟踪](2026-09-12-causal-point-tracking.md)已补读 TAPNext/++ 与 Track-On2 的方法、实验和相关附录；[微小目标新近邻](2026-09-12-recent-tiny-motion.md)进一步核对 CoWTracker 无 cost-volume 路线及其注意力成本。它们限制“显式相关体必需”和“无需相关体就整体线性”的两种相反误推。首轮表中的早期阅读深度按原记录保留。

**同日继续补读：** [TSM/GSM/GSF](2026-09-12-lightweight-temporal-routing.md)与[Taylor Videos/TDN](2026-09-12-temporal-difference-representation.md)补足经典轻量机制的全文和实现；本文第 5 节补 Motionformer。新的 [FreeFlow v1](2026-09-12-recent-tiny-motion.md)纳入 9 月 10 日新稿。以下首轮“仅摘要”记录不再代表这些条目的最新阅读深度。

## 1. 最直接改变创新判断的三项前史

| 工作与原始入口 | 本轮阅读层次 | 对本项目的具体约束 |
|---|---|---|
| [Devon: Deformable Volume Network for Learning Optical Flow](https://arxiv.org/abs/1802.07351)；[WACV 2020 正式论文](https://openaccess.thecvf.com/content_WACV_2020/papers/Lu_Devon_Deformable_Volume_Network_for_Learning_Optical_Flow_WACV_2020_paper.pdf) | 原始记录、正式正文方法 | 已针对金字塔粗尺度丢失快速小物体，使用高分辨率特征及可变形、扩张相关体。因此“保细节同时扩范围”不是新问题。arXiv 首稿 2018 与正式版本 2020 应分开记录。 |
| [Learning Optical Flow from a Few Matches / SCV](https://arxiv.org/html/2104.02166) | 正文，重点核对动机与 sparse cost volume 方法 | 已讨论快速小目标，并保留全局匹配中的稀疏高分相关值与坐标。它是“两端稀疏候选/高分辨率远搜索”必须对抗的近邻。存储稀疏不能自动证明建立候选也只需线性计算。 |
| [Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2026/html/Briedis_Efficient_All-Pairs_Correlation_Volume_Sampling_for_Optical_Flow_Estimation_CVPR_2026_paper.html) | CVPR 2026 正式页面、正文方法片段 | 已优化 RAFT 类相关采样的算子实现。必须与高效实现比较，不能仅打败朴素展开的全相关矩阵。其算子等价性和特定查询模式，不等于任意全局查询都获得免费线性复杂度。 |

这里不把三者直接认定为本项目最强球检测基线；它们首先限定新颖性。选定算子后，再决定最接近哪项、哪些组件需要公平适配。

## 2. 经典机制的原始来源与用途

| 机制 | 原始入口 | 核验层次与用途 |
|---|---|---|
| 时间通道移动 | [TSM](https://arxiv.org/abs/1811.08383) | 官方摘要；轻量时序基线的机制来源 |
| 门控通道移动 | [GSM](https://arxiv.org/abs/1912.00381) | 官方摘要；不能泛称“有选择地交换时序特征”为新意 |
| 门控移动与融合 | [GSF](https://arxiv.org/abs/2203.08897) | 官方摘要；与通用时序融合区分 |
| 差分式运动建模 | [TDN](https://arxiv.org/abs/2012.10071) | 官方摘要；差分可有效，但差分不直接给出位移对应 |
| 局部匹配推导位移与置信信息 | [MotionSqueeze](https://arxiv.org/abs/2007.09933) | 正文方法核验；核对相关体、kernel-soft-argmax 与置信信息，避免把原始局部相关写成本文贡献 |
| 时空自相似作为 motion 表示 | [Learning Self-Similarity in Space and Time as Generalized Motion for Video Action Recognition](https://arxiv.org/abs/2102.07092) | 正文概览；SELFY/STSS 的关系表示已不止一个位移向量 |
| 隐式轨迹注意力 | [Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers](https://arxiv.org/abs/2106.05392) | 官方摘要；Motionformer 的实验任务是动作识别，题名中的 ball 不代表已经验证本项目几像素球定位 |
| Taylor 式视频表征 | [Taylor Videos](https://arxiv.org/abs/2402.03019) | 官方摘要；作为表示前史，不能把识别任务证据外推到像素定位 |
| 光流引导跨帧聚合 | [FGFA](https://arxiv.org/abs/1703.10025) | 官方摘要；已有针对模糊与快速目标的特征聚合，本文必须定位到更窄的 tiny-support 问题 |
| 检测监督下时空可变形采样 | [STSN](https://arxiv.org/abs/1803.05549) | [2026-09-12补读全文](2026-09-12-task-supervised-alignment.md)；作者代码未确认。pair-conditioned采样不需光流标签，原训练随机取前后support，推理含未来帧；因果三帧只能作明确改写的机制对照 |
| 全局匹配与细化 | [GMFlow](https://arxiv.org/abs/2111.13680) | 摘要及方法概览；全局到局部不是新颖性本身 |
| cost memory / motion latent | [FlowFormer](https://arxiv.org/abs/2203.16194) | 摘要及方法概览；不能把相关体压成 latent 本身当作新的 motion 表示贡献 |
| 拒配/未匹配 | [SuperGlue 作者项目页](https://psarlin.com/superglue/) | 作者页面及论文片段；unmatched 是标准匹配问题组成部分，不能等同于球不可见真值 |

以上来源仅支撑对应的机制事实。没有运行它们在体育数据上的性能、显存或效率测试。

## 3. 新稿补检与纳入判断

检索同时使用题名/ID 与机制词组，重点覆盖 2025—2026，并补查 2026 年 8—9 月。下表中首稿日期来自 arXiv 提交记录；日期晚不代表相关性强。

| 工作 | 核验层次 | 判断 |
|---|---|---|
| [FlowIt: Global Matching via Hierarchical Transformers and Optimal Transport for Optical Flow](https://arxiv.org/abs/2603.28759)；v1 2026-03-30，v2 2026-05-31 | 官方题名、版本与摘要 | 全局匹配、最优传输与置信引导细化的近邻；未据摘要断言其准确复杂度或 tiny-ball 性能 |
| [SWIFT](https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Wang_SWIFT_Efficient_Warping-Only_Optical_Flow_via_Scale-Specialized_Refinement_CVPRW_2026_paper.html) | CVPR 2026 workshop 官方摘要检索结果；页面直接访问受限 | 效率路线补充；暂不推导未读正文的算子细节 |
| [MARCO](https://arxiv.org/abs/2604.18267)；2026-04-20 | 原文 HTML 摘要与方法概览 | 语义对应的局部监督泛化与细化近邻；不是跨帧球实例对应证据 |
| [NSFlow](https://arxiv.org/abs/2609.06074)；2026-09-05 | 官方摘要 | 可微神经/符号稀疏光流，用于视觉里程计；保留查重，非直接球检测基线 |
| [EgoSIS](https://arxiv.org/abs/2609.08938)；2026-09-08 | 官方摘要 | 相机参考、残差与可靠性概念的近期先例；原任务 UAV 问答，其切镜机制不进入本项目 |
| [PhysFlow](https://arxiv.org/abs/2609.08215)；2026-09-08 | 官方摘要 | 视频生成方向，排除出主要定位对照 |
| [MotionPyramid](https://arxiv.org/abs/2606.20705)；2026-06-15 | 官方摘要及方法概览 | 人形控制的动作层级，不是本文视觉 correspondence |
| [PACT / Following the Flow](https://arxiv.org/abs/2606.22378)；2026-06-21 | 官方摘要 | 事件相机小目标传播，可供机制查重；传感器条件不符 RGB 主任务 |
| [Event-based Gaze Control](https://arxiv.org/abs/2606.26780)；2026-06-25 | 官方摘要 | 主动事件视觉与旋转测量，排除出 RGB 定位主要基线 |

DINOv3 上采样的新稿、What Moves?、V-JEPA 2.1、ReMoRa 等详见现代 motion 附件；BIRD、MI-DETR 及红外/UAV 近邻详见微小目标附件。分开记录只为便于追溯，不表示不同任务的 motion 可以互相替代。

## 4. 本轮证据修订与边界

主报告另行核对了 WASB 原始 PDF 数据表：是 **Table 1，PDF 第 7 页**，不是 Table 7；19,835 的协议帧数与 TrackNet 原论文 20,844 的实验帧数属于不同口径。也核对了 TTNet 原文的低分辨率粗定位—原图裁剪精定位，以及 TOTNet 正文对完全遮挡帧采用轨迹一致插值的标签来源。OTHR 的球区域 EPE 数值经原文 Table 2 核对，不能当作真实体育视频结论。

本轮没有全量下载 arXiv 元数据，没有逐篇全文阅读所有外围条目，没有复现代码。最终接收状态无法从出版社确认时，保留预印本/作者声称状态。后续最值得补证的是最终所选算子的最近邻全文与实现，而不是无限增加题名数量。

## 5. Motionformer 全文补读：先保留时间索引，再做空间与时间池化

阅读日期：2026-09-12。Mandela Patrick et al., *Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers*，NeurIPS 2021；[arXiv v2](https://arxiv.org/abs/2106.05392v2)，2021-10-23。已读[方法 §3、主要实验 §4、附录 §6.1--6.2](https://arxiv.org/html/2106.05392v2)，以及作者 `TrajectoryAttention` 的实际前向；未运行模型。

### 原文中真正与当前问题相关的事实

每个空间时间 token 自作 query，先分别在各帧内做空间 softmax，得到每个时间的一份加权 feature；再沿时间注意力池化。这里的 trajectory 指 pairwise 的软匹配支持，不是显式多帧轨迹约束。精确版仍有 $S^2T^2$ 成对成本；Orthoformer 通过少量原型近似。原文默认用 $2\times16\times16$ tubelet，输入 16 帧，属于视频任务。[§3、式 (4)--(6)、(10)](https://arxiv.org/html/2106.05392v2#S3)

不能把它概括为“只做动作分类、完全没有密集证据”：附录还以冻结 attention 传播标签测 DAVIS 2017，J&F 为 60.6，同表 DINO-B/16 为 62.3；这是给定分割标签的传播，不是自动球中心定位。增加采样 stride 的分类实验也同时扩大观察时间跨度，不能单独归因于位移能力。[附录 §6.1](https://arxiv.org/html/2106.05392v2#S6.SS1)

### 作者实现还区分了历史路径与修正路径

[`vit_helper.py` 的 `TrajectoryAttention.forward`](https://github.com/facebookresearch/Motionformer/blob/main/slowfast/models/vit_helper.py#L146)先计算完整 `q_ @ k_.transpose(...)`，重排后每帧独立归一化，再计算时间 attention。它与“先全时空 softmax，再拆帧”的算子不同。

同一实现保留 `use_original_code` 两路：作者注释说明早期实现的时间 value 直接使用聚合后的 `x`，修正路才使用学习投影后的 `v2`。因此准备复用时，必须明确遵循论文公式、历史复现还是作者修正；不能把三者默认为完全一致。本轮只记录该作者明确披露的差异，未改本项目代码、未安装依赖或下载权重。[作者代码相应分支](https://github.com/facebookresearch/Motionformer/blob/main/slowfast/models/vit_helper.py#L242)、[作者使用说明](https://github.com/facebookresearch/Motionformer#training-the-default-motionformer)

### 对本项目的推论

1. **关系的组织方式可以独立于搜索范围。** 对同一对 logits $a_{u,v,t'}$，在每帧内归一化与在所有 $(v,t')$ 上一次归一化，施加了不同的竞争规则。前者每帧先分配单位总质量，再决定时间权重；因此“扩大候选域”不是唯一可研究的改变。这个思想已是明确前史，不能把逐时间保留匹配支持本身写成新贡献。
2. **软分布不等于一直保留多个可解码位置。** 第一阶段可以同时支持多个位置，但随后把它们汇成 feature；这与维护带地址的多假设集合不同。它既不必强选一个峰，也不保证不会混合球和背景。要主张后一种机制更好，需要真实自动定位证据，不能只拿软/硬二分作为论据。
3. **没有对应时，逐帧 softmax 仍会分配质量。** 后续时间 attention、残差或其他层可能抑制坏信息，所以不能由此断言系统必然失败；但这也不是显式 no-match 的证明。是否需要拒配应由本地失败与标签语义决定。
4. **原型数量与目标覆盖不是同一个量。** Orthoformer 选特征方向多样的原型，不等于监督保证每颗 tiny ball 都有原型。固定少量原型可降低算子成本；能否保持少量像素的球证据是另一问题。本轮没有测出覆盖不足，也不据此添加球原型分支。

Motionformer 使用稠密 token 自查询，与 Track-On2 的给定点 query 不同；没有 GT 点初始化并不自动使其成为球发现器。反过来，它说明“自查询 + 分帧全局匹配 + 时间融合”的概念组合早已存在。最终若采用这种关系组织方式，应说明本地自动候选、微小空间支撑和预算约束下具体改变了什么。

固定过去窗口只预测末帧时，窗口内双向交互仍可对末帧因果；若要跨窗口复用逐层时序状态，则须另行处理上下文依赖。原 Motionformer 的视频分类/标签传播结果没有给出本项目逐帧中心的这种部署协议，不应继承其分数或把论文整套训练变成当前必跑基线。
