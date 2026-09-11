# 训练期稀疏中心关系监督：DINO 特征适配与点对应近邻的边界

日期：2026-09-11。范围：对拟执行的训练期辅助监督做有界、对抗性近邻核对，只读 DINO-Tracker、LoFTR、TAPIR、CoTracker3 的正式/作者一手材料；MotionSqueeze 与 SELFY/STSS 的局部相关和时空 offset 分布先例已在 [local-correspondence-baselines](2026-09-10-local-correspondence-baselines.md) 核验，本次不重复展开。未读新数据或权重、未训练模型。该范围足以限制本辅助的表述，不是所有 correspondence/point-tracking 文献的穷尽检索。

## 决策先行

拟议辅助不是新的 correspondence 或 tracking 方法：当前 VC1 GT native 格共同定义局部 ROI；relation 臂从该格取当前实例 feature 作 query，并与历史 ROI 内候选 key 以温度 $\tau=0.1$ 的 centered cosine 排序。过去 VC1 GT 格只指定正确 offset：$\Delta1/R2$ 为 25 类，$\Delta2/R4$ 为 81 类，并以交叉熵训练。这属于已知的 **query-conditioned feature matching + 已知对应标签的候选分类/contrastive 学习**。DINO-Tracker 已以 DINO 特征 cosine cost 与候选负类的 softmax contrastive loss 适配特征；LoFTR 已以 GT 粗格对应对匹配置信矩阵做 NLL；TAPIR 与 CoTracker3 已把 query 到跨帧位置、可见性/置信度和局部相关作为点追踪的基本对象。这些是一般监督形式的先例，不能据此说它们覆盖本训练期、三真帧、无推理支路架构的全部细节；同样，本方案不能称首次“用中心对应监督 DINO”“用局部 offset 分类学 motion”或“以关系分布训练特征”。

本次仍有明确、可证伪的**实验问题**：在不改变既定 3 真帧定位头和推理图的条件下，训练期的双端 VC1 局部监督，是否比等维全局可训练向量的 CE 控制更能提升同域 local exact R@1/NLL，且该变化是否随自动定位指标改善而出现。若只改善 GT-query R@1/NLL、主任务自动定位没有改善，它只能说明辅助读出更会复现训练时的 oracle relation，不能声称模型在推理时建立了有用对应；若控制臂同样改善，不能把收益归因于空间条件 query；若两者均无改善，应停止这一辅助解释。

## 本次被审查的方案与它真正监督什么

保持共享可训练 DINOv3 ConvNeXt-Tiny 前两 stage、3 个真实帧 stack 和定位头不变。只在训练时，当前帧 GT native grid 定义两条时间关系共同的 local ROI；relation 臂在该当前格取 feature 作 query，并从过去 $\Delta1$ / $\Delta2$ 帧的同一空间 ROI 取 local keys。过去 GT native grid 只给出 ROI 内正确 offset；同一 clip、当前和过去均 VC1、该 offset 落在 $R2/R4$ 内才计算 CE，其余 OOB/不合格端点 mask。$\Delta1/R2$ 是 25 类，$\Delta2/R4$ 是 81 类。类别是局部 key 的**空间 offset**；非标注中心类只是 CE 的其余类别，并非已标注的背景、无球、遮挡或 no-match 类。OOB mask 也不是模型在推理时学会拒配。

该支路不进入推理图：自动定位时没有 GT 定义的 ROI、当前实例 query 或历史标签指定的正确 offset，也没有这一 25/81 类 softmax 的输出。因此辅助的 relation accuracy 是有用的优化诊断，却不是推理时可调用的 correspondence score。双端 VC1 限定只使标签语义更清楚，不能保证几个像素的球在特征格中可匹配，也不等于测试时自动检测已覆盖该球。

公平控制以全局可训练、归一化的 192D `ball` 向量替代从当前 GT 格读取的实例 descriptor；两臂都以同一当前 GT native 格定义 ROI，并保持 keys、标签、温度和 CE。relation 臂不新增辅助参数，`w` 臂才额外有 192 个训练参数；两臂只在训练 batch 上将辅助对共享 prefix 的 gradient norm 对齐，baseline 不含辅助。该控制可比较“当前实例 descriptor（及其回传到共享 prefix 的梯度）”与“全局共享外观向量”在同一标签条件 ROI 的作用，却不隔离全部关系机制：`w` 的专有参数、梯度方向、难例重权与共享特征的训练轨迹仍可不同。故它不是“参数化完全相同”的证明，也不能单靠梯度范数匹配排除这些差异；这些正是需要报告为设计范围的因素。

## 最近邻 1：DINO-Tracker 已适配 DINO 特征以服务点对应

Narek Tumanyan、Assaf Singer、Shai Bagon、Tali Dekel，**“DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video”**，ECCV 2024；[arXiv v2，2024-07-11](https://arxiv.org/abs/2403.14548)，[作者项目/代码入口](https://dino-tracker.github.io/)。本次实际阅读主文与附录 A--B（arXiv HTML 版本）。

**实际机制与监督。** 给定 query point $\mathbf x_q$，DINO-Tracker 双线性取 query feature，与目标帧全 feature map 算 cosine cost $S(\mathbf p)$，经 CNN refiner 和 spatial softmax 得 heatmap，再在峰值邻域加权读出坐标。[§3.1，式 (1)--(2)](https://arxiv.org/html/2403.14548#S3.SS1)。它在**单个测试视频上 test-time train**一个 CNN Delta-DINO 残差特征器：短距离正对应来自 RAFT flow 的 cycle-consistent tracklet；远距离粗对应来自 raw DINO feature 的 mutual-NN “best buddies”；以及 refined-feature best-buddy、cycle-consistency、DINO-prior preservation。DINO best-buddy 项正是温度 cosine 的 softmax contrastive loss：正配对相对目标帧全部候选 feature 的指数相似度归一化，其他位置是负类。[§3.2--§3.3](https://arxiv.org/html/2403.14548#S3.SS2)、[式 (5)](https://arxiv.org/html/2403.14548#S3.SS3)。

**与本方案相同和不同。** 它直接覆盖“预训练 DINO feature 经 query--key cosine relation 和候选负例监督后可为跨帧位置更可分”的一般动机，也明确表明 raw DINO 不天然够细位置。不同在于其标签不是本项目的球中心 GT：flow/best-buddy 是单视频自监督/伪对应；cost 搜索为全目标 feature map；模型训练后在该**同一测试视频**对任意 query 输出连续轨迹和可见性。拟议支路则是跨训练样本共享的 DINOv3 ConvNeXt prefix，只对双端 VC1 的 GT 中心附近做固定局部离散 offset CE，并在测试不执行 relation readout。这个差别足以防止把本实验说成复现 DINO-Tracker，却不足以把“DINO + cosine 对应适配”叫创新。

**成本与迁移限制。** DINO-Tracker 报 100 帧单视频约 1.6 小时、单 A100 的 test-time fitting；其所谓 inference cost 不含这一次必经的 per-video fitting。[附录 B](https://arxiv.org/html/2403.14548#S0.SS2)。本方案没有该测试期适配和额外推理 branch，成本条件实质不同。反过来，DINO-Tracker 的 tracking AJ/位置/visibility 结果不能替代 tiny-ball 自动发现或本方案的 F1：其评价有给定 query，且使用 TAP-Vid/BADJA 点轨迹协议。

## 最近邻 2：LoFTR 已以 GT 粗格对应训练匹配置信分布

Jiaming Sun、Zehong Shen、Yuang Wang、Hujun Bao、Xiaowei Zhou，**“LoFTR: Detector-Free Local Feature Matching With Transformers”**，CVPR 2021，pp. 8922--8931，[CVF 正式论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.pdf)、[作者项目](https://zju3dv.github.io/loftr/)。本次复核正文 §3.3--§3.5；不依赖其 supplementary。

LoFTR 在 $1/8$ coarse grid 建两图的全对候选 score，经过 optimal transport 或 dual-softmax 得 confidence matrix $P_c(i,j)$；训练时用相机位姿和深度将两边 coarse grid 的互为最近重投影点构成 $\mathcal M_c^{gt}$，并对 $P_c$ 的这些 GT 配对做 NLL。推理时再以 confidence threshold 与 mutual nearest-neighbor 选 coarse matches，随后才在 $1/2$ fine feature 的局部窗做 refinement。[§3.3、§3.5](https://arxiv.org/html/2104.00680#S3.SS3)。

这直接覆盖“用已知几何对应把一个离散候选集合训练成正确 offset/目标位置概率”的监督形式；拟议 25/81 类 CE 只是在候选范围、单向 normalisation 和 GT 来源上更窄。它不覆盖本项目的 3 帧运动输入或 tiny-ball 标签：LoFTR 是两张静态图、训练和推理都实际输出 match/confidence，并有深度/pose 派生的几何 GT；本方案只以已有中心标签做训练期特征塑形，不生产推理 match/confidence。因此不能把本方案的 softmax 值解释为 calibrated matching confidence，更不能因没有 no-match 类把它称为 LoFTR 式拒配。

## 最近邻 3：TAPIR 与 CoTracker3 表明 query 条件点监督、位置/可见性/置信度应分开

Carl Doersch 等，**“TAPIR: Tracking Any Point with Per-Frame Initialization and Temporal Refinement”**，ICCV 2023；[arXiv v2，2023-08-30](https://arxiv.org/abs/2306.08637)、[CVF 正式论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.pdf)。它接受给定 query point，先以 query 对每个目标帧全局 cost volume 得位置、occlusion 与 position-uncertainty，后以局部 correlation 跨时间迭代细化 query feature 和轨迹。[§3.1--§3.2](https://arxiv.org/html/2306.08637#S3)。其位置、occlusion 和 uncertainty 都有监督：uncertainty target 由预测离 GT point 是否超过阈值定义，推理时用 occlusion 和 uncertainty 共同作可用性判断。[式 (1)](https://arxiv.org/html/2306.08637#S3.SS1)。作者明确训练**仅**用带轨迹/可见标注的合成 Kubric MOVi-E，评测为给定 query 的 TAP-Vid 位置与遮挡指标，不是自动目标发现。[§3.3、§5](https://arxiv.org/html/2306.08637#S3.SS3)。

Nikita Karaev 等，**“CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos”**，ICCV 2025；[CVF 正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.pdf)、[arXiv:2410.11831](https://arxiv.org/abs/2410.11831)。它在合成预训练后用老师对无标注真实视频给 pseudo tracks；query 用 SIFT 从随机帧抽取且若可追踪点不足会跳过视频，明显不是“所有规则格/目标都可跟”。[§3.1](https://arxiv.org/html/2410.11831#S3.SS1)。模型迭代更新轨迹、confidence、visibility；轨迹以 Huber、confidence/visibility 以 BCE 监督，confidence GT 定义为当前预测距 GT track 小于 12 px。[§3.2--§3.3](https://arxiv.org/html/2410.11831#S3.SS2)。offline 版可以双向看全视频，online 版仅滑窗前向；二者时间可用性不可混用。[§3.2](https://arxiv.org/html/2410.11831#S3.SS2)。

二者共同限制本方案：GT current center 作 query 测到的是**给定正确查询条件下**的 relation 可读性，不能替代自动球发现；VC1/OOB mask 也不等于 visibility 或 confidence model。它们同时给出本项目应分开的实证层次：关系的 exact rank/NLL、自动定位、存在/误报，以及若今后加入拒配，拒配的明确 target 与推理输入。拟议方案没有输出 visibility/confidence/no-match，因而不要把 CE 降低称为可靠性或概率校准。

## 对训练解释和报告的约束

1. **已有先例的部分。** cosine/温度 softmax、以已知 GT/伪 GT point relation 监督候选排序、local refinement、为对应适配 DINO 特征、以及时间间隔中的点条件跟踪，均已有一般形式的直接先例。既有 MotionSqueeze/SELFY/STSS 还已覆盖局部相关与多 offset relation；不能从 25/81 类、两种 $\Delta$ 或 $R$ 的组合拼出“新 motion representation”。本方案的其余候选类别只是非标注中心类，不应把它们混称为语义明确的候选负类。
2. **有限且可测的区别。** 本方案的研究价值只能写成既定三真帧系统内的一项有限干预比较：同一共享 DINOv3 prefix 和定位头下，GT 双端 VC1 的稀疏、局部、训练期 relation CE 是否改变 DINO 特征的 local exact R@1/NLL，并在不增加推理算子时转化为自动球定位收益。全局 192D vector-control（仅该臂新增参数）、同 ROI/keys/labels/CE、训练 batch 的 prefix gradient-norm 匹配和只按主 F1 选 checkpoint，使这一具体比较可被推翻，却没有隔离全部 relation 机制，也不能包装为架构贡献。
3. **不能跨越的结论。** GT-query relation 变好不证明 inference-time query 已被发现；双端 VC1 结果不证明 VC2/VC3、遮挡、absence 或重入场；local label in-range 结果不证明长位移 coverage；CE logits 未经 held-out calibration 分析和明确定义的事件 target，不是 calibrated confidence。主 F1 是唯一 checkpoint 选择准则时，relation 指标应作为预先声明的诊断而非反复挑选模型的依据。
4. **最小有效结果表。** 报告两条辅助臂和无辅助主任务在同一 split/seed/训练预算下的 automatic F1 与逐帧定位；在独立固定的双端 VC1、in-range 诊断集合上报每个 $\Delta/R$ 的 exact R@1、NLL、有效样本数/OOB 数。再把自动定位的变化与 relation 的变化并列，不以任一替代另一项。若主 F1 无增益或退化，不把局部 relation 的正结果当作下游定位机制成立。

本次结论只限定拟议训练期辅助的最近邻位置和应检验的边界；没有建议引入 DINO-Tracker 的单视频 test-time fitting、LoFTR 的全局匹配器、TAPIR/CoTracker3 的 tracker/teacher，亦没有把已有点跟踪指标或论文成本移植为本项目结果。
