# 近期高速／微小目标 motion 近邻：四条会改变本项目判断的证据

**目的。** 本文是截至 **2026-09-12** 的定向补检，不是再次罗列已细读的 FlowIt、EgoSIS、MI-DETR、DeepPro、DMR、CMRTrack、BIRD/STSN，也不批准新的模型分支。它只补入四条能实质改变本项目问题表述、对照选择或可主张边界的近邻：一个自动检测加历史硬约束的 tiny-target 系统、一个带伪速度教师的极小目标 transport 系统、一个把背景而非目标作为主表征对象的系统，以及一个在高分辨率下以 warping 替代 cost volume 的 dense point tracker。

**检索边界。** 本轮在 arXiv 的 `sports ball detection`、`racket sports tracking`、`small target video motion`、`small object motion`、`long-range correspondence` 等主题检索中按提交／更新日期排序，并用题名追查一手论文、正式会议页或作者仓库。重点查看 2026 年 7--9 月更新和此前笔记未完整阅读的直接近邻。搜索结果没有证明不存在其他工作；以下仅是此范围内足以约束当前研究决定的最小证据集。

## 阅读范围、版本与结论先行

| 工作 | 一手阅读与版本 | 任务、时间语义和是否自动发现 | 对本项目最重要的限制 |
|---|---|---|---|
| Peng et al., *A Simple Detector with Frame Dynamics is a Strong Tracker* | [CVPR 2025 Anti-UAV Workshop 正式开放版](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Peng_A_Simple_Detector_with_Frame_Dynamics_is_a_Strong_Tracker_CVPRW_2025_paper.pdf) 全文；arXiv:2505.04917v1，2025-05-08 | 红外 UAV 框检测／tracking；`[t-2,t-1,t]` 的 signed frame difference 或 `t-1,t` Farneback flow；检测器全图出框，随后以历史框常速度过滤。挑战 Track 1 有初始框，Track 2 无初始框。 | 其完整三通道差分输入是三帧的可逆线性基变换；当前完整 frame stack 在首层可自由线性混合时已覆盖同一函数类。轨迹过滤则会把历史预测混回输出，不能当作当前帧 visual correspondence。 |
| Guo et al., *Following the Flow* / PACT | [arXiv HTML v1](https://arxiv.org/html/2606.22378v1) 全文、实验与消融；2026-06-21。作者 [仓库](https://github.com/fulongcai/PACT) 声称 ECCV 2026 接收，但本轮只见 README，未见可审计源码。 | event-stream 小目标分割／定位；在 8 秒事件窗中预测速度场、transport feature；不以 GT query 初始化，输出密集分割。 | “沿运动场运输弱证据 + transport residual 作可靠性”已有很近的 tiny-object 先例；但其 velocity teacher 来自同一实例的前景事件匹配，不能由球中心标签无歧义替代。 |
| Zhang et al., *Beyond Motion Cues and Structural Sparsity* / TenRPCANet | [arXiv HTML v2](https://arxiv.org/html/2509.07654v2) 全文、主要表格与消融；v2，2026-08-05。论文未给作者代码链接，本轮未把检索未命中写成“没有代码”。 | 多帧红外小目标 mask 与可见光空间碎片；8 帧等权、一次输出窗口所有帧；不需 GT 初始化、没有显式 correspondence／flow。 | 强烈反驳“微小高速目标必须首先建立目标 correspondence”的预设：背景的低秩／自相似可先成为主要判别证据；该假设在球场摇摄、球员与变焦条件下是否仍有用尚未被此论文证明，且其协议非因果。 |
| Lai et al., *CoWTracker: Tracking by Warping instead of Correlation* | [arXiv HTML v1](https://arxiv.org/html/2602.04877v1) 全文、附录运行／消融；2026-02-04。作者 [代码](https://github.com/facebookresearch/cowtracker) 提供推理接口，本轮未作代码复现。 | dense point tracking；参考帧每个像素都是 query，所有 target frame 共同用时空 attention 更新；无 GT point query，但仍不输出“哪个点是球”。 | 高分辨率 weak evidence 与大位移不必然要求显式 cost volume；高分辨率 iterative warping 是必要的竞争机制。不过它是大 backbone、全密集轨迹、离线窗口，不是可直接移植的自动球定位方案。 |

前三篇是 tiny-target／自动发现路线的主要补充；CoWTracker 只是“有限预算下何种 correspondence 形式有必要”的方法学反证，不能被误写成体育球 baseline。这里的 **A** 只表示本轮读到了论文方法和实验关键段，不表示本项目已经复现或认可作者指标。

从这四篇合起来得到的第一性原理修正是：

1. `frame difference`、dense flow、trajectory gate、feature transport 与 correspondence 是不同变量；已有系统往往混合它们才能取得最终 tracking 分数。
2. 真正仍待实证的不是“时间信息是否有用”，而是本项目公开球标签下，**当前球位置的视觉证据何时必须由跨位置匹配补足**，何时简单变化、背景建模或历史筛选已经解释全部收益。
3. “可信／no-match”也不能只看有没有一个 `[0,1]` gate：必须分清它是 track-error teacher、transport-consistency weight、历史空间 gate，还是可用于拒绝当前球视觉证据的校准决策。

## 1. Frame Dynamics：最应先排除的廉价自动检测竞争解释

### 可证实事实

Peng et al. 将当前红外帧与两个历史帧的**带符号差**拼接为 `cat(x_t, x_t-x_{t-1}, x_t-x_{t-2})`；另一变体拼接当前帧与 Farneback flow 的两个分量。二者都进入一个全图检测器，而不是先在某个 GT 目标附近 crop。论文在原始、frame-difference 和 flow 三种输入之间比较多个检测器；最终还组合多个 detector 的框，再进行加权框融合。[方法 §3.2](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Peng_A_Simple_Detector_with_Frame_Dynamics_is_a_Strong_Tracker_CVPRW_2025_paper.pdf)

它的 TC-filtering 是另一件事。先由前两帧输出框中心作常速度外推，在预测中心附近半径 `d_max` 的窗口外丢弃当前候选。若当前检测严重偏离，Track 1 可回退到 LoRAT 的 tracking 输出；作者明确说明 Track 2 因首帧没有目标位置而**不使用 LoRAT**。因此论文同时含有：

- 全局的当帧 detector，适用于“球／目标从哪里自动发现”；
- 两种很便宜的时间输入（差分、传统 flow），回答“哪里变化”；
- 基于已有位置的后验筛选／回退，回答“哪个历史轨迹更连续”。

最后一项并不是当前帧与历史帧真实视觉对应的证据。它会在 detector 漏检或误检时改变输出，故报告最终 AOA 却不报告 detector candidate recall 或每帧图像证据时，不能据它的最终 tracking 数字推导出某个 motion representation 具有对应能力。

论文使用 4th Anti-UAV 数据；训练集 223 条热红外序列，分辨率为 640×512 或 512×512。主检测测 AP，追踪测 AOA；最终方案把高分辨率 MMDetection 分支（1280×1024）和 640×512 YOLO 分支组合。其表格确实显示 frame-difference 输入在若干 detector／fold 上提升 AP，但这属于热红外框协议和 ensemble，不能移植成 BlurBall PCK 或 tennis 的逐帧中心定位承诺。[实验 §4、表 1--3](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Peng_A_Simple_Detector_with_Frame_Dynamics_is_a_Strong_Tracker_CVPRW_2025_paper.pdf)

### 对项目设计的直接影响

1. 不能把“现有 tiny target detector 只用单帧外观”作为动机。一个全局检测器已经可同时看 `t,t-1,t-2` 的差分，且不需要 GT 初始化。
2. 对本项目的 `current + 两个 signed difference` 而言，这恰好是完整 `[t-2,t-1,t]` frame stack 的可逆线性基变换：首层若能自由线性混合三帧，重参数化首层即可得到相同下游非线性 detector 的函数类。Peng et al. 的 IR 输入结果不推翻已完成的线性等价性判断，也不授权再做一次已经否决的 signed-difference 训练。只有在具体提案把差分放在首层混合**之前**的不可吸收非线性、归一化、量化或显式约束后，差分才另有待测的机制意义；当前协议没有这类改变。
3. TC-filtering 也说明“发生错误时使用历史轨迹拒绝候选”并不新。若球项目后续引入拒绝或多假设，必须把**当前视觉候选分数**与**历史运动先验**分开输出和评测；否则可能只是在把一个常速度 gate 重命名为 reliability。
4. 不应直接复刻其 Track 1 fallback：它依赖已知初始化框，和本项目逐帧自动球定位不是同一任务。Track 2 虽无初始化，但仍只有框 AOA，未分析 tiny ball 中心误差、大位移 bucket 或背景线条误配。

**当前决定。** 不启动该大型 ensemble、LoRAT 或新的 signed-difference 训练。当前完整 frame stack 已覆盖前述可逆差分变换的线性函数类；本论文只提醒我们，若日后出现实际不可吸收的差分操作或后处理，必须将其与历史先验和 detector 容量分开解释。

## 2. PACT：transport consistency 与伪速度监督的边界

### 可证实事实

PACT 面向事件相机的弱小目标，输入不是 RGB 帧，而是包含位置、时间、极性等属性的稀疏事件体。论文在 encoder 的每个尺度预测有界二维位移；从位移后的邻域采样 feature，并以 transported feature 与原 feature 的相对残差、再加对大速度的惩罚，构造连续权重 `g`。`g` 以凸融合方式选择保留原 feature 还是 transported feature；decoder 再沿该速度场传播以连接间断响应。[方法 §3.2--3.3](https://arxiv.org/html/2606.22378v1#S3)

其“运动真值”不是由 detector 自动产生：作者用相邻时间片中**同一前景实例**的最近前景 voxel 构造伪 velocity，以 Smooth-L1 监督预测速度；最终主任务仍是 voxel-level BCE segmentation。换言之，训练期存在比一个球中心标签更密、更具实例归属的目标区域和时间对应假设。[损失 §3.4](https://arxiv.org/html/2606.22378v1#S3.SS4)

论文在 EV-UAV 上报告平均目标大小约 6.8×5.4 px，99 train／24 test 序列，输入切为 8 秒事件窗和 1 ms 时间分辨率。它有 dense output，故自动发现不是 GT query tracking；但指标是事件分割的 IoU/Acc、定位 `P_d` 与 false alarm。作者的表中 PACT 为 2.9M 参数、58 ms／window；这不是“每个 RGB 帧 58 ms”，也没有给出与三帧球中心、视频解码、固定输入大小可比较的端到端延迟。[实验 §4](https://arxiv.org/html/2606.22378v1#S4)

### 对创新性和训练标签的压力测试

PACT 已封堵下列泛化表述：

- “首次让弱微小目标 feature 沿学习到的 motion 传播”；
- “首次以 transport residual 判断 motion evidence 是否可信”；
- “首次用 motion consistency 使背景触发不参与时序积累”。

不过它不等于球项目已有一个可直接使用的实现。事件极性、微秒时间、稀疏体卷积和 8 秒窗口改变了视觉证据；其成功还依赖伪 velocity teacher。以当前的 `x_t,y_t` 球中心监督，不能把同一中心位移扩张为一个球 blob 内每个像素／token 的真实 flow，再称作“原数据自带速度标注”。若希望研究此类机制，合法的最小版本只能明确为**中心点级的候选／关系监督**，或只以最终中心损失驱动；二者与 PACT 的可辨识性、误差来源不同，必须分别报告。

它的 `g` 也不是 no-match 标签：它是中间 transport 融合权重，没有单独“对应不存在”的语义、校准报告或下游拒绝动作。未来若项目提出 no-match，应先说明：拒绝的是哪一条 `(current location, history location)` 关系、拒绝后是否退回当前帧 appearance、以及如何不把不可见／未标注误作负例。

**当前决定。** PACT 是未来“candidate evidence transport”设计的硬近邻，不是当前数据协议下可直接复现的 baseline；不引入伪稠密 flow 标签，也不以它的 58 ms 作为效率依据。

## 3. TenRPCANet：背景结构可替代 motion matching 的强反证

### 可证实事实

TenRPCANet 的核心不是寻找 `u_t → u_{t-1}+δ`。作者把视频分解为低秩背景、稀疏目标和噪声，并主张：假警虽可能运动，其局部／非局部上下文在时间上通常稳定；目标作为背景结构的 outlier 更可辨。模型以局部子空间 embedding（LSE）和 Video Swin encoder/decoder 建模时空 self-similarity；输入先 stride-4 下采样，使弱目标在 Transformer 中大多被压制，Transformer 主要建背景，随后 PFR 用 2D 局部线性 refinement、阈值门控和上采样重建稀疏目标响应。[论文 §III-A--III-F](https://arxiv.org/html/2509.07654v2#S3)

它以 8 帧窗一次输出窗内**所有**帧的 segmentation，所有帧同等输入；论文未规定 target-frame 的因果锚点、因果 mask、未来帧禁用或 clip 边界处理。因此它只能作为离线序列模型的证据，不能被称作在线对照。作者的 window 消融显示从 2 至 8 帧与较大时空窗口可改善其红外指标；这并不能回答“在有大位移 RGB 球中，真对应在哪里”。[window 消融与限制](https://arxiv.org/html/2509.07654v2#S4.SS6)

在 NUDT-MIRSDT / HiNo（120 个、各 100 帧的红外序列）以及 SSTD 空间碎片数据上，论文报告密集 mask／object-level 指标。作者特别指出该 IR 数据目标相对面积极小，stride-4 后目的就是让 attention 不把目标外观当主要信息；其结论是背景低秩／自相似能在这些数据里比目标 motion 更稳健。[数据、指标与讨论](https://arxiv.org/html/2509.07654v2#S4.SS1)

### 为什么它会改变本项目的问题表述

这篇论文不证明“球场背景也低秩”，更不证明体育小球无需 time correspondence。它证明的是一个更关键的逻辑点：在弱、极小目标场景中，**先保住所有目标细节**并不是唯一合理的第一性原理；先建立稳定背景并在 residual 中定位目标也可能赢。当前项目若不先检验这一竞争解释，便不能把“高分辨率细节 + 长位移 correspondence”当作唯一因果故事。

但直接照抄其解释会犯三个错误：

1. 网球／羽毛球／乒乓球的镜头会摇摄、跟拍、缩放，背景线、人体和球拍也具有运动；TenRPCANet 的低秩／稳定上下文假设在这些条件下是否仍可成立，论文没有给出证据。
2. 模糊球的线状外观、网线、广告文字和反光点不一定是小而平滑的 PSF 点；不能把其“PSF 局部平滑”直接当 BlurBall 的物理观测模型。
3. 它输出的是 mask，中心位置、visibility、异常候选覆盖和精细 correspondence 都未测。即使最终 false alarms 下降，也无法单独证明对球中心定位或大 `rho` 位置有帮助。

因此，这条文献不允许预设 high-resolution current-frame evidence 是 history 收益的必要条件：当前帧很弱、模糊或暂时不可辨时，历史帧也可能提供新的视觉支持。更恰当的待检验分解是：当前证据已可读时，历史关系是否改进候选排序／细定位；当前证据很弱时，历史是否能在**没有 GT query**的条件下提供有效候选；以及稳定背景／变化抑制是否已解释其中任一收益。它也提示未来按相机／背景条件切分：若一个时间模块主要在稳定背景里提升，可由 background suppression 解释；若在连续相机运动中仍有独立收益，才支持更强的关系表示解释。

**当前决定。** 不将低秩 Tensor-RPCA 系统加入训练。已有预测若配有事前定义、可核验的背景/相机 proxy，可以用于描述错误相关性、提出下一假设；本项目当前没有由本次文献建立这样的 proxy。分组本身不能验证“背景建模可替代对应”的反事实，也不足以决定新增背景支路。真正提出背景机制后，再选择能够改变该决定的最小对照。

## 4. CoWTracker：高分辨率 long-range tracking 不需要显式相关体

### 可证实事实

CoWTracker 以第 0 帧每个 pixel 作为 dense query，预测该点在所有 target frame 的位置、visibility 和 confidence。它从零位移开始，反复在 target feature 上以当前位移 bilinear warp，拼接 reference feature、warped feature、状态和位移后，用交替的空间／时间 attention 更新 residual displacement。它没有 cost volume；论文将 head 描述为随 frame 数、feature 位置数和迭代次数线性缩放，但该概括不能视作任意帧数／分辨率下的全模型复杂度界。[方法 §3](https://arxiv.org/html/2602.04877v1#S3)

这一区别会改变计算结论。作者公开实现中的 temporal self-attention 对每个空间 token 在窗口的**全部** `T` 帧做 attention，未施加 causal／局部时间 mask；故每个 temporal block 的 attention interaction 随 `T` 为二次，而不是线性。每帧的 spatial attention 也在该帧全部 refinement token 上运行，仍有其空间二次项；feature warp 本身才是按 `T`、位置数和迭代次数线性。因而“无需 cost volume”不等于“全模型可任意扩展时空分辨率而线性”，只可说在固定短窗／固定 tokenization 下，省去了全局 all-pairs correlation tensor。[作者实现：temporal attention](https://github.com/facebookresearch/cowtracker/blob/main/cowtracker/layers/temporal_attention.py)；[video transformer](https://github.com/facebookresearch/cowtracker/blob/main/cowtracker/layers/video_transformer.py)

作者采用 VGGT backbone、DPT upsampler 和原 RGB 的小 U-Net，将 feature 提升至 stride 2；K=5 次迭代。训练使用 Kubric dense trajectory，visible／occluded position 用 Huber，visibility 和 confidence 另有 BCE。特别是 confidence 的 teacher 是“预测位置距 GT track 小于 12 px”的二元判定，visibility 是另一输出；它并不定义“没有对应”。[实现 §4.1](https://arxiv.org/html/2602.04877v1#S4.SS1)

作者在 TAP-Vid、RoboTAP 和 optical-flow 数据上报告结果，并在同一头内消融 backbone、upsampler、indexing resolution 与迭代：移除 upsampler 会下降，stride 1/2 的 indexing 高于 1/4--1/16；5 次迭代优于不迭代。其运行测量在 H100、336×560、FlashAttention-3 下进行，短 clip 约超过 30 FPS。它不能转换为本项目 DINO、12GB GPU、视频解码和自动球定位的端到端 FPS。[消融、运行附录](https://arxiv.org/html/2602.04877v1#S4.SS4) [附录 §8.2](https://arxiv.org/html/2602.04877v1#S8.SS2)

### 对本项目的精确含义

CoWTracker 是“高分辨率、large displacement、有限 cost”矛盾的最近通用解法之一。它禁止下列写法：

- “为了在高分辨率处理大位移，必须建立显式全局 correlation/cost volume”；
- “不用 all-to-all correlation 就不能建立长程对应”；
- “高分辨率 feature upsampling 只会做视觉插值，不能影响 tracking”。

但它没有解决本项目最难的一步：dense tracks 告诉每个 reference pixel 去哪里，**没有判断哪个 reference pixel 是球**。这不是 GT-initialized point tracker，却仍不是自动单球检测器；从它的 dense field 再选球中心会引入一个独立 candidate／detection 问题。报告模型的所有 target frames 联合进入 temporal attention，未见 causal mask；对窗口中间帧来说会使用未来 target frame，不能以其离线分数宣称在线可行。

它同样不构成“多假设”的证据。每轮 warp 在每个位置采样一个当前估计地址；多轮 residual refinement 不是显式保留多个互斥球位移候选。它的 confidence 又是 track-error 阈值 teacher，不能无修改套作 ball correspondence 的 no-match probability。

**当前决定。** 将 CoWTracker 作为未来若要拒绝高分辨率 cost volume 时的必要文献对照，而不下载／复现其模型。任何轻量球方法若主张“warping-only”，还须实测初始零位移如何跨越真实球的大 displacement，并与当前 repeat-current／history 介入结果相连；不能借用 CoWTracker 的 dense tracking 指标作替代证据。

## 5. 综合后的实验和论文纪律

四项证据给出的可执行判断不是“再加四个 baseline”。它们列出不同提案可能面对的竞争解释，以及在该解释与具体提案真正相关时应测什么：

| 可能待区分的解释 | 最近近邻 | 与具体提案相关时的诊断 | 不能借用的结论 |
|---|---|---|---|
| 历史帧仅以变化／appearance 提供增益 | Frame Dynamics | 完整 history 与 repeat-current；只有引入不可吸收差分操作时，才测该操作及当帧定位／候选误检 | 完整三帧线性 stack 上的 signed difference 不自动增加函数类。 |
| 沿估计运动运输弱证据 | PACT | 候选或关系是否在真实目标出现时可达；transport 前后 candidate recall、位置误差、拒绝后 fallback 行为 | dense event-instance velocity 不能由单个球中心伪造。 |
| 背景结构／稳定上下文已解释收益 | TenRPCANet | 在背景竞争、camera-motion proxy、位移与可见性分组上查看益处；同一真球的 current-frame 可读性 | IR／星图的低秩假设和 8 帧离线窗口不自动适用于体育镜头。 |
| 不显式相关体也能大位移细定位 | CoWTracker | 明确初始化、每轮 search/warp 地址、预算、因果时间范围；自动发现与给定点 track 分开 | dense all-pixel track 不是 ball detector，confidence 不是 no-match。 |

因此，后续每个具体提案都应明确其待证机制、最近的竞争解释和最小可区分实验，而不是预先把 correspondence 规定为唯一贡献，也不是要求每个提案逐一击败所有不相干的 baseline。若提案研究高分辨率 relation evidence，应保持自动逐帧球中心、时间范围、GT 初始化与伪标签的边界清楚；若提案研究背景、候选或 readout，则相应采用能区分该机制的对照。当前真实 history vs repeat-current 结果会决定哪一条问题值得继续，而不是由本页文献预先决定。

本轮不支持任何“首次 tiny-target motion”“首次 global/local decoupling”“首次 motion reliability”或“首个 high-resolution efficient large-displacement matcher”的表述。研究贡献须在最终实际提出的机制和证据上判断，不能用复杂模块或广泛的文献反例替代这个判断。

## 未解决项与检索充分性

- PACT 的作者仓库在本轮页面上仍写代码／权重将于 2026-09-01 发布，但可见内容不足以审计；不把 README 的预告写成已复现实作。正式 ECCV 版也未从会议论文页逐页核验。
- TenRPCANet 和 Frame Dynamics 未在本轮复现；前者未取得作者代码，后者论文给出的链接需要在真正准备复现时再确认可用性与数据许可。搜索未命中不是“没有实现”的证据。
- CoWTracker 已有公开推理仓库，却使用 VGGT、DPT、raw-image U-Net、dense full-frame field及 H100 评测；其显存、窗口与因果部署面均未映射到当前机器，故目前没有安装任何依赖或权重。
- 本文没有重读已在其他专题完整审查的 FlowIt、EgoSIS、MI-DETR、DeepPro、DMR、CMRTrack、BIRD/STSN，也没有把 RacketVision、TT4D 的已记录数据事实重写。它们仍是互补证据，不能因本页新近邻而被排除。
