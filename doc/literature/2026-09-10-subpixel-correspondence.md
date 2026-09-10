# 细位置读出与跨帧对应：有限近邻核对

核对日期：2026-09-10。本文只为当前决策服务：解释“细位置（subcell / phase）可读性”是否应与跨帧特征对应稳定性分开测量或分开表示。它不是新架构提案，也不是完整的局部特征、光流或 tiny-target 综述。

## 当前本地事实与问题边界

**本地开发实测（跨比赛泛化尚未验证）**：在[当前 DINO 冻结特征、局部 cost 拼接设置](../experiments/2026-09-10-local-cost.md)中，`PCK@8` 三个 seed 均下降；将融合 head 隐层调至 64 后仍下降。项目现转向不加 cost 的 prefix 微调。这只否定了该 cost 拼接和该 head 的组合，**不能**推出“特征不变性必然与精确定位冲突”、cost 一定无效，或已经需要两套表示。

这里的三个概念不能混为一个指标：

1. **细位置读出**：当前帧给定局部区域时，特征或 head 能否区分相邻像素／格内位置，并把中心读到足够小的误差。
2. **对应稳定性**：同一真实球在时间间隔 `Δ` 后，历史位置附近的候选是否在当前查询的排序支持内；它需要两端可信可见中心，但不等于 dense flow。
3. **最终自动定位**：还包含当前帧发现球、候选覆盖、背景假阳性与解码。前两者好不保证第三者好。

球中心标签只给出一个点对 `p_t ↔ p_(t-Δ)`；它不能为球 patch 的每个像素制造 dense-flow 真值，也不能把所有未选位置定为可靠负对应。全图 optical-flow EPE 也不能替代上述球点指标。

## 最接近的一手证据

### 1. RAFT：连续 lookup 与大范围 coarse correlation 已共存

**已读版本：** [ECCV 2020 正式论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf)，方法 §3；[作者实现](https://github.com/princeton-vl/RAFT)。

**论文事实。** RAFT 在 `1/8` 分辨率特征上构造 all-pairs 4D correlation pyramid；第一对空间维度保持不池化，后两维池化以扩展范围。以当前实值 flow 为中心的 lookup 使用 bilinear sampling；论文举例在最低层、半径 4 时覆盖原图 256 px。其输出是单一高分辨率 flow field，并用 learned convex upsampling 从低分辨率迭代结果生成全分辨率 flow。论文明确把传统 coarse-to-fine 的早期错误和漏掉小而快目标作为动机。

**对本项目的含义。** 这说明“大位移搜索 + 连续位置查询/输出”不是新机制，也说明离散格点相关响应不必迫使输出只取整像素。但 RAFT 的 matching feature 本身仍在 `1/8` 网格；其 convex upsampling 是 flow 的上下文读出，并未证明 2--5 px 球的原始视觉证据或真对应在该网格仍可判别。不能把 RAFT 在 Sintel/KITTI 的 dense-flow 改善当作球中心证据。

### 2. LoFTR：粗层全局筛选后，用更细局部特征做 sub-pixel refinement

**已读版本：** [CVPR 2021 正式论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.pdf)与[arXiv正文](https://arxiv.org/html/2104.00680)；[作者项目](https://zju3dv.github.io/loftr/)。

**论文事实。** LoFTR 先在 `1/8` coarse feature 上建立 dense matches，并从置信矩阵选出 coarse match；再对每个已选 match 从 `1/2` fine feature 裁局部窗，以相关方式将其细化到 sub-pixel。原文明确把 repeatable interest point 缺失、低纹理、重复图案与 motion blur 视为匹配困难，也明确此细化只作用于被 coarse stage 选中的候选。

**对本项目的含义。** 它是“wide/coarse correspondence 与细定位可以由不同空间尺度承担”的直接先例，不能将这种分工当作新意。它也暴露本项目的关键反证：若当前帧 coarse candidate 没有覆盖球，fine matcher 不会救回它。LoFTR 是静态图像局部匹配／几何任务，非 tiny-ball 自动发现；因此它支持把候选覆盖、对应排序和细位置误差拆开测，却不证明运动球必须训练两套独立表征。

### 3. FeatUp：高分辨率读出可被单独改善，但没有跨帧 repeatability 保证

**已读版本：** [ICLR 2024 正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/c5601d99ed028448f29d1dae2e4a926d-Paper-Conference.pdf)，与 [arXiv:2403.10516](https://arxiv.org/abs/2403.10516)；[作者实现](https://github.com/mhamilton723/FeatUp)。

**论文事实。** FeatUp 面向因 aggressive pooling 而空间分辨率不足的深特征，提出用多视图一致性学习任意分辨率 feature；给出单次前向 guided upsampler 与逐图 implicit 版本，并在 CAM、segmentation、depth 等 dense task 上验证。

**对本项目的含义。** 该工作支持一个温和判断：即使 backbone 深特征用于语义任务，空间读出仍应单独 probe；没有必要把低分辨率 feature 当作已失去一切细位置信息。反过来，FeatUp 并没有检验跨帧同一 tiny blurred object 的 descriptor 可匹配性，不能用其 dense-task 结果证明 ball correspondence 稳定，也不能证明上采样能凭空恢复已在 patch embedding 中混合掉的球信息。

### 4. WAFT：高分辨率 warping 可替代 cost volume，但依赖已有 flow estimate

**已读版本：** [arXiv HTML v3](https://arxiv.org/html/2506.21526v3)；[ICLR 2026 OpenReview 论文](https://openreview.net/pdf/b617c79f679d8fa2e40f2efac0282bfd355316ea.pdf)；[作者实现](https://github.com/princeton-vl/WAFT)。

**论文事实。** WAFT 以高分辨率 feature warping 取代 cost volume，指出高分辨率 cost lookup 的内存问题与低分辨率 indexing 的误差；用迭代更新估计 flow。作者报告 dense-flow benchmark 上的精度、内存与速度结果。warping 在每步按当前 flow estimate 取对应位置的 feature，而不是保留多候选相似度分布。

**对本项目的含义。** 它直接否定“要让细位置参与 motion 就必须拼接／保存 cost volume”的隐含前提；当前 cost 负结果不应被包装成一般性理论。但 WAFT 假定可以通过迭代 flow 获得可用对齐起点，目标是 dense optical flow。球自动发现、弱小球外观的初始化覆盖和多个假匹配的保留都没有被它解决；它不支持把无-cost prefix 微调预先称为更优对应机制。

### 5. OTHR / FlyingTO：tiny-object flow 有独立困难，但其监督与评测不等于球点定位

**已读版本：** [IJCAI 2025 正式论文 PDF](https://www.ijcai.org/proceedings/2025/0136.pdf)，重点为动机、FlyingTO 定义与评测；[作者实现](https://github.com/JaneEliot/OTHR)。

**论文事实。** OTHR 把 tiny-object optical flow 定为独立问题，FlyingTO 中一半对象面积小于 100 pixels；它以方向、明暗极性与时间延迟构造运动信息，并将其接到 flow model。评测包括 object／背景／对象邻域的 dense EPE，而非中心点检测。

**对本项目的含义。** 它支持 tiny spatial support 下通用 flow 机制可能失效、且应该局部评估的判断。它不能推出球的跨帧 descriptor 应与细位置 readout 分支化：球数据常只有中心标签，拖影与体育相机运动也不同。它反而要求项目只在可见且有两端标注的球点对上评价 correspondence，不用全图 EPE 或 OTHR 数字替代。

## 对“应分开表示”的对抗性结论

**已证实的前史。** coarse-to-fine / global-to-local、sub-pixel refinement、高分辨率 feature readout、cost-free high-resolution alignment、tiny-object-specific motion 都已有直接先例。任何后续模块若只是重述其中之一，没有新颖性基础。

**可作的逻辑推论。** 细位置读出与跨帧匹配受不同的可观测失败支配：前者可在单帧、给定球邻域中失败；后者即使单帧读出好，也可因大位移、重复背景、遮挡或 blur 而失败。故下一阶段应把它们**分开测量**，以避免最终 PCK 掩盖候选漏失或错误对应。这个推论不等于“两个任务存在不可避免的 representation conflict”。

**仍待证假设。** 当前 DINO 特征也许能经 prefix 微调同时提高细位置读出和可用 correspondence；也许只提高前者，或二者都不提高。现有三 seed cost 下降无法区分这些解释。更窄的、可推翻假设是：在固定 input、target frame、时间间隔、数据划分和定位 head 下，prefix 微调会提高至少一个独立观测量，而不会只以更强单帧 appearance 掩盖 motion 失败。

对该假设，最小证据应分别报告：同帧中心的局部细位置误差；两端 visible 标签下真历史中心是否位于 relation 的 top-`K`／半径 `r` 支持内；以及完整自动定位误差与漏检。若第二项没有改善，就不能把 prefix 的最终 PCK 增益解释为 correspondence；若只有 oracle 当前球 query 有改善，也只能称 correspondence probe，不能称自动球检测改进。

## 当前研究决策

不因本轮文献或本地负结果立即加 branch、cost volume 或 coarse-to-fine matcher。先将无-cost prefix 微调作为**诊断实验**，检验表示适配是否提高最终定位；单帧细位置与跨帧支持仍须独立测量，不能从最终PCK反推。若细位置上升而 correspondence 不变，优先研究 readout／候选问题；若 correspondence 上升但自动定位不变，检查当前候选覆盖和背景竞争；若两者均无改善，不再将该冻结cost组合的失败单独归咎于融合头，转向更强完整基线与输入限制的具体证据。没有新的错位证据时，不重复已经通过的标签/时间接线检查。

已在其他记录核对、因与本页问题不更近而不重复全文搜索的近邻包括 GMFlow/FlowIt 的 global-to-local matching，以及 DQAligner/MIST 的 tiny-target alignment；见 [通用 correspondence 证据](correspondence_evidence.md) 与 [tiny motion 证据](tiny_motion_evidence.md)。它们进一步限制“wide search / global-local / tiny-motion”层面的首次性，但不能替代本页对细位置与对应的区分。

## 来源与检索充分性

本轮只读五个一手来源，覆盖必要经典（RAFT、LoFTR）和与当前转向直接相关的 2024--2026 来源（FeatUp、WAFT、OTHR）。检索充分性仅限于“是否足以约束 prefix 微调前的解释”；不是对所有 2024--2026 sub-pixel correspondence 工作的穷尽保证。Midway、视频 MLLM 与未直接检验细位置或像素级对应的工作未纳入，因为它们不能改变这一阶段的判别实验。
