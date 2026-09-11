# 现代 motion / backbone 文献核验（2026-09-09）

## 范围、结论与证据等级

本笔记只核验蓝图所列的现代 motion 表示、光流与 DINOv3 骨干，不评价本地代码或替代依赖。初次检索日期为 **2026-09-09**；优先使用论文原稿、会议proceedings、作者官方仓库和模型卡。确认会议身份与读取正式刊载版分别记录：官方会议项目可确认身份，但不自动等于已取得出版社定稿；arXiv可访问也不自动确认接收。此组包含正式会议论文和近期预印本，检索与阅读深度按条目说明；文献扫描不等于数值结果已被本地复现。

核心结论：蓝图提出的“高分辨率局部证据 + 有限预算的大位移对应”依然是合理问题，但绝不能以“没有人做局部 motion / 高分辨率 match / motion latent / 多阶关系”为创新点。能争取的贡献必须落在：**对极小、稀疏、快速、易混淆目标，如何在已标注中心监督下同时测量并改善候选覆盖、正确对应排序、当前帧定位，而非仅改善视频语义任务**。

**2026-09-11局部补读：**本次深化MOSS、Midway、V-JEPA 2.1、What Moves?、MoAlign、COMET、MotionEnhancer与ReMoRa的全文机制、时间语义和评价边界，并修正末尾空间探针的过强推论；其余条目的检索日期和阅读限制仍按原记录。详见[多阶关系](2026-09-11-higher-order-motion.md)、[预测式运动潜变量](2026-09-11-predictive-motion-latents.md)、[密集视频预训练](2026-09-11-dense-video-pretraining.md)、[区域运动和全景参考](2026-09-11-region-motion-reference.md)、[运动子空间](2026-09-11-motion-subspace.md)、[方向差分](2026-09-11-directional-differences.md)、[扩散注意力教师](2026-09-11-diffusion-motion-teacher.md)与[压缩视频运动](2026-09-11-compressed-motion.md)，不表示已复现这些方法。

## 事实核验表

| 工作 | 状态（截至检索日） | 一手证据所建立的机制 | 对球定位的可迁移价值 | 不能推断 / 创新冲突 |
|---|---|---|---|---|
| [WAFT](https://arxiv.org/html/2506.21526v3) | **ICLR 2026**；arXiv v3 2026-02-06 | 将 RAFT 类显式 cost volume 换为迭代的高分辨率 feature warping；论文明确说 warp 本身不显式比较多候选，长程关系依赖 Transformer update。 | 是“不要假定高分辨率相关体必需”的最强反证；可作为教师审计或强 flow 参照。 | 其目标是稠密 flow，不是 2--5 px、常模糊小球的中心 correspondence。完整 WAFT 是 input encoder + DPT update + 多轮迭代，不能拿“无 cost volume”包装成轻量模块或宣称简单替换就解决大位移。 |
| [Midway Network](https://proceedings.iclr.cc/paper_files/paper/2026/file/8b301f565225e18c02af54308789dae4-Paper-Conference.pdf) | **ICLR 2026**；2026-09-11补读全文及固定作者源码 | source/target两帧生成10个全局latent token，条件化多层dense teacher-feature prediction；student与dynamics联合训练，teacher由EMA更新。 | 是dense predictive video-SSL的直接先例；分割与有监督微调flow的结果使它值得作为表示学习近邻。 | latent没有显式候选地址；全图token loss和flow EPE均不证明小球可读性。pair输入的因果性取决于输出锚在哪一帧，不能把已看到target的估计称为未来预测。详见[补读](2026-09-11-predictive-motion-latents.md)。 |
| [MoAlign](https://arxiv.org/html/2510.19022v1) | **ICLR 2026官方poster已确认**；本次方法阅读为arXiv v1，2025-10-21；2026-09-11补读全文及附录 | 冻结VideoMAEv2的768维token经64维瓶颈学习预测RAFT伪流，再对齐CogVideoX隐藏特征的时空余弦关系。 | 已覆盖flow目标约束的运动子空间及关系对齐；是DINO表示适配的直接近邻。 | 瓶颈和生成质量收益不证明appearance被排除或运动可辨识；49输入帧、24时间格、23流的目标映射未披露，不能猜RAFT时间stride。详见[补读](2026-09-11-motion-subspace.md)。 |
| [V-JEPA 2.1](https://arxiv.org/html/2603.14482v3) | **预印本 v3，2026-06-11**；2026-09-11补读全文及附录 | 同clip的masked与visible token均受EMA特征目标约束，四层深监督及其他recipe共同改善dense tasks；不是下一帧预测。 | 冻结线性depth/seg读出、GT初始mask的VOS、端到端STA分别提供不同强度的定位邻证。 | g/G主dense结果不自动适用于B/L蒸馏模型；B/L仅末层监督。token时空地址不是点对应，未测tiny-ball；双向attention的因果性须按输出时刻判断。详见[补读](2026-09-11-dense-video-pretraining.md)。 |
| [What Moves?](https://arxiv.org/html/2609.04383v1) | **ECCV 2026官方会议项目已列**；arXiv v1 2026-09-03，Springer定稿/页码未取得；2026-09-11补读全文及固定源码 | 外部region mask形成content query，与完整视频共同生成motion latent；TAPNext伪轨迹用于训练decoder。发布面为encoder。 | full-scene context与region query在actor action/生成控制任务有实际消融；是避免裁剪丢上下文的直接近邻。 | 不自动发现region，也未显式估计或分离camera motion。mask不必人工新标，但候选来源误差须另测；latent与初始化轨迹都不是自动球中心结果。详见[补读](2026-09-11-region-motion-reference.md)。 |
| [MotionEnhancer](https://arxiv.org/abs/2606.06853) | **CVPR 2026已核验**，pp. 2778–2787；2026-09-11补读全文及附录，未找到作者公开实现 | 冻结CogVideoX以QA问题为条件，经inversion/reconstruction离线提取注意力，选择head/token后对齐VLM attention。 | 是扩散attention作语义motion教师的直接先例；时空列可支持文字grounding，但需另测点对应。 | QA改进不证明球定位；原教师依赖问题文本，20–30秒/A100/样本的离线提取须计入。对角/同址筛选不自动等于运动可靠性，也不能预断它必然偏背景。详见[补读](2026-09-11-diffusion-motion-teacher.md)。 |
| [ReMoRa](https://openaccess.thecvf.com/content/CVPR2026/html/Yashima_ReMoRa_Multimodal_Large_Language_Model_based_on_Refined_Motion_Representation_CVPR_2026_paper.html) | **CVPR 2026**, pp. 31845–31855；arXiv v2；2026-09-11补读全文、附录及后期固定源码 | 论文先重编码为384²/16fps/H.264，稀疏I帧RGB与P/B帧MV分工；CoTracker3目标训练MV精炼器，再用于长视频QA。 | 是压缩域motion与appearance分工的直接先例；教师、编码、提取和推理需分别计时。 | Tennis发布包没有原始码流，后期提取代码也不等于论文重编码配方；MV参考关系和B帧未来依赖不能由模型窗口截断保证。未测球点误差。详见[补读](2026-09-11-compressed-motion.md)。 |
| [COMET](https://arxiv.org/html/2608.21030v1) | **预印本 v1，2026-08-21**；作者标注ACM MM 2026接收，正式刊载未核；2026-09-11补读全文 | 灰度一至五阶有限差分、幂/聚合及归一化混合，独立ViT经同位置时间attention融合；另有固定文字答案的正反序GRPO。 | 保留方向敏感变化作为对应方法的竞争解释；应选择兼容且不冗余的廉价原语，而非强制复刻完整MLLM。 | 完整五阶构造至少六帧，不能原样塞入当前三帧；语义消融不证明跨位置位移，参数百分比不是定位延迟。详见[补读](2026-09-11-directional-differences.md)。 |
| [GMoT](https://arxiv.org/html/2607.16322v1) | **预印本 v1，2026-07-15** | 对微手势的 spatial token 做运动感知门控/选择，任务为 MLLM recognition。 | 支持“小而弱的动态会被 pooling 稀释”的现象性动机。 | 微手势是较稳定人体部位的小振幅，不是几像素、可跨越大量 ball-size 的无纹理小球；gate/route 已不是新概念。 |
| [Motion-as-Prompt](https://arxiv.org/html/2608.11655v1) | **预印本 v1，2026-08-12** | 从原视频恢复 dense point trajectories，按 motion 选帧并将轨迹画回稀疏 frame 输入，供 frozen MLLM reasoning。 | 强调 `query/track coverage`：轨迹提示有价值的前提是上游点轨迹已经覆盖球。 | 它把 tracking 结果作为输入而非解决 tracking；不能以“显式轨迹提示”主张自动重捕球或没有候选漏检。 |
| [MOSS](https://arxiv.org/html/2604.20760v1) | **预印本 v1，2026-04-22**；2026-09-11补读全文 | 一阶STSS显式枚举局部时空offset，经编码再递归构造高阶关系，最后融合为每query格的feature。 | 直接限制多阶self-similarity本身的新颖性；原始关系轴与融合后特征的区别有助于审查对应读出。 | 高阶不是物理加速度或可靠性；融合后没有显式候选轴不证明地址信息全部消失。也不能将多阶直接等同于多假设后验，球中心coverage/rank仍未验证。详见[补读](2026-09-11-higher-order-motion.md)。 |

## DINOv3：可确认的骨干事实，而非性能许诺

官方 [MODEL_CARD](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md) 和仓库代码（本次核验 commit `6876159a11b4df116f30f667f8c9888617df0751`）支持以下事实：

* DINOv3 提供 web LVD-1689M 上训练/蒸馏的 ViT-S/S+/B/L/H+/7B 和 ConvNeXt-T/S/B/L；名称 `dinov3_vits16`、`dinov3_convnext_tiny`、`dinov3_convnext_small` 都存在。
* **ViT-S/16**：patch size 16、384 dim、12 blocks、6 heads；空间 token lattice 是 `H/16 × W/16`。输入长宽须是 16 的倍数，否则官方代码将裁到较小倍数。因而它没有原生的 stride-4 shallow map；所谓 high-resolution detail branch 必须是另加 adapter/side branch，不能写成 ViT 原生多尺度输出。
* **ConvNeXt-T** 是 `[3,3,9,3]` blocks、`[96,192,384,768]` channels；**ConvNeXt-S** 是 `[3,3,27,3]`、相同 channels。官方 stem 为 `4×4, stride=4`，后三层均为 `2×2, stride=2`，故四 stage 的 nominal strides 是 **4/8/16/32**。这使前两层较适合做球点 readout/probe，但仍不能保证 3 px 球在 stride-4 feature 中可分。
* 官方模型卡称其有高质量 dense features，并报告 dense evaluation；这只构成“值得作冻结特征 probe”的理由。它不是 ball detector，未报告运动 blur、跨帧 correspondence 或高速 tiny-target protocol。

可复现的外部源码证据（补充，不替代模型卡）：`facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751:dinov3/models/convnext.py:L155-L179`（stem/downsampling）；`facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751:dinov3/models/convnext.py:L319-L336`（T/S 配置）；`facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751:dinov3/models/vision_transformer.py:L332-L341`（ViT-S）。引用论文时仍应以 [DINOv3 paper](https://arxiv.org/abs/2508.10104) 和模型卡为主。

## 对论文设计的对抗性结论

### 1. 不可再宣称的“创新”

以下单独提出均会被上述工作或更早 STSS/flow 文献直接击穿：高分辨率 warping、无 cost volume flow、motion latent、motion subspace alignment、全局上下文下的局部 motion、多阶 self-similarity、appearance/motion 双分支、motion gate/token selection、显示轨迹作为 prompt、用 dense video SSL 改善局部特征。若论文采用任一项，贡献必须是具体的 **tiny-fast 失效诊断 + 机制 + 对应的实证**，而不是重新命名通用模块。

### 2. 目前最可防守的假说

在同一输入、同一 DINOv3 feature budget、同一定位头下：

1. 先测球中心附近的当前帧可读性（不同 layer/stride、不同输入分辨率）。若探针失败，只能说明在所测层、读出器和训练条件下尚未读出球；不能据此断言所有跨帧匹配都没有可用证据。
2. 在有连续真中心标签的帧对，单独报告真实位移是否落入候选集（coverage/recall）、真实候选 rank 或 matching NLL；随后再报告由 motion 带来的当前帧中心误差改进。这把“搜不到”和“搜到但认错”分开。
3. 按当前机制选择已有RGB/三帧对照及兼容的低成本时序竞争方法，不机械实现3D、差分、correlation、教师的全部组合。2026-09-11补读COMET后收紧：完整MLLM不是球定位必做项；归一化后线性首层前追加signed差分还可能被已有stack吸收，见[已完成的线性归因](2026-09-10-causal-baselines.md#本次采用与不做的归因)。若兼容的简单变化对照已解释增益，复杂对应方案就需另给机制证据。
4. 若声称 multi-hypothesis，必须比较 single soft-argmax、top-k/mixture 与一阶 relation；报告 ambiguity bucket（线条、反光、遮挡、blur、camera pan），而非只报总分。
5. 对 flow/point-track teacher，仅在中心邻域把预测 displacement 与真实**中心—中心**位移做可用性审计；不要把中心位移复制成 dense flow GT，也不要用全图 EPE 掩盖球区域失败。

这些测试可以推翻论文想法，正是它们有研究价值的条件。任何一个新模块都至少应有一个与其主张对应的可失败诊断：候选覆盖、rank、错误拒绝/校准，或完整端到端延迟；只有 endpoint accuracy 的增益不能判定表示机制。

### 3. camera motion 的最小边界

What Moves? 明确把局部 motion 定义为相对全局 reference（含相机/scene layout），但它并未解决自动对象发现。第一版保持 current-frame appearance 路径独立，固定机位主实验，并把连续 pan/zoom 作为分桶。只有在该分桶中定位确实下降、且全局 reference 条件真的提高对应 rank/coverage，才值得引入轻量全局 context/补偿；不要预设 SLAM、平面 homography 或“background subtraction”是正确答案。

## 额外检索（2025--2026，非穷尽）

检索式覆盖 `motion representation / localized motion / dense video SSL / motion latent / tiny object optical flow / large displacement small object`，在 arXiv 与官方会议入口进行；它是高风险近邻扫描，**不是完整 systematic review**。本轮没有发现可替代“球点大位移 correspondence”核心空白的、已确认会议版 2025--2026 方法；但有两条应补入蓝图的风险线：

* V-JEPA 2.1 已将“dense video pretraining + intermediate supervision + locally readable feature”做成明确主题，因而不能泛称“video pretraining 忽略 dense local motion”。可研究的是该表征在 tiny-ball 尺度的实际失效及低成本修复。
* What Moves? 已把“region-conditioned local motion with global context”做成明确主题，不能泛称全景上下文条件的区域运动无人研究。2026-09-11补读同时收紧这个引用：该文未显式完成相机/物体运动分解，不能拿它当此更强命题的直接证明。**point-supervised、推理无需GT mask、长相对位移且计算受限的实例化**仍是待检验问题；自动发现、对应与定位是否有效，要有各自实证。

未确认项必须保持未确认：MoAlign的正式排版版与作者代码、What Moves?的Springer定稿与页码、COMET的ACM MM最终刊载、GMoT/Motion-as-Prompt/MOSS的接收状态与独立复现。MoAlign已见ICLR官方poster，What Moves?已见ECCV官方项目，不能再写成只有作者自称接收，但这仍不等于读取出版社定稿。正式刊载未核实时，按arXiv的实际年份与版本引用并说明会议状态；不能统一写成2026，例如MoAlign的初稿是2025年。

## 特征上采样：必须新增的空间边界审查

这是一条独立于 motion 的强近邻线。它能改善**低分辨率特征的空间分配/读出**；使用高分辨率RGB的方案还引入额外视觉证据，不能全部视为只对旧特征插值。它们是否恢复小球定位要由实测确定，不能从输出变密推出成功。因而将其作为backbone adapter或空间基线是合理的；若把“feature upsampling”单独写成新贡献，已有工作会直接构成新颖性冲突。（2026-09-11推论修正。）

| 工作 | 状态与一手来源 | 已做什么 | 对本项目的严格解释 |
|---|---|---|---|
| [FeatUp](https://proceedings.iclr.cc/paper_files/paper/2024/file/c5601d99ed028448f29d1dae2e4a926d-Paper-Conference.pdf) | **ICLR 2024** | 通过多视角一致性，从冻结 backbone 的 jittered low-res features 学出任意分辨率 feature；有 feed-forward guided upsampler 和每图 implicit 版本。 | 已经是“把 VFM 深特征作高分辨率读出”的基础先例。它的 dense segmentation/depth 改善不能证明一个原本不可读的 3 px 球会被创造出来；必须在同一输入上以球点 probe 检验。 |
| [LoftUp](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_LoftUp_Learning_a_Coordinate-Based_Feature_Upsampler_for_Vision_Foundation_Models_ICCV_2025_paper.html) | **ICCV 2025**, pp. 9913--9923 | coordinate-based cross-attention 融合 high-res RGB、coordinates、low-res VFM features，并用 class-agnostic masks/self-distillation 建 pseudo high-res target。 | 直接封堵“坐标条件 high-resolution feature adapter”创新。它的 mask-guided pseudo target 可能把小圆球与附近线条/反光同化；不能以其相对大对象 dense-task 数字取代球点 audit。 |
| [AnyUp](https://arxiv.org/html/2510.12764v1) | **ICLR 2026 Oral**（[官方代码](https://github.com/wimmerth/anyup)；arXiv v1 2025-10-14） | 训练一次、推理时 encoder-agnostic 的 local-window attention upsampler；作者明确以 high-res RGB 把 coarse feature 分到像素。 | 最实用的 DINOv3 空间 adapter 候选之一：原文已报告 DINOv2-trained AnyUp 在 DINOv3 ViT-S+ 上的 dense probes。仍不是 DINOv3 tiny-ball/blur/temporal correspondence 证据，且其训练/推理成本必须计入。 |
| [RaysUp](https://arxiv.org/html/2606.22749v1) | arXiv v1 2026-06-22；作者仓库称 **ECCV 2026**，正式 CVF 条目仍需写作前核验 | VFM-agnostic、ray positional encoding 和 cross-attention 的轻量 upsampling；其表格明确含 DINOv3 语义分割/depth probe。 | 是 AnyUp 后的直接效率近邻，且已在 DINOv3 feature 做通用 dense-task 测试；但论文测的是 segmentation/depth，不是 tiny-ball，且 ray/pose 先验与单目体育视频的真实几何没有自动对应。 |

### 对空间路线的可证伪要求

1. 使用同一 RGB input、相同 DINOv3 layer、相同定位 head，比 `native/bilinear`、一个外部 upsampler（AnyUp 或 RaysUp）和原生 ConvNeXt stride-4 feature。上采样器本身不得同时更改 motion 算子，否则不能归因。
2. 报告中心点附近的 heatmap/probe 指标、误报（场线、反光、衣物）、blur 和球尺寸/位移分桶；不要只给全局 mAP 或视觉上更锐的 PCA 图。
3. 若upsampling带来增益而提高输入分辨率也有效，说明读出与尺度都是竞争解释；尚不能据此定位不可逆信息损失。即使所试方案都失败，也只能说这些表示、读出和训练条件未解决问题，不能断言损失发生在backbone最早空间支撑。（2026-09-11修正原文过强归因；参见[表示探针边界](second_pass_representation.md)与[本地实证修订](../research/2026-09-10-empirical-reframing.md)。）

对 DINOv3 最新直接证据的结论也应保守：本轮可靠检到的是 **AnyUp 和 RaysUp 在 DINOv3 的通用 dense probes**，并未检到在高速微小球、模糊球或跨帧 pixel correspondence 上验证 DINOv3 feature upsampling 的官方论文。这个空白可成为实验问题，不能预先称为贡献。

## 最小参考集合

1. [WAFT 原稿 / ICLR 2026](https://arxiv.org/html/2506.21526v3)；[官方代码](https://github.com/princeton-vl/WAFT)。
2. [Midway ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/8b301f565225e18c02af54308789dae4-Paper-Conference.pdf)。
3. [V-JEPA 2.1 v3](https://arxiv.org/html/2603.14482v3)；[官方代码](https://github.com/facebookresearch/vjepa2)。
4. [What Moves? v1](https://arxiv.org/html/2609.04383v1)；[官方代码](https://github.com/CompVis/WhatMoves)。
5. [MOSS v1](https://arxiv.org/html/2604.20760v1)。
6. [DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)；[官方代码](https://github.com/facebookresearch/dinov3)；[paper](https://arxiv.org/abs/2508.10104)。
