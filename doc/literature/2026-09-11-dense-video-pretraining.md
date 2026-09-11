# V-JEPA 2.1：密集预测预训练、可读性与球点对应的边界补读

日期：2026-09-11。范围：补读官方 V-JEPA 2.1 arXiv v3 全文、附录 A--D 与作者仓库的可固定源码；不运行模型、不下载权重或数据。本笔记展开 [modern_motion_evidence.md](modern_motion_evidence.md) 中 V-JEPA 2.1 的一行，避免重复把“dense feature”写成小球定位或对应已经成立。

## 结论先行

V-JEPA 2.1 是“视频 SSL 可以通过**同一 clip 内、按时空格定位的全 token latent prediction**来改善 dense readout”的直接先例。它使可见 context token 也接受 target 约束，并在四个 encoder 深度施加预测损失；作者用冻结 backbone 的线性 depth/semantic-segmentation probe，以及给定初始 mask 的 VOS label propagation，实证局部空间结构和时序稳定性改善。

这仍不能推出几像素球可由 V-JEPA 2.1 自动发现、精确读出中心或正确跨大位移匹配：论文没有 tiny-object/球大小分桶、中心误差、点轨迹、flow EPE、候选覆盖或 correspondence accuracy。只有 depth/semantic segmentation 是冻结 feature 上的线性 readout；VOS 是**给定第一帧 GT instance mask**的 cosine k-NN 传播；最接近框定位的 Ego4D STA 则是带 RPN/Faster R-CNN head 的端到端训练。三者均不等于球点的视觉信息必然存在或普通 cosine 已经正确。

## 版本、作者、读取范围与源码状态

Lorenzo Mur-Labadia、Matthew Muckley、Amir Bar、Mido Assran、Koustuv Sinha、Mike Rabbat、Yann LeCun、Nicolas Ballas、Adrien Bardes，**“V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning”**，arXiv:2603.14482v3，cs.CV。[arXiv 记录](https://arxiv.org/abs/2603.14482v3)显示 v1 为 2026-03-15、v2 为 2026-03-17、最新 v3 为 **2026-06-11**，未列 conference/journal reference，故本次按预印本 v3 引用。[官方 HTML 全文](https://arxiv.org/html/2603.14482v3)与[PDF](https://arxiv.org/pdf/2603.14482v3)是方法、实验及附录证据。

本次实际读到正文 §2--§3、附录 A（pretraining）、B（distillation）、C（全部 evaluation protocol）、D（multi-layer 与 pretraining resolution ablation）。作者[官方仓库](https://github.com/facebookresearch/vjepa2)可访问；为避免移动分支，源码证据固定至 2026-03-23 提交 `204698b45b3712590f06245fbfba32d3be539812`。该提交早于论文 v3，仓库没有声明它与 v3 完全同版；下述代码只用来核对论文已述机制，不用它补写 v3 未披露事实。

## all-token dense predictive loss 实际做了什么

### 遮挡、target 与 token 地址

原 V-JEPA 2 的 student `x` 是同一个 input 的 masked view：encoder 仅处理可见 token，predictor 再接上带时空位置的 learnable mask token；EMA `y`-encoder 处理未遮挡的同一 image/video。原始 \(L_{predict}\) 只在 masked index 集 \(M\) 上，以 stop-gradient target 的 L1 距离训练。[§2.1，式 (1)](https://arxiv.org/html/2603.14482v3#S2.SS1)；作者的诊断是这让可见 context token 无须保持局部信息，可能变成全局汇聚器。[§2.2](https://arxiv.org/html/2603.14482v3#S2.SS2)。

2.1 保留 masked-token \(L_{predict}\)，并新增对可见 index 集 \(C\) 的

\[
L_{ctx}=|C|^{-1}\sum_{i\in C}\lambda_i\lVert P(E(x),\Delta_y)_i-\operatorname{sg}(E(y)_i)\rVert_1,
\qquad L_{dense}=L_{predict}+L_{ctx}.
\]

其中 \(\lambda_i=\lambda/\sqrt{d_{min}(i,M)}\)：距任一 masked token 更近的 context patch 权重更大。[§2.3.1，式 (2)--(3)](https://arxiv.org/html/2603.14482v3#S2.SS3.SSS1)。它是**同一 clean clip 的 latent reconstruction/denoising**，并无“指定下一帧”或球中心 target；所有 token 被监督也不是每个 token 被赋予对象 ID、匹配标签或物理位移。

视频 tokenizer 是 `16×16×2` 的 3D patch/tubelet，故给定输入大小时原始 token 格具有 `(time-tubelet, row, column)` 地址；mask index 也按这个三维格计算距离。论文图 4 明说 mask token 带 spatio-temporal positional information。[§2.3](https://arxiv.org/html/2603.14482v3#S2.SS3)。固定源码进一步显示 patch 数为 `(T/2)×(H/16)×(W/16)`，并把 `is_causal` 传给各 transformer block：`facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812:app/vjepa_2_1/models/vision_transformer.py:L79-L90`、`L115-L125`；mask 距离代码从 flattened index 重建 frame/height/width：`facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812:app/vjepa_2_1/models/utils/masks_dist.py:L6-L74`。

这只建立了 token 格与其训练位置的显式对应，**不证明**最终 contextual token 内每个局部视觉细节可无损恢复、能输出点坐标，或为某个小球提供唯一跨帧匹配。相反，也不能因最终模型没有直接输出 `(x,y)` 而断言信息已消失；是否能由指定层/指定读出器线性读出，必须用本项目点监督 probe 测量。

固定仓库的 primary pretrain config（与 v3 的逐项一致性未声明）使用两类全时长 3D block：8 个空间面积 0.15 的 block 与 2 个空间面积 0.7 的 block，二者 `temporal_scale=(1,1)`；16 帧、256 输入、patch 16/tubelet 2 时相当于 8 个时间格、16×16 空间格。`predict_all: true`、mask 参数与 `is_causal: false` 见 `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812:configs/train_2_1/vitg16/pretrain-256px-16f.yaml:L17-L45`、`L72-L136`；训练循环分别对 `masks_pred` 和 `masks_enc` 求 loss，见 `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812:app/vjepa_2_1/train.py:L677-L703`。这解释了“all token”是**loss 覆盖 visible+masked**，不是不遮挡输入、也不是任何跨位置 candidate volume。

### 中间层监督

2.1 从三个 intermediate encoder block 加最终输出，按 channel concat 后由 MLP 融合并降维；predictor 产生四个 level 的输出，\(L_{predict}\) 与 \(L_{ctx}\) 都施加在每一级。[§2.3.2](https://arxiv.org/html/2603.14482v3#S2.SS3.SSS2)。论文附录 A 给出 pretraining 的四个等间隔层；固定源码对 24-block encoder 使用 `[5,11,17,23]`，见 `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812:app/vjepa_2_1/models/vision_transformer.py:L154-L159`。它是 encoder 各层的**训练 target**，不是推理时把四层自动拼成像素定位头。

## 2.1 相对 V-JEPA 2：真正变更与不可归因部分

论文把 2.1 定义为整套 recipe，而不是只替换一个 loss：

| 变化 | 论文证据 | 归因边界 |
|---|---|---|
| dense context loss | visible token 也以距离加权 L1 对齐 EMA target | 单加它令 ADE20K 从 22.2 至约 33.8/33.9 mIoU、NYUv2 从 0.682 至约 0.473 RMSE，却令 SSv2 从 72.8 降至 62.5；它不是无条件更好。 |
| deep self-supervision | 四层 target/prediction | 累积 ablation 加它后 ADE20K 38.6、NYU 0.463、SSv2 72.1；说明可恢复部分 global performance，但不能把之后的最终增益全归到它。 |
| data / tokenizer / scale / cooldown | VisionMix-163M、图像 2D 与视频 3D tokenizer、ViT-G、较高分辨率 cooldown | 图 5/表 1 是按此顺序的累计 recipe：最终 47.9 mIoU、0.307 RMSE、77.7 SSv2 同时还混入数据、模型与分辨率变化。它不隔离“loss 本身”在 tiny object 上的效应。 |

[§2.3 与 Fig. 5/Table 1](https://arxiv.org/html/2603.14482v3#S2.SS3)是上述累积消融的一手来源。附录 D.1 还显示：有 deep supervision 的 ViT-L 只用 last layer 已较强，四层拼接的额外收益较小；无该监督时反而依赖多层 readout。[Table 13](https://arxiv.org/html/2603.14482v3#A4.T13)。附录 D.2 仅说明 384×384 cooldown 在作者任务上优于 256×256，[Table 14](https://arxiv.org/html/2603.14482v3#A4.T14)，不说明 3--5 px 球在 stride-16 格中仍可分。

## 可得 B/L、蒸馏与 dense evaluation 的实际训练条件

作者项目列出 384 输入的 ViT-B/16（80M）与 ViT-L/16（300M）2.1 checkpoint，及 ViT-g/16（1B）、ViT-G/16（2B）。[固定 README 的模型表](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md#v-jepa-21-pretrained-checkpoints)是“权重可得”的证据，**不是本项目已下载或适配完成**。

附录 B 说明 B/L 来自 ViT-G teacher 的 distillation：EMA target 改 frozen teacher，predictor 改为 12 block；**不使用 deep self-supervision**，仅在最后 encoder layer 计算 loss。第一阶段是 16 帧 256，第二阶段是 64 帧 384。[Appendix B](https://arxiv.org/html/2603.14482v3#A2)。因此 B/L 不能被不加说明地当作拥有与 g/G 完全相同的四层监督事实；论文的主 dense Table 8 列的是 2.1 ViT-g/G，并没有 B/L 的相同 dense benchmark 行。

下游评估也必须分开：

| 论文任务 | backbone 是否冻结 / 另训什么 | 对本项目能支持什么 |
|---|---|---|
| NYUv2/KITTI depth、ADE20K/Cityscapes/VOC segmentation | frozen **final-layer** patch features；只训带 BatchNorm 的 dense linear projection，且使用 image tokenizer；不取中间层 | 这是“局部空间信息可由指定 frozen readout 线性读出”的最强直接证据。 |
| DAVIS/YouTube-VOS | frozen patch features，无可学习参数；输入第一帧 GT mask，以 cosine similarity、top-k、温度和 first/past frame context 的 label propagation；超参在 DAVIS train 搜索 | 这是 GT 初始化的时序稳定性/相似度证据，非自动球发现、点轨迹或无标签对应。 |
| Ego4D short-term object-interaction anticipation | 四层 attentive probe、frame-guided pooling、FPN、RPN、RoIAlign；**end-to-end** Faster R-CNN/verb/TTC losses | 有 bbox/未来交互 mAP，但不能归结为 frozen V-JEPA token 自发给出精确框，更不是点读出。 |
| 分类、EPIC action anticipation | frozen encoder 上另训 attentive/focal-loss probe | 语义/动作预测，不给像素 correspondence 证据。 |

depth/segmentation 的 frozen-linear 规则在[§3.5](https://arxiv.org/html/2603.14482v3#S3.SS5)与[Appendix C.1--C.2](https://arxiv.org/html/2603.14482v3#A3.SS1)；VOS 的初始 GT 与非参数协议在[§3.6](https://arxiv.org/html/2603.14482v3#S3.SS6)和[Appendix C.3](https://arxiv.org/html/2603.14482v3#A3.SS3)；STA 的最后帧空间对齐、RPN 和端到端训练在[Appendix C.4](https://arxiv.org/html/2603.14482v3#A3.SS4)。附录 C 还覆盖分类与 action anticipation；正文另报告 VQA、real-robot grasping 与 navigation world-model planning。这些都已读到，但不额外列作本项目应实现的 baseline。

## 视频时间语义：窗口双向性不等于目标必看未来

预训练输入是完整 clip 的 masked/clean 两个 view。附录 A 的 primary phase 是 16 帧、4 FPS、256×256，cooldown 是 64 帧、384×384；patch size 16、tubelet 2、predictor 24 block。[Appendix A](https://arxiv.org/html/2603.14482v3#A1)。早于 v3 的固定仓库 primary config 将 `is_causal: false`，且 `local_window=(-1,-1,-1)`；它支持这样一个受版本边界限制的实现判断：就**提供给该 encoder 的 clip**而言，attention 未受因果遮罩约束。这是双向 representation learning；论文 loss target 是同 clip 的 token，不是“窗口最后一帧的未来状态”。

下游 target 必须另看协议。Ego4D STA 要让预测的空间坐标与**最后输入帧**对齐：full-clip 3D tokens 作为 key/value，最后帧 2D tokens 作 query；但该任务的 video clip 是发生在待预测 action 前的 context，默认 anticipation gap 为 1 秒。[Appendix C.4](https://arxiv.org/html/2603.14482v3#A3.SS4)。因此不能只见 backbone 是 non-causal，就断言这个“末帧 anchor”的输出看到了该末帧之后的帧；论文给出的 temporal order 正相反。另一方面，VOS label propagation 明确只从第一帧和一小组**过去**帧向 subsequent frame 传播标签，但论文没有在该段拆开说明送入 encoder 的 temporal window 是否另含未来帧，故不能把 VOS 数值标作严格端到端因果保证。

## 哪些证据接近定位，哪些仍不存在

论文量化了 segmentation/depth、VOS `J&F`，以及 STA 的 box-IoU/时间/语义综合 mAP；其中 VOS 的 69.0 DAVIS-17、72.7 YouTube-VOS 是给定初始 GT mask 后的 patch-similarity propagation，作者还以“fast motion”例图展示 mask 连续性。[§3.6](https://arxiv.org/html/2603.14482v3#S3.SS6)。这些是空间结构/实例掩码的有价值邻证。

但全文未报告 tiny-object 或对象尺度分桶、自动点定位、中心误差、点轨迹正确率、真中心候选 rank/coverage、flow EPE，亦未把 VOS 的 dense mask 分数分解为小物体或大位移对应质量。traffic-light contour、VOS fast-motion mask 等属于定性图示，不是球点量化。故不能将 “temporally consistent dense features” 改述为“高速微小球仍有可读中心”或“跨帧 cosine correspondence 已可靠”。

## 计算量：预训练 recipe、下游系统开销与通用推理成本不能混用

论文的 135k iteration primary 加 12k cooldown、16→64 帧和 256→384 分辨率是**预训练**投入，不是部署 inference latency。[§2.3](https://arxiv.org/html/2603.14482v3#S2.SS3)与[Appendix A](https://arxiv.org/html/2603.14482v3#A1)。固定公开配置含 `nodes: 16`、`tasks_per_node: 8`（primary）或 `nodes: 32`、`tasks_per_node: 8`（cooldown）。这些仅是训练配置声明，不能当作已核验的实际训练时、端到端 FLOPs 或本项目预算。

论文报告的 navigation planning 10× speed-up 是 V-JEPA representation 上另训 diffusion world model 的 sampling/planning 比较，不是 encoder 单次 forward；STA 也含 4 attention blocks、FPN/RPN/RoIAlign 与端到端训练。官方全文没有给 B/L/g/G 在“单窗口 tiny-ball 自动定位”条件下的 FPS、显存或延迟。因此不能用这些系统级数字称 V-JEPA 2.1 是本项目的低成本 backbone/motion module。

## 对本项目 DINO 表示适配的可证伪边界

1. 不能把“video SSL + dense all-token loss + intermediate supervision”当成新机制；V-JEPA 2.1 已覆盖。可研究的问题更窄：在同一输入尺度、中心监督、层选择、读出器与训练条件下，DINO 各层对几像素球的可读性如何，及 video-dense teacher/adapter 是否改善该点。某个冻结 probe 失败只说明该层、读出器和训练条件下尚未读出，不能证明信息已丢失。
2. 不可将 DINO 的当前 failure 直接归因于“缺少 temporal prediction”：2.1 的最终 gains 混合 context loss、deep supervision、数据、tokenizer、模型规模和 cooldown；而 B/L 蒸馏又不含四层深监督。任何适配器的因果归因需逐项消融，并保留 RGB-only/current-frame probe。
3. 先做冻结、明确层和空间格的点监督 probe；若候选 B/L 或 g/G 也不能在球中心邻域稳定读出，停止把它称为 dense-feature 修复。若只在线性 probe 中改善，也只能说该模型/层/输入下的球点可读性增强，不能说已解决跨帧对应。
4. 后续若测 temporal relation，须独立报告真中心是否被候选覆盖、其 rank/误差和 current-frame 定位；VOS 的 GT 初始化 mask 与 STA 的端到端 box detector 不能替代这三段证据。暂不据此堆叠 predictor、FPN、RPN、world model 或 dense-pretraining module。

本次证据面是 v3 论文及其 A--D 附录、再加早于 v3 的固定作者代码；没有把作者的 PCA、mask 或 VOS 图示扩写成小球结论，也没有主张完成任何本地特征 probe。
