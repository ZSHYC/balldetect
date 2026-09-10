# 第二轮表示审查：冻结 VFM 的 correspondence probe 到底在测什么？

**范围。** 本文只补第一轮遗漏的表示与评测边界；不重复 FeatUp/WAFT 等已审查模块，也不提出实现。检索截至 2026-09-09。核心判断是：对冻结 DINOv3/ViT 作 tiny-ball correspondence probe 时，失败既可能来自球证据丢失，也可能来自 patch phase、位置偏置、层选择、归一化和过早 top-1；成功也不能自动证明模型得到可用于当前帧定位的 motion。必须把它们逐项拆开。

## 先给可执行结论

1. **不要只测 final-layer、双线性上采样、top-1 cosine。** DINOv3 自己报告全局训练会损害 dense locality，且其 Gram anchoring 正是为此设；最新 frozen-DINOv3 correspondence 原稿发现 intermediate layer + position debias 后，真对应常在小 top-K 而非 rank-1。
2. **不要把 feature-grid 上采样解释成 sub-patch 定位精度。** 空间插值只改变 readout lattice；它不能增加原 patch embedding 中没有的球-背景可分信息。只有在输入 RGB 引导/另一路 stride-4 feature 被控制后才可能带来额外证据。
3. **显式 motion 不是逻辑上必需的。** 静态 descriptor 的真球 top-K coverage 只是判断它能否提供候选的一个诊断；是否值得引入显式 motion 还必须由同预算下的**当前帧定位增量**决定。低 SNR、无纹理、模糊球会使候选条件失效，不能从普通物体/人脸 correspondence 外推。
4. **预训练污染是可报告的风险而非可由黑盒测试消除的事实。** LVD-1689M 的实例清单未公开；在可获得元数据的范围内记录近重复、日期/赛事分隔和从零/旧监督 backbone 对照，并声明“未能排除 web-pretraining overlap”。membership inference 的阴性结果不能证明没有泄漏。

## 证据卡（8 项）

| # | 一手来源、状态 | 阅读深度 | 证据与对本题的约束 |
|---|---|---|---|
| 1 | [DINOv3 paper](https://arxiv.org/html/2508.10104)，arXiv 2025；[官方模型卡](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md) | **深读**：dense-feature、Gram anchoring、register/outlier 附录 | 原文显示长训练可让 global metric 继续升、dense task 降；patch similarity map 变得不局部，Gram anchor 约束 patch-patch Gram matrix。register tokens 缓解高-norm patch outlier，但不等于消除所有 feature-dimension/局部性问题。故“DINOv3 很强”不足以跳过 layer sweep；probe 应至少取早/中/末层、`norm=False/True` 与相同 head。官方只披露 LVD-1689M 为大规模 web 图像的策展集合，未给可逐实例审计清单。 |
| 2 | [Preserve, Then Resolve](https://arxiv.org/html/2604.23670v3)，arXiv v3，2026-08-30；[代码](https://github.com/LIAS-CUHKSZ/preserve_then_resolve) | **深读**：abstract、Sec.2、Appendix A/E | 以 frozen DINOv3 ViT-L 做几何 correspondence，作者发现 corrected layer-19 中真对应常不居 cosine rank-1，但在 small top-K；K=1→5 在其 NAVI-Wild 分桶增加 correspondence recall 32.0--44.5 pp。它还给出 MKNN 的 recall--association-count 曲线。**直接改变本项目实验：** multi-hypothesis 不可仅作为新意，首先是必须具备的公平 baseline；报告 `K=1..Kmax` 的球中心 coverage/rank 与候选数。局限：相机几何充分的普通物体/跨视角数据，不是无纹理模糊小球；其 RANSAC 解析不能照搬到单球帧对。 |
| 3 | [INSID3 原稿](https://arxiv.org/html/2603.28480)；[CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.pdf)，pp.21638--21648 | **深读**：Sec.3.1、4.2、Appendix A | 原文发现 DINOv3 跨图 similarity 会在相同绝对坐标产生与语义无关的激活；用一张 noise/low-complexity image 的 feature SVD 提取 positional subspace，并投影到其正交补后仅用于跨图 matching，保留原 feature 作图内 grouping。该工作报告 semantic correspondence PCK 最多 +6.6%。故球项目应把 raw 与 corrected DINOv3 分开；但 correction 对球的影响仍未知，且不能把位置去偏的定义搬到图内定位/当前帧外观通路。 |
| 4 | [Making Vision Transformers Truly Shift-Equivariant](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Making_Vision_Transformers_Truly_Shift-Equivariant_CVPR_2024_paper.html)，CVPR 2024；[Reviving Shift Equivariance in ViTs](https://arxiv.org/abs/2306.07470)，ICLR 2024 | **中读**：abstract/method claims | 两篇一致指出 strided patch tokenization、positional encoding、subsampled/window attention 会破坏 shift equivariance；后者明确把 patch embedding、position 和 subsampled attention 列为来源。故对 1--8 px 平移做一次 probe 不是 augmentation 花样，而是有效性检查。不要声称“球移动一个像素，feature 应只平移一个位置”。也不能由此推出改造 equivariant ViT 是论文贡献：已有完整架构先例。 |
| 5 | [Probing the 3D Awareness of Visual Foundation Models](https://openaccess.thecvf.com/content/CVPR2024/html/El_Banani_Probing_the_3D_Awareness_of_Visual_Foundation_Models_CVPR_2024_paper.html)，CVPR 2024 | **中读**：protocol/limits | 该工作把 VFM descriptor 的 3D/跨视角可配准性当作需测量而非默认存在的性质，是 `descriptor quality != geometry` 的直接先例。它支持使用真实中心构成的、带几何容差的 correspondence recall，而不能把 detection heatmap accuracy 当 correspondence 成功。其测试对象可见、纹理/形状较丰富；对小球只提供评测思想。 |
| 6 | [Data Leakage in Visual Datasets](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Ramos_Data_Leakage_in_Visual_Datasets_ICCVW_2025_paper.html)，ICCVW 2025 Findings, pp.6368--6378 | **中读**：摘要/数据重叠定义 | 视觉 benchmark 的图像重叠会破坏公平评估这一点已有专门审查。用于本题的最小含义是：clip/rally 切分只防**下游**相邻帧泄漏，不能保证 DINOv3 web pretraining 未见过公开视频/赛事帧；应把两类风险分表。该工作不是 DINOv3 的 LVD 成员证明。 |
| 7 | [Blind Baselines Beat Membership Inference Attacks for Foundation Models](https://arxiv.org/html/2406.16201v1)，NeurIPS 2024 | **深读**：problem/setup/conclusion | 论文针对未知 web-pretraining corpus 的 membership inference，结论是常用 MIA 评估不足以说明泄漏，且 blind baselines 能击败既有攻击。对本项目是重要的**负面方法论**：不得以“跑了 MIA 没检测到”声称 TrackNet/BlurBall/OpenTTGames 未被 VFM 预训练。可审计的是近重复、公开视频 URL/哈希、发布日期和公开 corpus manifest；不可审计的部分应如实保留。 |
| 8 | [Emergent Region-Level Facial Correspondence in Frozen VFMs](https://arxiv.org/html/2607.14423v1)，arXiv v1，2026-07-15 | **中读，预印本** | 该原稿在 DINOv3 ViT-L/16 人脸视频上发现 correspondence 最强层可在 intermediate block 18；其 final block 的全局混合更利于某些区域、却损害眉眼类细结构。其价值在于支持“layer is a scientific variable”；但人脸有固定拓扑、初始 FaRL labels 和大区域，远比球容易，不能作为 ball-motion 成功证据。 |

## 三个容易混淆的命题

### A. feature interpolation、high-resolution input、high-resolution evidence

它们不是同一件事。

* **interpolation:** 从 `H/16×W/16` token grid 产生更密格 readout；没有新增观测，若独立于 RGB 只能是重采样。
* **RGB-guided feature upsampling:** 可利用原始 RGB 的边缘/纹理把 coarse semantic feature 分配到像素；新增的是 RGB guidance 的使用，不是从 coarse feature 凭空复原球。
* **提高输入分辨率或使用 ConvNeXt stride-4 stage:** patch/stride 前的原始采样支持真的改变；但会改变总计算、目标像素数和预训练分布，必须列为空间预算而非免费 decoder 改进。

最小反证：固定原 RGB、固定 DINO layer 与 head，对比 native-grid readout、bilinear 与仅从同一 feature grid 重采样的版本；再单列 RGB-guided upsampling、真实提高输入和 stride-4 分支，因为它们使用了不同的信息来源。只有前一组可支撑“纯 readout/interpolation”的结论；后两组的增益应归因于额外 RGB/更密原始采样，并分别报告，不能以收益大小判定是否恢复信息。

### B. semantic matching、geometric correspondence、球的 detection

冻结 DINOv3 能把“眼睛对眼睛、嘴对嘴”或物体部件留在近邻，说明 semantic descriptor 有用；却不蕴含一个白色小球在不同背景、不同模糊长度、不同曝光相位下有唯一 descriptor。对于无纹理小球，局部外观近似同质，候选分布是否可辨只能来自：球与邻域的联合外观、当前帧 detector prior、运动历史/物理可行域，或其他观测。若这些都没有，则 correspondence 本身是多解问题；复杂 attention 不会改变观测不足。

因此 motion 不是必需“输入通道”，但必须把问题写对：静态 pairwise descriptor 的 candidate coverage/rank 只说明候选证据是否存在；显式 motion 的必要性取决于它在相同预算下是否带来可重复的**当前帧定位**改善。若静态 descriptor 已进入 top-K，优先审查 ambiguity resolution；若连 top-K 也不覆盖，优先审查证据保留/候选生成，不能只改 soft-argmax head。

### C. patch phase / position bias 与真运动

一个球平移 1 px 时跨过 ViT 16-px patch 边界，patch content 的混合比例突变；position encoding 又可令同样内容在不同 absolute coordinate 的 descriptor 不同。该效应会让“feature 相似度下降”看起来像 motion failure，也会让模型恰好记住图像坐标。DINOv3 使用 RoPE 并训练时做 RoPE-box jitter，仍不能免除在实际小球尺度上测量。position debias 也可能移除对球场先验有用的位置，因此必须在 test-only transform 上报告收益/损失，而非默认开启。

## 不实现模型也应先完成的最小 probe 套件

以下只需公开中心标签和连续帧；都是对最终模型的前置否证，不是额外人工标注。

1. **亚 patch 平移曲线。** 从可见球帧裁固定全图/ROI，对原 RGB 作 `dx,dy∈[-8,8]` 平移（正确更新中心、避免 padding 区评分）。每层分别测：中心 heatmap/logit 位移误差、真中心 descriptor 相对原 descriptor cosine、top-K candidate coverage。用 ConvNeXt-T stage-1 (s=4)、ViT-S/16 中/末层和 bilinear readout；结果按 `(dx mod 16, dy mod 16)` 汇总。目的：识别 patch phase，而不是追求 shift augmentation 高分。
2. **位置偏差 test-only 对照。** 固定图像内容，把同一有效区域平移到多个合法绝对位置；测 cosine map 的 peak、rank、定位误差。raw / 可复现 INSID3 correction / 简单 per-channel normalization 三组。INSID3 subspace 只用训练/验证数据或无语义合成图像预先估计；不要用测试中心或测试标签选择 rank。
3. **候选曲线而非单点。** 对每个有相邻标签的可见帧对，以真实 `p_t` query，统计 `recall@K`、mean reciprocal rank、候选数量和 false-candidate 类别，K=1,2,3,5,8。用真实中心只作**诊断 oracle query**；另报 detector-query coverage，避免把 oracle 结果误称完整系统性能。
4. **静态-时序必要性检验。** 保持 feature/head 容量，比较单帧定位、两帧 descriptor-only candidate readout、帧差、明确 correspondence。若简单差分在当前预算下已达到复杂模块的定位表现，应优先采用它并收缩复杂表示的主张；这不能判定问题的物理本质只是变化检测。
5. **低 SNR 合成敏感性。** 在整张图作受控 blur/contrast/noise 扫描，保留原中心标签并单列报告；同步记录球附近/背景的 descriptor separability（正负 similarity 分布）。它不模拟真实曝光物理，也不声称保持拖影中心语义，只测模型对受控退化的敏感性。
6. **污染与可复现记录。** 对可获得的 test-clip 来源 URL/赛事/上传或发布日保存 manifest；在合理成本内作 perceptual-hash 或公开近重复检查，并报告命中规则、排除数量及未能审计的 LVD 部分。再加至少一个不依赖现代 web-VFM 的公平空间对照。零命中只能表述为“本次公开审计未发现”，绝不能表述为“证明未预训练”。

## 对论文创新边界的更新

`top-K 保留多个可能对应`、`intermediate feature probe`、`training-free positional correction`、`shift robustness` 都已有直接近邻，不能单独作为贡献。仍值得研究的窄命题是：**在已有球中心标注、clip 内连续时序和严格计算预算下，冻结/轻调现代特征为何会在 tiny-fast 的 phase、blur、relative-displacement 条件下丢失真候选；一种机制能否在不牺牲当前帧证据的情况下改善 coverage--clutter--localization Pareto。**

成功的最低证据链应是：空间层/相位诊断发现明确失效 → 机制提高真实中心候选 coverage 或 rank（非仅总热图）→ 在同 backbone/input/head/budget 下转化为定位收益 → 在 blur/large-displacement/相机连续变化分桶中不靠某一容易污染来源支撑。
