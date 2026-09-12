# 原生匹配置信度：位置误差、共视与无匹配分别怎样学习

日期：2026-09-12。范围：补读 PDC-Net/PDC-Net+、RoMa/RoMa v2、最新 RoMa-Ω 与 PWarpC，回答“没有初始相机几何时，模型能否直接估计对应风险；这种风险与自动球定位是什么关系”。另以LayeredFlow区分物理多层与单目标的不确定候选。与[整体匹配误差及 CoRe 后验](2026-09-12-coarse-fine-uncertainty.md)互补。

## 结论

**原生预测对应误差、区分误差尺度、处理背景运动与独立物体竞争、利用现代冻结 backbone 做全局匹配，均已有直接先例。** PDC-Net 系列与 RoMa v2 的图像对推理并不需要预先给定相机几何；不能把 CoRe 需要初始投影残差的限制泛化给所有可靠对应方法。

这些论文也说明“confidence”不足以定义研究问题。PDC-Net 的输出是围绕单一 flow 的误差密度；RoMa v2 分开预测共视和条件位置误差。背景点完全可以具有正确的对应和很高的置信度，自动球定位仍须判断哪个位置属于球。另一方面，论文已有有效定位、错误排序和下游几何实验，应承认其作用，不能用任务差异抹去正证据。

## 一、来源与阅读边界

| 来源 | 版本与正式状态 | 本次阅读 |
|---|---|---|
| Prune Truong 等，*GOCor: Bringing Globally Optimized Correspondence Volumes into Your Neural Network* | [NeurIPS 2020 正式论文](https://proceedings.neurips.cc/paper_files/paper/2020/file/a4a8a31750a23de2da88ef6a491dfd5c-Paper.pdf)及[补充](https://proceedings.neurips.cc/paper_files/paper/2020/file/a4a8a31750a23de2da88ef6a491dfd5c-Supplemental.pdf) | 主文、补充的优化目标/消融/迭代成本与固定 global/local 实现 |
| Prune Truong 等，*Learning Accurate Dense Correspondences and When To Trust Them* | [CVPR 2021 正式论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Truong_Learning_Accurate_Dense_Correspondences_and_When_To_Trust_Them_CVPR_2021_paper.pdf)，即 PDC-Net | 主文方法、训练与推理定义；用于确认机制已出现于初版 |
| Prune Truong、Martin Danelljan、Radu Timofte、Luc Van Gool，*PDC-Net+: Enhanced Probabilistic Dense Correspondence Network* | [arXiv 2109.13912v2](https://arxiv.org/abs/2109.13912v2)，2021-09-29；[作者仓库](https://github.com/PruneTruong/DenseMatching)列 TPAMI 2023，正式 DOI [10.1109/TPAMI.2023.3249225](https://doi.org/10.1109/TPAMI.2023.3249225) | 本次正文细节与表格来自 v2：§3、§4.1–4.3、§4.6、附录 A–C；未将其逐页等同于期刊排版版本 |
| Johan Edstedt 等，*RoMa: Robust Dense Feature Matching* | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html)，[arXiv 2305.15404v2](https://arxiv.org/abs/2305.15404v2)，2023-12-11 | 预训练特征、coarse 多峰、anchor 读出及核心实验 |
| Johan Edstedt 等，*RoMa v2: Harder Better Faster Denser Feature Matching* | [arXiv 2511.15706v3](https://arxiv.org/abs/2511.15706v3)，2026-07-06，标明 ECCV 2026 camera-ready | 主文与附录的结构、监督、covariance、效率及消融，并核对公开 inference 路径 |
| David Nordström 等，*RoMa-Ω: What Feed-Forward 3D Models Know About Image Matching* | [arXiv 2609.09507v1](https://arxiv.org/abs/2609.09507v1)，2026-09-08，预印本 | 正文与补充的特征条件化、不同 probe、受控训练、动态反例与成本；另读双图输入及匹配源码 |
| Prune Truong、Martin Danelljan、Fisher Yu、Luc Van Gool，*Probabilistic Warp Consistency for Weakly-Supervised Semantic Correspondences* | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Truong_Probabilistic_Warp_Consistency_for_Weakly-Supervised_Semantic_Correspondences_CVPR_2022_paper.html)；[arXiv v2](https://arxiv.org/abs/2203.04279v2)更新于 2023-10-31 | 正文、正式补充材料与固定 SF-Net 训练/bin/推理路径；以下补充表格按 CVPR 版编号 |
| Hongyu Wen、Erich Liang、Jia Deng，*LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow* | [ECCV 2024作者全文](https://layeredflow.cs.princeton.edu/static/files/main.pdf)；[arXiv 2409.05688](https://arxiv.org/abs/2409.05688) | 主文、GT定义、全部表格和固定Multi-RAFT数据/输出/去重路径；未取得所引补充材料 |

PDC 作者代码固定为 [`DenseMatching@b054fe9`](https://github.com/PruneTruong/DenseMatching/tree/b054fe9f7988c70db1e0e7347d0b1bafe135cc27)，2023-04-20。RoMa v2 固定为 [`RoMaV2@95c9968`](https://github.com/Parskatt/RoMaV2/tree/95c9968145c8906b7b59383258e9f73b02853d89)，2026-04-20，早于所读 camera-ready；源码事实和最终论文实验分别引用。原 RoMa 的读出核对使用 [`RoMa@77f8d68`](https://github.com/Parskatt/RoMa/tree/77f8d68803526dcddfd9b7a46bc76125bdc25f15)。

PDF、提取文本与少量源码保存在 `outputs/literature/` 的 `pdcnet-*`、`roma-*`、`romav2-*`、`pwarpc-*`。PWarpC 源码使用上述同一个 DenseMatching 版本。本次没有运行这些模型，表内数字均为作者报告。

RoMa-Ω 源码固定为 [`RoMa-Omega@24c693a`](https://github.com/davnords/RoMa-Omega/tree/24c693a47004ab7dda19005e8e3d0040fecda58d)，2026-09-10，晚于所读 v1。它和早于论文的 RoMa v2 代码是两份独立的来源记录，不能假定所有配置完全对应最终论文。

GOCor 的少量源码固定为 [`GOCor@b1de663`](https://github.com/PruneTruong/GOCor/tree/b1de663391b0060d70d08fa182462a9d18bbd3a1)，2023-02-23；论文及代码缓存为 `gocor-*`。作者只发布测试代码，以下训练配方来自论文，未将其写成已经复核公开训练工程。

## 二、PDC-Net：为一个预测位置学习不同尺度的误差

### 2.1 从图像到对应的实际路径

PDC-Net+ 以 GLU-Net-GOCor 为基础。L-Net 读取缩放到 256×256 的图像，在 16×16 feature grid 上进行全局 GOCor，再在 32×32 做局部匹配；H-Net 读取较高分辨率图像，在 1/8、1/4 尺度继续细化。后三层的 local correlation 为前级 flow warp 后的 9×9 范围。GOCor 自身还包含可学习的优化过程，不能把全部成本写成一次普通点积。[附录 C](https://arxiv.org/pdf/2109.13912v2)

这里已经组合了粗全局、细局部、跨尺度 flow 与 uncertainty 传播。9×9 是该层读取的相关值范围，前级 flow、空间 decoder 和连续 residual 回归还会影响输出，不能据此宣称最终 flow 只能落在固定原图邻域内。

附录与实际配置还有一个归因细节：plus 第一阶段使用冻结、共享的 VGG；第二阶段微调时设置 `make_two_feature_copies=True`，为 256 输入和原分辨率输入分别建立 VGG pyramid。初版 PDC-Net 共享 backbone。因此“宏观架构相同”不等于完整训练期间参数共享方式与可训练容量相同。[stage1](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/train_settings/PDCNet/train_PDCNet_plus_stage1.py)、[stage2](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/train_settings/PDCNet/train_PDCNet_plus_stage2.py)

### 2.2 同均值 mixture 的概率语义

对每个 source pixel，模型输出一个二维 flow 均值 `mu`、分量权重 `alpha_m` 和方差 `sigma_m²`：

`p(y|X) = sum_m alpha_m [1/(2 sigma_m²)] exp(-sqrt(2) ||y-mu||_1 / sigma_m)`。

每个分量内部的两轴独立、方差相同，所有分量共享同一个 `mu`。这是一种**单峰、不同尾部尺度的误差模型**，不是分别指向球网、球拍和球的多个地址假设。两轴等方差也不代表密度旋转对称：这里的等密度轮廓由 L1 距离决定，没有任意拖影方向的 covariance 参数。[§3.2、附录 A](https://arxiv.org/pdf/2109.13912v2)

完整模型用两个分量，第一分量 `sigma_1²=1`，第二分量限制在 `2≤sigma_2²≤HW`。分量的 variance 范围分开，既固定角色，也避免 NLL 被接近零的方差任意压低。权重是该潜在分量的概率；即使作者称其为 inlier/outlier，也不能直接将某个权重等同“误差小于 4px”的概率，两种 Laplace 的支撑都覆盖整个平面。

论文实际使用的置信度是积分：

`P_R = P(||y-mu||_infinity < R) = sum_m alpha_m [1-exp(-sqrt(2) R/sigma_m)]²`。

这是每轴半宽 R 的**方框**，本项目 PCK 用的则是二维欧氏距离圆。相同数字 R 不能直接互换。在同一坐标系和同一个预测密度下，有：

`P_infinity(R/sqrt(2)) ≤ P_2(R) ≤ P_infinity(R)`。

这只是集合包含关系，不需要新模型实验。迁移时还必须明确 R 是输入像素、原图像素还是归一化坐标。`P_R` 的模型含义清楚，不意味着在新的球数据分布上已经校准。

源码的默认 Laplace helper 与上述公式一致，见 [`mod_uncertainty.py`](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/models/PDCNet/mod_uncertainty.py)。不能将旁边 Gaussian 分支的圆内概率套用到默认 Laplace 实验。

### 2.3 “逐 query 处理 cost”不是整个 uncertainty 分支完全局部

论文先把每个 source pixel 对应的二维 cost slice 独立处理：将 source 空间维移入 batch，对 displacement 维卷积，得到匹配不确定性特征。这确实为该 query 的匹配分布保留了直接输入路径。

随后还有一个空间卷积 predictor，接收上述特征及 flow decoder 信息，并传播前一尺度的 flow/uncertainty。附录 C 与实际默认实现使用的是 flow decoder 倒数第二层特征，而不只是两个 flow 数值。最终模型因而仍能使用空间邻居与前级先验；不能称为“纯视觉 cost、不受运动外推影响”。[§3.3、附录 C；源码 `estimate_uncertainty_components`](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/models/PDCNet/PDCNet.py)

这恰好说明应当研究的是各信息路径怎样影响判断，而不是把所有上下文视为不可信。作者 ablation 中，cost-only 分支在 YFCC pose 上很好，但在含独立运动物体的 KITTI 上，AUSE 为 0.418，完整 uncertainty decoder 为 0.205；共同 decoder 则为 0.171。完整分支在所有数据和指标上并非最好，不能将“更强调本地证据”写成普遍定理。[Table 8](https://arxiv.org/pdf/2109.13912v2)

### 2.4 GOCor 的背景竞争来自优化 filter，不是新增位置

PDC 的相关层自身也已有针对重复外观的设计。GOCor 用每个 reference 位置的 filter `w_p` 代替原 feature `f_p^r`，输出 `C(p,q)=w_p^T f_q^q`，global/local 版本仍读取原来那组 query 地址。内循环目标是：

`L(w)=L_r(C(w,f^r))+||R_theta * C(w,f^q)||²+||lambda_theta w||²`。

`L_r` 在 reference 自相关中鼓励本位置响应高、相似竞争位置响应低，并学习对正/负响应不同的惩罚和距离函数；第二项以可学习 4D convolution 正则 query cost 结构。它们通过下游任务 loss 学习，推理内循环只读取图像特征，不读取 GT 对应或相机几何。少数步最速下降采用 Gauss–Newton 近似步长，并不保证求得数值全局最优。[§3.3–3.6](https://proceedings.neurips.cc/paper_files/paper/2020/file/a4a8a31750a23de2da88ef6a491dfd5c-Paper.pdf)、[`global_gocor.py`](https://github.com/PruneTruong/GOCor/blob/b1de663391b0060d70d08fa182462a9d18bbd3a1/GOCor/global_gocor.py)

因此，“用 reference 中的相似区域以及跨地址一致性来重新评价匹配”已有直接先例。4D 正则不等于严格一对一指派或物理运动定律；GOCor 输出的完整 cost 可以保留多个地址分数，但没有独立的 null、球身份或拒绝动作。这些仍由后续任务决定。

这里可给出一个限定清楚的代数边界：若固定 query 特征在两个地址恰好相同，即 `f_a^q=f_b^q`，任意 filter 均有 `w_p^T f_a^q=w_p^T f_b^q`。仅改这个 filter 不能打破两地址在该相关输出中的平局。它不证明实际球特征相同，也不限制另有坐标/上下文的 decoder 或可训练 backbone。反过来，只是原始 cosine 的 top-K 漏球，**不**足以排除 GOCor：只要球地址仍在完整 cost 的定义域内，重标可能改变排名。不能把已被硬删的地址与尚未排进 top-K 的地址混为一谈。

动态数据训练的主文 Table 6 有正消融：BaseNet→global `L_r`→再加 global `L_q`→再加 local GOCor，KITTI-2015 AEPE 为 **8.93→8.50→7.87→7.10**；对应 HPatches PCK5 为 **69.22→71.29→71.21→78.30**，并非每一步所有指标都改善。静态训练的补充 Table 8 中，给 local GOCor 也加 `L_q` 后，KITTI-2012/2015 AEPE **4.02/9.92→4.24/10.20**，HPatches PCK5 却 **74.80→75.26**。最终 local 路径没有采用这项正则。正则作用的位置和训练条件确实重要，不能用全局项的收益替局部项担保。

补充还区分固定模型增加推理迭代和重新训练更多迭代：global 训练3步时，推理改多或改少均可下降；这不等于训练更充分永远无益。Titan X、KITTI-2012 图像对、quarter-resolution flow 再上采样的完整 GLU-Net，普通相关 **154.97ms**，GOCor **261.90ms**（global3/local7）。这些运行时优化属于 forward 成本，不是一次免费的预处理。[主文 Table 5–6；补充 E.1、F](https://proceedings.neurips.cc/paper_files/paper/2020/file/a4a8a31750a23de2da88ef6a491dfd5c-Supplemental.pdf)

## 三、PDC-Net+：训练目标会决定什么被视为可信运动

### 3.1 不可见扰动与背景捷径已有直接先例

PDC-Net 发现：只用平滑的人工 homography/TPS，网络可能靠邻居外推并在纹理贫乏区域过度自信。它对真实图片构成的人工图像对加入局部弹性 flow 扰动；纹理区域会产生可观察变化，均匀区域的外观可能几乎不变，但标签位移已改变。NLL 因而鼓励后一类区域输出更大的不确定性。[§3.4、附录 B.3](https://arxiv.org/pdf/2109.13912v2)

这已经涉及“正确外推不等于存在直接视觉证据”的问题，但只是特定训练分布下的设计和证据，不能视为可观测性的自动判别定理。它需要已知人工 flow、合成 warp、COCO 分割物体及随后加入的 MegaDepth 对应；论文的 self-supervised 不等于仅用本项目球中心标签就能原样训练，也不等于完整模型完全无 GT。

### 3.2 Injective mask 不是简单的 visibility mask

PDC-Net+ 在图像对中放入多个独立运动物体，并将造成多对一映射的遮挡背景位置从损失中移除，保留可见物体对应，避免更容易预测的背景 flow 压过物体 flow。

但它并不删除全部遮挡位置。§3.6/Fig.7 的例子明确规定：若物体只出现在 reference，该覆盖区域采用背景 flow 监督，以学习不可见背景的外推；只出现在 query 的物体所遮住的背景区域，也可以保留背景 flow。超出 query 图像但仍满足单射的 flow 同样保留。

因此，它的 GT 在这些位置包含人为定义的外推任务，不能全部解释为同一可见物理点的直接对应。其训练 mask 也不是模型推理时获得的目标 mask。对体育球，消失、遮挡或只有一帧有标签时，不能照搬为“用背景 flow 代替球的真实运动”。这会改变监督对象，而不是补齐现成球标注。

**所读论文与固定源码在一个实际分支上不同。** 上述只在 reference 出现的物体，是 v2 论文 Fig.7 的监督规则。公开增强代码使用 target→source flow；当物体的 source mask 为空时，它保留背景 flow 数值，却把 target 物体 mask 加入 `mask_of_reprojected_object_from_source_to_target`。默认 plus 配置启用 `compute_object_reprojection_mask`，最终监督 mask 为这个排除区域的反集；该处若未再被后续合成物体改写，就被排除，而不是按背景 flow 参与损失。[增强代码](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/datasets/object_augmented_dataset/synthetic_object_augmentation_for_pairs_multiple_ob.py#L382-L416)

已沿两阶段配置、batch 处理和 actor 的 H-Net/L-Net loss 确认 mask 被实际消费；第二阶段 `MixDatasets` 不会将其覆盖为全真。该默认增强分支也会替换原背景 correspondence mask，并未另外与 `valid_flow` 相交，因此不能把它概括成完整可见性过滤。[stage2](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/train_settings/PDCNet/train_PDCNet_plus_stage2.py)、[batch 处理](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/training/actors/batch_processing.py)、[actor](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/training/actors/self_supervised_actor.py)

这一区别说明：**flow 数组里存在一个背景数值，不等于该位置实际接受背景监督。** 当前仅确认 2021 年 arXiv v2 与 2023 年公开代码的这个差异，没有复现出现频率、性能影响或核对期刊版对应段落，不据此判定哪一版效果更好，也不修改尚未接入项目的上游实现。

### 3.3 实证既有收益，也有目标之间的取舍

固定同一个 PDC-Net+ (D) 的预测，Fig.11 在 MegaDepth 上给出 AEPE/PCK5 的 AUSE：cycle error 为 **0.2953/0.0266**，mixture variance 为 **0.2663/0.0171**，`P_R` 为 **0.2482/0.0112**。这是置信度读出改善错误排序的直接正证据。AUSE 对保持排序的严格单调变换不变，所以它本身不能证明数值概率已经校准。

Table 9 对第一阶段训练的 ablation：

| 训练变化 | KITTI AEPE / Fl / AUSE | MegaDepth PCK1 / PCK5 / AUSE |
|---|---|---|
| 1 个物体 | 6.70 / 19.61 / 0.123 | 52.62 / 73.78 / 0.215 |
| 多个物体 | 5.55 / 16.75 / 0.136 | 54.89 / 75.92 / 0.238 |
| 无 mask | 6.01 / 18.07 / 0.139 | 53.66 / 75.05 / 0.247 |
| Injective mask | 5.55 / 16.75 / 0.136 | 54.89 / 75.92 / 0.238 |
| 完整 occlusion mask | 9.64 / 19.64 / 0.103 | 56.41 / 77.37 / 0.247 |

前两行比较物体数量，后三行比较 mask，不能拼成逐步加模块的同一条路径。多个物体改善 flow，却使 AUSE 变差；完整遮挡 mask 的 KITTI AEPE 明显更差，但其 AUSE 更低，MegaDepth PCK 反而更高。应接受 injective mask 在作者目标下的综合优势，同时承认遮挡外推和只评价可见对应的目标不同。

完整训练的 PDC-Net→plus 也同时变化物体数、mask、backbone 共享/微调方式和步数（330k→575k），不能把最终 KITTI AEPE 5.22→4.53 全归于某一个机制。

### 3.4 NLL 不是仅多报告一个置信度

当 flow 和尺度共同学习时，扩大误差分量会降低相应位置对均值学习的直接压力。对固定 mixture 参数、非零坐标残差，`-log p(y|X)` 对 `mu_d` 的偏导为：

`sign(mu_d-y_d) sum_m gamma_m sqrt(2)/sigma_m`，

其中 `gamma_m` 是给定当前残差后的分量 posterior。它描述均值的直接梯度；不是对共享网络参数的全部梯度，也不证明某类样本一定被忽略。

这个性质可带来鲁棒定位，作者也明确利用它降低大离群误差的影响。但对于本项目，若被宽分量吸收的恰好是论文要改善的高速微小球，整体似然更好仍可能伴随该组召回不足。这是未来采用此类损失时应检验的竞争解释，当前没有相应本地失败实测。

PDC 的 D 路径一次图像对 forward 就输出 flow 与置信度。H 路径另做 confidence 筛选、homography 估计和第二次 forward，MS 再加多尺度尝试。作者在 RTX 2080 Ti、480×480 的 Table 1 分别报告 **88/284/1017ms**；相应结果不能与 D 的时间混用。这也说明“原生 uncertainty 需要先估相机”对 D 不成立，对 H/MS 则确实还存在额外几何流程。

## 四、RoMa v2：DINOv3 全局证据与高分辨率细化的强近邻

### 4.1 远处地址从哪里进入表示

原 RoMa 组合冻结 DINOv2 和 VGG19 fine feature，用 coarse `64×64` anchors 表示可能多峰的匹配分布；最大 anchor 及其邻域读出后，仍得到单一 warp。多峰内部表示不等于向后续视频模型持续输出多条轨迹。[RoMa §3.3–3.4](https://arxiv.org/pdf/2305.15404v2)

RoMa v2 改为冻结 DINOv3 ViT-L/16 第 11、17 层特征，经 Multi-view Transformer 在两图之间交换信息。代码显式计算全部 A/B patch 的 cosine similarity，按每个 A query 对所有 B 地址 softmax，将 B 的 Fourier 位置编码加权汇总；DPT 同时使用该匹配表示、Multi-view token 和 DINO feature，直接预测 stride-4 warp 与 overlap。[§3.2](https://arxiv.org/html/2511.15706v3)、[`matcher.py`](https://github.com/Parskatt/RoMaV2/blob/95c9968145c8906b7b59383258e9f73b02853d89/src/romav2/matcher.py)

因此它确实读取了全局地址分布。默认 precise 的 800×800 coarse 输入对应约 50×50 patch grid，单方向相似度矩阵约 2500² 项；还有 Multi-view global attention。它没有高分辨率 all-pairs，但也不是固定 K 候选的全局算法。

这种多地址位置编码汇总不能简单等同二维坐标 soft-argmax：Fourier 特征可保留不同于一阶坐标均值的分布信息。不过它也没有保证完整分布可逆恢复，最终 inference 仍为每 source pixel 单一 warp。

### 4.2 高分辨率读取的具体范围

v2 的 fine VGG19-BN 路线读取 stride 4/2/1 特征。固定公开实现中，前两级 local correlation 半径分别为 3、1，即每 query 49、9 个目标地址；stride 1 不再计算新的 correlation grid，但仍使用已经 warp 的 target feature、source feature 与 displacement 表示。[`refiner.py`](https://github.com/Parskatt/RoMaV2/blob/95c9968145c8906b7b59383258e9f73b02853d89/src/romav2/refiner.py)

这些范围描述逐 query 的直接相关值读取。空间卷积、feature receptive field、前级 warp 和连续 residual 仍然存在，因此不能写成整网不可越过的硬半径。第二次高分辨率 pass 同样是细化，不是在高分辨率全图恢复所有地址比较。

作者用专门 local-correlation kernel 降低细化成本；Table 8 在 H200、640×640、batch8 报告含 kernel 的完整模型 30.9 pairs/s、4.8GB。默认 precise 还有 800→1280 的两次分辨率处理和可选双向预测，不能用某个局部 kernel 或单尺度吞吐冒充这一配置的体育端到端速度。

### 4.3 共视与条件 covariance 有不同监督

`overlap` 采用二元 BCE，GT 来自一致的 depth，或 flow 数据的 warp cycle consistency。它定义的是该 source 位置在另一图中的共视/对应有效区域，涵盖有正确监督的动态点，并非只允许静态背景。[§3.3](https://arxiv.org/html/2511.15706v3)

共视为真时，模型仍可能匹配错误；背景共视为真时，该点也仍然不是球。故 overlap 既不能直接当定位误差概率，也不能当球存在概率。visibility、未标注和出界等本项目状态仍按原数据协议解释。

另三个输出参数经 Cholesky 构成正定 precision `P=Sigma^-1`，以 Gaussian NLL 学习二维 warp residual。论文明确：**只在共视且 residual 范数小于 8px 的位置训练 covariance，并 detach residual**。这更接近条件内点误差的尺度/方向，不是全局 coarse miss 或 no-match 概率。detach 仅切断该 loss 通过 residual 回到位置回归的直接路径，不代表共享特征完全不受 uncertainty loss 影响。

各尺度相加 precision 是作者采用的参数化。它不自动证明不同尺度提供独立观测，也不能解释为对三份独立运动证据完成了严格贝叶斯融合。

Fig.7 对图像施加人工线性 blur kernel，预测椭圆沿模糊方向增大，是方向响应的正定性证据；它没有用真实球曝光轨迹监督，更不等于帧间位移方向。Table 12 在 Hypersim 的 AUC1°/3°/5° 为：原 pipeline **54.9/79.5/85.9**，covariance refit **75.8/89.0/92.6**，另用于 RANSAC scoring 后 **76.4/89.3/92.8**。这支持该几何流程的实际用途；表中未单列普通无权重 refit，不能把全部差值都理解为一种置信度数值本身的独立贡献，也不是球坐标修正的实验。

### 4.4 现代 backbone 的正结果和反例都要保留

v2 Table 1 的冻结特征线性变换加 kernel matcher，在 MegaDepth 上将 DINOv2 的 EPE **27.1**、32px 内比例 **77.0%** 改善到 DINOv3 的 **19.0/86.4%**。这反对“现代语义特征天然不能做精确对应”的先验判断，但没有测几像素球。

Table 10 的 coarse-only backbone 消融也有非单调项：DINOv2→DINOv3，Hypersim 指标 **78.1→79.2**，WxBS 则 **35.6→34.2**。不能只引用前者宣称所有域都改善。完整 v2 还变化 matcher、监督、训练数据、refiner 与 EMA，整体增益不能全部归于 backbone 更新。

动态场景并非完全未测：FlyingThings3D 报告 EPE **0.93**、PCK1/3/5 **89.4/95.2/96.8**，并有小动态 guitar 的定性实例。它们支持动态对应能力，但没有球尺寸、rho、真实球拖影、自动发现召回和 game-level 球定位结果。

### 4.5 RoMa-Ω：最新的强表征证据仍须分清输入条件

2026-09-08 新稿 RoMa-Ω 将 RoMa v2 的冻结视觉基础换为 VGGT-Ω，并研究不同读出。关键区别是：VGGT-Ω 在图像 encoder 后交替做 frame-wise 和跨视图 global attention，其后层 feature 已依赖输入图像对。固定代码把 `img_A,img_B` 堆叠为 `(B,2,3,H,W)`，一次送入冻结的 `vggt.aggregator`；DINOv3 支线才分别编码两图。[§3.1、§4.2](https://arxiv.org/html/2609.09507v1)、[`matcher.py`](https://github.com/davnords/RoMa-Omega/blob/24c693a47004ab7dda19005e8e3d0040fecda58d/src/romaomega/matcher.py#L287-L326)

因此比较同时涉及预训练表征与**双图条件化**，不能称为完全相同输入计算下的独立单帧 backbone 替换。冻结也不表示跨视图计算免费，更不能把换过配对图的 feature 当成原单帧缓存复用。

论文区分了四种测量：

| 测量 | 实际读出 | 对本项目的含义 |
|---|---|---|
| Raw cosine / NN | 对冻结、已条件化的 patch feature 取全图相似度最大值；MegaDepth-1500 的 Fig.5 用 PCK32 | 测原始相似度几何，不测几像素球或自动发现 |
| Linear probe | 学习线性投影，再经 small decoder 输出 dense field | 不是纯线性位置头；正文/补充未给该小 decoder 的完整层数、训练集和损失，不补写其成本 |
| Raw geometry | 从模型预测 depth/camera 得到 warp，或对预测 3D points 做 mutual NN，再估计相对 pose | 测 3D 输出能否提供对应；与 raw descriptor cosine 是两条路径 |
| Full coarse matcher | 冻结 VGGT-Ω 或 DINOv3，控制可训练 decoder 架构，并比较不同 decoder 深度 | 能验证适当受训读出后的匹配能力；不能消除 feature 先前是否跨图交互的差别 |

Fig.5 的 VGGT-Ω 后层 raw NN 较差，但受训 probe 较好；Fig.6 的同 decoder 比较也支持其特征可被有效利用。**这是 raw cosine 不可读不等于信息完全消失的直接证据。** 它没有实验证明本项目所有失败的线性头都能被另一个头救回；“一个受限探针失败不足以证明不存在任何可用信息”仍是逻辑边界，而不是无需实验的收益保证。[§3.2–3.4](https://arxiv.org/html/2609.09507v1)

Table 6 进一步控制粗匹配器训练为 100k steps，RoMa v2 对照采用作者相同训练数据。读出与层选择会明显改变结果：

| 粗匹配设置 | Hypersim PCK3 | ETH3D PCK3 |
|---|---:|---:|
| 重训 RoMa v2 对照 | 61.8 | 85.8 |
| VGGT-Ω + DPT | 28.3 | 47.5 |
| 再加 transformer decoder | 64.6 | 82.3 |
| 采用最深两层 `{18,24}` | 68.5 | 87.7 |
| 再扩大 DPT 至完整 2048 宽度 | 65.1 | 85.3 |

受控结果支持“表征、层与读出的配合”，不支持“大模型接一个头就自然更准”或“更宽一定更好”。最终系统仍保留可训练跨图 decoder、全局 all-pairs 相似度与位置编码、DPT 和 RoMa v2 refiners，不是细分辨率上固定 K 次远搜索。[§4、Table 6](https://arxiv.org/html/2609.09507v1)

完整训练另外采用 coarse 500k steps、batch128、约 2 天/16×A100，refiners 300k、batch64、约 2 天/8×A100；最终数据 mix 比 RoMa v2 略大。不能用上述 100k 的受控消融代替完整训练成本，也不能把最终增益完全归于骨干名称。[§4.3、补充 A](https://arxiv.org/html/2609.09507v1)

正结果很强：在 Ω 论文自己的 Table 2 中，WxBS/HardMatch/RUBIK 从 RoMa v2 的 **64.8/46.5/85.6** 提高为 **72.9/50.0/88.0**；这些数值只在该表协议内比较，不与 v2 自己论文的另一套表格拼接。六个 dense test 中五个改善，但动态 FlyingThings3D 是反例：EPE **0.93→1.07**，PCK1/3/5 **89.4/95.2/96.8→88.5/94.6/96.3**。论文没有用因果消融确定这是 3D 先验、训练分布还是其他因素造成，不能替作者归因。[Table 2、Table 4](https://arxiv.org/html/2609.09507v1)

同一 A100、560×560、batch8 的 Table 5，RoMa v2 为 **21.6 pairs/s、4.0GB**，Ω 为 **12.1 pairs/s、29.9GB**；这不是此前 v2 的 H200/640 设置。该成本与动态图反例都要求谨慎迁移，但 batch8 的显存数值不证明 batch1 一定无法在本地运行。当前保留它作为最新文献和探针解释对照，不因此启动 3D 模型路线。

## 五、PWarpC：显式 null 状态与实际拒绝动作

### 5.1 它在匹配分布里保留了什么

PWarpC 针对不同实例、同一语义类别的图像匹配。由真实图像对 `(I,J)`，将 I 作已知人工 warp 得到 I'，要求 `I'→J→I` 的概率映射经中间位置边缘化后符合已知 `I'→I`，同时直接监督后者。这里的矩阵 composition 保留跨空间候选的概率关系，训练目标不只约束一个回归坐标。[§4.1–4.2](https://arxiv.org/html/2203.04279v2#S4.SS2)

它给 cost volume 增添可学习标量 logit 的 null bin，令 `P(null|null)=1`，使经过中间图已经 unmatched 的路径继续保持 unmatched。这是显式未匹配状态及其组合规则的清晰前史。

可见区域训练 mask 则是另一回事：按 two-hop 分布在已知 warp 落点的概率排序，选取最高的一部分 query。真实共同可见区域没有密集标签，预测排名是其代理；不能把该 mask 当作真实球 visibility。论文也承认全 cost volume 的内存成本使其受限于较粗分辨率。[§4.3、§6](https://arxiv.org/html/2203.04279v2)

### 5.2 Null 的直接负监督是不同类别的整图

PNeg 取一张不同类别的图 A，将 `(I,A)` 的所有像素以 BCE、目标值 0.9 引向 null。固定训练代码用 batch 移位和 `category_id` 不同筛选负对；bin 是全图共享的可学习标量，其 softmax 概率仍会随各 query 的其他 logits 改变。[§4.3–4.4](https://arxiv.org/html/2203.04279v2#S4.SS3)、[`NegProbabilisticBin`](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/training/losses/probabilistic_warp_consistency_losses.py)

这是语义任务的无对应监督，不能直接译为同一 clip 的球不存在、未标注、误匹配或视觉证据不足。强监督 keypoint 版本为保持标注条件一致，也没有使用这一依赖类别标签的 bin/PNeg 分支。

本项目可以重新定义**以球身份为条件**的 null：例如指定球 query 没有可接受的球对应。但这时必须明确已改变条件事件，并用球标签和相应输出规则解释它。把非球位置记为“不是目标球”可以合理；把同一位置真实存在的背景几何对应记为“不存在任何对应”，则是另一个命题。

### 5.3 源码中有 bin，仍可能每次输出坐标

固定 [`SFNetWithBin.estimate_flow`](https://github.com/PruneTruong/DenseMatching/blob/b054fe9f7988c70db1e0e7347d0b1bafe135cc27/models/semantic_matching_models/SFNet.py) 的默认 argmax 路径先保留包含 bin 的完整 softmax，随后以 `[:, :h*w]` 去掉 bin，再从空间位置取最大值。因此 dense-flow API 仍输出一个坐标。

另一函数 `estimate_flow_and_confidence_map` 读取完整分布，返回最大空间分数的倒数，以及 `(1-P_bin)×max_spatial_score` 的倒数作为 uncertainty，还另跑反向 flow 得到 cycle error。它提供了使用 bin 的分数路径，但没有在这个 API 内以 bin 胜出或某阈值直接返回“拒绝位置”。这不是实现缺陷：dense flow 加质量分数本来可以由后续任务使用；只是“学到 null”与“最终定位不输出错误位移”尚隔着决策规则。

### 5.4 应当接受的有效性证据

正文 Table 2 从已有正对 PW-bipath、visibility mask、warp supervision 的 IV 到加入 bin/PNeg 的 V，PF-Pascal `alpha_img=0.1` 的 PCK **84.9→87.6**，SPair-71K `alpha_bbox=0.1` **30.7→33.5**。这里的容差按图像或物体框尺寸归一化，不是本项目固定几像素容差。[Table 2](https://arxiv.org/html/2203.04279v2#S5.SS4)

[正式补充材料](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Truong_Probabilistic_Warp_Consistency_CVPR_2022_supplemental.pdf)还在相同正对损失下比较不同负损失；PNeg 优于所列 max-score/min-entropy 对照。补充 I.4 的 PF-Pascal sparsification 报告，PWarpC-SF-Net matching-score AUSE **0.0496**，SF-Net matching-score 为 **0.0673**，mapping forward-backward 为 **0.0685**。这些支持语义对应精度与错误排序的改进，未直接评价 bin 的二元概率校准或球检测。

## 六、LayeredFlow：多个物理对应不是同一目标的多峰后验

LayeredFlow的层按透明表面深度顺序定义，各层都有自己的二维位移。真实采集经AprilTag测量、移动场景/相机再拍摄，标注稀疏且每个标注像素只给一层；完整多层训练真值来自修改后的ray tracing。不能从这种前后状态推导连续视频的曝光间隔。[§3–5](https://layeredflow.cs.princeton.edu/static/files/main.pdf)

Multi-RAFT共享图像特征/相关体，并用独立context分支输出四组flow；少于四层时重复末层训练，推理按相邻输出距离0.5px去重。它的层数和已知层误差评价，不是对四个互斥地址取best-of-K；输出也没有给出对应概率。Table4支持多层任务上的条件收益，未评价球中心。[§5.2–6.2](https://layeredflow.cs.princeton.edu/static/files/main.pdf)；[固定实现](https://github.com/princeton-vl/LayeredFlow/blob/57ff3f18814d5201ecb440aaf2a04d849938a9d4/MultiRAFT/core/raft.py#L175-L236)和[去重](https://github.com/princeton-vl/LayeredFlow/blob/57ff3f18814d5201ecb440aaf2a04d849938a9d4/MultiRAFT/evaluate.py#L65-L80)。本地原文与源码缓存为 `outputs/literature/layeredflow-*`，未运行算法。

**对本项目的推论。** “一像素永远只有一个物理对应”不是普遍成像事实。但透明表面的多个同时成立真值、单个球中心的竞争候选、同一位移的误差尺度、曝光内不同时间相位，仍是四种不同对象。球拖影是时间积分，不能仅因混合背景就采用透明层GT；LayeredFlow也不能单独证明其机制解决了单目标后验歧义。

因此今后若保留多个motion hypothesis，应说明哪个是条件于球身份的候选、它是否有概率含义、最终怎样选择或拒绝，以及候选保留是否改善自动定位。既有单球中心标签不会因内部输出多个槽而变成多层flow监督；当前不新增物理分层任务。

## 七、对本项目研究决策的影响

可复用的思想已经很具体：直接评估 query 的 cost 分布；让训练中的不确定性不能只靠平滑外推蒙混过关；分开共视、位置误差和目标身份；以现代粗特征进行全局搜索，再读取细空间证据。这些都应作为近邻，而非待宣布的新贡献。

如果后续需要强对应探针，RoMa v2 是有依据的候选，但是否运行取决于当前实验暴露的瓶颈与实际资源。GT source 球点下的成功，证明的是该条件下的对应能力；自动球定位还多出目标发现、类别判别、候选排序、读出与时序融合，不能只由两者差值断言唯一瓶颈。

评价也应贴近最终动作：拒绝一部分对应可能改善剩余样本误差，却降低球召回；精确的背景对应不能直接提高球 F1；用背景几何衡量独立运动点的误差还可能改变随机量。以上判断不要求现在新增 head 或校准分支。当前继续完成已锁定的 BlurBall 真历史/重复当前帧对照，再决定哪个失败值得研究。
