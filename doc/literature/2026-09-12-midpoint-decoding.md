# 热图坐标解码：BlurBall 三帧中点基线的直接先例与最小诊断

> **范围。** 本笔记只审查已经训练完成的 \(288\times512\) 空间类别加缺失类、hard one-hot CE 输出，能否用**固定输出的坐标读出**诊断长拖影中点偏差。它不提出新的 blur/motion 模块、不改训练目标，也不把人体/人脸关键点结果当作球定位性能证据。检索与一手全文阅读截至 **2026-09-12**。
>
> **当前证据边界。** 现有保存物只有空间 argmax 与 \(q\)，没有完整空间 logits/probability map；因此任何非 argmax 读出都不能离线从这些记录重建，必须用同一冻结 checkpoint 再做一次 forward 才能评价。本笔记未运行该 forward 或任何数据实验。

## 结论先行

固定 heatmap/logit 读出早已有直接先例，不能写成新颖机制；但现有论文也**不能证明**它必然改善本项目的 hard-CE 小球中点。

* 全局 integral / soft-argmax（DSNT）是为端到端坐标损失设计的期望读出；将其直接接到冻结的超大空间分类器，会受远处低 logit、多个峰及 missing 类混入影响。
* DARK 和局部 soft-argmax 都可以在推理阶段改变坐标，却把收益建立在 Gaussian heatmap、连续 Gaussian 编码与（通常）像素级 heatmap 回归的条件上。当前 hard one-hot CE 并未要求 logits 在球周围近似 Gaussian。
* 因而先做一个**局部**、保留当前缺失判定的无 GT 输入读出对照是合理诊断；若没有收益或只在条件可见样本上改善，停止把它当作解释，更不能据此推出 blur motion 已被建模。

## 三个最直接先例

| 工作与实际阅读范围 | 算子、训练关系与作者代码 | 对当前 hard-CE 输出能否直接迁移 |
|---|---|---|
| **DSNT / Numerical Coordinate Regression with Convolutional Neural Networks**，Nibali et al., arXiv:1801.07372v2（2018-05-03）。已读全文 §4--§6.2。 [论文](https://arxiv.org/html/1801.07372)；[作者 PyTorch 实现](https://github.com/anibali/dsntnn)。 | 把归一化二维 heatmap 当离散概率分布，输出其坐标期望；作者实现以全空间 softmax 归一化后作边缘期望。论文训练的是坐标 Euclidean loss，并可加方差或 JS-Gaussian distribution regularization，而不是冻结模型后的单纯后处理。 | **不能把其性能结论直接迁移。** 固定 logits 的全局 expectation 在数学上可算，却没有受到本模型 CE 训练的坐标误差约束；远处重复纹理或另一候选的很小质量也会拖动坐标。这提示“全局 soft-argmax 不是免费改进”的风险；论文未测试当前分类器，不能称为本模型上的实测反例。 |
| **DARK / Distribution-Aware Coordinate Representation for Human Pose Estimation**，Zhang et al., CVPR 2020；arXiv:1910.06278v1（2019-10-14）。已读全文坐标解码、编码与相关消融。 [CVPR 论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.pdf)；[作者代码](https://github.com/ilovepose/DarkPose)。 | 推理中先对**非负 heatmap** Gaussian blur、取 log，再在 argmax 邻域以 Hessian/一阶导 Taylor 求亚像素极值；因此读出部分不必再训练。可是其推导显式假设预测分布与训练目标同为 Gaussian；完整 DARK 还把 GT heatmap 改为非量化的连续 Gaussian，原实验是 heatmap regression。 | **可作第二个纯读出诊断，不能照搬为理论保证。** 当前 logits 既不保证非负，也不保证 Gaussian，直接 blur--log--Taylor 可能数值不稳或沿拖影长轴得到非物理极值。若以后测试，必须明确它只是 post-hoc decoder，不能称为 DARK 复现或把其 pose 增益外推给球。 |
| **Subpixel Heatmap Regression for Facial Landmark Localization**，Bulat et al., BMVC 2021；arXiv:2111.02360v1（2021-11-03）。已读全文 §3.2--§4.3。 [论文](https://arxiv.org/html/2111.02360)；论文指向的[作者项目/代码入口](https://github.com/1adrianb/face-alignment)。 | 先保留空间 argmax，再截其附近 \(d\times d\) patch，对 patch logits/heatmap 以温度 softmax，作局部期望并加回 offset。该步骤本身在推理时不读 GT；论文在 \(5\times5\) 窗口上取得最好结果，且其消融中 global soft-argmax 劣于 local。可是主方法同时采用连续 Gaussian 编码、像素级 heatmap 损失与 Siamese consistency training；局部窗口又明确按其目标 Gaussian 尺寸选择。 | **最接近本轮的最小对照。** 它避免 global expectation 把远处模式平均进来，且不需改变候选/缺失决策。其收益仍不能归因给 decoder 单独造成，更不能说明拖影质心就是球中点；仅可检验当前 peak 邻域是否含有对中点有用的连续不对称证据。 |

### BCIR 给全局 soft-argmax 的额外边界

Gu et al. 的 **Bias-Compensated Integral Regression (BCIR)** 提供另一个直接理论限制：[TPAMI 2023 版本的开放预印本](https://arxiv.org/html/2301.10431) 已读 §III--§VII-C。它指出全局 softmax + expectation 会因全域非零质量产生位置偏置，并给出不使用 GT 的补偿项；但完整 BCIR 仍是 end-to-end integral-regression 框架，并配合 heatmap shrinkage regularizer 或 Gaussian prior。其“软期望可被背景质量拖偏”的分析与本项目相关；其补偿公式建立在局部 support 模型上，不能证明对 hard one-hot CE 的冻结 logits 有效，也没有成为首个应做的对照。

## 对 hard one-hot CE 的可识别边界

当前输出可视为 \(K=288\cdot512\) 个位置类别加一个 missing 类。空间 argmax 是该分类目标的 mode；把 logits 重新归一化并取 expectation，等于在**不同的连续坐标损失/决策规则**下解释同一个分类器。CE 并不约束 peak 的局部二阶形状、拖影轴向质量或次峰是否应参与中点估计。

因此，以下说法都不成立：

* “有 softmax 概率，所以全局期望必然比 argmax 更接近欧氏中心”；多峰时，期望可落在没有球的位置。
* “DARK 是模型无关 plug-in，所以任意 logits 都满足其 Gaussian--Taylor 假设”；作者的训练/编码条件正是该假设的一部分。
* “局部重心沿拖影轴移动就证明模型读出了 motion”；它也可能只是分类峰的非对称、resize/量化或背景纹理造成的偏移。

## 对本轮固定无GT输入解码的影响

采用**argmax-centered local soft-argmax**作为唯一首轮对照，实际规则以[已锁定协议](../protocols/blurball-local-readout-v1.md)为准。文献中的5×5来自其人脸Gaussian尺寸，不能机械移植；本项目选择15×15格、T=1，半径17.5原图像素，对应已经识别的4–16px近位置误差范围。没有在当前数据上比较两种窗口并选优。

原q与0.5输出判断完全保持，缺失类不参加空间质量。全部目标均计算空间读出，所有V1继续纳入raw位置诊断，即使被q拒绝；仅真正输出的子集另报emitted指标。推理不使用GT位置、轴或半长度。局部窗口以模型原argmax为中心，不能恢复窗口外的候选，也不保证局部双峰的平均对应真实球中点。

它检验的是当前peak邻域是否已有可恢复的中点质量，改变的不只整数格量化；15×15允许多个格范围的移动，因此不能把任何阳性都称为纯亚像素精化。这里不改q，不能救回原本拒绝的输出，但可能通过位置变准改变严格定位TP/FP1与检测recall；笼统说“recall完全不能改变”不正确。

若无收益，只能排除这一固定解码配方，不得据此认定hard-CE logits或图像没有几何信息。若有收益，仍须报告全体和各match、原l组的救回/破坏，不以有输出的条件分数掩盖拒绝或远错误，也不声称motion表示已经改进。

## 可复用来源与状态

1. Nibali et al. **Numerical Coordinate Regression with Convolutional Neural Networks**. arXiv:1801.07372v2, 2018-05-03. [全文](https://arxiv.org/html/1801.07372)，[作者实现](https://github.com/anibali/dsntnn)。作者实现确有全局 flat_softmax 与坐标 expectation；本轮未运行。
2. Zhang et al. **Distribution-Aware Coordinate Representation for Human Pose Estimation**. CVPR 2020；arXiv:1910.06278v1, 2019-10-14. [正式论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.pdf)，[作者代码](https://github.com/ilovepose/DarkPose)。本轮核对论文解码公式与作者 inference.py 的 blur--log--Taylor 流程；未运行。
3. Bulat et al. **Subpixel Heatmap Regression for Facial Landmark Localization**. BMVC 2021；arXiv:2111.02360v1, 2021-11-03. [全文](https://arxiv.org/html/2111.02360)，[作者代码入口](https://github.com/1adrianb/face-alignment)。本轮核对局部 soft-argmax 公式、连续 Gaussian 编码、Siamese 训练和窗口消融；未复现。该入口目前未定位到可确认对应本论文 local-soft-argmax 的独立实现，故本文不把它作为源码实现事实。
4. Gu et al. **Bias-Compensated Integral Regression for Human Pose Estimation**. TPAMI 2023；arXiv:2301.10431v1, 2023-01-25. [开放全文](https://arxiv.org/html/2301.10431)。本轮只作全局 softmax/expectation 边界核对；未取得可确认作者代码。

**未覆盖。** 本笔记没有系统检索所有 sub-pixel/keypoint decoder，也没有寻找把 hard one-hot spatial classification、缺失类和高速运动模糊球联合评测的论文；当前三篇足以限定“decoder 不是新概念”及上述最小对照的适用边界。若局部读出结果显示稳定且实质的残差模式，再以该模式为检索问题补文献，而不是先扩张为模糊模块。
