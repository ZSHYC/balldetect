# 空间位置分布集中度作为只读定位质量诊断：有界最近邻审查

日期：2026-09-11。状态：文献审查完成；本项目的空间分布forward与统计尚未运行。

## 问题、范围与先验判据

[读出集中度实验](../experiments/2026-09-11-readout-concentration.md)针对当前DINO `SpatialProbe` 的一个具体矛盾：联合softmax的存在分数 `q` 可以很高，但空间`argmax`仍可能远离标签；[全量HRNet–DINO比较](../experiments/2026-09-11-full-hrnet-dino-comparison.md)已记录168例阈上错位输出和97例VC0误报。拟测量的只是条件空间分布 `c=softmax(z)` 的 `max(c)`、归一化熵和预测格邻域质量，以及联合量 `q*m16` 的事后排序能力。

这不是调阈值、训练质量头、改loss或提出新方法。`m16`以**预测**argmax为中心，不读取GT；它描述模型已输出位置分布在该处的质量，不能等同“16px内正确的已校准概率”。一个高而错误的背景峰会同时有高`q`、高峰值和低熵，故任何正向相关都必须由固定的AUROC、分组分位数和逐clip报告实证，不能由形式推导。

本次问题属于“从已预测的离散位置分布读出定位质量”的最近邻审查，不是时空correspondence或运动可靠性审查。结论先行：**该诊断仍有实证价值，但“利用分布集中度估计定位质量/置信度”早已有直接先例；这个宽泛概念本身不足以成为本项目的独立创新点。** 它的价值是区分“多峰/分散导致高`q`错位”与“集中却错误的背景峰”，从而决定是否有理由继续研究读出或运动证据。

## 直接近邻

### 1. GFLv2：离散框边分布统计指导定位质量预测

Xiang Li et al., **“Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection”**, CVPR 2021；arXiv:2011.12885 v1（2020-11-25）。[CVPR论文](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html)，[论文PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.pdf)，[arXiv记录](https://arxiv.org/abs/2011.12885)。

- **实际读到的方法与分布。** 论文§3.1将每个box边距离散为General Distribution；四个`l,r,t,b`边各有一个在固定bin上的分布，坐标由期望求出（式2）。§3.2/式4从每边取Top-k概率及其均值，以反映分布“sharp/flat”，拼为统计特征；两层FC（ReLU、Sigmoid）预测IoU标量`I`。这不是中心热图、点格类别分布，更不是相邻帧flow或correspondence。
- **质量如何得到、监督与输入可用性。** DGQP以四条已预测边分布的统计量为输入，将分类`C`与IoU质量`I`相乘为`J=C×I`；训练时以Quality Focal Loss监督该联合分类–IoU表示，推理时`J`作NMS分数（§3.2、式3–5）。因此它是一个**新增、经GT IoU监督的质量预测器**，不是从固定输出直接宣布`max`或熵为质量。
- **它实际证明了什么。** §4.3报告正样本的预测IoU与真实IoU的Pearson相关，并称DGQP相对GFLv1提高0.026（表6）。论文没有报告ECE、reliability diagram、coverage或任何“概率已校准”保证；“reliable LQE”在这里是IoU质量估计与检测排序改进，不能移植为位置格条件概率校准。
- **对本项目的限制。** “从离散定位分布的峰/平坦统计得到定位质量”已是明确先例，不能作为创新措辞。当前实验反而更窄：不加DGQP，不用GT训练新分数，只检验空间位置读出是否补充现有`q`。由于GFLv2的四个独立bbox边分布与本项目单个72×128条件位置分布不同，它也不支持“tiny ball时空对应可靠”的结论。

论文原文给出的作者代码地址为`https://github.com/implus/GFocalV2`；在本次2026-09-11检索中该地址返回404，故未以非作者fork替代，也没有声称读到可复现的作者源码。

### 2. D-FINE：边界分布细化；作者代码仍含LQE读出

Yansong Peng et al., **“D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement”**, ICLR 2025 Spotlight；arXiv:2410.13842 v1（2024-10-17）。[ICLR论文PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/6cf58a87e3097e7d1f9be3e8693a93de-Paper-Conference.pdf)，[arXiv记录](https://arxiv.org/abs/2410.13842)，[作者项目](https://github.com/Peterande/D-FINE)。

- **实际读到的论文方法与分布。** 论文§4.1的FDR为每个候选框四条边维护离散offset分布`Pr^l(n)`；每一decoder层把残差logits加到前层logits后softmax（式3），以非均匀`W(n)`加权期望来修正框边（式2、4）。FGL以GT边offset附近两个bin的插值CE训练，并按IoU加权（式5）。这是一种**训练中的bbox回归中间表示**，不是本项目空间softmax的质量诊断，也不输出球位置邻域概率。
- **质量、监督和校准边界。** 论文的核心证据是COCO box AP/速度与回归细化；其GO-LSD把最终层的细化分布蒸馏给浅层。论文提到DDF对未匹配预测按分类confidence加权，但没有将FDR的熵、峰值或邻域质量作为定位质量分数来评估，也未报告概率校准指标或校准保证。因此不能把“D-FINE使用distribution”转述成“它证明任何分布集中度是校准定位置信度”。
- **作者代码的补充证据（非论文主张）。** 读取作者仓库在2026-08-19的固定提交`956d1709314c2c6a4df6f34de232054578a7449f`：[固定版本的Integral与LQE实现](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_decoder.py#L274-L313)的`Integral`对四条边分布softmax并作加权期望；`LQE`对每边取Top-4和均值，经MLP得到`quality_score`并加到检测score。调用处在[decoder中的调用](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_decoder.py#L427-L440)。这显示作者实现复用了“分布统计→质量读出”的家族，但该源代码位置本身不提供校准实验，且不能替代论文的证据范围。这里引用固定作者代码版本，不将它与2025年论文发表时的代码状态混同。
- **对本项目的限制。** 即使只引用当前作者实现，Top-k/mean+MLP质量读出也不是可主张的新方向；但本项目尚未新增MLP或改变检测分数，故本次只读`max`、entropy、邻域质量仍是必要的失败归因，而不是D-FINE式方法复刻。

### 3. 热图坐标峰值可作为不确定性读出，但不是校准证明

Lawrence Schobs, Andrew J. Swift, Haiping Lu, **“Uncertainty Estimation for Heatmap-based Landmark Localization”**, *IEEE Transactions on Medical Imaging*, 2022；arXiv:2203.02351 v2（2022-12-19）。[arXiv记录及全文入口](https://arxiv.org/abs/2203.02351)，[作者代码/数据](https://github.com/schobs/qbin)。

- **实际读到的方法与分布。** 附录A–B明确其U-Net回归以landmark为中心的Gaussian heatmap；PHD-Net由patch vote形成heatmap。单模型S-MHA取heatmap的argmax，使用该峰激活的倒数作为不确定性（式2–3）；E-MHA则先平均多个模型的heatmap（式4–5），E-CPV使用ensemble坐标离散度（式6–7）。这比box边分布更接近“位置图上的峰值”，但仍是医学landmark、Gaussian/投票热图与ensemble设置，不是本项目的K+1互斥softmax。
- **质量如何得到、监督与输入可用性。** 读出不训练新的质量头；S-MHA/E-MHA直接用预测热图，E-CPV需多个独立模型。论文在hold-out validation上以Quantile Binning划分不确定性等级，并用isotonic regression估计误差边界；它实证发现低不确定性子集可有更低误差，且最有用的信息主要在分布两端（§4.2–4.4、§6.3–6.4）。这支持“先测已有读出能否排序错误”的实验形状。
- **校准边界与限制。** 作者把heatmap activation称为pseudo-probability，并特意指出E-CPV可避免由热图失校准或Gaussian目标假设带来的偏差；误差边界是按该模型和留出集拟合，内层quantile并不稳定。故它没有证明单模型峰值、entropy或任意局部概率是普适校准概率。它限制本项目将`q*m16`称为置信度校准；本次没有进行独立校准或风险–覆盖评估，不能据分布数值直接承诺可迁移的拒绝规则。

## 2024–2026窄检索的覆盖与边界

检索日期为2026-09-11。只使用论文/作者项目等一手来源，未采用教程、博客或非作者复现。检索目标不是给所有检测器列综述，而是寻找**直接**把单模型离散中心热图、点格分布或其`max`/entropy/邻域质量用作定位正确性质量的近期工作。

- 检查了arXiv检索式`"heatmap localization" AND (uncertainty OR confidence OR entropy)`与`"point localization" AND (uncertainty OR confidence OR entropy)`，按submitted date倒序各取最多25条；并以`2024/2025/2026 heatmap localization uncertainty entropy confidence`、`2024/2025/2026 point localization heatmap confidence calibration`交叉检索。
- 在可访问结果中，D-FINE（2024预印本、2025发表）是唯一需深入核对的近期、与**离散定位分布**直接相邻的工作，但它属于bbox边回归与细化。[`Consistent-Point: Consistent Pseudo-Points for Semi-Supervised Crowd Counting and Localization`](https://arxiv.org/abs/2503.12441)（arXiv:2503.12441 v1，2025-03-16）只从标题/摘要筛出，其“instance-wise uncertainty calibration”用于半监督伪点的类别一致性；未见它将预测位置热图集中度作为定位质量，故未纳入深入来源，也不把它包装为最近邻证据。
- 没有在这个窄而可访问的检索面中找到2024–2026论文，能直接证明“单模型中心位置softmax的max、entropy或预测邻域质量”已校准为点定位正确概率。**这只是本次检索未命中，不是完整无遗漏的否定结论。** 2022的Schobs等被保留，是因为其方法段直接检验了热图argmax峰与定位误差，相关性高于仅因发布日期更新而纳入的泛化不确定性论文。

## 对当前只读诊断的研究判断

1. 保持既定一次forward和预先固定的分组/排序，不调阈值、不选新checkpoint、最终测试集不进入本项测量，统计不改变正在运行的训练。它直接回答`q`遗漏的是分散/多峰证据，还是高而错误的背景峰；已有`q`、坐标CSV无法回答这个问题。
2. 若`q*m16`在整体、旧阈上子集与多数clip均不优于`q`，或错例同样低熵且高局部质量，应停止“仅给现有空间读出加集中度”这条解释；此时重点转向背景竞争、输出网格/监督或时序视觉证据，而非增加质量模块。
3. 即使排序改善，也只能报告“该固定模型的现有空间分布在此验证协议中含有与16px正确性相关的额外读出信息”。不能宣称新定位质量方法、新运动可靠性、时空correspondence、概率校准、F1提升或可迁移拒绝规则；更不能把bbox分布工作等同于微小球的大位移时空匹配。

因此当前最小动作已经足够：收集并分析既有logits。跳到DGQP/LQE head、loss、阈值搜索、ensemble或D-FINE式边分布，既不能由上述文献推出，也会破坏这次诊断的归因。
