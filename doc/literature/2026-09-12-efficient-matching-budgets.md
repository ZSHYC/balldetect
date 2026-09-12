# 高效对应的不同改变：消息、候选、相关值执行与 cost memory

日期：2026-09-12。本文围绕当前问题“看得细、找得远、计算可承担”，补读 Efficient LoFTR、CasP 和 Briedis 等人的相关体采样工作；第五节进一步补足 SCV 和 FlowFormer 的原始信息路径。主文、相关补充和公开实现的阅读范围如下；未运行这些模型，没有新装依赖、下载权重或改变正在进行的 BlurBall 对照训练。

## 结论：不能把不同预算改动都叫作新的 motion representation

这三篇直接限制了三种宽泛创新说法：“压缩全局注意力再恢复细节”“粗尺度全局发现、细尺度只看少量候选”“不保存全相关体而按需计算”。它们已有明确先例，而且有各自任务上的有效实验。对本项目仍未解决的是：**自动球定位中，哪些真实历史证据没有被有效利用，改变哪一步能改善严格位置误差，付出多少实际成本。**

| 工作 | 主要改变 | 仍存在的全局步骤 | 对微小球研究的约束 |
|---|---|---|---|
| Efficient LoFTR，CVPR 2024 | 聚合 query/key/value，减少 attention 成对计算，再融合原位置特征 | 最后的 1/8 全图相似度矩阵 | 消息压缩不等于删掉所有细节；attention 变便宜不等于末端全局匹配也消失 |
| CasP，ICCV 2025 | 1/16 全局生成 top-k prior，限制 1/8 interaction 和 matching 的离散地址 | 1/16 全图相似度；粗特征融合到细层 | 是粗全局加细候选的直接前史；离散地址缺项不自动成为连续坐标召回上限 |
| Briedis et al.，CVPR 2026 | 将 RAFT 已请求的 correlation lookup 按硬件友好的块计算并直接输出 | 采样器没有新增全局发现；沿用原 flow 网络的更新路径 | 数学上保留所请求算子，可减少存储/执行成本；它不负责找到尚未请求的远处球 |

这与[ASpanFormer 的范围和样点问题](2026-09-12-adaptive-search-support.md)互补：消息中保留什么、实际访问哪些地址、如何执行这些访问、最后允许输出哪里，必须沿完整路径判断。本文的形式分析是对方法的推论，不是已测得的本地失败。

## 一、来源与实际阅读范围

| 工作 | 直接来源与版本 | 实际阅读 |
|---|---|---|
| Yifan Wang et al., *Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed* | [CVPR 2024 正式论文，pp. 21666–21675](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.pdf)；[arXiv 2403.04765 v2，2024-03-11](https://arxiv.org/html/2403.04765v2)；[作者页](https://zju3dv.github.io/efficientloftr/) | 主文方法与实验、arXiv 附录，尤其 Table 10/12 原页；作者 attention、coarse/fine matching、FPN、配置与监督源码 |
| Peiqi Chen et al., *CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance* | [ICCV 2025 正式论文，pp. 28063–28072](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_CasP_Improving_Semi-Dense_Feature_Matching_Pipeline_Leveraging_Cascaded_Correspondence_Priors_ICCV_2025_paper.pdf)；[arXiv 2507.17312 v2，2025-08-01](https://arxiv.org/html/2507.17312v2) | HTML 全文方法、监督、实验和 Table 6/7；作者 prior 索引、partial matching、局部 patch、分类及 homography 回归。正式 PDF 已缓存，文本转换异常，未冒充完整读出 |
| K. M. Briedis et al., *Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation* | [CVPR 2026 正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Briedis_Efficient_All-Pairs_Correlation_Volume_Sampling_for_Optical_Flow_Estimation_CVPR_2026_paper.pdf)；[作者补充材料](https://studios.disneyresearch.com/app/uploads/2026/05/Efficient-All-Pairs-Correlation-Volume-Sampling-for-Optical-Flow-Estimation-Supplemental-Paper.pdf)；[arXiv 2505.16942，首稿 2025-05-22](https://arxiv.org/abs/2505.16942) | 正式主文与相关补充，Table 2、复杂度式 (3)/(4) 原页；补充 Listing 2–4 的 CuTe DSL 计算/采样/线程协作。未编译 kernel 或复现速度 |

公开仓库按已有 Git 版本固定引用，不新增内容指纹：

- Efficient LoFTR：[`zju3dv/EfficientLoFTR@ffd4a46`](https://github.com/zju3dv/EfficientLoFTR/tree/ffd4a4644064354468eb1f0c7a3e732233cb732f)，2025-07-30。
- CasP：[`pq-chen/CasP@cee1dc3`](https://github.com/pq-chen/CasP/tree/cee1dc30e5133683360fcb790d6e431e3767916f)，2026-01-04。版本晚于论文，只用于核对机制，不替代论文实验事实。
- Briedis：本轮没有找到独立可核验仓库，但补充材料**已经公开 kernel 清单**。不能把“未找到仓库”写成“没有公开实现”；完整接入、编译环境与 backward 尚未在本地验证。

PDF、提取正文与只读源码保存在 `outputs/literature/` 下的 `efficient-loftr-*`、`casp-*`、`efficient-correlation-volume-sampling-*`，按现有项目规则不提交缓存原件。

## 二、Efficient LoFTR：压缩消息路径，保留原位置路径

### 实际结构

输入为一对图像，先用 RepVGG 提取多尺度特征。在 1/8 特征上交替做 self/cross 的 aggregated attention；公开默认聚合边长为 4：query 经 stride-4 depthwise convolution，key/value 经 4×4 channel-wise max pooling，再进入相应投影与 attention。更新消息双线性上采样，与未聚合的当前位置特征拼接，经 FFN 后残差相加。[论文 §3.2](https://arxiv.org/html/2403.04765v2)、[固定版 `transformer.py`](https://github.com/zju3dv/EfficientLoFTR/blob/ffd4a4644064354468eb1f0c7a3e732233cb732f/src/loftr/loftr_module/transformer.py)

若同尺度两图各有 P 个 token，双方聚合后各剩 P/16，vanilla attention 的成对项由 P² 变成 P²/256。这只是该 attention 项的计数；不是整网快 256 倍，也不是相对原 LoFTR 的 linear attention 有这个倍率。

这里有两个容易误解的信息问题。

第一，channel-wise max 不等于“挑中窗口内某一个最显著像素的完整描述子”。例如两个位置的两通道向量为 `(4,0)` 和 `(0,5)`，池化可得 `(4,5)`，它不属于任何一个原位置。这个数学例子说明聚合消息不再保留逐位置身份；没有证明真实球信号一定被抹掉。

第二，完整更新仍保留未池化的 feature 残差。池化分支的信息损失不能直接推成整层、整个 backbone 或最终定位的信息不可恢复；原特征也可能已包含较大范围上下文。其限制应表述为“跨图消息以何种空间粒度组织”，而非“网络只看得到池化后的图”。

### 最终匹配并未变成稀疏全局搜索

更新后的 1/8 特征仍构造完整两图 `L×S` 相似度矩阵。full 配置使用双向 softmax、阈值与 MNN；opt 配置跳过双向 softmax，但仍计算相似度矩阵。删去归一化改变候选竞争与阈值语义，不是语义完全不变的 kernel 替换。[固定版 `coarse_matching.py`](https://github.com/zju3dv/EfficientLoFTR/blob/ffd4a4644064354468eb1f0c7a3e732233cb732f/src/loftr/utils/coarse_matching.py)

fine stage 通过 FPN 融合浅层特征，再按 coarse pair 抽局部 patch。默认 source 是 8×8、target 是带边圈的 10×10；前 C−8 通道先建立 64×100 相似度并双向归一化，随后裁去 target 外圈，用中央 64×64 confidence 选一个 pixel pair。最后 8 个通道为选中 target 邻域的 3×3 回归提供信息，DSNT 输出一个连续修正。不是最终保留九个不同 motion hypothesis。[`fine_preprocess.py`](https://github.com/zju3dv/EfficientLoFTR/blob/ffd4a4644064354468eb1f0c7a3e732233cb732f/src/loftr/loftr_module/fine_preprocess.py)、[`fine_matching.py`](https://github.com/zju3dv/EfficientLoFTR/blob/ffd4a4644064354468eb1f0c7a3e732233cb732f/src/loftr/utils/fine_matching.py)

论文描述 fine MNN 后取 top-1，源码直接取矩阵全局最大；全局最大本身也是行列最大项，不能仅因代码未单独写 MNN 就制造矛盾。训练会补入 GT coarse pair，推理没有该补入，所以训练 fine 样本覆盖不等于推理候选覆盖。

它最终输出通用图像对应，不选择“哪一对是球”。即使局部 patch 内背景获得了正确且稳定的几何对应，也没有完成球实例识别。反过来，某个真值 coarse cell 未被选中，也不够证明最终球点不可能出现：必须检查其他保留 patch 及完整细化坐标域，不能把单个索引的漏选直接写成全模型硬失败。

### 应接受的实证与成本边界

附录 Table 10 在 ScanNet 上给出 AUC@5°：默认卷积 query / 池化 key-value 为 19.2，两端都卷积为 18.6，两端都池化为 18.3；把消息上采样移到 FFN 之后为 17.3。这支持其具体交互和细节融合方式的条件收益。它未单列几像素球，也不证明混合聚合普遍最佳。

主文 Table 1 在 RTX 3090、640×480 图像对上分别报告 full FP32 40.1 ms、mixed 34.4 ms；opt FP32 35.6 ms、mixed 27.0 ms。对应 AUC@5°，full 到 opt 在 MegaDepth 为 56.4→55.4，在 ScanNet 为 19.2→17.4。**速度与精度取舍是实测事实，不应只取最快一列。**

附录 Table 12 还将耗时分到 backbone、coarse transform、coarse matching、fine fusion 和 refinement；其中 full 40.1 与 opt 27.0 ms 同时改变归一化和数值精度。因此不能把 13.1 ms 差全部写成删掉双向 softmax 的收益。以上为作者计时，不能转换成本机球定位 FPS。

## 三、CasP：细层地址限制明确，最终坐标域却不是同一个集合

### 粗全局到细候选的链条

CasP 先在 1/16 特征上构造全局相似度，每个 token 双向取 top-k prior；默认 k=8。每个粗地址映射到四个 1/8 子地址，所以 RSCA 中每组四个 query 对 32 个 target token 做交互。1/8 matching 的推理归一化也只在相应 prior 地址内计算，并要求匹配满足双向 prior 约束。粗特征在此之前已融合到 1/8 特征，不能将细特征误作纯局部原图裁剪。[论文 §3.3、式 (5)–(14)](https://arxiv.org/html/2507.17312v2#S3.SS3)

这与 Efficient LoFTR 的“压缩 attention，最终仍全图 1/8 matching”不同；CasP 真正省去了细层完整匹配矩阵。但 1/16 全局相似度仍有空间二次项，不能把整个系统写成随图像像素数线性。

训练时 coarse prior 会注入 GT 对应，推理没有；两尺度匹配监督由相机姿态和深度生成，另有 fine pixel/subpixel loss。它没有 GT 球初始化，但也没有球类别和本项目 visibility/no-match 标签。因此既不能称为给定球点 tracker，也不能称为完成自动球发现。

### 为什么不能草率画一条“候选覆盖 = 最终召回上限”

作者实现先按保留 coarse pair，在 1/2 特征上用 `unfold(kernel=5,stride=4,padding=2)` 取局部 patch；先做 8×8 网格分类，再由 `FineHomo` 回归变换。最终 `fine_reg_biases * 4` 加入坐标，源码没有按候选 cell 或 patch 半径作输出裁剪。[`fine_preprocess.py`](https://github.com/pq-chen/CasP/blob/cee1dc30e5133683360fcb790d6e431e3767916f/src/models/nets/casp/fine_preprocess.py)、[`fine_homo.py`](https://github.com/pq-chen/CasP/blob/cee1dc30e5133683360fcb790d6e431e3767916f/src/models/nets/casp/homo/fine_homo.py)、[`casp.py`](https://github.com/pq-chen/CasP/blob/cee1dc30e5133683360fcb790d6e431e3767916f/src/models/nets/casp/casp.py)

因此必须分别判断：

- **细层直接匹配地址。** 某个地址不在相应 prior 中，就没有该项 RSCA/partial-softmax 的直接计算。双向候选约束会进一步限制可输出的离散 match。
- **特征已携带的上下文。** 粗全局交互、跨尺度融合与卷积感受野仍可能带来远处信息，不能说未读取的 fine cell 在整个网络中毫无信息路径。
- **最终连续坐标。** 它可通过已有 patch 的回归越过 cell 边缘；没有固定裁剪就不能从 cell 集合推导固定最终坐标边界。坐标越界也不自动证明找回了该处视觉对应。

一般地，只有证明最终输出 `p_hat ∈ D(S)`，其中 D(S) 是已保留候选 S 的**全部**解码坐标域，才能用真值落在 D(S) 的覆盖率约束最终位置成功率。对容差 ε，应使用 D(S) 的 ε 邻域。若最后是未裁剪回归，这个界可能不提供任何有用约束。若保留单帧定位回退分支，也必须将其输出域纳入，而不能只看 motion 分支。

这是推理边界，不是替 CasP 宣称成功恢复了漏选球。需要检验实际错误时，离散候选覆盖、直接采样支撑、最终位置误差应各自报告；不要给一个不存在的有限召回上限赋值。

### 候选归一化的数值不是跨预算通用的正确率

在有限 logits 不变时，若 A 是完整地址集合、S 是包含 j 的子集，则

`exp(z_j) / sum_{i∈S} exp(z_i) ≥ exp(z_j) / sum_{i∈A} exp(z_i)`。

仅由缩小归一化集合造成的数值上升，本身不证明对应 fine logit 或视觉证据增强；候选集合 S 的生成仍可能携带额外的粗层视觉证据。CasP 的双向 partial softmax 也受各自竞争集合影响；完整模型中 feature 同时变化，不能用该不等式预测实际分数变化大小。这只是说明不同 K、不同归一化和不同模型的 confidence 不应直接当成同一已校准正确率。当前不据此新增校准 head。

### 不能忽略作者已经做出的组件替换证据

Table 6/7 不只有完整 CasP 对完整 Efficient LoFTR：还把 cascaded matching 接入 ELoFTR。ETH3D 室内、标准 RANSAC 的 AUC@5°，EL w/ DS 为 49.1，EL+CM-full 为 52.3；对应 FP32 耗时 238.3→144.6 ms。这支持**级联匹配组合**在该设置的收益，不能笼统说完全没有机制消融。

同时参数为 16.0→17.6 M，且该替换包含 coarse prior、细层 interaction 与匹配规则；它并未孤立证明“只删 top-k 之外的地址”就有这个收益。完整 CasP-lite 又改变低层通道及交互结构，365.1 GMAC、108.1/67.7 ms 的结果也应按整系统解释。

上述 Table 7 使用论文的单张 V100、1152 long-side 条件，FP32/FP16 分列。不能与上一节原论文 RTX 3090、640×480 的 ELoFTR 数字混成速度排名。[CasP §4.5、Table 6/7](https://arxiv.org/html/2507.17312v2#S4.SS5)

## 四、Briedis：保持所请求相关值的数学定义，改变实际执行

### 它计算的是已请求地址，而不是替模型发现新地址

RAFT 的某次 lookup 对每个源位置，根据当前 flow 中心读取 target 上的局部邻域及金字塔。若固定 feature、query center、半径、插值与边界约定，这些结果可以从完整相关矩阵取，也可以只计算所需内积后插值。Briedis 改善后一条路径的硬件执行。[正式论文 §3–4](https://openaccess.thecvf.com/content/CVPR2026/papers/Briedis_Efficient_All-Pairs_Correlation_Volume_Sampling_for_Optical_Flow_Estimation_CVPR_2026_paper.pdf)

作者采用 patch-major 排列，使二维邻近特征在相关矩阵中更集中；将请求合并为块乘法，在 fused kernel 内决定下一个所需块、计算、采样并直接写出结果。省去完整相关体以及显式块 mask 的物化。Sintel 中约 1.6% 的访问比例是作者在特定 flow/迭代设置的观测，不是所有视频或高速球的定律。

这里的“保留”是数学算子层面的：没有通过 top-k 抛弃这次 lookup 本来请求的值。实际 kernel 使用 BF16 输入、FP32 累加，主文 §5 报告相对官方实现约 0.03% 的 EPE 差；故不能声称逐位一致，也不能用全图 EPE 接近来代替未来真实球区域的精度验证。已有 CuTe 清单使实现可查，不表示本地编译、完整反向传播与球区域精度都已通过。

### 线性复杂度成立在什么条件下

设 P 为每幅 feature map 的位置数，D 为 feature channel，K=2r+1 为 lookup 边长，B 为二维特征块边长。每对源/目标块包含 B²×B² 个相关项，块乘法成本为 B⁴D。补充 §8.5 的时间与空间上界分别是：

`O(P (K+B)² B² D)`，`O(P (K²+D+B²))`。

固定 K、B、D 及层数/迭代数时，对 P 线性。它不是任意大范围全局 matching 的通用线性算法；若为覆盖全图而让 K 随图像边长增长，这个条件已经改变。默认 r=4、B=8 的最坏块请求数为每个源位置 9 个；算法还通过同块源位置的请求复用减少实际工作。[补充式 (3)/(4)、Algorithm 1、Listing 2–4](https://studios.disneyresearch.com/app/uploads/2026/05/Efficient-All-Pairs-Correlation-Volume-Sampling-for-Optical-Flow-Estimation-Supplemental-Paper.pdf)

单步 lookup 外的地址没有被这个采样器发现；下一次 flow update 可以移动中心，所以也不能将一次漏覆盖直接当成全网络不可恢复。高分辨率改善还包含另一个动作：作者为 SEA-RAFT 引入 test-time cascaded initialization，在输入最短边超过 800 px 时递归缩小图像、估计 flow，再给高分辨率阶段初值。大位移质量改善因此不能单独归给新的采样 kernel。

### 性能证据的具体口径

主文 Table 2 对完整 flow 方法按 Default / On-Demand / Ours 分列。8192×3432 的 SEA-RAFT，dense 超过该表 80 GB 的可运行边界，on-demand 为 2.63 s / 11.82 GiB，新采样器为 1.78 s / 11.82 GiB。正文默认计时硬件为 GH200；80 GB 是此处的可运行界限，不能凭表中 OOM 推断计时机器只有 80 GB。

这组数值针对普通 SEA-RAFT。带 cascaded initialization 的 8K SEA-RAFT 在 Table 1/补充 Table 11 中另报 1.91 s，并伴随不同 EPE；不要将其质量与前一行 1.78 s 拼成一个不存在的配置。

8K CHARGE 的 335 帧、332 对 forward flow 是作者合成评测；补充 §9.2 和渲染脚本明确关闭 motion blur。高分辨率、全图 EPE 和大位移分组不是 BlurBall 曝光模糊或球中心误差的证据。主文结论又明确只优化 forward，backward 留待后续；本项目若未来端到端训练，不能继承其推理内存作为训练预算。

## 五、SCV 与 FlowFormer：保留稀疏地址，还是压缩关系再按需读取

这两篇此前只有摘要或方法概览，现补读原文和源码。它们直接回答“motion latent 里究竟还保留了什么”，不扩展到整个后续家族。

### SCV：全局 top-k 保存之后，还有一层位移编码

Shihao Jiang et al., *Learning Optical Flow from a Few Matches*，CVPR 2021，pp. 16592–16600；[arXiv 2104.02166 v1，2021-04-05](https://arxiv.org/html/2104.02166v1)，已读全文与作者 [`zacjiang/SCV@9f80991`](https://github.com/zacjiang/SCV/tree/9f809910e17125701b28dc1054fbc7648b801957)（2021-04-08）的 KNN、稀疏相关、位移编码与迭代输出路径。缓存为 `scv-2104.02166v1.{pdf,txt}`、`scv-source/`。

**真正存储的是带地址的匹配值。** 每个 source 在 1/4 feature grid 上对 target 全图选择 top-k，默认 k=8；一次计算相似度并保存地址/值，后续按当前 flow 平移相对位移。源码的候选索引选择在 `no_grad` 中，对选中 feature pair 重新计算可求导内积。不能据此说未选中的 target feature 在整个网络没有梯度，它仍可能被其他 source 选中，骨干参数也共享。[`knn.py`](https://github.com/zacjiang/SCV/blob/9f809910e17125701b28dc1054fbc7648b801957/core/knn.py)、[`sparsenet.py`](https://github.com/zacjiang/SCV/blob/9f809910e17125701b28dc1054fbc7648b801957/core/sparsenet.py)

**稀疏存储不消除候选生成的全局计算。** 原文 §3.1 明确使用 Faiss brute-force exact search，源码调用 `faiss.bfKnn` 和 inner-product metric。对 P 个 source/target 位置，它仍有全对搜索工作；Nk 存储不是整个 pipeline 的线性时间证明。原文还报告当时 Faiss 约 2 GB 的固定开销，故在 1/8 尺度下节省并不明显；这属于当时实现，不是今天所有 KNN 库的固定开销。

**保存在 sparse volume 中，不等于该轮已经传给 GRU。** 每轮把相对候选坐标缩放为五个尺度（1、1/2、1/4、1/8、1/16），双线性 splat 到各自 9×9 位移格，再拼成 405 通道的 motion 输入。多个候选可以落入相同格并累加；超出所有当前窗口的候选不会在这一轮贡献直接相关值。源码先对四个整数邻格筛选，边缘可能只保留部分 splat 权重，不能把它简化成对连续坐标的一次硬圆形半径截断。

但窗口中心会随迭代移动，source context 与邻域 GRU 也仍存在；不能从单轮漏读推导最终 flow 必然失败。更直接的是，原文 §4 已明确说明最终解不必位于初始 top-k 集合。这里的 recurrent iteration 是同一图像对内的更新，不是额外视频帧，也不自动提供长期运动轨迹。

**正、负证据都很接近本项目问题。** 原文明确讨论 small objects moving fast，且 Fig. 5 展示细竹竿的改善。Table 2 在同一 SCV 家族中将 feature 从 1/8 提至 1/4，Sintel clean EPE 1.55→1.29、final 3.07→2.95；KITTI EPE 却由 5.74→6.80。k=1/4/8 的消融支持保留多个候选的价值，但不是每个指标都严格随 k 改善。Fig. 6 又给出模糊头发的失败，作者将其解释为歧义使正确对应未进入 top-k；它是具体定性证据，不是对所有模糊场景的失败率统计。

Table 3 的 correlation-value 字节估算与整网训练显存分开报告；1/4 特征、400×720 crop、batch=1 时，作者测得 RAFT 10.6 GB、SCV 6.1 GB。该事实支持特定设置的实际节省，不代表可忽略索引、Faiss、其他 activation 或本项目 batch 条件。它没有球中心/visibility 指标，也不把通用正确对应自动识别为球。

### FlowFormer：latent 外面仍保留 raw cost

Zhaoyang Huang et al., *FlowFormer: A Transformer Architecture for Optical Flow*，ECCV 2022；[官方正文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770672.pdf)、[arXiv 2203.16194 v4，2022-09-21](https://arxiv.org/html/2203.16194v4)、[官方补充](https://drinkingcoder.github.io/publication/flowformer/images/FlowFormer-supp.pdf)。已读正文/补充及 [`FlowFormer-Official@6ba7ea8`](https://github.com/drinkingcoder/FlowFormer-Official/tree/6ba7ea82b45394d3a7a25808399695427c9febd8) 的 encoder、decoder、AGT 和配置。缓存为 `flowformer-eccv2022*` 与 `flowformer-official-6ba7ea8/`。

**先全局构造，再学习压缩。** 1/8 feature grid 首先形成完整四维 all-pairs cost。每个 source 地址各自拥有一张 target cost map；它经过 stride-8 patchification 和 target 位置编码，接受共享的 K 个 learned query 聚合，形成该 source 的 latent memory。默认 K=8、token dimension=128；八个 token 不是 top-8 target 地址，也没有与八个峰一一对应的保证。[`encoder.py`](https://github.com/drinkingcoder/FlowFormer-Official/blob/6ba7ea82b45394d3a7a25808399695427c9febd8/core/FlowFormer/LatentCostFormer/encoder.py)

源码的 patchification 是三层 kernel=6、stride=2 的重叠卷积并增加通道，不是简单的非重叠 8×8 平均。它不再显式枚举每个 target 格点，但不能仅从 stride 下降断言球信息必定丢失；可学习通道仍可能编码细位置。有限 latent 对真实窄峰是否有足够可读性，需要实验，既没有无损保证，也不能凭压缩名称证明丢失。

AGT 先在同一 source 的 latent tokens 内交互，再沿 source 空间网格让同一 latent index 交换信息，并利用 source appearance。source 地址仍由 memory 外层索引保留，target 区域信息则通过 patch 特征与位置编码汇入 latent。它不是把整场景压成一个无地址向量，也不等于建立了显式目标身份。

**默认 decoder 读取两条关系路径。** `data['cost_maps']` 一直保留原始相关图；每轮在当前 `p=x+flow(x)` 周围取 raw 9×9 cost，以其编码和当前位置形成 query，对该 source 的 latent memory 做全局信息检索。默认还把检索结果与 raw patch 拼接后送给 GRU，输出一个 residual flow。[`decoder.py`](https://github.com/drinkingcoder/FlowFormer-Official/blob/6ba7ea82b45394d3a7a25808399695427c9febd8/core/FlowFormer/LatentCostFormer/decoder.py)

所以 compressed memory 不代表整个原始相关图被删除；局部 9×9 也不代表整个网络无法作远距离更新。远处细峰是否得到利用，取决于 feature、latent 检索、迭代位置和最终 raw 读取的共同作用。AGT 的 latent residual 又是另一条保留路径，不能与 raw-cost bypass 混写。

初始全局相关仍保留二次项。完整高分辨率推理若采用补充材料的 paired tiling，则每个 tile 的全局仅针对其实际输入域；跨出该 target tile 的地址不在该次相关图中。是否发生取决于实际裁剪和位移。当前也没有本机证据证明项目输入上的 1/8 all-pairs 必定不可承担，不能以此提前排除基线或强制采用压缩。

**结构收益不能一概归给更大模型。** Table 2 分开了 K/D、骨干预训练与 AGT 的配置；Table 3 又用 small FlowFormer（6.2 M）对 GMA（5.9 M）：Sintel clean/final 为 1.20/2.64 对 1.30/2.74，KITTI 也有改善，并提供扩大 GMA 和更换 Twins 的对照。这是有效的归因证据，应接受其支持完整设计的条件收益；它没有独立检验微小球远峰保留，也没有把 raw bypass 单独作为本项目变量。

### 对本项目的共同约束

SCV 已经保留少数带地址的远距假设，FlowFormer 已经将全局关系压成 memory 后结合局部 raw evidence。不能分别把“多假设”或“全局 latent 加局部细化”写成首次。二者的标准任务都依赖 dense flow 监督；通用 flow 成绩不替代公开球中心标签条件下的自动定位。

本项目更值得检验的是：**候选被产生、被保存、在本轮被读到、被最终用于球定位，是四件不同的事。** 这些环节只有在实际模型中出现时才测；当前不预建四套模块或把所有近邻列为必跑基线。

## 六、这些证据实际改变什么决定

**先把成本算在正确位置。** 若以后实际采用少数地址 lookup，应先复用成熟的按需计算路径。没有必要为了论文动机构造一个低效 P×P 张量，也没有必要现在就移植专用 CuTe kernel。只有本地 profiling 证明相关值执行是瓶颈，专门优化才值得投入。

**把搜索与读出问题收窄。** 当前不能只问“窗口是否够大”。应在所选模型中明确：远处候选如何产生，是否保留带地址的多个假设，进入细层的球证据是否有效，背景正确对应会不会压过球，最后坐标允许从哪里产生。具体测量随实际模型决定，不预建统一诊断框架。

**区分更便宜与更有信息。** 保持同一 lookup 的数学结果而减少时间，属于执行改进；改变 token 聚合、候选集合、关系分布或归一化，才改变模型收到的匹配信息。两者都可以有价值，但论文归因不同，不能用同一个“motion增强”覆盖。

**当前继续完成已有对照。** 本轮没有证据要求立即采用三者之一。BlurBall 真历史对重复当前帧的完整训练与配对评价先完成；之后只围绕实际收益/残留错误选一条有区分力的机制实验，避免为了排除所有理论可能性而复现整套图像匹配谱系。

## 七、最新稿件的覆盖边界

本轮同时定向筛查 2025–2026，包含 2026 年 8–9 月 arXiv 条目；没有做穷尽检索。Briedis 与 CasP 因直接改变本轮预算判断而深读，并补足作为前史的 Efficient LoFTR。

初筛后的 [Semi-Dense Matching Uncertainty Is Not Just Local Confidence v2（2608.08685）](2026-09-12-coarse-fine-uncertainty.md)已在同日另行深读，记录整体残差模型、几何输入、有效消融及公开后验公式的版本差异。[REDI-Match（2606.24330）](https://arxiv.org/abs/2606.24330)仍仅初筛，涉及旋转等变蒸馏；EDM（ICCV 2025）暂未深读。后两项不计入已核实的机制结论；搜索未命中也不作为不存在的证据。
