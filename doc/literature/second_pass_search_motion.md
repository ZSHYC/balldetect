# 第二轮深检：弱证据的时序积累、Track-Before-Detect 与稀疏全局对应

> **问题边界。** 本轮只查一个更窄的问题：一个目标在任何单帧都不够显著时，能否先沿可能轨迹积累证据、再声明候选/目标；在大位移下怎样不穷举所有轨迹；以及这种逻辑与 learned sparse global matching 的关系。它不重复第一轮已写的 OTHR、MOCID、DQAligner、MIST、DeepPro、DMR、CMRTrack、BIRD、MI-DETR、FlowIt、SCV、Devon 或 CVPR 2026 EACVS。
>
> **检索与证据规则。** 主体检索截至 **2026-09-09**，重点窗口为 2024–2026；**2026-09-12**在第9节补读FRT与FaXT的光学弱目标全文。A = 已读取原始官方全文/PDF的相关方法和实验段；B = 官方出版页/官方全文网页或摘要可访问，但没有逐式审计 PDF；C = 仅官方摘要或书目信息，不能据此推断细节。主体列10项近邻，增量阅读单独标明，不声称穷尽。

## 结论先行

“在显著性/候选筛选前利用时间积累弱球证据”不是空白；它在 **track-before-detect (TBD)** 中是经典主张，在最新 learned TBD 中已具有很接近的形态：从原始序列提 frame feature，做时序 accumulation，预测 trace-center confidence，再回归整条轨迹，最后 threshold。最直接的先例是 [NeuralTBD, IJCAI 2024](https://www.ijcai.org/proceedings/2024/0674.pdf)。

因此，不能仅凭下列宽泛概念声称“首次”或把它们本身当作新颖性；在具体任务约束下，更有效或更可验证的机制改进仍可能成立：

* “不先单帧检测、而是先时序积累再产生候选”；
* “用轨迹中心监督训练时序积累和轨迹 proposal”；
* “用 learned dynamic programming / state-aware pruning 控制 TBD 预算”；
* “用 sparse affinity / sparse global matching 替代 dense all-pairs”；
* “在不可靠 correspondence 下改用区域支持、融合 prior 与 appearance”。

对本项目仍可能成立、但必须实验建立的窄命题是：

> 对于连续体育 clip 中的高速微小球，现有前筛选（单帧 objectness、低分辨率 query、局部时间混合）会在保留足够的细粒度空间证据前丢掉正确路径；一个 **只在有限、显式计量的 wide-displacement path/support 集上** 积累高分辨率局部证据的机制，能在不使用测试时 GT query、不过 clip 边界、且不假定匀速的条件下，提高 pre-decision true-path coverage 与最终中心定位。

这不是“把 TBD 搬到球检测”。最强反证很现实：短窗 frame difference / motion-map、NeuralTBD 式 accumulate-then-propose、或者普通 coarse-to-fine detector 已得到相同 recall；另一反证是大位移和击球/反弹使 path state space 爆炸，所提有限候选并没有覆盖真实中心。

## 1. 最接近的学习式 TBD：候选之前的 accumulation 已被完整实现

### 1.1 NeuralTBD：Learning-Based Tracking-before-Detect for RF-Based Unconstrained Indoor Human Tracking

**Z. Wu et al., IJCAI 2024；证据 A。** [官方论文 PDF](https://www.ijcai.org/proceedings/2024/0674.pdf)，[IJCAI 条目](https://www.ijcai.org/proceedings/2024/674)。

这是本轮最重要的遗漏近邻。它不在 RGB 视频中做球定位，而是在多人、遮挡、多径干扰的 RF heatmap 上做轨迹检测；但其 computation order 与候选研究问题高度一致。

原文实际做了什么：

1. 输入固定长度 heatmap sequence；先抽取逐帧 deep feature。
2. 用 LSTM temporal accumulator 跨扩展时间积累 feature。
3. 由 accumulated feature 预测一个 **trace-center confidence heatmap**；以全轨迹中心的 Gaussian heatmap 加 focal loss 监督。
4. 对每个 trace center 回归每个时间步的二维 offset，形成 trace proposal；此后才按 confidence threshold 去假警。

因此它不是“先每帧 box detector，再用 Kalman filter”。原文明确将其对立于 detect-then-track：TBD 直接处理 raw/minimally processed data，跨帧沿可行轨迹累积能量；同时说明其网络采用 fixed window，对长序列滑窗运行。论文 Table 3 的例子还显示增加时窗从 6 到 24 帧时指标继续提升、48 帧不再稳定提升——时窗并不可以无限加长。

**对新颖性的否证。** 若本项目提出“时序 feature accumulate → heatmap candidate → trajectory/path offsets → threshold”，这几乎是 NeuralTBD 的视觉任务迁移，不能称为新的计算次序。若提出 “center supervision protects temporal evidence”，NeuralTBD 已直接使用 trace-center supervision 和 offset regression。

**不能直接照搬的差异，也是实际风险。**

* RF heatmap 的多径/噪声与 RGB 纹理、球场线、高光、motion blur 不同；它的效果不能转移为球结果。
* 它训练时拥有序列内完整 trace center 与逐时刻 offset。TrackNet Tennis、BlurBall、Shuttlecock 等密集中心 clip 可以构造此监督；OpenTTGames 的未标注帧不可以补零成 background，也不能回归完整 trace。
* 固定窗口内的 human movement 与球在击球、反弹、遮挡处的加速度/方向突变不同。以 constant-velocity 或任意单峰 state 压缩 ball path 都可能失效。
* 它的滑窗是否因果取决于窗口对输出帧的配置；本项目必须分别声明 causal 和 offline（可用未来帧）协议，不能把后者的结果称作实时。

**建议的最小比较。** 若论文主张“候选前的时序积累”是关键，在同一 backbone/输入/窗口下比较 `(a)` per-frame heatmap + temporal head，`(b)` difference/motion-map + current locator，`(c)` **受 NeuralTBD 启发的思想适配对照**（不是声称忠实重现其 RF 算法），`(d)` 所提 sparse wide support accumulation。输出不只最终 F1：应报告 accumulation 之后、threshold 之前的 `GT path/center coverage@K`，再报告 path-to-center / final localization。若 `(c)` 已达到 `(d)`，新增 matcher 的必要性就很弱。

### 1.2 Learn to Track-Before-Detect via Neural Dynamic Programming

**E. Fishel, N. Tsarov, T. Tapiro, I. Nuri, N. Shlezinger；ICASSP 2024；证据 B。** [机构出版记录和 DOI](https://cris.technion.ac.il/en/publications/learn-to-track-before-detect-via-neural-dynamic-programming/)，DOI [10.1109/ICASSP48485.2024.10448128](https://doi.org/10.1109/ICASSP48485.2024.10448128)。原 PDF 本轮不可获取，以下只复述正式出版摘要。

该工作把 Viterbi/DP 与 DNN 结合：DNN 不需手写精确 measurement model，state-aware pruning 用来降低 TBD 的计算量，实验在物理一致的 range–Doppler radar measurement 上进行。

**对新颖性的否证。** “传统 DP 路径过多，所以学习一个评分网络来稀疏/剪枝轨迹状态”在 2024 年已是正式工作。若你的 wide-search 是随时间维护少量 path hypotheses，必须承认 Neural DP/TBD 是直接先例。

**球任务的真正分歧。** 它的运动 state 在雷达坐标/物理模型中定义；体育 RGB 中像素位移还受相机摇移、变焦、透视和球的非匀加速影响。不能无证假定一个小离散 velocity state space 足以覆盖球。应实际测：每个时间间隔下，top-K support/path 是否覆盖标注中心；当击球/反弹时是否突然下降；扩大 K 的真实 memory/latency 代价多少。

## 2. TBD 不是一句口号：它的基本交换、存在不确定性和状态爆炸已有清楚反例

### 2.1 Unified Analysis for Dynamic Programming Track-Before-Detect Algorithms: Error Convergence and Spatial Uncertainty

**N. Bampton, T. J. Ma, M. N. Do；arXiv:2512.11170 v2（2026-06-23）；证据 A，但预印本。** [官方 arXiv](https://arxiv.org/abs/2512.11170)。

这篇工作面向 1–9 pixel 的高空 IR target，明确讨论 DP-TBD 对弱路径作多帧积分、以迭代最大化降低计算。它在自身分析模型、阈值构造和假设条件下给出一个对项目很有价值的提醒：随观测增加，target **existence** 的置信度可以上升，而 **location** 的不确定性可能上升；其 NPI 以 sequential observation 的 similarity 而非直接 intensity integration 建路径。这不是“更多观测必然使 Bayes 最小风险变坏”的一般信息论结论。

**对新颖性的否证。** “积分多个帧增强微弱目标”甚至“用 observation similarity 做路径积分”均不是新；不能将 temporal evidence accumulation 说成一种新原理。

**对设计的硬约束。** 不应只报“是否找到了球”或只有 final heatmap 最大值。至少分开：

* path/existence score 是否把真球路径排到前 K；
* 条件于真 path 被保留时，每一帧 center 的定位误差；
* K、时间窗长度、允许位移范围变大时，location uncertainty / false path 如何变化。

这也是多假设的合理性：多 path 并非为“更复杂”服务，而是避免早期把不确定位置平均成一个不存在的轨迹；但 NPI/DP 已使“保留路径不确定性”失去首次性。

### 2.2 Adaptive pseudo-spectrum based track-before-detect for targets with uncertain existence

**P. Li, G. Zhou；IET Radar, Sonar & Navigation 18(5), 2024；证据 A（官方全文）。** [原文](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12502)。

原文针对连续滑窗中 target appearance/disappearance time 不确定：若把 noise-only frames 也累积，会污染目标能量、带来 decision delay。它以 appearance/disappearance time 与 velocity 一并匹配，只向“informative frames”积累。

**对球项目的具体否证。** “没有可靠 correspondence 时不要强行累积/应拒绝无效帧”不是新的 reliability 想法；它是 TBD 中已明确的问题。并且这比二元 confidence gate 更精确：是 `frame-in-path support` 是否纳入积分的问题。

**可迁移但必须谨慎的实验。** 有可信 visible center 的 dense clip 可按一个 pair/时间位置是否对应来做 support mask 诊断；标签明确不可定位时禁止 correspondence accumulation loss。未标注帧不能叫 noise-only 或 target-absent，因此 OpenTTGames 只能把这些帧作视觉上下文，不能拿来监督 “skip”。此外，球短时在画面不可见与摄像机跟球导致近零 displacement 的情形不能同化。

### 2.3 Message Passing for Track-Before-Detect

**M. Liang, F. Meyer；arXiv:2506.18177（2025-06-22）；证据 B，预印本。** [官方 arXiv](https://arxiv.org/abs/2506.18177)。

该工作不是图像论文，而是 radar/sonar/communications 的 factor-graph TBD。它的直接启发不是模块移植，而是问题定义：detect-then-track 的前端 measurement extraction 是**不可逆信息压缩**，对 weak/close objects 尤其有害；TBD 在 raw data 上做 inference，但必须处理 amplitude fluctuation 与 measurement correlation。

**对项目的影响。** 可以用它支持“候选阈值不应在全部 temporal evidence 使用之前把低分球删掉”的问题陈述，不能据此声称端到端 optical/RGB 方法已有相同结果。它也指出某个必要的反实验：

* 对相同计算，early threshold 的 candidate pipeline 与 late threshold 的 path/support pipeline 比较召回；
* 记录被 early filter 删除、但在后续时序证据下应变为 true 的球；
* 同时记录 late accumulation 多出的 false paths，否则只说明召回交换而未说明算法更好。

### 2.4 DK-TBD：two-stage detection method based on improved DP-TBD for marine weak extended targets

**Digital Signal Processing 160 (2025) 105054；证据 B。** [出版方摘要](https://www.sciencedirect.com/science/article/pii/S1051200425000764)。

DK-TBD 先以 DP-TBD 的 virtual nodes 粗检测，后以 kernel correlation filter re-track/refine。它的动机是经典 DP-TBD 在环境因素导致孤立 target 时会断掉；其 virtual candidate association 让断点进入轨迹生成。

**对新颖性的否证。** “粗 trajectory/path support 后再局部 template/相关性精修”“候选暂时丢失时应以虚拟节点维持路径”已有明确 two-stage TBD 先例。

**为何仍不能直接当球方案。** 目标是海面 weak extended radar target，KCF 模板在高速、低纹理球上可能没有稳定外观；若本项目套用，应把它作为对照思想而非宣称对应。最关键的测试是人为把一帧或一个短段的球 evidence 削弱/遮挡：系统是以正确 path bridge 过去，还是以错误背景高光 bridge 过去？两者需要分别计数。

## 3. 有限预算 wide displacement：稀疏全局关联和小点匹配的新增直接先例

### 3.1 LEAP-Track：Learning to LEAP: Efficient Dense Point Tracking by Focusing Where It Matters

**C. Zhao, W. Wang, B. Zhang, W. Wang；AAAI 2026；证据 A。** [AAAI 论文页与官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/38311)，[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38311/42273)。

这是“sparse global matching supports saving”的最新强近邻：它针对 TAP 自监督点跟踪，明确把 dense all-pairs attention/correlation volume 视为会引入 irrelevant feature、降低效率与学习信号质量的问题。方法用 curriculum sparse attention 选相关 key，再把 contrastive random walk 的 affinity graph 逐步改成 k-NN sparse transition。PDF算法段将 dense `(HW)^2` affinity 变成 `HW×k` sparse affinity。

**对新颖性的否证。** “不做所有高分辨率 all-to-all，而在 learned/adaptive sparse pairs 上保留 global correspondence；以 progressive top-k/kNN 保持关系”不是球项目的新结构。也不能只报告 FLOPs：LEAP 的动机本身包含 irrelevant correspondence 会污染学习这一质量问题。

**球任务仍有的差别。** TAP 是给定 point 查询的 tracking，ball localization 必须先自动发现 point；将 GT ball center 作 query 只能是 oracle diagnostic。并且 kNN feature match 的 top-K 可能全是球场线/广告纹理而无真球；“稀疏”绝不是“覆盖”。

**必须的实验。**

* 固定 `K` 与 relation elements，比较 random/grid、local-window、coarse global top-K、learned top-K；报告历史 GT center 的 rank/coverage 和最终定位。
* 以自动 current-frame candidate 和 GT center query 各跑一次；两者落差就是 discovery bottleneck。
* 计入 top-k selection、sparse gather、backbone、decoder 的端到端 latency/peak memory；不可只写 correlation head FLOPs。

### 3.2 Sparse Global Matching for Video Frame Interpolation with Large Motion

**C. Liu, G. Zhang, R. Zhao, L. Wang；CVPR 2024；证据 A。** [官方 CVPR PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Sparse_Global_Matching_for_Video_Frame_Interpolation_with_Large_Motion_CVPR_2024_paper.pdf)，[项目页](https://sgm-vfi.github.io/)。

该论文在 high-resolution feature 上先取局部细节并估初始 flow，随后只对初始 flow 的失败处做 sparse global matching compensation，最后 adaptive merge local flow 和 global compensation。它针对的大位移瓶颈与项目高度同构，但任务是 VFI 与 intermediate-flow，不是弱目标自动检测。

**对新颖性的否证。** “细特征保证细节 + sparse global branch 补偿 local mismatch + adaptive merge”是一条成熟组合，不能作为球论文的主要结构创新。

**可以保留的球问题。** VFI 有图像重建损失和大量像素的 signal；小球可能只占数像素，且初始 flow failure detector 可能在球已经消失时根本不给它预算。若项目以它为机制参考，必须额外证明 sparse compensation **在哪里被启动**：由 current objectness、由跨帧 relation uncertainty、还是由全局背景 failure？每种触发器对 candidate recall 的影响应报告。

## 4. 一项最新多帧 weak-target 紧邻：imperfect correspondence 下的 region support

### 4.1 MISA：Multiframe Infrared Spatiotemporal Aggregation for Small Target Detection Under Imperfect Interframe Correspondence

**IEEE JSTARS 2026；线上 2026-08-10；证据 B。** [IEEE 出版记录和开放 PDF 入口](https://ieeexplore.ieee.org/abstract/document/11646933/)，DOI [10.1109/JSTARS.2026.3722252](https://doi.org/10.1109/JSTARS.2026.3722252)。IEEE PDF 本轮未逐页解析，故只据出版方摘要陈述。

MISA 的起点几乎逐字命中本项目风险：weak appearance、background/platform motion，temporal accumulation 在 motion estimation 不可靠时会把 misalignment residual 与目标一同放大。它用 region-based temporal prior generation，替代 isolated point support 的 multiregion joint tracking，再用 asymmetric fusion 避免 temporal prior 干扰 appearance。

**对新颖性的否证。** “因为单点/单一对应不稳，所以用多个空间 support 生成较可靠 temporal prior”“appearance 与 temporal prior 不对称融合”均已有 2026 近邻。多 support / relation reliability 不可当新名词。

**项目应从它吸取的反证，而不是模块。** 多区域背景 support 对 IR platform motion 可能可靠，对球场的相机跟球、非共面球、快速局部球员运动未必成立。建议把任何 multi-support prior 当作可拒绝的辅助，不设为 ball existence 的必要条件；与单点、无 prior、错误 support/random support 做控制。只有在低候选分数、高 displacement 或明确 camera-motion proxy bucket 同时降低误匹配，才值得保留。

## 5. 最新深层时序模型只是必要强基线，不是 correspondence 的证明

### 5.1 A Mamba-like spatio-temporal attention network for detecting infrared moving small targets

**W. Zhang et al.；Geo-spatial Information Science，online 2026-06-26；证据 B。** [出版方全文页](https://www.tandfonline.com/doi/full/10.1080/10095020.2026.2686054)，DOI [10.1080/10095020.2026.2686054](https://doi.org/10.1080/10095020.2026.2686054)。网页可读主体与实验摘要，但不以网页摘录重构全部网络细节。

该文明确以 long-range interaction、inter-frame self-attention、cross-layer connection 处理多帧 IR small target，特别声称 cross-layer connection 可防 pooling 后 tiny feature 丢失；其时窗敏感性分析还显示更多帧不必单调提高，快速目标/平台运动场景需要不同 T。

**对项目的否证。** “现代 sequential backbone + long-range temporal module + high-resolution/shallow skip 就可保 tiny target”已经是近期发表的完整路线；不能在没有 2×2 归因的情况下把 DINO + 时序层的增益称作 wide correspondence 的收益。

**最低基线。** 同主 backbone下，必须包含一条没有 explicit cross-position matcher 的 long-range temporal mixer / state-space 或 temporal-attention baseline，并让它拥有同一 high-resolution detail branch。若它已经获得所有收益，wide correspondence 不成立为必要机制。

## 6. 从这批工作得到的第一性原理：积累不是免费午餐

设每帧 high-resolution evidence 为 `e_t(u)`，一种 TBD 形式是在所有允许路径 `z=(u_1,…,u_T)` 上评价

`S(z) = Σ_t a_t e_t(u_t) + Σ_t ψ_t(u_t, u_(t+1))`。

这只是统一记号，不是新公式。文献已经展示三个不可回避的交换：

| 交换 | 为什么对球更尖锐 | 不能靠什么掩盖 |
|---|---|---|
| 弱证据可经多帧变强，但可行路径数随位移/时间增长 | 球可小而 `rho` 很大，击球/反弹让 velocity transition 突变；固定小 window/path grid 必漏。 | 不可只展示成功轨迹；要测每 K、每 Δ 的真实 GT coverage。 |
| 放宽 path state 防漏球，也积累假路径 | 球场线、高光、衣服纹理可在多个帧形成“稳定但错误”的 evidence。 | 不可只测 recall；要测 false path count、precision/risk-coverage、错误类别。 |
| 长窗加强存在证据，却可能降低位置/时刻确定性 | 滑窗、未来帧、同一轨迹中可见性变化会把一个当前帧定位任务变成后验轨迹平滑。 | 不可把 offline result 写成 online/real-time；须报 causal delay 和每帧中心误差。 |
| 剪枝降低计算，也可能在弱球尚未显著时删去唯一真路径 | 高分辨率稀疏方案特别容易出现 “candidate before evidence” 漏检。 | 不可只报 head FLOPs 或 oracle-query matching；须报 pre-prune true-path survival。 |

### 对本项目的一项重要区分

* **TBD evidence accumulation** 回答“这一条时空路径是否包含足够总证据，可被声明为球？”
* **correspondence** 回答“当前位置在历史哪个位置是同一个对象？”
* **trajectory prior** 回答“在还没有视觉证据时，哪些路径在运动上可行？”

三者可合作，但不能混用。一个强 trajectory prior 可以在球被遮挡时输出位置，却没有提供当前帧的直接视觉定位证据；一个 global match 可以给出一帧配对，却未积累弱信号；一条 blur streak 是曝光内积分，也不是帧间 path。论文应明确所提分支输出的是 evidence、relation 还是 prior，并为三者各做消融。

## 7. 面向现有公开球标签的可证伪实验门槛

### 7.1 不实现复杂 TBD 之前必须做的三个诊断

1. **early-vs-late decision 曲线。** 在同一 dense-label clip，将 per-frame candidate threshold 改为保留 top-K / lower threshold，再做固定廉价 temporal accumulation。画 `candidate/path recall@K`、false candidates、F1/center error、latency；若 late decision 没有净收益，TBD 主线停止。
2. **可行路径覆盖而非匀速假设。** 直接从标注中心测相邻帧 displacement、加速度 proxy 和方向突变。对每个 proposal/path transition family，测其是否容纳真实 `p_(t+1)`；击球/反弹/高 displacement 分桶不可被平均分数掩盖。
3. **高分辨率证据是否真存在。** 用 GT center 仅作 probe，测试不同 feature level 中真历史中心的 rank；然后换 auto candidate。若 oracle probe 可行而 auto candidate fail，论文应优先检查 proposal/TBD；若 oracle 都不可行，只能说明当前 feature、匹配尺度、评分或 matcher 的至少一环失效，不能据此排除 matcher 是瓶颈。

### 7.2 允许的监督与不可声称的监督

| 资料状态 | 可以训练/评测 | 不可以做 |
|---|---|---|
| 连续 dense ball centers，端点均可信可见 | center heatmap、endpoint offset/rank、短 window path-center supervision | 称为 dense optical-flow GT 或把中心扩成 patch flow |
| 遮挡/不可见但数据集给出合法、可验证的位置协议 | 按该协议保留定位/轨迹监督并单列结果；屏蔽不可验证的视觉 correspondence loss | 当成 no-ball / noise-only frame 来惩罚所有 path，或把推断位置伪称直接视觉对应 |
| 明确不可定位且没有可验证位置 | 关闭直接定位与 correspondence loss；可单列 evaluate visibility 语义 | 当成 no-ball / noise-only frame 来惩罚所有 path |
| OpenTTGames 未标注中间帧 | 作为 model input/context；保留真实 time index | 重新编号成连续标签、当背景或 trace 负样本 |
| clip/rally 边界 | 重置 state/window/path cache | 用跨边界 path 训练“wide displacement” |

NeuralTBD 的 fully annotated long trace 不能由稀疏公开球标签免费得到。若没有连续 dense labels，优先将 TBD 当作 **test-time feature aggregation 的受限模块**，并只在可监督子段训练/验证其 path loss；不要把缺标签位置伪装成 “no match”。

### 7.3 一个足够小但有辨别力的比较矩阵

固定 backbone、输入像素、frame window 和总 relation/path budget：

| 组 | 决策时刻 | 跨帧结构 | 目的 |
|---|---|---|---|
| S0 | 每帧 | 无显式 time relation | 现代空间基线 |
| S1 | 每帧 | difference / temporal mixer | 测便宜 time cue 是否已足够 |
| S2 | 后筛选 | 受 NeuralTBD 启发的 accumulate→center/path proposal 对照，不声称忠实复现原算法 | 排除“先积累后候选”这个已有解释 |
| S3 | 后筛选 | 所提 sparse wide support/path relation | 只在 S3 增益被 S0–S2 排除后才有研究价值 |

主张时序积累或稀疏 path search 时，所选对照应尽量报：完整 pipeline 位置指标；`pre-decision center/path recall@K`；条件于真 path/candidate 被保留的 relation/center accuracy；false path/candidate count；端到端 latency、peak memory；按 displacement、blur、visibility 和 camera-motion proxy 分桶。`S2` 是最贴近的思想适配对照，但不要求每个阶段性项目逐一实现全部 TBD、Mamba 或其他外部模型；应选择最能反驳当前主张且资源可承受的对照。

## 8. 文献清单与检索充分性

1. Wu et al. **Learning-Based Tracking-before-Detect for RF-Based Unconstrained Indoor Human Tracking**. IJCAI 2024. [Official PDF](https://www.ijcai.org/proceedings/2024/0674.pdf). A.
2. Fishel et al. **Learn to Track-Before-Detect via Neural Dynamic Programming**. ICASSP 2024, pp. 9586–9590. [Official institutional record](https://cris.technion.ac.il/en/publications/learn-to-track-before-detect-via-neural-dynamic-programming/). B.
3. Bampton, Ma, Do. **A Unified Analysis for Dynamic Programming Track-Before-Detect Algorithms: Error Convergence and Spatial Uncertainty**. arXiv:2512.11170v2, 2026. [Official arXiv](https://arxiv.org/abs/2512.11170). A, preprint.
4. Li, Zhou. **Adaptive pseudo-spectrum based track-before-detect for targets with uncertain existence**. IET Radar, Sonar & Navigation 18(5), 2024. [Official full text](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12502). A.
5. Liang, Meyer. **Message Passing for Track-Before-Detect**. arXiv:2506.18177, 2025. [Official arXiv](https://arxiv.org/abs/2506.18177). B, preprint.
6. **A two-stage detection method based on improved DP-TBD for marine weak extended targets**. Digital Signal Processing 160, 2025, 105054. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S1051200425000764). B.
7. Zhao et al. **Learning to LEAP: Efficient Dense Point Tracking by Focusing Where It Matters**. AAAI 2026. [Official proceedings/PDF](https://ojs.aaai.org/index.php/AAAI/article/view/38311). A.
8. Liu et al. **Sparse Global Matching for Video Frame Interpolation with Large Motion**. CVPR 2024. [Official PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Sparse_Global_Matching_for_Video_Frame_Interpolation_with_Large_Motion_CVPR_2024_paper.pdf). A.
9. Zhang et al. **A Mamba-like spatio-temporal attention network for detecting infrared moving small targets**. Geo-spatial Information Science, online 2026. [Publisher full-text page](https://www.tandfonline.com/doi/full/10.1080/10095020.2026.2686054). B.
10. **MISA: Multiframe Infrared Spatiotemporal Aggregation for Small Target Detection Under Imperfect Interframe Correspondence**. IEEE JSTARS 2026. [IEEE record](https://ieeexplore.ieee.org/abstract/document/11646933/). B.

**未确认 / 刻意未纳入。** 本轮没有把雷达/声呐 TBD 的一般成果误写成 RGB ball 定位的实证；它们仅定义问题和算法空间。MISA、Neural DP、DK-TBD 的全文细节有访问限制，故没有臆测 loss、backbone 或全部实验设置。也没有把已经由其他笔记覆盖的 SCV、Devon 与 EACVS 再次作为“新增”。未进行代码复现、数据下载或跨任务复验；本笔记不能证明上述思想会在网球/羽毛球/乒乓球数据上获益。

## 9. 2026-09-12 补充：直接沿运动假设累积证据

前述TBD近邻之外，本次继续沿[9月10日astro-VAE新稿](2026-09-12-recent-tiny-motion.md#同日增量补检2026-09-08-至-09-12)的直接引用，补读两个光学弱目标来源。目的不是将天文算法照搬到球场，而是检查：**微弱目标必须先建立逐像素对应，才能使用motion吗？** 两篇提供的反例是先按受限路径族累积观测，再产生候选。

### 9.1 单帧拖影：Fast Radon Transform 的收益与离散支撑

Guy Nir、Barak Zackay、Eran O. Ofek，[*Optimal and Efficient Streak Detection in Astronomical Images*](https://arxiv.org/abs/1806.04204v2)，AJ156(5):229，2018；已读v2方法、预处理、模拟/真实实验及附录。缓存为 `outputs/literature/streak-frt-1806.04204v2.*`。

它对PSF展宽的直线做匹配滤波，以短线段的shift-and-add复用，将N×N图的离散长直线搜索从约 `O(N³)` 降为 `O(N² log N)`。灰度先累积、后阈值，避免先把每个弱像素变成二值候选。最优性依赖指定模板、已知PSF和去背景后的独立高斯噪声；不自动覆盖未知形状、结构背景和全模板搜索的检测率。[§II–III、附录A](https://arxiv.org/html/1806.04204v2)

短线搜索读各二分尺度的中间和，端点落在不利分块位置时SNR可减半，长度只是近似；降低候选阈值再局部细搜会增加后续工作。真实盲检有效，但仍需点源/背景处理，且论文的极低误警率来自分布尾部拟合，不是实测了百万张无目标图。[§IV.3、VI–VII](https://arxiv.org/html/1806.04204v2)

它限制“首次沿拖影方向保留弱信号”的主张，同时保留一个具体问题：粗触发、采样覆盖和精细位置不是同一件事。这里的线是**曝光内拖影**，不能当帧间位移；背景线也可被正确检出，却仍不是球。其预处理和方向屏蔽不应直接成为体育球的保留条件。

### 9.2 FaXT：未知起点与速度的四维搜索

T. Nguyen、D. F. Woods、J. Ruprecht、J. Birge，[*Efficient Search and Detection of Faint Moving Objects in Image Data*](https://doi.org/10.3847/1538-3881/ad20e0)，AJ167(3):113，2024。已完整阅读[第一作者提供的九页PDF](https://tamz.umd.edu/publication/nguyen-2024-efficient/)，缓存为 `faxt-nguyen-2024-author.pdf`、`faxt-nguyen-2024.txt`；未运行实现，也未确认arXiv版本。

FaXT在所有 `(x0,y0,dx,dy)` 上累积帧值，**无需GT起点或速度**，之后才筛选四维峰。它递归共享短路径，但其整数路径由二分构造，和VMF逐帧直线插值再取整并不完全相同；不能称为逐数值等价的加速。[作者全文§2–3](https://tamz.umd.edu/publication/nguyen-2024-efficient/)

在每轴速度0–1px/frame、论文所用 `1/N_f` 速度分辨率及二次幂帧数下，Eq.7为 `Q=2N_xN_yN_f(N_f−1)`；各维同尺度增长时为 `O(N⁴)`，VMF为 `O(N⁵)`。最终假设内存仍约 `N_xN_yN_vxN_vy`。高斯点目标模拟中，256³数据约90s对80min，检测率接近。[§2–3](https://tamz.umd.edu/publication/nguyen-2024-efficient/)

TESS盲搜先去静态背景，以128帧累积并用64个CPU节点；2145条轨迹中1997与目录关联，148未关联的性质尚未确定，**93%不能称为precision**。更快目标、进场与非线性轨迹是扩展问题；作者提出预移位或检出后细化，并未证明当前基线已覆盖它们。[§4–5](https://tamz.umd.edu/publication/nguyen-2024-efficient/)

因此它是受限运动族中“先累积、后发现”的强先例，不是高分辨率任意运动的免费全局搜索。短窗口、宽速度范围、反弹和动态背景会改变其可用条件；不能仅靠帧数短就假定该方法适合当前三帧模型，也不能因为来自天文就忽略其计算复用思想。

### 9.3 第一性原理上的三个区分

**路径、证据和身份。** 给定路径 `p_theta(t)`，累积 `E_t(p_theta(t))` 与比较两帧descriptor相似度是不同统计量。前者可使用每帧很弱的目标响应，而不要求存在唯一、稳定的表面点descriptor；它同时依赖路径族和响应的目标含义。正确沿某条运动背景累积也不能自动确认球身份。以上是机制区别，不证明哪种在本地更好。

**运行复用和误警机会。** 对固定观测、固定各模板打分，候选集合 `A` 包含于 `B` 时，有 `max_A score <= max_B score`。在同一阈值下扩大搜索域不会降低空场景的触发概率；共享计算不会消除这些候选。这个结论不需要候选独立，但要求原分数不因候选集改变而重新计算。带联合attention或重新softmax的模型不直接满足此条件。

**检测能量和中心信息。** 已知正确模板时累积SNR提高，不等于未知路径更容易找到，也不等于找到后中点误差足够小。[连续拖影模型的推导与CPU核对](second_pass_measurement.md#25-2026-09-12-补充检测拖影与定位中点需要不同信息)进一步表明：固定单位长度亮度时，延长拖影可增加检测能量和垂轴信息，却不持续增加沿轴中心信息；固定总通量时又是另一组变化。不能用一种SNR图代替严格位置、召回和背景误报。

当前没有新增FRT、FaXT、VAE或轨迹模板训练分支。如果后续实际研究累积，应先明确受限路径族、真实窗口、保留的观测及完整搜索成本，再在同一自动定位协议上检验。现有中心标签只能监督其定义的位置，不能因此生成未标注帧的假真值。
