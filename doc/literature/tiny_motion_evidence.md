# 高速微小目标 motion / correspondence：对抗性文献核验笔记

> **范围与日期。** 本笔记只审查“高速微小球定位”最接近的外部证据：微小目标光流、移动红外小目标检测、UAV 小目标跟踪、视频检测中的跨帧采样、高分辨率稀疏计算，以及大位移光流。检索和网页读取截至 **2026-09-09**；不是穷尽性综述。尤其是 2026 年论文中有数篇只是 arXiv 预印本，不能在论文中写成已同行评审结果。
>
> **阅读深度。** A = 已读官方全文/官方 PDF 的方法与实验关键段；B = 已读官方摘要和官方作者仓库 README，足以限定主张，未逐式审计；C = 仅核对官方书目信息/摘要，不能据此复述实现细节。链接均为原始论文、出版方或作者官方仓库；仓库不能替代论文证据。

## 0. 先给结论：哪些表述已经不能作为创新

以下五种“模块级新颖性”都已有非常近的先例，不能作为本文的首要贡献：

| 候选表述 | 最强已知反证 | 对本项目的含义 |
|---|---|---|
| 高分辨率但只在少量位置计算 | QueryDet 从低分辨率粗 query 引导高分辨率稀疏计算；它直接以小目标的高分辨率计算冗余为动机。 | 不能声称首个高分辨率稀疏搜索。球任务仍可研究的是：**候选尚未被检测到时**如何保证跨帧真对应覆盖，而非把 QueryDet 式目标 query 直接搬过来。 |
| 先远后近 / global-local matching | GMFlow 先全局匹配再局部残差细化；FlowIt 用全局最优传输初始化、置信度引导细化。 | “大位移全局匹配 + 局部精修”本身不新。真正困难是球在高分辨率里弱、全局特征里近乎不可辨，且不允许用 GT query。 |
| 全局相机/背景运动与局部目标运动分离 | DMR 显式建模全局 coherent motion、以其引导局部 anomaly；EgoSIS 又把图像平面 transition、残差支持、可靠性因子化。 | 不能声称首个 global/local motion decoupling 或 camera residual。可研究“这种残差在像素级球定位中是否实际提高候选/对应”，但要避免把二维背景流残差称为物理球运动。 |
| 多个对应假设、置信度或 no-match | SELF(Y)/STSS 保留时空相似关系；FlowIt 输出 occlusion/confidence；CMRTrack 用反事实历史学习可靠 motion；光流已长期处理遮挡。 | top-k、soft distribution、confidence gate、occlusion/no-match 都不是新概念。必须明确 no-match 的语义、监督和最终定位收益。 |
| 运动上下文/位移建模可提升 tiny target | OTHR、MOCID、MIST、MI-DETR、BIRD、DQAligner 都针对微小且复杂/快速运动。 | 不能以“既有微小目标方法没有 motion 或只用帧差”为动机。应把贡献收束为一个可测的 **高分辨率弱证据 × 大位移候选覆盖 × 自动发现球** 失效。 |

更可防守的论文命题不是“发明一种 multi-hypothesis sparse matcher”，而是条件式：

> 在现有公开球中心标签可验证的连续 clip 中，发现并量化：当当前帧的细粒度位置证据仍可读出、但局部窗口不覆盖真前帧中心时，现有空间/时间基线如何在 **自动候选阶段** 漏掉真球或把高响应背景当对应；随后证明一个限定预算的候选—关系表示在相同 backbone、输入分辨率、候选数和端到端延迟下提高真对应覆盖和最终中心定位。

这仍可能有研究价值，但不是预先成立的创新。最可能的反证是：提高输入分辨率、合适的前景重加权、短 temporal convolution / frame difference，或者 QueryDet 风格的粗检测，已经解释全部收益。

## 1. 已核验的直接近邻

### 1.1 OTHR / FlyingTO — tiny optical flow 的最强“不要把 loss 和结构混在一起”反证

**证据与状态：A。** [IJCAI 2025 正式论文页](https://www.ijcai.org/proceedings/2025/136) 与 [官方 PDF](https://www.ijcai.org/proceedings/2025/0136.pdf)；[作者仓库](https://github.com/JaneEliot/OTHR)。官方页称对象小于 100 pixels，并把 OTHR 接到 RAFT/FlowFormer；仓库 README 只明确提供 FlyingTO **sample** 的下载入口，不能假设完整数据已经可无障碍复现。

* 它针对的问题正是普通 flow 在 tiny object 上失败；论文指出多尺度 pyramid 对大位移有效，但对 tiny object 没有显著优势（PDF p.2）。OTHR 利用明/暗极性、方向和时间延迟构造方向选择性运动算子，再接入已有 flow network。
* 它不等价于球中心定位：监督是稠密光流，评测也含 tiny object 邻域的 flow 指标；球数据多为中心点，不能把一个中心位移复制为对象 patch 的稠密 flow 真值。
* 它是对“只加方向算子/前景 loss 就是贡献”的强反证。原蓝图摘录的 Table 2 数值应在正式写作前重新从 PDF 表格逐格转录；本轮已读 PDF 的相关工作与方法段，但**未逐单元格复核该表**，所以不在此把具体 EPE 数字作为二次证据。

**对设计的压力测试。**

1. 若方法加入 polarity / direction filter，必须和“同样参数的普通卷积/差分”、对象平衡 loss 分开 2×2 消融；否则可能只是监督重新分配。
2. 不能拿全图 EPE 或背景匹配精度证明球对应有效。应报告中心邻域（仅在有可信中心时）的候选召回、真中心 rank、点误差和假阳性。
3. OTHR 的对象阈值（<100 pixels）不是球直径标注；本项目不能把它外推为自己的球尺寸统计。

### 1.2 MOCID — motion context 与 frame displacement 已是正式 tiny-target 检测主线

**证据与状态：B。** [AAAI 2025 官方 proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/33087)（发表 2025-04-11，pp.10022–10030，摘要/元数据已读）；[官方仓库](https://github.com/TanzanOY/MOCID) README 几乎没有可审计实现说明。

* MOCID 从 clip level 用 Fourier-inspired spatio-temporal attention 获得 motion context，再把它写入动态卷积核；frame level 用 temporal interpolation 与 displacement-aware scan 建模帧间位移。
* 因此“现有多帧 tiny-target detection 没有明确 motion/displacement guidance”不成立。它同时提醒：概念名为 Mamba/Fourier 并非贡献边界；必须讨论实际的候选搜索或对齐行为。
* 其输入是红外小目标检测，通常目标/背景统计和球赛 RGB 不同；但它足以否定泛化的“motion context + displacement”的首次性。没有阅读全文/代码，不能声称它保留了多峰 correspondence 或实现了全局 query。

### 1.3 DQAligner — large-motion tiny target 的 global query、跨尺度双向 attention 与 deformable alignment

**证据与状态：B。** [IEEE TGRS 2026 记录/摘要](https://ieeexplore.ieee.org/document/11363482/)（DOI `10.1109/TGRS.2026.3657842`，线上发表 2026-01-26）；[官方仓库](https://github.com/dengfa02/DQAligner_MIRSTD) README、训练/测试入口已读。IEEE 页面受 JavaScript 限制，未取到全文；因此以下只复述出版方摘要和作者 README 明示内容。

* 论文明确针对 complex motion 与 large displacement，提出 global random large-displacement augmentation、cross-scale bidirectional shared attention、动态 receptive-field pyramid deformable convolution，以及迭代 class query memory（CQM）。出版方称其从 rigid alignment 转为 flexible matching。
* 这直接覆盖“高分辨率 feature 上做 global dynamic query / 随后 dynamic alignment”的模块级说法。其仓库还明确警告 spatial deep supervision 和 `track_loss` 默认关闭，因为可能与深浅 query learning 冲突；这只是作者实现观察，不可泛化成理论结论。
* **2026-09-10源码补读修正：** DQAligner 的类query从可学习初态出发，在窗口内读取全空间特征，输出dense mask；公开路径没有先从当前检测器取硬候选。不能给它强加“当前候选先漏”的结构判断，也不能说query必然恢复球。窗口内GRU、跨窗口参数未实际使用以及因果采样边界见[专门审查](2026-09-10-dqaligner-query.md)；论文全文仍未取得，源码证据不替代全文。

**修正后的比较边界：** 若今后采用稀疏时序候选，报告自身candidate recall@K及条件匹配，并与自动dense系统比较共同任务的最终定位/检测。DQAligner没有必需的硬候选阶段，不要求它报告虚构的内部候选召回；从dense输出另取top-K时需明确这是额外诊断规则。GT query结果不能代替自动发现结果。

### 1.4 MIST / MISTNet — irregular fast motion 的多尺度隐式补偿和困难子集

**证据与状态：B。** [IEEE TIP 条目](https://ieeexplore.ieee.org/document/11511399/)；[官方仓库](https://github.com/ShuCvlab/MIST) README 已完整读取。仓库说明 2026-04-27 接收、数据/代码/权重于 2026-03-08 发布；发表页因 JS 未读到全文。

* MIST 是合成引擎构建的 airborne IR benchmark，目标有姿态/尺寸/强度变化与真实背景融合；不能把其性能直接和 RGB 球赛结果比较。
* MISTNet 的 Shifted Neighborhood Compensation Block (SNCB) 声称以多尺度 correspondence 做隐式 motion compensation，Progressive Distillation Decoder 过滤与目标无关的信息。
* 其官方 hard split 是 11 条测试序列，条件为 SCR≤1、轨迹高度不规则、速度 >7 pixels/frame。这个**分桶思想**很适合项目：主报告不能只有均值；应预注册由真中心算出的 displacement/速度分桶、blur bucket、camera-motion proxy bucket。但切勿挪用 `>7 px/frame` 作体育球“高速”的物理阈值。

**对创新的冲突。** “不规则/快小目标需要多尺度匹配补偿”和“压制无关 motion”已是已发表或近发表系统的主题。可区分之处必须来自可验证的球中心对应和计算-空间分辨率权衡，而不是词汇。

### 1.5 DeepPro — 最危险的效率反证：时间 profile 可能已足够

**证据与状态：B。** [arXiv:2506.12766](https://arxiv.org/abs/2506.12766)（最初 2025-06-15）和[作者仓库](https://github.com/TinaLRJ/DeepPro)已读；蓝图所列 IEEE TPAMI 链接可访问性受限，故在本文将正式发表状态保守标为“作者仓库给出 TPAMI 2026 citation，未由本轮出版全文复核”。

* 它把红外小目标任务重构为每个空间位置的长时 1D temporal profile anomaly detection，强调 temporal saliency/correlation，并主要只沿 time 做计算；仓库给出 40-frame 训练序列设置。
* 这不是跨位置 correspondence：固定像素位置的 time profile 在相机移动、球有大位移时可能完全不保留同一目标。因此它不能否定球的 wide-search 需求。
* 但它强迫项目做廉价强基线。若一个 time-only/temporal-profile 分支（加当前帧 spatial head）在网球、羽毛球已达到方法结果，那么大规模 correspondence 并非必要贡献。

### 1.6 DMR — camera/background coherent motion 与 local anomaly 的最直接概念冲突

**证据与状态：A（但仅预印本）。** [arXiv HTML v1，2026-06-13](https://arxiv.org/html/2606.15286) 已读摘要、动机、方法概览和关键公式；没有正式会议/期刊版本可据此声称。

* DMR 的核心观察是背景运动在空间上 global coherent，而 tiny target 呈 sparse localized anomaly；它以 pretrained optical-flow prior 显式 branch 建 coherent motion，以 deformable alignment 隐式 branch 建 target-sensitive local motion，再以两者一致性压制背景 false response。
* 这已经覆盖“camera/global motion 与 object/local motion decoupling”“用全局运动先验指导局部对齐”“背景一致的假警应抑制”的实质表述。即使球项目不用 IR，也必须引用它并说明域与监督差异。
* **最强反证/警告：** 对体育球，球的图像运动并不必是“全局背景流的局部异常”：相机跟球时球可近乎静止；滚动快门、变焦、视差、球在空中与场地平面不同深度，都会令“global coherent + sparse residual”不成立。故不得监督或解释为物理 `object motion = observed flow - camera flow`，除非有额外可验证的几何/相机证据。

**可保留的较窄问题。** 可以把 global motion 仅作为一个**可拒绝的视觉上下文/背景干扰描述符**：它不得是球存在的必要条件；在 current-frame appearance 支路独立输出的前提下，测试它是否降低候选中背景线、广告、球员衣服的错误对应。没有收益便删除该分支。

### 1.7 CMRTrack — reliability 不是新词，counterfactual target-erased history 是很近的训练先例

**证据与状态：B（预印本）。** [arXiv:2607.23209](https://arxiv.org/abs/2607.23209)，提交 2026-07-25；摘要已读。为得到训练操作细节，本轮还读了该文可检索 PDF 片段，故以下机制表述有来源支撑，但未完成整篇全文审计。

* 面向 infrared UAV tracking；它已经拥有 template/search 的目标先验，因而不能等同于自动球检测。
* 训练时从历史 search region 擦除目标，构造 counterfactual history；比较 factual 与 target-erased history 的 motion response，学习目标一致 motion，而非任意 temporal change。推理时移除 counterfactual branch，并把 learned motion 与 appearance score 做 reliability-aware fusion。
* 这否定“我们首次判断 motion 是否可靠 / motion 可靠性门控”。若使用 history-erasure、相对 response gap、可靠性融合，必须明确是改造/任务迁移，且要做其直接消融。

**本项目能否使用其思想？** 只有在训练标签定义允许时：有可信中心的相邻帧，才可遮蔽历史中心邻域做 counterfactual；不能对 OpenTTGames 等未标注帧擅自把未知位置擦除，也不能把擦除后低分当“球不存在”。更干净的主指标是匹配分布的 calibration：例如 `P(true correspondence in top-K | predicted support/confidence bin)`，以及 candidate recall、定位 F1/中心误差。可靠性分数若不能预测其中任一风险，便只是额外 head。

## 2. sparse / global-local correspondence 的通用先例

### 2.1 QueryDet — high-resolution sparse computation 的候选漏检上限

**证据与状态：A。** [CVPR 2022 官方 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf)；[官方实现](https://github.com/chenhongyiyang/querydet-pytorch)。

* QueryDet 先在低分辨率 FPN 层预测粗位置，递归选择 query，在高分辨率 feature 上稀疏运行检测头；官方摘要报告 high-resolution inference 的加速和 small-object AP 提升。
* 它不是时序 matching，不包含球的 large displacement；因此并未直接占领项目主问题。
* 但它给出方法学上的硬限制：任何“高分辨率仅在候选处匹配”的成本主张，必须连同**候选生成全部成本**计入。高分辨率稀疏模块的 speedup 不能隐去 backbone、dense coarse head、scatter/gather 与 memory overhead。
* 对球的独特危险更严重：低分辨率的 objectness 可能恰好漏掉 2–5 像素球；此时后续 high-resolution matcher 无法恢复。必须画 `K` 从小到大的 candidate recall—latency 曲线，且同样允许单帧 QueryDet 基线使用相同 `K` 与 high-res branch。

### 2.2 STSN — detection supervision 可学跨帧可变形采样，但并不保证真实 correspondence

**证据与状态：A。** [ECCV 2018 官方 PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Gedas_Bertasius_Object_Detection_in_ECCV_2018_paper.pdf)，[arXiv](https://arxiv.org/abs/1803.05549)。

* STSN 用跨时间 deformable convolution，从邻帧学空间采样位置，在 object detection loss 下端到端优化，不需 flow supervision；明确目标是抵抗 occlusion/motion blur。
* 因而“只以定位损失训练的时序 deformable sampler”不是新。更关键的是，检测损失只要求最终框/热图正确，不保证 offset 是同一物体的真实匹配；背景捷径和从未来帧拷贝外观都可能得到同样 loss。
* 本项目若提出可解释 correspondence，不能只展示 learned offset 可视化。需要有标注两帧中心时的 **oracle correspondence test**：将当前位置或自动候选固定，测历史真中心（带可见性/时间间隔）是否在 search top-K/offset neighbourhood；再分离它与最终 detector 的增益。

### 2.3 GMFlow 与 FlowIt — long range global matching、局部 refinement、occlusion/confidence 早已成熟

**GMFlow，证据 A。** [CVPR 2022 官方 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)。它用 Transformer feature enhancement 和全局 matching 得初始 flow，再 warping 后在局部范围预测 residual；所以 global-then-local 本身不能构成球方法贡献。

**FlowIt，证据 C（预印本）。** [arXiv:2603.28759](https://arxiv.org/abs/2603.28759)，2026-03-30 官方摘要。它明确针对 large pixel displacement，以 optimal transport 做 global initialization，同时导出 occlusion/confidence map，再用高可信运动引导低可信区 refinement。没有阅读全文，不能对其 exact complexity/损失作任何断言。

二者同样不解决球任务的三项条件：tiny feature 可能没有判别性、全局 all-to-all 在高分辨率太贵、球检测不能以 GT point 初始化。但它们已将“global range + local refine + confidence/no-match”占为通用 correspondence 设计空间。

**论文必须回答的不是“为什么局部窗口不够大”而是：** 为什么一个固定预算的全局候选机制能在弱小球 feature 上提供比 objectness-only / GMFlow-style coarse matching 更高的 top-K 真对应覆盖，且不会因高重复背景产生等量假匹配。若没有此证据，复杂 matcher 应删除。

### 2.4 2025–2026 扩展检索命中：BIRD、MI-DETR、FlowIt、EgoSIS

以下四项是本轮为补齐 2025–2026 文献新检索得到、且不在原蓝图清单内的关键近邻。它们按相关性纳入，不表示全面覆盖。

| 工作 | 状态/证据 | 对本项目最重要的影响 |
|---|---|---|
| **BIRD**, *Bidirectional Temporal Information Propagation for Moving Infrared Small Target Detection* | **A，arXiv:2508.15415**。已读官方 HTML 的方法与消融段；未见本轮可核验正式出版版本。 | 将 local deformable temporal fusion 与 whole-clip forward/backward propagation 合并，显式批评滑窗只用邻帧、整段多次处理的开销。它是“用更远的时间帧补救当前弱目标”的直接反证。球项目若自称 long-range temporal evidence 新颖，必须与此类递归 propagation 相比，并说明是否可 causal、边界如何 reset、是否跨 clip。 |
| **MI-DETR**, *A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration* | **C，arXiv:2603.05071，2026-03-05**，官方摘要已读。 | 以 retina-inspired cellular automaton 将 raw sequence 转为与 appearance 同网格的 motion map，只用 bbox supervision。这是“简单 motion map + appearance 双路即可成为强 baseline”的危险反证；先跑等预算差分/运动图基线，再声称 correspondence 必要。 |
| **FlowIt** | **C，arXiv:2603.28759，2026-03-30**，见上。 | 覆盖 global matching、置信度和 occlusion，尤其提醒 no-match 不能只是一个任意 sigmoid。 |
| **EgoSIS**, *From Factorized Visual Ego-Transitions to Motion-Canonical Spatial Evidence for UAV Reasoning* | **C，arXiv:2609.08938，2026-09-08**，在截止日前一天；官方摘要已读。 | 用 RGB-derived bidirectional flow 拟合 robust image-plane transition，并产出 residual-support/reliability factor。任务是 UAV VQA，不是检测或球定位；不能当性能 baseline，却是 camera residual 表述的最新概念冲突。 |

另外，检索到 **OMFlow**（Pattern Recognition Letters 2026，occlusion motion estimation）等纯 flow 工作，但其目标/评测没有 tiny automatic detection 的可比性，未列为主近邻；它只补强了“occlusion/no-match 已有大量前史”，不足以支持球方法的具体机制。

## 3. 五个候选设计逐项的创新冲突、可识别性和最小验证

### A. `high-resolution sparse wide search`

**冲突。** QueryDet 覆盖 high-res sparse compute；DQAligner 覆盖 global dynamic query + cross-scale attention + deformable alignment；GMFlow/FlowIt 覆盖 wide/global matching + refine；STSN 覆盖从 detection loss 学跨帧采样。

**尚有的窄空白（假设，未证实）。** 这些方法没有共同证明以下联立条件：当前帧仅有像素级 tiny-ball center supervision、真球不先作为 query 给出、high-res feature 要保持定位、wide search 的全部成本受限、且要对 sports 的相机平移/变焦和拖影保持有效。空白是一个任务—失效—验证空间，并不是算法词汇的首次性。

**可识别性。** 有两帧可信可见的中心标注时，`p_t - p_(t-Δ)` 可以作为对象**中心位移**，足以评价 candidate coverage。但它只标识一个中心对应；无法标识球 patch 每个 token 的 dense flow、也不能告诉模型多个视觉相似候选中哪个 feature 应相似。中心监督可监督 search distribution 的落点或 ranking，却不能把全体 non-selected positions 自动当作可靠负匹配（可能是同色球场线、模糊、未标注球、repeated texture）。

**最小试验。**

1. 固定 DINO/decoder/input/temporal context，先测 current-frame locator、frame difference、local correlation、coarse-to-fine query、提议 sparse wide search；禁止每种方法用不同分辨率或候选预算。
2. 对每对有中心标签的帧：`coverage@K,r = 1[∃ candidate within r pixels of p_(t-Δ)]`；分别报告 oracle-current-center、自动当前候选、完整系统三种条件。三个数必须分开。
3. 同时报告 latency/peak memory/FLOPs（含 candidate generator、feature backbone、index/gather、解码），以及 K 的 Pareto 曲线。
4. 若 simple coarse detector 的 coverage 已相同，或增加 K 才获得收益且 costs 超预算，停止该方向。

### B. `multi-hypothesis correspondence`

**冲突。** SELFY/STSS 已将时空 self-similarity 看作 relation distribution；MotionSqueeze 已从 local correlation 得 displacement/置信信息；flow 的 correlation volume、GMFlow soft match 与 FlowIt transport 均不要求立即单一 displacement。多峰不是新。

**真正要证明的命题。** 不是“top-K 比 argmax 更灵活”，而是：在相同 relation element 数、相同通道与相同自动候选 recall 下，多峰延迟决策能降低具体的 ball ambiguity（线条、高光、球员衣物、拖影）导致的中心误差，而不是只增加模型容量。

**可识别性与 no-match。**

* 若 `p_t` 与 `p_(t-Δ)` 均可信且可见，真中心是一个正 relation；若前/当前帧不可见、出界或没有标签，不能强制任何候选为正，也不能把所有匹配压成 no-match。
* `no-match` 至少区分三类不可混淆事件：(i) target 无有效视觉对应（遮挡/强 blur/出界），(ii) target 对应存在但 proposal/search coverage 漏失，(iii) 当前帧根本不是球或标签缺失。一个二元 gate 不能从中心点标签自动分辨 (i) 和 (ii)。
* 因此可监督的 no-match 最保守定义是：**在两端均有可信 visible 标签且真历史中心不在候选集时，称为 proposal miss；在标签明确不可定位时，禁止 correspondence loss。** 不可从未标注帧制造 no-match 负标签。

**最小试验。** 同总 `K×scales×Δ` 关系预算，比较 point estimate、soft expectation、top-K token、top-K+reject；报告 true-center rank、top-K coverage、NLL/Brier 或 ECE（只在定义良好的可见配对上）、最终定位。若 top-K 只有 oracle query 有用、自动候选没有提升，论文应诚实写“relation probe”，不能称 detection algorithm。

### C. `reliability / reject / don't hallucinate motion`

**冲突。** CMRTrack 直接以 target-erased history 构造 counterfactual motion reference，FlowIt 有 occlusion/confidence，DMR 用 coherent-motion consistency suppress false response。故不能 claim first reliability-aware motion。

**可用且可测的更窄定义。** 将 reliability 定义为条件概率估计，而非“模型说它可靠”：

`r_t ≈ P(true previous center lies within chosen search support | current/history features, proposal state)`。

该定义可在有标签、可见的配对上用 coverage 校准测试；在 absent/missing label 只可当 unlabeled，不能评测概率正确性。

**关键风险。** 若 r 同时用 current detector score 训练，可能只是在重命名 appearance confidence；若以未来信息训练却部署为因果模式，必须拆分 offline/causal 协议。若完全拒绝 motion 后仍由强 current-frame head 正确定位，reliability 对最终任务没有独立价值。

**最小试验。** reliability bin 的 empirical coverage diagram；risk–coverage curve（拒绝多少帧、保留帧的 matching/center error）；以及 `with/without reliability` 在相同 proposal and matcher 下的 difference。counterfactual erase 若使用，和普通 patch dropout、中心附近随机遮挡以及不使用历史分开比较。

### D. `camera residual / global-local`

**冲突。** DMR 是最直接的 small-target detector 先例；EgoSIS 是最新 pose-free image-plane transition/residual/reliability factorization；传统 global motion compensation 更早已存在。

**第一性原理限制。** 画面运动不是球 object motion。对于非共面球、透视、rolling shutter、变焦，单一 2D homography/flow 只可描述部分背景；球相对背景残差是观测性特征，不是几何分解真值。相机跟球时，所需的 correspondence 甚至可以接近零位移。因此把 residual 设为 detect 的必要条件会系统性伤害最需要的序列。

**推荐状态。** 第一篇主线不把 camera residual 放入核心贡献；把它放成预注册的可选分析：用粗 background transition/flow 的 robust fit 生成 `support quality` 和 `residual magnitude`，在静态与明显运动相机 clip 分桶。只有当它在每个 bucket 都能减少明确类型的 false correspondence，才保留。不要声称相机 motion recovery，更不能使用不可获得的 camera ground truth。

### E. `center supervision protects motion`

**支持与冲突。** OTHR 明确有 tiny-object-aware evaluation 与前景权重；MOCID/DQAligner/MIST 都以目标检测监督改变时序表征；STSN 已显示 detection loss 可学习时空采样。中心点监督当然可以约束有标签的 endpoint displacement，却不唯一识别 feature-level motion。

**最低限度的监督协议。**

| 帧对状态 | locator loss | correspondence/rank loss | 可以作什么结论 |
|---|---|---|---|
| 两端有可信、可见中心，且时间索引连续/已知 Δ | 可以 | 可以把历史真中心当正 endpoint | 仅中心级 correspondence 有监督 |
| 一端标签明确不可定位/不可见 | 按数据集标签语义 | 默认禁用 | 不能训练 no-match 为“无球” |
| 一端未标注（尤其 OpenTTGames） | 禁用 | 禁用 | 视频仍可作输入，绝非负样本 |
| 跨 clip/rally 边界 | 禁用跨帧 loss/缓存 | 禁用 | 防止人为制造大位移或漏配 |

仅有中心点时，不应报告“flow supervision”“pixel correspondence ground truth”“object-size normalized motion”除非相应标签真实存在。对 blur，曝光内长条与 frame-to-frame center displacement 也必须分开评测。

## 4. 对蓝图的具体修订建议

1. **把 H2 改得更难。** 由“保留少量多峰对应优于过早唯一位移”改为“在固定关系元素数、自动候选 recall 与端到端成本下，延迟压缩的 relation distribution 在高 displacement/blur/重复纹理分桶提高 true-center rank、校准与定位；其收益不由增加容量或 oracle query 解释。”
2. **把 H4 拆为两个不可互相遮蔽的门槛。** `candidate recall@K` 是发现门槛；在正确候选已给定的 `conditional match accuracy` 是对应门槛。二者应分别测量，但不要求同时提升；需要证明覆盖与条件匹配的组合，以及完整定位表现，在同预算下改善。
3. **不要把 global/local/camera 放到首版模型名或贡献里。** DMR/DQAligner/EgoSIS 的概念距离很近，且体育球的非共面几何让强解释不安全。先做数据分桶的诊断；没有显著收益就删。
4. **考虑与实际机制对应的强基线。** `(a)` current-frame high-res locator，`(b)` difference/MI-DETR-style motion-map + appearance，`(c)` local deformable sampling（STSN/BIRD 类），`(d)` 稀疏候选（QueryDet）或全局动态query/dense对齐（DQAligner）。后两者不是相同候选机制；2026-09-10源码补读据此修正原归类。按当前实验问题选必要控制，不机械实现整份列表；若收益已被简单系统解释，停止加模块。
5. **把 long-term temporal propagation 作为明确对照而非遗漏。** BIRD/DeepPro 表明“更远帧信息”和“便宜时间 profile”都可能解释收益。实验应写清 offline 还是 causal；若使用未来帧，不可把 latency 写成实时。
6. **论文 related work 加一个“任务差异而非免引用”的段。** 红外是低纹理/低 SNR，球赛 RGB 是小尺寸、motion blur、复纹理、相机移动；这些差异说明为何不应直接迁移其结论，却不使相同机制的 prior art 消失。

## 5. 检索充分性、未确认与下一轮需要补的原始证据

**本轮已覆盖。** 直接小目标 motion：OTHR、MOCID、DQAligner、MIST、DeepPro、DMR、CMRTrack；通用机制：QueryDet、STSN、GMFlow、FlowIt；2025–2026 新补：BIRD、MI-DETR、EgoSIS。检索关键词包括 `tiny/small target`, `large displacement`, `global dynamic query`, `sparse correspondence`, `motion reliability`, `occlusion/confidence`, `global-local motion`，时间窗口覆盖 2025-01-01 至 2026-09-09。

**明确未确认。**

* 不是系统综述式全库检索，不能声称“所有” latest papers 已覆盖；高质量但未检索到的 remote sensing/infrared/point-tracking 工作仍可能存在。
* DQAligner、MIST、DeepPro 的出版方全文本轮受访问/JS 限制；细节只以摘要/官方 README 为界。CMRTrack、DMR、FlowIt、MI-DETR、BIRD、EgoSIS 中除 DMR/BIRD 已读较多正文外，均应保守标为预印本或摘要级证据。
* 本轮未复现任何代码、未验收数据下载，也未验证这些小目标 benchmark 的标签定义与球数据完全一致。
* 未覆盖用户蓝图中的体育球专门方法、TrackNet 系列、视频 foundation model、数据集划分与许可；这些由其他研究笔记负责，不能从本笔记推论。

## 6. 可复用引用清单

1. Ji, Wang, Wang. **Optical Flow Estimation for Tiny Objects: New Problem, Specialized Benchmark, and Bioinspired Scheme**. IJCAI 2025. [Paper](https://www.ijcai.org/proceedings/2025/0136.pdf), [project code](https://github.com/JaneEliot/OTHR). A.
2. Zhang et al. **MOCID: Motion Context and Displacement Information Learning for Moving Infrared Small Target Detection**. AAAI 2025. [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/33087). B.
3. Deng et al. **Learning Global Dynamic Query for Large-Motion Infrared Small Target Detection**. IEEE TGRS 2026. [IEEE record](https://ieeexplore.ieee.org/document/11363482/), [official code](https://github.com/dengfa02/DQAligner_MIRSTD). B.
4. Gao et al. **MIST: A Benchmark and Baseline for Multi-frame Infrared Small Target Detection in Complex Motion**. IEEE TIP 2026. [IEEE record](https://ieeexplore.ieee.org/document/11511399/), [official code/data](https://github.com/ShuCvlab/MIST). B.
5. Li et al. **Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better**. arXiv:2506.12766; author repository cites TPAMI 2026. [arXiv](https://arxiv.org/abs/2506.12766), [official code](https://github.com/TinaLRJ/DeepPro). B.
6. Zhang et al. **Decoupled Motion Representation Learning for Moving Infrared Small Target Detection**. arXiv:2606.15286, 2026. [HTML](https://arxiv.org/html/2606.15286). A, preprint.
7. Chen. **Counterfactual Motion Reliability Learning for Robust UAV Tracking**. arXiv:2607.23209, 2026. [arXiv](https://arxiv.org/abs/2607.23209). B, preprint.
8. Yang, Huang, Wang. **QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection**. CVPR 2022. [Official PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf). A.
9. Bertasius, Torresani, Shi. **Object Detection in Video with Spatiotemporal Sampling Networks**. ECCV 2018. [Official PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Gedas_Bertasius_Object_Detection_in_ECCV_2018_paper.pdf). A.
10. Xu et al. **GMFlow: Learning Optical Flow via Global Matching**. CVPR 2022. [Official PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf). A.
11. Safadoust et al. **FlowIt: Global Matching via Hierarchical Transformers and Optimal Transport for Optical Flow**. arXiv:2603.28759, 2026. [arXiv](https://arxiv.org/abs/2603.28759). C, preprint.
12. Luo et al. **Bidirectional Temporal Information Propagation for Moving Infrared Small Target Detection**. arXiv:2508.15415, 2025. [arXiv](https://arxiv.org/abs/2508.15415). A, preprint.
13. Liu et al. **MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration**. arXiv:2603.05071, 2026. [arXiv](https://arxiv.org/abs/2603.05071). C, preprint.
14. Yang et al. **EgoSIS: From Factorized Visual Ego-Transitions to Motion-Canonical Spatial Evidence for UAV Reasoning**. arXiv:2609.08938, 2026. [arXiv](https://arxiv.org/abs/2609.08938). C, preprint.
