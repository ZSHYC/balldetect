# 高速微小目标 motion / correspondence：对抗性文献核验笔记

> **范围与日期。** 本笔记只审查“高速微小球定位”最接近的外部证据：微小目标光流、移动红外小目标检测、UAV 小目标跟踪、视频检测中的跨帧采样、高分辨率稀疏计算，以及大位移光流。检索和网页读取截至 **2026-09-09**；不是穷尽性综述。尤其是 2026 年论文中有数篇只是 arXiv 预印本，不能在论文中写成已同行评审结果。
>
> **阅读深度。** A = 已读官方全文/官方 PDF 的方法与实验关键段；B = 已读官方摘要和官方作者仓库 README，足以限定主张，未逐式审计；C = 仅核对官方书目信息/摘要，不能据此复述实现细节。链接均为原始论文、出版方或作者官方仓库；仓库不能替代论文证据。

2026-09-10至11日局部更新：MOCID已补读官方全文方法与实验；DQAligner、MISTNet已补读固定作者源码，全文仍不可访问。2026-09-11又完成DeepPro arXiv v5全文及固定源码、CMRTrack v1全文、FlowIt v2全文与固定作者源码，以及 EgoSIS v2 全文的定向补读，修订见§1.5、§1.7、§2.3、§2.4及所链专题。其余条目维持原阅读范围，没有据局部更新宣称全库重检。

## 0. 先给结论：哪些表述已经不能作为创新

以下五种“模块级新颖性”都已有非常近的先例，不能作为本文的首要贡献：

| 候选表述 | 最强已知反证 | 对本项目的含义 |
|---|---|---|
| 高分辨率但只在少量位置计算 | QueryDet 从低分辨率粗 query 引导高分辨率稀疏计算；它直接以小目标的高分辨率计算冗余为动机。 | 不能声称首个高分辨率稀疏搜索。球任务仍可研究的是：**候选尚未被检测到时**如何保证跨帧真对应覆盖，而非把 QueryDet 式目标 query 直接搬过来。 |
| 先远后近 / global-local matching | GMFlow 先全局匹配再局部残差细化；FlowIt 用全局最优传输初始化、置信度引导细化。 | “大位移全局匹配 + 局部精修”本身不新。真正困难是球在高分辨率里弱、全局特征里近乎不可辨，且不允许用 GT query。 |
| 全局相机/背景运动与局部目标运动分离 | DMR 显式建模全局 coherent motion、以其引导局部 anomaly；EgoSIS 又把**图像平面代理**的 transition、残差支持、可靠性因子化。 | 不能声称首个 global/local motion decoupling 或 camera residual。可研究“这种残差在像素级球定位中是否实际提高候选/对应”，但要避免把二维背景流残差称为物理球运动。 |
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

**证据与状态：A，2026-09-10补读。** [AAAI 2025记录](https://ojs.aaai.org/index.php/AAAI/article/view/33087)（2025-04-11，pp.10022–10030）；[官方全文](https://ojs.aaai.org/index.php/AAAI/article/view/33087/35242)。研究代理读了方法、实验与主要消融；根代理复读pp.10024–10027的输入、公式1–13、损失与Table2。作者仓库未提供可审计的完整模型实现，本轮没有代码复现。

* FISTA先做空间频域滤波，再沿固定空间位置的时间轴滤波，用所得上下文调制卷积核（公式1–6）。这是时空调制，未显式输出位置对应。
* DAM用两帧的3D中心差分生成扫描参数，将池化后的target/reference同索引token交错后双向扫描（图4、公式9–13）。这里interpolation是特征序列交错，不是生成真实中间视频帧；扫描可传播跨位置上下文，但不提供明确的u→v对应地址。
* 论文输入为过去4帧加当前帧，输出当前检测；DAM位于YOLOX头之前，无当前硬候选前提。训练描述为BCE与IoU损失，未列GT光流/位移监督。Table2支持最终检测组件收益，未测球中心对应或候选覆盖；片段边界细节未给出。[方法与实验原文](https://ojs.aaai.org/index.php/AAAI/article/view/33087/35242)

**对本项目的判断。** “运动先改变空间表示”与“差异感知扫描”已有直接先例；不能把名称中的displacement当作物理位移测量，也不能把没有显式cost volume解释为不能利用运动。若采用类似机制，仍要用任务收益区分背景抑制、外观适配与对应改善。另一个工程推论是：卷积核依赖整个clip时，同一帧在不同窗口中的特征未必相同，不能照搬当前逐帧DINO前缀的验证去重；RGB仍可复用。这不要求现在增加MOCID分支。

### 1.3 DQAligner — large-motion tiny target 的 global query、跨尺度双向 attention 与 deformable alignment

**证据与状态：B。** [IEEE TGRS 2026 记录/摘要](https://ieeexplore.ieee.org/document/11363482/)（DOI `10.1109/TGRS.2026.3657842`，线上发表 2026-01-26）；[官方仓库](https://github.com/dengfa02/DQAligner_MIRSTD) README、训练/测试入口已读。IEEE 页面受 JavaScript 限制，未取到全文；因此以下只复述出版方摘要和作者 README 明示内容。

* 论文明确针对 complex motion 与 large displacement，提出 global random large-displacement augmentation、cross-scale bidirectional shared attention、动态 receptive-field pyramid deformable convolution，以及迭代 class query memory（CQM）。出版方称其从 rigid alignment 转为 flexible matching。
* 这直接覆盖“高分辨率 feature 上做 global dynamic query / 随后 dynamic alignment”的模块级说法。其仓库还明确警告 spatial deep supervision 和 `track_loss` 默认关闭，因为可能与深浅 query learning 冲突；这只是作者实现观察，不可泛化成理论结论。
* **2026-09-10源码补读修正：** DQAligner 的类query从可学习初态出发，在窗口内读取全空间特征，输出dense mask；公开路径没有先从当前检测器取硬候选。不能给它强加“当前候选先漏”的结构判断，也不能说query必然恢复球。窗口内GRU、跨窗口参数未实际使用以及因果采样边界见[专门审查](2026-09-10-dqaligner-query.md)；论文全文仍未取得，源码证据不替代全文。

**修正后的比较边界：** 若今后采用稀疏时序候选，报告自身candidate recall@K及条件匹配，并与自动dense系统比较共同任务的最终定位/检测。DQAligner没有必需的硬候选阶段，不要求它报告虚构的内部候选召回；从dense输出另取top-K时需明确这是额外诊断规则。GT query结果不能代替自动发现结果。

### 1.4 MIST / MISTNet — irregular fast motion 的多尺度隐式补偿和困难子集

**证据与状态：B，加2026-09-11固定源码补读。** [IEEE TIP 条目](https://ieeexplore.ieee.org/document/11511399/)；[官方仓库](https://github.com/ShuCvlab/MIST) README 已完整读取。仓库说明2026-04-27接收；本次采用[2026-05-18源码c95bb208](https://github.com/ShuCvlab/MIST/tree/c95bb20805b823ddbf86a0e9864147f2a43f3a68)。发表页仍未取得全文，定向检索未找到作者公开PDF/arXiv；下面的机制来自代码，不称已审阅论文全部消融。

* MIST 是合成引擎构建的 airborne IR benchmark，目标有姿态/尺寸/强度变化与真实背景融合；不能把其性能直接和 RGB 球赛结果比较。
* MISTNet的SNCB已由源码确认是局部异位置软对应：完整current特征作Q、past作K/V，NATTEN计算邻域相关，softmax后加权聚合；没有检测器硬候选输入。它先将past通道分八组，padding、平移、裁回后混合，默认位移3、邻域3/5/7。每个邻域输出聚合特征，下游不接收显式位移多峰列表。各尺度操作不能直接换算为一个原图物理搜索半径。[SNCB](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/deepmist/models/multiframe/MISTNet/model_MISTNet.py#L6-L64)
* 四级encoder均执行补偿，包括首次pooling之前的浅层；末帧为query，窗口中各帧含当前自身都参与，随后按时间聚合。decoder逐级上采样并融合最浅层，输出单通道mask。浅层路径存在不等于体育球精细定位已经验证。[encoder](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/deepmist/models/multiframe/MISTNet/base.py#L83-L97)、[补偿与decoder](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/deepmist/models/multiframe/MISTNet/model_MISTNet.py#L67-L170)
* 当前MIST配置使用过去4帧加当前帧、末帧mask、frame_padding=False；序列开头四帧不生成窗口，没有取模回绕。默认损失为SoftIoU加0.01倍SufficiencyLoss，配置未使用GT光流或位移监督。没有复现运行，不能用这些代码事实补写未取得的论文消融数值。[数据窗口](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/deepmist/datasets/MISTDataset.py#L33-L55)、[正式配置](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/configs/train_MISTNet_MIST.yaml#L16-L39)
* 其官方 hard split 是 11 条测试序列，条件为 SCR≤1、轨迹高度不规则、速度 >7 pixels/frame。这个**分桶思想**很适合项目：主报告不能只有均值；应预注册由真中心算出的 displacement/速度分桶、blur bucket、camera-motion proxy bucket。但切勿挪用 `>7 px/frame` 作体育球“高速”的物理阈值。

**对创新的冲突。** “不规则/快小目标需要多尺度匹配补偿”和“压制无关 motion”已是已发表或近发表系统的主题。可区分之处必须来自可验证的球中心对应和计算-空间分辨率权衡，而不是词汇。

源码补读使冲突更加具体：浅层细节、多尺度局部attention、分组平移与隐式补偿已有直接组合先例。但不能反过来强称这种dense方法存在硬候选漏检阶段；若本项目采用稀疏候选，候选覆盖是自己的额外责任。新机制仍需在自动球定位中证明收益，而非仅在GT查询下提高条件匹配。当前不增加MISTNet复现或模块。

### 1.5 DeepPro — 最危险的效率反证：时间 profile 可能已足够

**原阅读状态（2026-09-09）：B。** 当时只读[arXiv:2506.12766](https://arxiv.org/abs/2506.12766)摘要和[作者仓库](https://github.com/TinaLRJ/DeepPro)，正式出版状态未独立确认。

**2026-09-11更新：A（开放全文与源码）。** 已完整补读[arXiv v5，2026-03-27](https://arxiv.org/pdf/2506.12766v5)及固定作者源码；正式书目信息为TPAMI 48(8):10157–10175（2026），[DOI](https://doi.org/10.1109/TPAMI.2026.3683258)，但没有取得IEEE排版全文，也未假定它与v5逐表相同。来源、版本与时间语义见[temporal profile专题](2026-09-11-temporal-profiles.md)。

* 基础DeepPro在固定空间位置用学习的`T×T`矩阵混合时间特征，输出输入窗内每帧mask；它不是特征相似度生成的空间cost volume，也没有对象位移输出。基础版仍有空间pooling/上采样，不能把time-only写成没有任何空间操作。
* 默认40帧全时间混合，重叠4帧取最大响应。非末帧输出可以用未来，作者没有报告只取末帧的因果版本；离线吞吐不能与本项目三帧当前输出的延迟直接比较。
* **修正原判断的逻辑跳步：** 本条原以“同一固定像素不保留移动目标”弱化其对wide search的挑战。它确实不建立对象对应，但目标经过该像素形成的瞬态脉冲恰是检测证据；论文toy分析明确同尺寸下速度增加会缩短脉冲。缺少对应不等于不能检测高速目标。
* 所以fixed-pixel变化/profile是“对应搜索是否必要”的真实竞争解释；相机跟球、背景扫描及RGB纹理下能否成立仍未验证。当前先完成既定时序输入控制，只有结果需要区分这一解释时再选择必要的廉价对照，不由一次补读启动所有长窗/离线路线。

### 1.6 DMR — camera/background coherent motion 与 local anomaly 的最直接概念冲突

**证据与状态：A（但仅预印本）。** [arXiv HTML v1，2026-06-13](https://arxiv.org/html/2606.15286) 已读摘要、动机、方法概览和关键公式；没有正式会议/期刊版本可据此声称。

* DMR 的核心观察是背景运动在空间上 global coherent，而 tiny target 呈 sparse localized anomaly；它以 pretrained optical-flow prior 显式 branch 建 coherent motion，以 deformable alignment 隐式 branch 建 target-sensitive local motion，再以两者一致性压制背景 false response。
* 这已经覆盖“camera/global motion 与 object/local motion decoupling”“用全局运动先验指导局部对齐”“背景一致的假警应抑制”的实质表述。即使球项目不用 IR，也必须引用它并说明域与监督差异。
* **最强反证/警告：** 对体育球，球的图像运动并不必是“全局背景流的局部异常”：相机跟球时球可近乎静止；滚动快门、变焦、视差、球在空中与场地平面不同深度，都会令“global coherent + sparse residual”不成立。故不得监督或解释为物理 `object motion = observed flow - camera flow`，除非有额外可验证的几何/相机证据。

**可保留的较窄问题。** 可以把 global motion 仅作为一个**可拒绝的视觉上下文/背景干扰描述符**：它不得是球存在的必要条件；在 current-frame appearance 支路独立输出的前提下，测试它是否降低候选中背景线、广告、球员衣服的错误对应。没有收益便删除该分支。

### 1.7 CMRTrack — reliability 不是新词，counterfactual target-erased history 是很近的训练先例

**原阅读状态（2026-09-09）：B。** 当时仅摘要与可检索PDF片段。**2026-09-11更新：A（仍为预印本）。** 已读[arXiv:2607.23209v1全文](https://arxiv.org/pdf/2607.23209v1)，提交2026-07-25；未找到作者源码。公式、消融和ROI边界见[反事实运动专题](2026-09-11-counterfactual-motion.md)。

* 面向给定初始化的infrared UAV单目标tracking，有template/search和跨帧跟踪先验；它不解决从全图自动发现球。
* Motion map实际来自当前/历史search crop的通道平均绝对像素差，经轻量卷积和当前GT热图监督。训练再按**历史GT框**以全局均值擦除历史像素，用事实/反事实前景与背景响应差监督内部融合gate；推理不执行擦除。该gate不是已校准的对应概率或no-match拒绝输出。
* 已有tiny-size、fast-motion、背景属性、擦除填充值及模块消融，不能说作者完全未测小目标或背景。但没有球中心协议、原像素位移/尺寸分桶和自动候选覆盖实验。
* 反事实响应差受擦除区域、当前GT加权支撑和卷积/resize有效范围影响；原视频位移又不等于跟踪ROI坐标位移。源码与ROI对齐细节未公开，不能把差值天然解释成大位移correspondence可靠性，也不能据条件推导断言论文存在实现bug。

**对当前决策的影响。** 基本的history-erasure、响应差监督和可靠性融合已有直接先例，不能单凭这些机制声称创新。中心标签也不直接给出作者所需的GT框面积；定义中心邻域代理是另一个需要证据的选择。当前继续同一参数化下真实历史/重复当前帧的重训控制，不立即引入跟踪状态、擦除监督或gate；该控制只能测固定学习规则下的输入增量，不能证明目标物理运动因果。将来若研究可靠性，再明确它预测的是候选覆盖、对应误差还是最终定位风险，并验证相应排序/校准；不能把这些不同事件共用一个无定义的“可靠度”。

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

**FlowIt，证据 A（arXiv v2；公开源码）。** [arXiv v2，2026-05-31](https://arxiv.org/abs/2603.28759v2)已读方法、训练与消融；[作者项目页](https://kuis-ai.github.io/FlowIt/)称 BMVC 2026 Oral，当前可读论文仍为 arXiv 版本。作者有公开模型实现；此处固定到 `sadrasafa/FlowIt@a6fa46829b1b5ae3fe0ff665f2d3771caf62bf9f`（2026-08-28）。

* **不是稀疏 global search。** CNN/FPN/MRT 最终在 \(1/4\) 分辨率形成 feature，再构造完整 \(N\times N\) 4D cost/probability volume，\(N=HW/16\)；没有候选裁剪。它以降采样压低代价，但 all-pairs 元素数仍随原图面积平方增长。S 版在 FlyingChairs 消融中报 344 GMAC、99 ms（L40S）；该数不等于本项目端到端成本。[论文方法/表4](https://arxiv.org/html/2603.28759v2#S3)
* **OT 与两个不同的分数。** Sinkhorn entropy-regularized OT 加 source/target dustbin；真实像素可把质量送入 dustbin。初始 `confidence` 是 argmax 周围 \(r=2\) 小窗内的有效概率质量；论文名为 `occlusion` 的图实为所有有效 target 的边缘质量，高值表示**非**遮挡/可匹配、低值表示质量流入 dustbin。随后三步局部 refinement 以 flow、图像特征、局部相关、confidence 和该可匹配分数为条件，更新三者；源码为每轴 \(r=4\) 的局部 correlation lookup 与 ConvGRU。[迭代代码](https://github.com/sadrasafa/FlowIt/blob/a6fa46829b1b5ae3fe0ff665f2d3771caf62bf9f/core/flowit.py#L96-L128)、[ConvGRU](https://github.com/sadrasafa/FlowIt/blob/a6fa46829b1b5ae3fe0ff665f2d3771caf62bf9f/core/refinenet.py#L98-L115)。
* **有 dustbin，但不是输出层的拒绝。** 固定源码把 dustbin 行/列从返回的概率张量删去，再在剩余有效位置 argmax+局部期望以产生每个 source pixel 的 dense flow；confidence/可匹配分数参与 refinement，并作为辅助 dense maps 返回，但没有据其阈值化或拒绝输出 flow 的分支。这是“已有 unmatched mass 与辅助分数”的直接先例；若要作显式 `no-match` 决策，还需另定义并评价拒绝规则。[OT与初始化](https://github.com/sadrasafa/FlowIt/blob/a6fa46829b1b5ae3fe0ff665f2d3771caf62bf9f/core/submodules.py#L216-L329)、[最终返回值](https://github.com/sadrasafa/FlowIt/blob/a6fa46829b1b5ae3fe0ff665f2d3771caf62bf9f/core/flowit.py#L159-L167)。
* **监督与证据范围。** 输入严格为两张 RGB 帧、目标为 dense two-frame optical flow；flow GT 直接监督。non-occlusion 标签由 forward--backward consistency（<2 px）构造；confidence 是 endpoint error <4 px 的指示量。初始 flow/confidence 只在 non-occluded pixel 监督，三次 refinement 的 flow/confidence 覆盖全图，非遮挡分数以 \(L_1\) 全图监督。它在 Sintel、KITTI、Spring、LayeredFlow 等 dense-flow 数据上测 EPE/像素阈值，未报告 tiny/small-object 分桶、自动球发现、球中心或 blur 评测。故这是**未验证 tiny RGB 自动定位**，不是“其机制已被证实不能保留 tiny 特征”；不应因创新边界把未知写成失败。

二者已将“global range + local refine + confidence/occlusion”占为通用 correspondence 设计空间。FlowIt 的 dense flow 与全图 forward--backward 监督不能由现有球中心标签完整提供；这是监督条件的限制，不是 dustbin 算子本身不能复用。论文也没有验证其 \(1/4\) feature 或全图 flow 能否自动定位数像素球。若以后使用预训练 FlowIt 作 dense-flow 对照，应另测自动候选覆盖、球中心误差和实际高分辨率成本；用 GT 中心采样光流的结果必须标为 oracle 探针，不能当自动定位性能，也不能预设其为性能上界。

**论文必须回答的不是“为什么局部窗口不够大”而是：** 为什么一个固定预算的全局候选机制能在弱小球 feature 上提供比 objectness-only / GMFlow-style coarse matching 更高的 top-K 真对应覆盖，且不会因高重复背景产生等量假匹配。若没有此证据，复杂 matcher 应删除。

### 2.4 2025–2026 扩展检索命中：BIRD、MI-DETR、FlowIt、EgoSIS

以下四项是本轮为补齐 2025–2026 文献新检索得到、且不在原蓝图清单内的关键近邻。它们按相关性纳入，不表示全面覆盖。

| 工作 | 状态/证据 | 对本项目最重要的影响 |
|---|---|---|
| **BIRD**, *Bidirectional Temporal Information Propagation for Moving Infrared Small Target Detection* | **A，arXiv:2508.15415**。已读官方 HTML 的方法与消融段；未见本轮可核验正式出版版本。 | 将 local deformable temporal fusion 与 whole-clip forward/backward propagation 合并，显式批评滑窗只用邻帧、整段多次处理的开销。它是“用更远的时间帧补救当前弱目标”的直接反证。球项目若自称 long-range temporal evidence 新颖，必须与此类递归 propagation 相比，并说明是否可 causal、边界如何 reset、是否跨 clip。 |
| **MI-DETR**, *A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration* | **A，arXiv:2603.05071v1，2026-03-05**；已读官方全文方法/实验和作者公开源码的固定提交。仍是预印本。 | 它不是 correspondence/flow，而是廉价、因果、带状态的差分—累积 motion map 加双路融合；它是“显式大范围匹配是否必要”应面对的竞争解释，但不能以其 IR bbox 结果替代 RGB 球中心定位证据。 |
| **FlowIt** | **A，arXiv:2603.28759v2，2026-05-31**；已读全文关键方法/实验与固定作者源码。作者项目称 BMVC 2026 Oral，论文可读版本仍为预印本。 | 它在 \(1/4\) 特征做 dense all-pairs OT，真有 dustbin 与监督的 confidence/可匹配分数，但最终仍强制输出 dense flow；因此覆盖 global matching、可靠性辅助与 unmatched mass，不等于已验证的球 `no-match` 或 tiny 自动定位。 |
| **EgoSIS**, *From Factorized Visual Ego-Transitions to Motion-Canonical Spatial Evidence for UAV Reasoning* | **A，arXiv:2609.08938v2，2026-09-09**；已读官方全文方法与实验，仍是预印本。论文未链接作者代码；本次未定位到可确认的作者公开实现。 | 冻结 VideoFlow/MOFNet 双向 flow 后以 Huber-IRLS/MAD 拟合 affine **image-plane proxy**，再把 residual/static-support 与手工可靠性门控用于 VQA 时空证据；它不是相机姿态或物体 motion 的可识别分解，也未验证像素定位、tiny ball 或端到端 flow 成本。 |

另外，检索到 **OMFlow**（Pattern Recognition Letters 2026，occlusion motion estimation）等纯 flow 工作，但其目标/评测没有 tiny automatic detection 的可比性，未列为主近邻；它只补强了“occlusion/no-match 已有大量前史”，不足以支持球方法的具体机制。

#### MI-DETR 定向补读（2026-09-11）：它实际排除了什么、没有排除什么

**证据范围。** [arXiv v1](https://arxiv.org/abs/2603.05071v1) 于 2026-03-05 提交；以下已读其[方法 RCA/PMI](https://arxiv.org/html/2603.05071v1#S3)和[实验/复杂度](https://arxiv.org/html/2603.05071v1#S4)。作者确有公开源码；本条固定到 2026-03-11 提交 `24257e4774c8f328738e88142c9a3cdabfd7afa5`。其[README](https://github.com/nliu-25/MI-DETR/blob/24257e4774c8f328738e88142c9a3cdabfd7afa5/README.md#L39-L46)明确把appearance与预先生成的motion图一一配对为6-channel输入，模型配置在P3做双向`TransformerFusionBlock`。固定树中未找到从原始序列生成RCA map的脚本；复现包要求另取retina-processed data。因此算法事实以论文为准，公开代码只能核验其下游双路detector与预处理输入约定。

* **实质算子。** 对当前帧先作阈值/侧抑制/ON--OFF contrast，随后取 `|C_t-C_{t-1}|`；该差分经 `S_a=0.8S_{a,t-1}+0.2R_t` 作指数累积，再与当前 contrast 经固定 Mexican-hat（约 \(5\times5\)）中心环绕滤波、阈值、双边滤波和归一化合成为 \(M_t\)。所以它绝非 raw one-step frame difference，也不只是多帧相加：有固定的空间抑制和状态去噪；但其时间核心仍是**同一像素地址**的一步绝对差分加 EMA，而非 feature correspondence。
* **时间与因果性。** 输出/检测目标是当前 \(I_t\)；每一步只读当前帧、\(C_{t-1}\) 和 \(S_{a,t-1}\)，无未来帧，故可因果部署。序列开始时状态清零，首帧用空间梯度初始化；作者把 \(\alpha=0.8\) 解释为约五帧记忆。表中“1 frame”只表示没有显式缓存输入帧，**不等于无历史**。按EMA公式推导，约五帧是有效尺度，不是硬截断：充分长序列中五个最近输入之外仍有`0.8^5=32.768%`的线性累计权重；后续非线性处理还会影响实际响应。这是公式推论，不是论文报告的额外实验。任何同三帧球基线的对照都必须同样在rally及内部时间边界reset，并明确限制或报告真正可用的历史。
* **监督、几何与相机边界。** 训练只有标准 detection 的类别、L1 box 与 GIoU 损失，没有 flow、位移、motion-map 或对应监督；论文统一 letterbox 到 \(512\times512\)。\(M_t\) 与 \(I_t\) 逐像素对齐，却是单张非负标量图：不含速度正负、物理位移向量、跨位置 candidate 或多假设。作者没有给出 camera-motion compensation 或静态相机假设；由同址差分可直接推知，连续平移/缩放造成的背景变化也会进入响应，不能把它解释为 object-only motion。
* **实验与成本的可比性。** 工作评估三套 moving-IR **bbox** 数据，用 mAP@0.5、P/R/F1；不是 RGB 视频的点中心或 blur 评价。IRDST-H 上报告 70.30 mAP@50、72.70 F1、32.44M parameters、93.90 GFLOPs、34.60 FPS（RTX 3090），但论文明确 FPS **不含一次 RCA preprocessing**。故其数值不能同 BlurBall 的中心容差、位置误差或端到端三帧吞吐直接排列，也不能据此称其在运动模糊 RGB 球上有效。

**对当前决策的限制。** 不据此改变正在运行的三帧 DINO 基线，也不把 RCA/PMI 加入当前版本。若后续结果要宣称“wide correspondence 必要”，先用同一数据切分、输入尺度、目标帧和真实因果历史比较：(i) current-only，(ii) 只加可见的差分/EMA motion 图，(iii) motion 图加双路交互；否则会把固定 map、额外历史和 32.44M/93.90G detector 融合容量混为“motion mechanism”。该比较须计入图生成时间，并按本项目中心协议报告漏检/误检和定位，而不是移植 IR box mAP。即使 (ii) 已解释收益，也只否定该设置下昂贵匹配的必要性；它不证明或反驳球的真实跨位置 correspondence。

#### EgoSIS 定向补读（2026-09-11）：二维稳定参考的概念先例，非球定位机制证据

**证据范围。** [arXiv v2](https://arxiv.org/abs/2609.08938v2) 于 2026-09-09 修订；以下已读官方[方法](https://arxiv.org/html/2609.08938v2#S2)与[实验/消融](https://arxiv.org/html/2609.08938v2#S3)。论文页面及全文未提供作者代码链接；以题目和方法名检索时未定位到可确认的作者公开实现，故没有源码事实可报告。

* **transition 与失败处理。** 冻结 VideoFlow/MOFNet 提供相邻 \(I_t,I_{t+1}\) 的双向 flow；论文未单独交代该 flow estimator 的完整输入上下文。只保留有限、端点在界、反向支持存在且通过像素单位 forward--backward 阈值的对应。随后在归一化坐标以 Huber-IRLS 加 MAD rejection 拟合 \(3\times3\) affine \(A_t\)；一致点过少时，该 edge 保持时间对齐但支撑和 confidence 置零。输出 packet 包含归一化的 residual median/MAD 统计；static support 与有效支撑、双向一致性、仿射条件数共同进入 \(c_t\)，而 cut score 还用 RGB photometric warp disagreement。方法段列出这些组成量，却未给 \(c_t\) 的具体组合公式，也没有对象/相机真值或单独校准监督。
* **时间语义与可识别性。** ReTEM 按 edge 顺序以 \(g_t=c_t(1-q_t^{cut})\mathbf1[valid]\) 更新、只安全组合 geometry，并在 cut、持续低 confidence、不安全候选、质量下降或 segment horizon 时重锚；这是一条**有界、按时间正向**的 transition chain。VQA 实验把视频采样为 2 FPS、截为 8--32 帧；但正向递推和 causal LM loss 不代表逐帧视觉输入无前瞻。论文将 edge \(t\) 对齐源帧 \(t\)，该 edge 至少需要 \(t+1\)；EASE 的公式8又对同一 segment 内全部有效视觉组聚合，再以四个池化 context tokens 回注视觉切片。按该公式直接用于较早帧输出时可能包含其后的视觉组，这是公式的时间依赖推论，不是实测在线延迟。作者明确称表示是 image-plane proxy，不是 metric pose 或 3-D map：\(A_t\) 拟合通过筛选的优势二维变换，把它解释为背景运动还依赖足够背景支撑；不能把 robust fit 或 residual 自动当作真实 camera/object motion 分解。
* **输出、监督、成本和消融边界。** 训练目标仅为 assistant answer token 的 causal language-modeling loss，Qwen vision 与 flow 均冻结；评测是 SIS-Bench 13 项 UAV VQA 的问答 accuracy，未报告点/框定位、tiny-object 指标或球 blur 条件。F/FR/FRE 的表 3 是逐级冻结继承模块、每级追加优化的 stagewise comparison；作者自己说明仍需 checkpoint-controlled progressive ablation 才能隔离 ReTEM/EASE。因此只能说论文报告含这些机制的整套训练流程获得更高 VQA 分数，额外优化与各组件的贡献尚未分离，不能将增益干净归因给 motion、reset 或可靠性本身。全文未报告 latency、FLOPs、显存，也未单列或计入 flow estimator 成本。

**对本项目的限制。** EgoSIS 使“用鲁棒二维 background transition 生成可拒绝的 support-quality 描述符，并在时间边界 reset”不再是新概念；它不排除、也未验证其对高速微小 RGB 球的候选覆盖、中心误差或误匹配抑制。若未来采用同类分析，只能把它作为 background-supported image-plane proxy，按球的定位协议另测收益和端到端成本，不能写成物理 camera/object decomposition。

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

**冲突。** DMR 是最直接的 small-target detector 先例；EgoSIS 是最新 pose-free **image-plane proxy** transition/residual/reliability factorization；传统 global motion compensation 更早已存在。

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
* 截至原2026-09-09检索，DQAligner、MIST、DeepPro出版方全文受限，CMRTrack只读摘要及片段。2026-09-10至11的局部补读已更新：DQAligner/MIST仍无全文但有固定源码；DeepPro开放v5全文与源码已读、TPAMI书目信息已确认；CMRTrack完整v1已读但仍为预印本。其余论文保持各条目实际阅读深度，不把预印本状态和全文阅读与否混为一项。
* 本轮未复现任何代码、未验收数据下载，也未验证这些小目标 benchmark 的标签定义与球数据完全一致。
* 未覆盖用户蓝图中的体育球专门方法、TrackNet 系列、视频 foundation model、数据集划分与许可；这些由其他研究笔记负责，不能从本笔记推论。

## 6. 可复用引用清单

1. Ji, Wang, Wang. **Optical Flow Estimation for Tiny Objects: New Problem, Specialized Benchmark, and Bioinspired Scheme**. IJCAI 2025. [Paper](https://www.ijcai.org/proceedings/2025/0136.pdf), [project code](https://github.com/JaneEliot/OTHR). A.
2. Zhang et al. **MOCID: Motion Context and Displacement Information Learning for Moving Infrared Small Target Detection**. AAAI 2025. [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/33087), [全文](https://ojs.aaai.org/index.php/AAAI/article/view/33087/35242). A（2026-09-10补读方法与实验）。
3. Deng et al. **Learning Global Dynamic Query for Large-Motion Infrared Small Target Detection**. IEEE TGRS 2026. [IEEE record](https://ieeexplore.ieee.org/document/11363482/), [official code](https://github.com/dengfa02/DQAligner_MIRSTD). B.
4. Gao et al. **MIST: A Benchmark and Baseline for Multi-frame Infrared Small Target Detection in Complex Motion**. IEEE TIP 2026. [IEEE record](https://ieeexplore.ieee.org/document/11511399/), [official code/data](https://github.com/ShuCvlab/MIST). B.
5. Li et al. **Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better**. TPAMI 48(8):10157–10175, 2026. [DOI](https://doi.org/10.1109/TPAMI.2026.3683258), [arXiv v5](https://arxiv.org/pdf/2506.12766v5), [official code](https://github.com/TinaLRJ/DeepPro). A（2026-09-11补读开放v5与固定源码，未取得IEEE排版全文）。
6. Zhang et al. **Decoupled Motion Representation Learning for Moving Infrared Small Target Detection**. arXiv:2606.15286, 2026. [HTML](https://arxiv.org/html/2606.15286). A, preprint.
7. Chen, Lan, Wei. **Counterfactual Motion Reliability Learning for Robust UAV Tracking**. arXiv:2607.23209v1, 2026. [arXiv](https://arxiv.org/abs/2607.23209), [全文](https://arxiv.org/pdf/2607.23209v1). A（2026-09-11补读全文；源码未取得）, preprint.
8. Yang, Huang, Wang. **QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection**. CVPR 2022. [Official PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf). A.
9. Bertasius, Torresani, Shi. **Object Detection in Video with Spatiotemporal Sampling Networks**. ECCV 2018. [Official PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Gedas_Bertasius_Object_Detection_in_ECCV_2018_paper.pdf). A.
10. Xu et al. **GMFlow: Learning Optical Flow via Global Matching**. CVPR 2022. [Official PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf). A.
11. Safadoust et al. **FlowIt: Global Matching via Hierarchical Transformers and Optimal Transport for Optical Flow**. arXiv:2603.28759v2, 2026-05-31. [arXiv](https://arxiv.org/abs/2603.28759), [project](https://kuis-ai.github.io/FlowIt/), [fixed author source](https://github.com/sadrasafa/FlowIt/tree/a6fa46829b1b5ae3fe0ff665f2d3771caf62bf9f). A（2026-09-11补读全文关键段与固定源码）；作者项目称 BMVC 2026 Oral，当前论文来源为 arXiv v2。
12. Luo et al. **Bidirectional Temporal Information Propagation for Moving Infrared Small Target Detection**. arXiv:2508.15415, 2025. [arXiv](https://arxiv.org/abs/2508.15415). A, preprint.
13. Liu et al. **MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration**. arXiv:2603.05071v1, 2026-03-05. [arXiv](https://arxiv.org/abs/2603.05071), [fixed author source](https://github.com/nliu-25/MI-DETR/tree/24257e4774c8f328738e88142c9a3cdabfd7afa5). A（2026-09-11 定向补读全文与固定源码）, preprint.
14. Yang et al. **EgoSIS: From Factorized Visual Ego-Transitions to Motion-Canonical Spatial Evidence for UAV Reasoning**. arXiv:2609.08938v2, 2026-09-09. [arXiv](https://arxiv.org/abs/2609.08938v2), [官方全文](https://arxiv.org/html/2609.08938v2). A（2026-09-11 定向补读全文方法与实验；未定位到可确认的作者公开代码）, preprint.
