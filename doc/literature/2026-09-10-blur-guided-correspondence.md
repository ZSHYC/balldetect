# 拖影轴能否约束帧间搜索：已有先例与候选边界

核对日期：2026-09-10。性质：围绕一个候选问题的有界一手文献审查。

问题来自[BlurBall训练侧轴统计](../experiments/2026-09-10-blurball-axis.md)：曝光内拖影无向轴与短间隔位置差存在联系，这种联系能否帮助有限计算下的correspondence搜索。本文不提出新模型，也不重复已完成的[标注几何核验](second_pass_measurement.md)。

## 当前判断

值得保留为待检验候选，但目前没有新颖性或自动定位收益证据。已有方法早已从blur恢复运动、用blur改善光流、从快速小物体的拖影恢复曝光内轨迹。无向轴、曝光轨迹、帧间位移和球的自动发现仍是不同问题。

本轮未检到专门将“单帧无向blur轴约束”用于自动tiny-ball帧间候选搜索的一手工作。这个有限检索结果不能支撑“首次”：机制近邻跨越光流、去模糊、fast moving object和曝光内跟踪，搜索词也未必与最终实现同名。

## 最接近的五项一手证据

### Optical Flow in the Presence of Spatially-Varying Motion Blur

Portz、Zhang、Jiang，CVPR 2012，pp.1752–1759。[作者项目及论文](https://pages.cs.wisc.edu/~lizhang/projects/blurflow/)。实际阅读摘要、引言、blur-aware数据项、四帧核构造及局限。

该方法直接在模糊帧之间估计光流，用四帧I0… I3的初始flow及duty cycle构造分段线性模糊核，加入blur-aware匹配；不是把人工GT轴作为输入。其核方向来自跨帧估计，不能视作单帧无向轴已决定运动符号。小于1px的核回退原方法，非分段线性运动会破坏近似。

这项工作已经封堵“首次利用blur改善correspondence/flow”。它还指出单帧曝光内运动信息不能单独确定任意相邻帧的准确对应，直接限制将BlurBall轴统计当作帧间位移真值的解释。

### Exposure Trajectory Recovery from Motion Blur

Youjian Zhang等，TPAMI 44(11)，7490–7504；本轮读[arXiv v2，2021-10-04](https://arxiv.org/html/2010.02484v2)，并查[作者代码](https://github.com/yjzhang96/Motion-ETR)。阅读范围为摘要、I、III-A–D、V-A、V-C与VI。

ETR从单张模糊图恢复曝光轨迹，以blur/sharp配对训练，使用offset及线性、二次等轨迹约束，服务于再模糊、去模糊和子帧提取。它不要求运动GT，却也不做两个视频帧的自动球关联。曝光积分丢失时间顺序，作者的视频EPE在正反方向中取较小者；大的相机抖动、高动态运动及域外blur仍有限制。

它限制“单帧恢复motion latent/轨迹即新贡献”的说法，也说明即使输出一条看似有方向的轨迹，评价可能仍容许整体时间反转。球项目需要保留这种歧义。

### Intra-Frame Object Tracking by Deblatting

Jan Kotera等，ICCV Workshops 2019，pp.2300–2309。[正式论文](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kotera_Intra-Frame_Object_Tracking_by_Deblatting_ICCVW_2019_paper.pdf)、[作者代码](https://github.com/rozumden/tbd)。已读摘要、§1、§3及官方评测说明。

TbD面对fast moving object，包括体育中的高速球，通过盲去模糊、matting和一维轨迹约束恢复曝光内轨迹、形状与外观。高帧率真值用于评测，不能把它说成推理输入。它提供的是曝光内物体跟踪，不能直接等同跨帧自动对应。

因此“从高速体育球的拖影恢复运动轨迹”已有直接先例。论文若要使用blur几何，需要在推理条件、监督、计算与自动定位误差上说明实质差异，仅将物体换成tiny ball不足以成立贡献。

### Single-Image Deblurring, Trajectory and Shape Recovery of Fast Moving Objects With Denoising Diffusion Probabilistic Models

Radim Spetlik等，WACV 2024，pp.6857–6866。[正式论文](https://openaccess.thecvf.com/content/WACV2024/papers/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.pdf)、[作者代码](https://github.com/radimspetlik/SI-DDPM-FMO)。2026-09-10 初读限摘要、§1及README；2026-09-12 已升级为下述全文与关键源码补读，未复现模型。

该方法以单张模糊图生成快速物体的子帧轨迹与形状，使用合成FMO训练。论文明确单帧无法确定轨迹方向，物体与背景也存在歧义；推理不需要GT轴或邻帧。仓库所述完整训练数据需要自行生成，不能因为有代码就当成本项目现成廉价基线。

它是单帧FMO轨迹恢复的直接现代先例，未提供有限预算下的跨帧ball correspondence证据。本项目当前也不因此引入扩散模型。

#### 2026-09-12 补读：单帧 FMO temporal super-resolution 的真实输入与评测边界

**阅读升级与来源。** 本次读完 [WACV 2024 正式论文](https://openaccess.thecvf.com/content/WACV2024/papers/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.pdf) §3--§5、[作者 README](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/README.md)和官方 benchmark runner；源码固定于 `9e5afcba`，不下载权重或数据。根代理复读方法、Table 1 图像与 ROI/方向源码；PDF 和文本保留在 `outputs/literature/siddpm-fmo-wacv2024.{pdf,txt}`。原 README 的 arXiv 引用实际指向旧 DeFMO，不将其误作 SI-DDPM-FMO 的预印本版本；本文以 WACV 正式版本为准。

**事实（输出与曝光内时间）。** 给定一张含 FMO 的 RGB crop，模型从噪声生成 `K=24` 组 residual image 与 alpha mask；论文 Eq.(15) 用输入图加上“各时间 alpha 的平均值 × 该子帧残差”合成外观，轨迹取 alpha-mask 质心。配置为 `256×256`、100 个 diffusion step。为与高帧率评测对齐，论文将每三张生成子帧平均成一张；公开代码还按 `nsplits` 支持两张一组，所以不能把 24 个输出都称作独立实测时刻。[论文 §3、§5](https://openaccess.thecvf.com/content/WACV2024/papers/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.pdf)；[评测中的采样、时间平均及质心](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/fmo_deblurring_benchmark.py#L137-L231)。这是曝光积分内的 temporal super-resolution，并不是相邻视频帧的 object correspondence。

**事实（时间方向）。** 作者明确说单张模糊图的时间方向不可判定。FMO benchmark 的 TIoU 对估计轨迹和其完整反转都计算，再取较大者；外观评测也以高帧率 GT 决定是否翻转生成序列。[方向同步和 TIoU 代码](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/benchmark/loaders_helpers.py#L224-L231)，[TIoU 的正反最大化](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/benchmark/reporters.py#L181-L186)。这是**评价时使用 GT 消除方向歧义**，不是推理时已知的运动正负号，也不能拿来约束后续帧的球位置。

**事实（是否需要背景、ROI 或初始化）。** SI-DDPM-FMO 网络本身以单张 FMO 图为条件，不额外输入背景估计；比较的 SI-DeFMO baseline 才从单图生成背景再送入 DeFMO。可是公开 benchmark runner 并非全图自动检测：它从 `GroundTruthProcessor` 取 GT trajectory、半径和 box，再用高帧率 GT 与背景收紧 `bbox_tight`，将该 ROI 扩展、crop、resize 后才调用模型。[runner](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/benchmark/benchmark_loader.py#L46-L83)与[ROI 构造](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/benchmark/loaders_helpers.py#L161-L186)。因此论文证明的是**已知 FMO 区域内**的单图去模糊/轨迹恢复，不是从复杂整帧自动发现球或给多候选关联。

**事实（训练与真实评测）。** 训练为合成数据：Blender 渲染 ShapeNet 的 50 个类别、DTD texture、VOT background 与 6D trajectory，论文报告 50,000 张训练图；作者 README 说明完整 24-timestamp 训练集约 1 TB、需自行生成且受 ShapeNet license 限制。[论文 §4](https://openaccess.thecvf.com/content/WACV2024/papers/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.pdf)，[作者数据说明](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/README.md#L50-L77)。真实评测使用 FMO benchmark：TbD 的 240 fps 原始视频经时间平均为 30 fps，并配有高帧率 appearance、mask 与完整轨迹；TbD-3D 和 Falling Objects 也用高帧率真值做评测。网络没有额外输入高帧率 RGB，但上述 ROI 构造依赖高帧率 GT 与背景，因此端到端评测仍有 GT 条件，不能笼统称为全程只用一张原始图。

**事实（多样本和选择）。** 本次读到的正式实验和官方 `fmo_deblurring_benchmark.py` 路径为每个输入调用一次 diffusion sampling，未发现将多次 DDPM samples 以 GT 挑 best 的路径。配置中的通用 `num_samples: 100` 未被该评测路径读取，故不能据此声称论文做了多样本 best-of-N。反之，**方向** best-of-two 是明确存在的 GT 辅助评测选择。未检查的 demo/分支不能据此断言完全没有其它采样用法。

**事实（实际计算）。** 经 Table 1 原 PDF 列对齐核对，SI-DDPM-FMO 为 **5 fps 的生成图像吞吐**；`0.001 fps` 属于 TbD-3D，不能错移到本方法。表注定义的是每秒生成多少图像，不能直接解释成每秒处理 5 个完整输入视频帧；本地尚未测量实际延迟。论文报告多 GPU 训练，README 要求自行生成大型合成集。主线暂不采用它的依据是任务、输入与监督条件不同，以及引入整套曝光重建需要额外论证；不能用误读出的极慢速度否定它。[正式表1](https://openaccess.thecvf.com/content/WACV2024/papers/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.pdf#page=6)、[作者训练说明](https://github.com/radimspetlik/SI-DDPM-FMO/blob/9e5afcba5fdcd95884b5132595da341283bcd33c/README.md#L43-L67)。

**推论（对本项目真正的约束）。** 恢复出的 alpha-mask 质心只有在 FMO 已被正确裁入 ROI、生成 mask 与真球而非背景 streak 对齐时才是有用的球中心候选；该链条并不证明自动发现、absence 判断或逐帧中心容差正确。它更强地限制“单帧拖影恢复曝光内轨迹/形状”这种宽泛新颖性，却不构成大位移跨帧 correspondence、可用方向 sign 或无 oracle 自动定位的证据。

**另一个不能跳过的推论。** 时间方向不确定，不等于 BlurBall 标注中点不可定位。若目标定义为两个拖影端点 `a,b` 的几何中点，`(a+b)/2` 在交换端点后不变。完整曝光轨迹、方向和形状的恢复比估计这个中点要求更多；不能把前者的不可辨识性或计算量直接转嫁给后者。背景竞争、端点本身是否可辨、遮挡和标注定义仍可能限制中点估计，这些需要真实图像与当前任务评价。

**当前研究决定。** 保持“不把扩散模型接入当前主线”。若将来研究单帧 blur 轴，按图像可得的**无方向**曝光内证据处理；若另借历史判方向，需明确新增的时间证据。未来模型有硬候选时再分开测候选正确/错误条件；直接 dense 输出则测其实际定位路径。不把 benchmark 的 GT ROI、方向翻转选择或高帧率评测真值带入正常推理。

### MoTDiff: High-resolution Motion Trajectory estimation from a single blurred image using Diffusion models

Wontae Choi等，[arXiv:2510.26173v1，2025-10-30](https://arxiv.org/html/2510.26173v1)，按预印本引用。全文HTML可读；本轮读摘要/I、II、III-A–D、IV-A、IV-C及消融、V，并非只读摘要。所读正文未见作者代码链接，不推断代码一定不存在。

MoTDiff以单张模糊图的多尺度条件生成高分辨率运动轨迹，使用合成GoPro patch、PSF与高分辨率轨迹数据，服务于blind deblurring和coded exposure摄影。它不评价自动跨帧球关联。

这进一步封堵“高分辨率单帧blur轨迹”层面的宽泛新颖性；其生成轨迹质量不能替代球发现、对应候选覆盖或严格定位指标。

## 与本地统计合并后，真正剩下的问题

当前BlurBall合法中心对中，约三分之一因l=0没有可用标签轴。对于有定义的轴与非零位移，Δ1有86.42%落在15°轴误差内，Δ2为78.26%。较短拖影的角度误差更大，最长拖影组还存在来源组成差异。这些数值已在训练侧测量，未使用最终测试，不能当作推理中可得信息。

如果将±15°当作硬搜索方向限制，即使用标签轴，也不能保证覆盖所有现有真中心对。由像素估计的轴还会增加误差；轴本身没有给出搜索距离、速度正负或曝光外的反弹轨迹。单独提高轴预测精度，也不保证球所在查询位置能被自动找到。

因而任何后续使用——硬候选带、软空间先验或其他形式——都需要回答三个实际问题：

1. 轴来自推理时的图像预测，还是GT中心/GT轴？只有后者有效的结果仍是oracle。
2. 当前候选已经是背景线条时，估计出的轴是否只是背景方向？不能先用GT找到球再隐去这个前提。
3. 在相同搜索面积或计算预算下，它是否提高真历史中心覆盖和最终定位/检测，而不是只生成更漂亮的轨迹？

无轴时不能把l=0解释为球静止；可能是短曝光、运动量不足、标注定义或当前图像不能提供明显拖影。强blur也不保证直线运动：曝光内加速度、反弹、相机运动和物体遮挡都会改变关系。当前不定义可靠性网络，也不预选一种回退或融合结构。

## 对候选试验的约束

若全量强基线结果支持继续该方向，应先明确实际残留错误和推理输入，再固定同预算的必要对照。例如轴无关候选、随机轴候选与图像预测轴候选可用来区分形状/预算变化和真实方向信息；并不是现在就要把三个模块都实现。

标签轴最多用于有标签训练监督或明确标注的上界诊断。当前帧未知球位置、历史未标注、VC3遮挡位置等情况不能凭空制造正视觉对应。比较同时保留严格位置、检测与候选漏失；不能以舍弃困难目标换取剩余样本的匹配准确率。

只有图像可获得的轴在这些条件下提供额外收益，才有保留具体机制的理由；若只在GT轴下有效，或相同预算的普通候选已经解释收益，就停止该实例化。即使有效，也需围绕实际实现再作最近邻核对，不能沿用本文的“未检到”当首次性证明。

## 检索范围与限制

实际检索式包括：

```text
"motion blur" "optical flow" blur direction arXiv
"motion blur" "feature matching" blur kernel direction correspondence
"blur-guided" tracking motion blur direction object tracking
site:arxiv.org/abs "motion-blurred" "optical flow" 2024 OR 2025 OR 2026
```

随后按TbD、ETR和MoTDiff准确题名补检，深读限制为上述五项。筛选保留会改变“曝光内轴约束帧间关系”判断的工作，没有把一般图像去模糊论文堆成列表。最早先例可比仅按年份追新更直接；同时纳入可访问的较新预印本，不把出版或代码状态未核实之处补成事实。

本页不覆盖所有运动模糊、光流、点跟踪或扩散轨迹论文，也不提供穷尽检索保证。BlurBall、TrackNet和其他tiny-motion先例仍见既有专题；本次结果只决定这个候选应带哪些约束继续被审视，不决定主模型架构。

## 2026-09-11 补检：状态自适应与上下文路线已经有近邻

本次只补检 2025--2026 年能改变下一步决策的自动 RGB 体育目标定位工作；不重读 BlurBall，也不把一般图像去模糊列为相关工作。结论是：**没有发现推翻“先测自然 blur 条件下的误差，再选 motion 机制”的直接证据；相反，两项近邻把该顺序变得更必要。**

实际检索包括`"fast-moving object" localization "motion blur" 2025 arXiv`、`"fast moving object" detection tracking motion blur 2026 arXiv`、`deblatting fast moving object 2025 2026 arXiv`、`sports ball localization motion blur 2025 2026 arXiv`，随后按MoSA-Det、PLUCC、LDINet题名补查全文与作者代码。arXiv检索结果不足以保证穷尽；本次只深读下列两项与当前决策有关的新增来源。

### MoSA-Det：运动状态已被用于自适应时序融合，但不是 tiny-ball 点定位

Lulu Yang、Wenqing Sun、Jinkui Ren，*MoSA-Det: motion state adaptive object detection for sports videos*，Scientific Reports 16, 15969，2026-04-03。[正式文章](https://www.nature.com/articles/s41598-026-43231-2)，[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13194943/)。阅读深度：[P] 摘要、引言、运动状态估计/MAAF/SGTA、损失、三帧设置、ball 消融、伪标签敏感性和局限。

作者以帧间 feature difference 与 local correlation 估计像素级 `static/slow/fast` 状态；该状态同时调节多尺度/可变形采样与跨帧注意力。快速区域的历史权重被压低，且另有位置补偿分支。状态监督却来自**相邻帧 GT box displacement** 的 one-hot pseudo-label；其三帧 `SGTA`、SoccerNet-Tracking/SportsMOT box-mAP 设置与本项目的自动微小球中心定位不同。论文报告 SoccerNet 的 Ball 类提升，但没有报告 BlurBall 式中点、逐帧像素容差、候选漏失、streak-length 条件或有/无球中心 heatmap 的证据。

它限制了“首次按 motion state 选择时序聚合”“首次用 feature difference/local correlation 产生状态先验”这类宽泛表述，也表明以 GT 位移生成状态监督已经是已知训练路径。静/慢/快是运动状态，不等于对应正确概率或校准的可靠性；不能据此把所有 motion reliability 主张都视为已被同一机制覆盖。它**不**证明在真实 tiny-ball 视觉证据稀弱时，状态估计可发现球、对应可信，或应扩大搜索范围。若未来研究状态门控，必须先与该近邻区别在于：状态来自何种推理时可得证据、是否对球的定位/漏检而非 boxes 有效，以及相同计算预算下的条件收益；当前不据此加入模块。

### PLUCC：模糊微小目标可由上下文改善，但其证据不能替代 motion 诊断

Liam Salass 等，*Ice Hockey Puck Localization Using Contextual Cues*，CVPRW CVSports 2025；[正式论文](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Salass_Ice_Hockey_Puck_Localization_Using_Contextual_Cues_CVPRW_2025_paper.pdf)，[arXiv HTML](https://arxiv.org/html/2506.04365)。阅读深度：[P] 摘要、§1--§4.6 和数据/指标限制。

该工作以单帧全分辨率 puck heatmap 与预训练 player detector/segmenter 生成的上下文图融合；作者明确将 small size、blur 与遮挡作为问题，并在其私有冰球数据上报告像素距离/AP。它没有使用帧间 correspondence；加入上下文预处理后的流程约6 FPS，必须计入成本，不能据此直接断言本机无法训练。其私有数据不能作为本项目现成公开2-D标注的直接复现基线。

因此不能把“高分辨率 heatmap + 人/场景上下文帮助高速微小目标”写成新颖贡献。它也不要求现在加入球员检测或分割：本项目尚未测得 BlurBall 的自然 blur 残差究竟来自局部证据缺失、跨帧错配还是背景 FP。只有结果显示当前视觉/历史对应不足而语义上下文能以可接受成本区分错误时，才有理由把上下文作为另一条受控路线；届时必须与 motion 机制分开归因。

### 其他检索命中没有新增决策证据

2025 的 LDINet（Haodong Fan 等，*Journal of Visual Communication and Image Representation* 109:104439，[DOI](https://doi.org/10.1016/j.jvcir.2025.104439)）继续单张 FMO deblatting 的 latent decomposition/interpolation 路线，使用合成前景 blur 与背景条件。实际只读取出版社页面Highlights、摘要、引言/结论片段，未全文深读；其中没有提供自动跨帧球发现或有限搜索逐帧中心定位的证据，未改变上文结论。精确题名与作者代码检索未发现可核实的作者代码链接，不等于证明没有代码。2026 的 YOLO-Ball 仅能从作者期刊页的摘要确认其在自建网球数据上以 box mAP 处理 blur/occlusion，缺少公开数据、像素级中心协议与可复核完整方法证据，不作为当前设计依据。[来源](https://doi.org/10.1177/17543371261423768)。

**当前决定。** BlurBall DINO 三帧中点基线仍先按 `l`、可见中心、原图像素误差、FP1/FP2/FN 和帧间 `d_px` 诊断自然残差；不从 MoSA-Det/PLUCC 直接移植状态门控、可变形采样、attention 或人类上下文。未来任何选择先用该条件残差提出可证伪问题，再做与这两项近邻可区分的同预算对照。
