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

Radim Spetlik等，WACV 2024，pp.6857–6866。[正式论文](https://openaccess.thecvf.com/content/WACV2024/papers/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.pdf)、[作者代码](https://github.com/radimspetlik/SI-DDPM-FMO)。实际阅读摘要、§1及仓库README任务/数据说明，未声称完整复现。

该方法以单张模糊图生成快速物体的子帧轨迹与形状，使用合成FMO训练。论文明确单帧无法确定轨迹方向，物体与背景也存在歧义；推理不需要GT轴或邻帧。仓库所述完整训练数据需要自行生成，不能因为有代码就当成本项目现成廉价基线。

它是单帧FMO轨迹恢复的直接现代先例，未提供有限预算下的跨帧ball correspondence证据。本项目当前也不因此引入扩散模型。

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
