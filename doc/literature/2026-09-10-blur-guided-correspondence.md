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
