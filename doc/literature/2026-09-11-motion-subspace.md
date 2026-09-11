# MoAlign：以 RAFT 流约束的 motion subspace，与球点定位的边界补读

日期：2026-09-11。范围：补读 MoAlign 官方论文全文及 A.1--A.5 附录、[ICLR 2026 官方 poster 页](https://iclr.cc/virtual/2026/poster/10009784)和作者公开发表页；未运行模型，未下载权重或数据。本文只展开 [modern_motion_evidence.md](modern_motion_evidence.md) 的一行，审查“motion subspace / feature adaptation”是否能构成当前高速微小球项目的主张，不能将它改述为自动检测或点对应结果。

## 结论先行

MoAlign 已是一个直接近邻：它从**冻结 VideoMAEv2**时空 token 经可训练 3D 卷积压缩出 64 维 $M$，要求其经 decoder 预测 RAFT 计算的稠密 flow；随后将 **CogVideoX-2B 第 18 个 MM-DiT block**的 diffusion hidden state 投影到同形状的 64 维张量，并匹配空间内和跨帧 token 的余弦相似关系。这足以限制“低维 motion subspace + 以 motion target 对齐另一骨干特征”作为项目的新颖性说法。

但它不是已证明的 appearance/motion 可辨识分解，也不是小球跨位置 correspondence。所谓 `disentangled` 的实际机制是**容量瓶颈加 flow 预测的任务对齐**：没有 appearance 对抗器、互信息/独立性目标、干预测试或“无法从 $M$ 读 appearance”的量化。其 flow supervision 只是作者未具名 RAFT 产生的二维稠密伪目标；不含球中心、目标 ID、无匹配标签或物理速度真值。Stage 1确有预测flow的训练decoder；最终文本到视频生成接口则不以$(x,y)$、逐点匹配、候选cost volume或流场为推理输出。论文也没有tiny object、自动点定位、中心误差、点轨迹、flow EPE或大位移correspondence指标。

因此，MoAlign 支持一个较窄且可证伪的问题：在本项目确定的 DINO 层、输入尺度和球中心监督下，投影/适配后能否提升球邻域的**可读性**或真实中心对应；它不支持立刻把整套视频 diffusion 微调作为必做 baseline，更不能称“flow-guided subspace”“appearance/motion 解耦”本身为贡献。

## 版本、状态、来源与读取深度

Aritra Bhowmik、Denis Korzhenkov、Cees G. M. Snoek、Amirhossein Habibian、Mohsen Ghafoorian，**“MoAlign: Motion-Centric Representation Alignment for Video Diffusion Models”**。官方 [arXiv 记录](https://arxiv.org/abs/2510.19022v1)显示目前可得版本为 **arXiv:2510.19022v1，2025-10-21**；其[官方 HTML 全文](https://arxiv.org/html/2510.19022v1)和[PDF](https://arxiv.org/pdf/2510.19022v1)是以下方法、附录和结果的主要证据。ICLR 官方页将同题同作者列为 **ICLR 2026 poster**，页面元数据给出发布日 2026-02-06；作者 [Amirhossein Habibian 的发表页](https://habibian.github.io/publications.html)也标作 ICLR 2026，并链接该 arXiv PDF。

本次实际读到正文 §3--§6、Appendix A.1--A.5；A.2 列出 VBench/VBench-2.0 全指标，A.3/A.4 给出两阶段结构、输入与训练细节。论文列出的作者项目链接 `https://qualcomm-ai-research.github.io/moalign/` 在本次读取时返回 GitHub Pages “Site not found”；GitHub 的精确题名和 arXiv ID 搜索未发现作者发布的代码仓库。因此没有用未固定第三方实现推断模型细节，也没有源码层面的可复现性证据。ICLR/作者入口确认的是会议状态，不替代对 arXiv v1 方法文本的逐项版本核对。

## 数学对象与监督边界

### Stage 1：VideoMAEv2 token 的低维流预测表示

对真实视频 clip $x_0\in\mathbb{R}^{F\times H\times W\times C}$，冻结视频 encoder $V$ 给出

\[
S=V(x_0)\in\mathbb{R}^{F''\times H''\times W''\times D_v},\qquad
M=M_\psi(S)\in\mathbb{R}^{F''\times H''\times W''\times D_m},
\]

其中 $M_\psi$ 是可训练投影头，论文用 $D_v=768,D_m=64$。轻量 decoder $F_\omega$ 将 $M$ 解码为 $\hat O$；正文抽象地将目标 $O$ 定义为在 $x_0$ consecutive frames 间计算的 flow，并以 $L_{flow}=\lVert\hat O-O\rVert_1$ 训练。这是论文 [§3.2 式 (3)--(4)](https://arxiv.org/html/2510.19022v1#S3.SS2) 的完整对象：$M$ 是带时空 token 格的中间张量，**不是**位移向量、目标轨迹或任意 target 的对应地址。

附录 A.3 进一步说明，冻结的 VideoMAEv2 输出为 `[B,768,24,H,W]`；$M_\psi$ 先以 $3\times1\times1$ temporal convolution 映至 256 channel，再以 $1\times1\times1$ 映至 64 channel，均接 SiLU。flow decoder 先将 24 个时间格变为 23，再两次 2 倍空间转置卷积，最后将 flow 插值为 `[B,23,2,128,192]`。[A.3](https://arxiv.org/html/2510.19022v1#A3) 报告输入视频 resize 为 $160\times240$、截为 49 帧、center crop；它**没有**交代 49 个输入帧如何映为 VideoMAE 的 24 个时间格和 23 个 flow target：49 帧若逐原帧相邻计算应有 48 对，文中未给 temporal sampling、聚合、target index 或 RAFT flow 的时间 stride。故不能确认这 23 个 target 是原视频逐相邻帧 flow，亦不能推断 RAFT stride 为 1 或 2。它同样没有说明所用 RAFT 的论文/代码版本、checkpoint、置信度/遮挡处理、RAFT 原生输入网格或该 `[128,192]` target 如何由 RAFT 产生。能确认的只是“作者在 decoder 输出形状上对 RAFT-computed target 做 L1”，不能把 `128×192` 写成 RAFT 的原生 grid，亦不能称它为人工 GT flow。

论文称 flow 是 “ground-truth optical flow”，其可操作含义仍是 RAFT 计算的 dense pseudo-flow。它可约束网络再现该估计器所给的 flow target；由于上述 49→24→23 时间映射未披露，不能进一步断言每项都对应哪一对原始相邻帧。它不能保证在遮挡、背景竞争、小于 token 尺度的球或大位移时是球的真中心位移，不能给出无匹配（no-match）判断，也不能取代有 ID 的真实 point correspondence 监督。

### Stage 2：relation alignment，而非坐标或匹配输出

MoAlign 建立在 **CogVideoX-2B** 文本到视频 latent diffusion 上；它的 3D VAE 在时间轴压缩 4 倍，MM-DiT 使用 bidirectional spatio-temporal attention。[§3.1 与 §4.1](https://arxiv.org/html/2510.19022v1#S3.SS1) 给出的 $Y_t\in\mathbb{R}^{\tilde F\times\tilde H\times\tilde W\times\tilde D}$ 是 noisy latent $z_t$ 经 denoiser 的 hidden state。作者在第 18 个 MM-DiT block 取 $Y_t$，用可训练 $P_\zeta$ 得到与 $M$ 同形状的 $Z\in\mathbb{R}^{F''\times H''\times W''\times64}$。

对每帧，把其 $H''W''$ 个 token 保留索引后组成所有 token-pair 的 spatial cosine-similarity matrix；再把所有帧 token 展平，对跨帧 token-pair 也计算 cosine-similarity matrix。intra-frame pair 的 temporal 权重为 0，其他 pair 按 frame distance 乘 $\exp(-\Delta_{ij}/\tau)$，最后以两个矩阵的**平均绝对误差**形成

\[
L_{align}=\frac1{F''}\sum_f\lVert S^{spatial}_Z(f)-S^{spatial}_M(f)\rVert_1+
\lVert W\odot S^{temporal}_Z-W\odot S^{temporal}_M\rVert_1,
\qquad L_{total}=L_{diff}+\lambda L_{align}.
\]

精确定义见[§3.2 式 (5)--(9)](https://arxiv.org/html/2510.19022v1#S3.SS2)。因此 loss 确实跨全空间 token pair、跨时间 token pair 做 MAE（后者有 $W$）；但它比较的是 teacher/student 的**关系结构**，没有把某一对 token 监督为目标真实匹配，更没有将 relation matrix 作为模型 API 输出。训练时这些 token 索引保有其格点地址；最终模型未导出坐标/匹配图，亦不能反过来证明 latent 中的地址信息必然不可恢复。是否能经特定 readout 提取球点，仍需本项目实验。

附录 A.4 将该 projector 具体化为 18th block 的 1920→256→64 temporal $3\times1\times1$ convolution、再 $1\times1\times1$ convolution，时间轴 trilinear 上采样 2 倍，空间端用 stride-3 的 $3\times3$ convolution 对齐 VideoMAE 格。[A.4](https://arxiv.org/html/2510.19022v1#A4) 还明确 Stage 2 冻结 VideoMAEv2 和 Stage-1 compressor，只更新 **CogVideoX transformer 与 $P_\zeta$**；这不是冻结 diffusion feature 的线性 probe。

### “disentanglement”究竟证明到哪里

作者的逻辑是 768→64 bottleneck “biases” $M$ 保留对 flow 预测有用的信息，并宣称压制 static appearance；消融比较未改 VideoMAEv2 feature、无 soft-TRD 及不同 MM-DiT layer 的 VideoPhy2 分数。[§5.2、Table 4--5](https://arxiv.org/html/2510.19022v1#S5.SS2) 表明，在其生成质量任务及该训练数据下，两项设计优于其 VideoREPA/FT 对照，18 层最佳。

这是一项**任务对齐效果**证据，并非表征可辨识性证明：全文没有 appearance 标签读出、去除 appearance 的反事实干预、子空间独立性统计、任意可逆重参数化处理，或证明 $M$ 不能保留对象/场景信息。特别地，flow 本身可受外观纹理、相机运动、遮挡和 RAFT 误差影响。故写作宜说“为 RAFT pseudo-flow prediction 而训练的 compressed motion-oriented subspace”，不能说已分离出纯物理运动或可靠球对应。

## 时间语义、可用视频与输出任务

Stage 1 的正文抽象定义将 flow 置于同一真实 $x_0$ 的 consecutive frames 间，但实现中 49→24→23 的映射未知，不能写成原视频逐相邻帧监督；Stage 2 的 diffusion denoiser 对同一完整训练 video latent $z_t$ 使用双向时空 attention，并同时接受文本 condition。它没有“以第 $t$ 帧为预测锚、只读过去 $k$ 帧”的检测协议，生成时从噪声迭代得到**完整视频**。论文所称不需额外 inference-time conditioning，是指生成模型在推理时不需要外接 flow/pose/control 输入，[Figure 2/§3.2](https://arxiv.org/html/2510.19022v1#S3.SS2)；不等于它提供因果末帧球定位或实时流估计。

训练数据是 Open-Sora Plan video dataset 的 350K subset 加 Wan2.1-14B 生成的 16K synthetic videos；论文未在此处报告为小物体、体育球、点位或大相对位移专设的采样/标注。[§4.1](https://arxiv.org/html/2510.19022v1#S4.SS1)。项目若采用历史窗口，必须另定义窗口边界、target frame、是否可看未来和真实球中心标签；不能把 MoAlign 的整 clip 生成训练当作这些选择的依据。

## 作者实际测了什么，以及没有测什么

官方任务是文本到视频的物理合理性/生成质量：VideoPhy2（591 extended prompts，以 auto-evaluator 的 semantic adherence、physical commonsense 和 joint score）、VideoPhy（343 prompts）、VBench、VBench-2.0 和 50 个视频、672 次 pairwise preference 的 blind user study。[§5.1](https://arxiv.org/html/2510.19022v1#S5.SS1)；附录 A.1 是人跳跃、吃布丁、倒水等 qualitative video，A.2 给出 VBench 的 subject/background consistency、motion smoothness、object class、多对象、dynamic spatial relationship、camera motion 等 benchmark 项。

这些是生成内容的语义、物理/感知质量指标。即使其中出现 coin、glass、spoon 或 “object” score，也没有目标像素尺寸分桶、自动对象发现 precision/recall、中心点误差、初始化 point tracking、遮挡/no-match、flow EPE、候选覆盖/rank 或大位移对应 accuracy。不能用 VBench 的 motion/对象命名指标或图示替代微小球逐帧定位证据。

## 成本与可复现缺口

论文报告的是真实两阶段**训练**配置：Stage 1 为 50,000 iteration、batch 128、4×H100 80GB、AMP；Stage 2 为 4,000 step、batch 32、4×H100 80GB、AMP，$\lambda=0.5,\tau=10$；前者输入为 49 帧 $160\times240$。[§4.1](https://arxiv.org/html/2510.19022v1#S4.SS1)与[A.3--A.4](https://arxiv.org/html/2510.19022v1#A3)明确这些训练资源。论文没有报告单视频生成秒数、编码/RAFT preprocessing 时间、显存峰值或“自动小球定位窗口”的 latency/FLOPs；不可将其 H100 配置包装成本项目可负担的实时 motion module。

复现尚缺官方代码、准确 RAFT release/checkpoint 与预处理、VideoMAEv2 具体 checkpoint、CogVideoX 权重/采样参数，以及 350K/16K 数据筛选的完整可复现描述。本次不以猜测补齐。

## 对本项目会改变的研究判断

1. 不能提出“将 backbone feature 压至 motion subspace，再以 flow/关系对齐另一特征”作为新机制；MoAlign 已覆盖这一范式，并已在生成任务中做过 motion-feature、soft relation 和 layer 选择消融。若 DINO adaptation 继续存在，贡献必须落在**小球尺度、明确中心/存在标签、自动发现与跨帧真中心对应的实证差异**，而不是“subspace”名称。
2. 首先做低成本、可反驳的诊断：固定输入尺度与时间锚，比较原 DINO 层和指定投影/适配后的球中心邻域 readout、候选覆盖、真中心 rank/误差，以及无球/背景竞争。投影较好只可称该条件下球点可读性改善；还要单独验证它是否改善跨帧 correspondence，不能由 RAFT 伪流 loss 推出。
3. 若上述读出没有改善，停止把 motion-subspace adaptation 作为首要解释；若改善且观测窗口实际延伸到末帧 target 之后，才须按末帧 target 重测因果版本。窗口仅含至末帧 target 为止的已见帧，即使 encoder 在该窗口内双向 attention，时间上仍可合法。不要因这篇 2B diffusion 生成方法而堆建 video diffusion、关系矩阵或 RAFT pseudo-label pipeline；它们的成本、输出和监督对象都与自动几像素球定位不同。

本次证据面限定于官方论文、ICLR/作者页面。没有把作者对“ground-truth flow”“disentangled”的措辞扩展为可辨识物理运动，没有把无显式坐标输出误写为信息消失，也没有将论文的生成质量提升改写为 tiny-ball 检测成功。
