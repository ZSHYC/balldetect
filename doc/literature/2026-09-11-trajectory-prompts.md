# Motion-as-Prompt：把既有点轨迹画回帧的 MLLM 提示，与自动微小球定位的边界

日期：2026-09-11。范围：只补读 Xikai Sun 等的 Motion-as-Prompt（MaP）一篇，实际阅读 [arXiv v1 HTML 全文](https://arxiv.org/html/2608.11655v1)、[arXiv 摘要/版本页](https://arxiv.org/abs/2608.11655) 和 [作者公开仓库](https://github.com/SunVictor23/MaP)。本文展开 [modern_motion_evidence.md](modern_motion_evidence.md) 的一行；未下载权重或数据、未运行 tracker/MLLM，且不把其问答结果改写为球点定位证据。

## 结论先行

MaP 的直接贡献是：先在完整视频上以**冻结的 CoTracker3**恢复规则格点轨迹，再用这些轨迹的相机补偿残差选 keyframe，并将相邻已选帧之间的轨迹段画到较晚一帧，作为冻结 MLLM 的像素提示。它解决的是稀疏 MLLM 输入丢失中间运动这一展示问题，不训练 MLLM。**最终 MLLM 的问答接口和论文评价**不输出物体中心、点对应、检测分数或轨迹置信度；这不否认内部 tracker 输出每个 query 的点位置与 visibility。[论文 §3](https://arxiv.org/html/2608.11655v1#S3)；作者固定源码也把 tracker 明确称作 frozen motion sensor，而非 VLM 的可学习部分：[`SunVictor23/MaP@e150b0433613ec789f48b3d7d67f9c1ef535e57f:map_kit/data/cotracker_runner.py:L0-L7`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/data/cotracker_runner.py#L0-L7)。

因此它是“已有**上游轨迹**怎样在稀疏视觉输入中保留跨帧运动”的直接先例，也限制本项目把画线、轨迹视觉提示或 motion-guided frame selection 单独写成创新；它**不是**自动发现几像素球、证明稠密查询覆盖球、或将显式画线当作当前帧球中心读出的先例。一个规则网格能否恰好在球上初始化、CoTracker3 对该点能否持续可见、随后 motion filter 会不会保留它，都需要本项目实测。论文没有球/小目标的 query-coverage、中心误差或对应准确率，不能无证据改称“稠密 query 一定漏球”。

## 版本、身份、阅读面与可访问性

论文完整题名为 **“Motion-as-Prompt: Enhancing Motion Reasoning in Multimodal Large Language Models via Motion-Guided Cross-Frame Visual Prompting”**，作者 Xikai Sun、Kebin Liu、Haotian Wang、Li Liu、Xu Wang、Yunhao Liu。[arXiv:2608.11655v1](https://arxiv.org/abs/2608.11655) 于 **2026-08-12** 提交，标题页写明 “Preprint. This manuscript is currently under review”；截至本次补读未见作者列出会议、OpenReview 或 CVF 正式条目，故只能称预印本，不能附会会议身份。

本次还实际打开 [arXiv v1 PDF](https://arxiv.org/pdf/2608.11655v1)：它共 **9 页**，正文至结论在 PDF 第 1--7 页、其后为参考文献；没有 HTML 漏掉的 appendix 或附录页。正文虽称完整伪代码在 supplementary，但 arXiv 版本页只提供 PDF/HTML/TeX source，PDF 亦没有附录附件链接；本次未能取得独立 supplementary，故不以它补写参数或实验。实际阅读面为 PDF 全 9 页（§1--§5、算法 1、表 1--3、图 3--5、参考文献）。作者仓库可访问、没有 release/tag，当前 `main` 固定到 `e150b0433613ec789f48b3d7d67f9c1ef535e57f`（2026-08-13，论文 v1 后一天）。以下源码仅是后续公开实现的交叉证据，尤其不能反向认定为 v1 的精确实验配置。

## 上游轨迹：谁跟、在哪里初始化、何时才可用

论文在全帧率视频 $V=(I_1,\ldots,I_N)$ 上，以冻结 point tracker $\mathcal T$ 跟踪 $G\times G$ query grid；固定间隔（例为每秒）重新播种，以接纳 clip 中途进入的对象。输出是每点位置 $\mathbf p^i_\ell\in\mathbb R^2$ 与可见标志 $o^i_\ell\in\{0,1\}$。[§3.2 Dense tracking](https://arxiv.org/html/2608.11655v1#S3.SS2) 的实验配置只明确 tracker 是 **CoTracker3**、$G=10$，即每次播种 100 个规则 query；并没有报告 CoTracker3 checkpoint、输入分辨率、实际 tracking fps、重新播种的精确周期、每个 clip 的 query 总数，或该 10×10 格与对象尺度的 coverage 统计。[§4.1](https://arxiv.org/html/2608.11655v1#S4.SS1)。

作者代码的同一层操作是可读的，但与论文可确认配置有版本差异：`_grid_queries` 在 $t=0$ 及每个 `round(reseed_every_s*fps)` 帧放置内缩规则格；每个 query 带自己的 seed-frame index，输出 `(T,N,2)` tracks 和 `(T,N)` visibility。[`cotracker_runner.py:L116-L140`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/data/cotracker_runner.py#L116-L140)、[`L142-L195`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/data/cotracker_runner.py#L142-L195)。但这个提交的库默认 `grid_size=30`、`tracking_fps=8`、最多 200 tracking frames，而 README 的论文 runner 默认 `grid_size=10`、`segment_track=on`、`online=off`；论文只可证实后者的 10×10，不可把库的 30/8/200 误写成论文设置。[`pipeline.py:L55-L79`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/core/pipeline.py#L55-L79)、[README 的论文 runner 说明](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/README.md#L272-L283)。

可见性并非可省略的装饰：论文以相邻两帧均 visible 的点，拟合能最好解释这些位移的**全局 similarity transform** $\mathbf p\mapsto s\mathbf R\mathbf p+\mathbf b$（平移、旋转、尺度）；以 $(a,b,t_x,t_y)$ 参数化，解线性 least-squares，少于 3 点时 camera term 置零，随后才从 raw displacement 中减去拟合的 camera flow，得到 residual velocity $\mathbf v^i_\ell$。[§3.2](https://arxiv.org/html/2608.11655v1#S3.SS2)。这是明确的相机补偿前例，但该 residual 仍依赖 tracker、全局模型和可见点集合，不能写成已验证的“纯物体运动”。源码也在画线时跳过接触 invalid（pre-seed、离场或被 tracker 判不可见）的线段，避免把填补后的坐标当真轨迹：[`track_marker.py:L491-L530`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/core/track_marker.py#L491-L530)。这只能说明实现意识到 visibility 边界，不能给出任何 tiny ball 的 visibility/occlusion 准确率。

## 从轨迹到选帧和画线：实际数学对象

每个时刻的 motion energy 不是球的 score，而是所有可见 query 的相机补偿后残差速度、相邻速度差与转向角的均值；此残差也不能自动等同为纯 object motion：

\[
M(\ell)=\widehat{\Phi_3\!\left(
\widehat{\mathrm{spd}}_\ell+
\widehat{\mathrm{acc}}_\ell+
\widehat{\mathrm{cur}}_\ell\right)}\in[0,1].
\]

三项各以 95th percentile 归一化、等权相加、长度 3 的移动平均后再归一化；它是视频整体的运动强度，绝不是某一对象的可靠 correspondence 或球存在概率。[§3.2 Motion energy](https://arxiv.org/html/2608.11655v1#S3.SS2)。

在帧预算 $B$ 下，MaP 先取 $n_a=\max(2,\lceil\alpha B\rceil)$ 均匀 anchor（实验 $α=1/4$），其余从 $M$ 的峰值按 $r=\max(1,\lfloor N/(2B)\rfloor)$ 做时域 NMS；不足再按 motion energy 由高到低填满。每个选中 index 都映射回原视频时间戳。[算法 1 和 §3.3](https://arxiv.org/html/2608.11655v1#S3.SS3)。这保持同一 $B$，但不是逐目标的候选保留机制：背景、相机补偿残差或其他对象可决定峰值，论文没有测球/点在选帧集合内的 recall。

对选中序列 $s_1<\cdots<s_B$，作者把从 $s_{k-1}$ 到 $s_k$ 的**可见 query**轨迹段 $\gamma_k^i=(\mathbf p^i_\ell)_{\ell=s_{k-1}}^{s_k}$ 画到较晚的 $I_{s_k}$，以 polyline 加终点圆标记提示 MLLM。论文只保留在后续 $\Delta$ 帧最大位移超过 $\tau W$ 的点，$τ=0.03$，按总运动排序且最多 $K$ 条，但没有给出 $K$ 或 $\Delta$ 的数值。[§3.4](https://arxiv.org/html/2608.11655v1#S3.SS4)。所以该可视化输出**保留图像坐标**，但只是上游 tracker 的位置绘图；它没有反过来为 MLLM 产生可评测的点地址、对象 ID 或 match/no-match 决策。

后续源码细节也应只作版本受限的解释：它使用约一秒窗口的位移阈值 $0.03W$ 去掉静点，再按总运动排序，可选择 cap；此处的代码路径与论文的“从每个 sampled frame 看 following $\Delta$”文字不完全同构。[`track_marker.py:L121-L199`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/core/track_marker.py#L121-L199)。不能以此后续实现填补论文未给的 $K/\Delta$，更不能将筛掉的轨迹解释为“球不可靠”。

## 时间锚、未来信息和成本口径

论文的任务是对**完整视频**先做 dense tracking、全局 $M(1{:}N)$ 选帧，再交给 frozen Qwen3-VL-2B 或远程 GPT-5.5 问答；论文没有定义单个目标帧的在线输出时刻。[§3.1](https://arxiv.org/html/2608.11655v1#S3.SS1)、[§4.1](https://arxiv.org/html/2608.11655v1#S4.SS1)。虽然画在 $s_k$ 上的线段本身只覆盖 $s_{k-1}\ldots s_k$，但 $M(\ell)$ 用到 $\mathbf p^i_{\ell+1}$，而全视频峰值排序、anchors 与 NMS 也在看到完整 $M$ 后执行。因此原论文是 offline 视频理解；若把它移至“预测第 $t$ 帧球中心”，不能仅因一段 polyline 终止于 $t$ 就宣称因果，必须另定义选择、tracker 更新和可用输入是否截至 $t$。

源码确实提供 `online` CoTracker predictor，却把它和 paper runner 的 `segment_track` 设为互斥；`segment_track` 在已知 selected indices 后，可对每两个 selected frames 独立重跟一次。adaptive path 先跑一次完整 tracking 得 $M$ 和 selected indices，再在 segment mode 再跑一次以画区间轨迹。[`cotracker_runner.py:L50-L90`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/data/cotracker_runner.py#L50-L90)、[`pipeline.py:L200-L239`](https://github.com/SunVictor23/MaP/blob/e150b0433613ec789f48b3d7d67f9c1ef535e57f/map_kit/core/pipeline.py#L200-L239)。这是公开代码的计算/时间语义，不是 v1 已报告的 online benchmark；它反而说明画 prompt 前已有 tracker 与选帧开销，不能把轨迹线说成免费输入。

表 3 报的是**每视频 preprocessing**：CLEVRER（约 5.12s）783ms、SSv2（约 4.21s）609ms、TempCompass（约 10.88s）1920ms；平均 GPU memory 0.35GB、peak 3.41GB。文字将低内存归因于 98MB tracker 和 window-based streaming tracking，故表的 MaP 路径包含其 frozen tracker 的 preprocessing；但表没有按 decode、full-frame-rate I/O、tracker、相机拟合、绘制和 MLLM 分别计时，也没有计入 Qwen/GPT-5.5 推理、远程传输/排队、权重加载或训练（MaP 无训练）。[§4.6/Table 3](https://arxiv.org/html/2608.11655v1#S4.SS6)。本次已查 PDF 全文，未见 GPU 型号、CPU、驱动、batch、分辨率或计时协议，故这些设备条件**未知**，不能推算本机效率或直接改写为本项目逐帧球定位的端到端在线延迟。

## 已有量化证据，以及它没有测什么

作者在 CLEVRER（MVBench 的 object existence、moving direction/count/attribute、counterfactual inference）和对象名称被抽象为 “something” 的 SSv2 四选一动作理解上报 accuracy；另把 TempCompass 用作“任务不 specifically target object motion”的 broader video-reasoning transfer check，检验其 motion-guided sampling 是否伤害作者所称的 non-motion-oriented tasks。[MaP §4.1、§4.3](https://arxiv.org/html/2608.11655v1#S4)。这只是 MaP 作者对该实验用途的界定：原始 [TempCompass](https://aclanthology.org/2024.findings-acl.517/) 本身是覆盖 action、speed、direction、attribute change、event order 等 temporal aspects 的 Video LLM temporal-perception benchmark，不能把整个基准定义为“非 motion”。GPT-5.5 相对 uniform 的平均分提升分别为 4.2 和 8.9 个百分点；在 CLEVRER 的逐步消融中，trajectory marking 单独加 1.3/1.8 个百分点，motion-guided sampling 再加 1.0/1.0 个百分点。[表 1、§4.2--4.5](https://arxiv.org/html/2608.11655v1#S4)。这支持“冻结 MLLM 能利用已画出的跨帧运动提示做问答”，不支持 MLLM 或 MaP 正确读出所有轨迹。

作者没有报告任何小对象尺寸分桶、球/点 bbox、自动点定位 AP、中心距离、point-track PCK/AJ、track visibility precision/recall、光流 EPE、遮挡退出重捕、query coverage，或大位移对应准确率。CLEVRER 的球体是合成推理场景元素，其问答分数仍不是 GT 中心点评估；SSv2 也被构造成动作选项分类。图中的轨迹覆盖只是定性展示，不能作为 tiny object 或自动点定位的量化证据。

## 对本项目的研究判断

1. MaP 已覆盖“把**先验已得的**跨帧轨迹以可见标记附到稀疏帧，帮助冻结视觉语言模型理解运动”这一主张。本项目若试图用轨迹图、timestamp 或 motion-energy 选帧辅助高层分析，必须把它定位为 MaP 的受限近邻，而非自动球定位机制。
2. 高速微球项目的关键反事实不同：球中心只有标注、其视觉支撑可能低于 $10\times10$ 规则 query 间距、又会被 blur/遮挡和大位移影响；要使 MaP 形式成立，需先证明一个 query/track 在目标邻域的覆盖、可见性与位置误差。这是待检验条件，当前不能从“dense tracking”或图示推出成功，也不能推断必然失败。
3. MaP 的 offline 全视频选帧不回答当前协议中的逐帧/末帧可用时间问题；若某个球点任务允许固定前瞻，仍须把 track window、target frame、提示绘制区间与等待延迟写清。若目标为在线，第 $t$ 帧以后用于 $M$、全局选帧或 point tracker 的数据都不可静默使用。
4. 最有实证价值的不是立即堆叠 MLLM 或画线模块，而是在已有球标注下单独测量：规则 query 对球邻域的 coverage、冻结 tracker 的球邻域轨迹误差/visibility、motion 筛选是否把含球时间保留，以及渲染后是否遮挡微球证据。任何一个失败都改变“轨迹提示能帮助定位”的解释；问答 accuracy 不能替代这些测量。

本次结论基于单篇 arXiv v1 和其后一天的作者源码。作者所称 supplementary 在本次入口不可取得，且没有独立的球点量化，因此未知项保持未知；本笔记不将更晚代码参数、稀疏 grid，或 MLLM motion-QA 改称球 correspondence/检测证据。
