# What Moves?：region-query motion latent 保留全景参考，但不做自动球发现

日期：2026-09-11。范围：只补读 Frank Fundel 等的 *What Moves? Localized Motion Representations for Compositional Scene Control*；不复现、不下载权重/数据、不新增模型或训练任务。本笔记补充[现代 motion / backbone 文献核验](modern_motion_evidence.md)的 What Moves? 条目，重点核对 region prompt、时间语义和 global scene/camera reference 的实际边界。

## 可直接使用的判断

What Moves? 的直接先例是：**给定一个已知 region，使用其 mask 形成 content query，同时让 motion encoder 看完整视频，从而得到每 region、每时间步的 compact motion latent。** 它证明在其 actor/action 与生成控制任务中，“裁剪局部画面”或“先做全局特征、再事后遮罩”不等同于带 full-scene context 的 region-conditioned encoding；这足以封堵“局部 motion 必须裁剪，或全局/局部 context 并用”这一类泛化新颖性主张。

它没有自动从全图发现 region、输出球中心、候选分数、每像素 flow/correspondence field 或显式 camera-motion component。region-localized 模式需要外部 mask；没有 mask 时当前发布 encoder 可以输出**全局** motion latent，但这不是 localized representation。故它的 mask 假设绕开“在哪里找球”的 candidate/assignment 轴，却**不证明**球在全图 feature 中不可读，也不要求本项目新增人工 mask 标注：mask 可以来自已有实例标注、交互提示、外部 segmentation，或未来真正的候选机制；这些来源及其误差必须另行定义和测量。

“保留相机/场景参考”在本文应窄读为架构与任务证据：完整视频 token 与 region content query 在同一个 transformer 中交互，作者在 crop/global-mask 消融和多 actor action classification 中看到下降。论文并未估计、输出、监督或量化分离 camera trajectory、homography、object displacement 与 background flow。它支持“裁掉上下文会让局部 motion 含义歧义”的问题设定，不支持“已完成 object/camera 解耦”或“已解决体育 pan/zoom”。

## 版本、状态、来源与已读范围

* 题名为 *What Moves? Localized Motion Representations for Compositional Scene Control*；作者为 **Frank Fundel、Malek Ben Alaya、Thomas Ressler-Antal、Stefan Andreas Baumann、Björn Ommer**（前三位 equal contribution）。[arXiv:2609.04383v1](https://arxiv.org/abs/2609.04383)于 2026-09-03 提交，只有 v1。
* 正式状态已能从 [ECCV 2026 官方 Videos 列表](https://eccv.ecva.net/Conferences/2026/Videos)核验其题名，且 arXiv 的 journal reference 标为 ECCV 2026。ECCV [Accepted Papers](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)页仍称其列表等待出版方检查；本轮未取得 Springer 正式 proceedings 章节。因此本文写作 **ECCV 2026 已出现于官方会议项目，Springer 定稿/页码未取得**，不把 arXiv v1 当作已核验的最终排版版。
* 方法与实验阅读范围是 [arXiv HTML v1，§3--4、Appendix A--B](https://arxiv.org/html/2609.04383v1)。作者入口是 [CompVis/WhatMoves](https://github.com/CompVis/WhatMoves)。源码只读固定提交 [`CompVis/WhatMoves@5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7`](https://github.com/CompVis/WhatMoves/tree/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7)（2026-09-07；无 tag/release）。该仓库是 inference/app 发布面，未含论文 encoder/decoder 预训练脚本或训练数据管线；代码事实与论文发表实验分开记录。
* 未取得：训练阶段四个 masks/video 的生成器与失败率、内部 16M clips 的可访问组成、TAPNext 轨迹质量、正式 proceedings 定稿、每帧/每窗口 GPU latency、FLOPs、显存峰值及任何 tiny-ball 原始评测。以下不会猜补。

## mask 从哪里来，何时是必要输入

论文定义 `V={x_t}_{t=1}^T`，并在第一帧给定 `K` 个 region masks `S={s_k}`；目标是每个 region 的 persistent motion token `m_{k,t}`。content encoder 把一帧和 `s_k` 编为实体的静态 content token `c_k`，它是 motion encoder 的 entity query，也是 tracks decoder 的 appearance attachment。motion encoder 本身读取完整 `V` 与所有 `c_k`；它**不是**把视频先裁到 `s_k`，也不是对一个已完成的全局 latent 做 mask pooling。

训练/评估所使用的 mask 并不总是同一个来源：

| 场景 | 已发表/发布材料所说的 mask 来源 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 方法与预训练 | §3 假设输入是第一帧 region mask；Appendix A 表 A.1 只给 `4 masks/video`，没有说明 OpenVid-1M/内部数据怎样产生 masks。 | localized encoder 的训练依赖 region supervision/prompt。 | 不能说作者要求人工逐帧手标，也不能评价其上游 mask recall/漏球。 |
| localized action classification | A2D 的 actor segmentation mask 与 actor action label；每个 actor--action instance 用对应 mask 取 embedding。 | 这是**给定 actor region**后的 action retrieval/classification。 | 不是自动 actor/球检测，亦没有把错误 mask、未知候选或中心误差纳入指标。 |
| motion-transfer benchmark | MTBench source subject mask 用 SAM2 提取；生成时还需指定 target region。 | published generation tests不是纯手工 mask oracle。 | SAM2 prompt/初始化/失败的细节与该 benchmark 的 mask quality 未报告；不能把它说成无提示发现。 |
| 当前官方 encoder | `encode_motion(video, masks)`可收 static `[B,K,H,W]` 或 temporal `[B,T,K,H,W]` mask；static mask默认对应第 0 帧，temporal mask 在每 window 取第一个可用 mask，不可用时复用已有 content。 | region prompt 可从一帧而非逐帧手标开始，也支持后续更新。 | 仍需要每个 region 在首 window 至少有一个有效外部 mask；没有内部 detector 自动创建它。 |
| 当前官方 encoder 无 mask | `encode_motion(video)`返回 global motion stream。 | 同一 encoder 可作 global representation。 | 它放弃“哪个实体”的地址，不能把 global output 作为自动 region discovery。 |

源码的输入契约直接支持这一区分：[`what_moves/model.py:L551-L595`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/model.py#L551-L595)规定 static/temporal/none 三种输入和 first-frame 默认；[`L625-L739`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/model.py#L625-L739)显示 temporal content 的更新/复用与无 mask 的 global query。文档说“user-specified region”是交互应用语义，**不应**误译为“本项目必需新增人工标注”：关键是 inference 要已有某种 region address；该地址来自何处、是否真实可得，是本项目另一个可失败问题。

## 真实输入、输出与空间地址

论文的三个训练组件为：

\[
c_k=C_\phi(x_i,s_k),\qquad
\{m_{k,t}\}=E_\theta(V,\{c_k\}),
\]

再以当前 frame、当前 query point、content token 和 motion token 解码未来轨迹：

\[
D_\psi(x_t,p_{k,t},c_k,m_{k,t})\rightarrow p_{k,t+1:t+L},\qquad
\mathcal L=\|D_\psi(\cdot)-p_{k,t+1:t+L}\|_2^2.
\]

训练 points 仅在 `S_k` 内稀疏采样。论文将 target 称作 ground-truth trajectories，但它们由 **TAPNext** 获得；就来源而言这是外部点跟踪器产生的 teacher/pseudo-track supervision，文中未报告其误差或人工真值核验。loss 只在有效 future point positions 上计算。Appendix 的默认是 8 frames、256 tracks/decoding step、prediction horizon `p=3`，并将 decoder 输出表述为相对 source point 的 displacement。注意：这些 trajectory 是**训练 decoder 的监督与输出**，不是发布 motion encoder 在本项目任务中的直接输出。

发布 encoder 的实值输出为：有 region mask 时 `[B,T-3,K,384]` motion latent，无 mask 时 `[B,T-3,384]` global latent；content output 是 `[B,K,512]`。空间顺序要精确区分两次 resize：外部任意尺寸 video 先在 WhatMoves model 里双线性 resize 到 `256×256`；随后每帧进入 DINOv2 wrapper，先裁为正方形、再 resize 到其固定 `224×224` 输入，最后由 `14×14` patch 得到 `16×16` grid。也就是说，**不是**直接在 `256` 上作 14-patch 得到 16 格；固定源码的 `256→224→16×16` 才是实际发布路径（[`what_moves/model.py:L302-L319`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/model.py#L302-L319)、[`L604-L624`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/model.py#L604-L624)、[`what_moves/dinov2.py:L235-L273`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/dinov2.py#L235-L273)）。该发布接口输出的是latent，未输出每pixel `u→v`向量场、query heatmap、per-region proposal score或center readout；这由接口与读出定义，而非单由patch尺寸推出。

当前仓库明确说 release checkpoint 为 compact encoder-only，decoder parameters 被忽略；`from_pretrained` 也把模型设为 inference/eval。不能以论文训练 decoder 的 `p_{k,t+1:t+L}` 公式声称当前 released encoder 会输出球 track。反过来，**没有显式 candidate/center 输出只说明该信息没有被这个接口读出或评测，不等于原始 RGB / full-scene feature 中该信息已不可读。**

## “local query + global reference”实际怎样实现

论文主张 local motion 对 camera motion、scene layout、遮挡和 interaction 需要全局参考。它的实际机制不是显式 background subtraction：content encoder 把 mask 投影/聚合成每实体 query；motion encoder 把所有 frame 的 spatial tokens 与每时间、每实体的 content query tokens 送入一个 12-layer spatiotemporal transformer。因而 masked query 可 attend full-frame/full-time token。当前 fixed code 中，所有 `t×h×w` frame tokens 及 query tokens concat，再以 3D positional encoding 作 self-attention；attention mask 只屏蔽无效 region query，不按时间或背景/前景分开（[`what_moves/model.py:L121-L219`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/model.py#L121-L219)）。

这建立两条**已测**的较窄证据：

* 在 A2D localized action classification，full-video + actor mask 的 Ours 为 `29.9` Accuracy / `42.0` F1；把输入裁为 actor crop 的 `Ours_rgb` 为 `20.0/31.1`，把 query 变为 global mask 的 `Ours_global` 为 `23.4/35.1`。这是 authors 的 actor action 任务内证据，说明二者缺一在该协议下降。
* motion transfer 的 source/target subjects 与其 mask 已给定；作者报告自己的 semantic localized conditioning 使 multi-source motion-selectivity 为 `0.4114`，并在 qualitative cases 讨论 camera following。它支持“控制结果的 leakage 降低”，不是相机轨迹/球位移的估计准确度。

以下仍**没有**被论文测到：camera parameter、homography、egomotion、背景 flow、object-minus-camera displacement、camera-motion bucket 的误差或局部位置精度。作者在 Appendix C 中将“confuses subject motion for camera motion”作为生成案例的失败描述，不等于存在可检验的 camera/object 分解变量或监督。故可说“该 representation 可条件化地利用 global context”，不可说“已显式消除了 camera motion”或“object 与 camera 可辨识地分离”。

## 时间锚点：训练 future supervision 不自动等于因果预测

论文的 tracks decoder 在给定 `x_t,p_{k,t},c_k,m_{k,t}` 后，训练预测 `p_{k,t+1:t+L}`；Appendix 令 `p=3`。这是 future-trajectory **监督目标**。但 motion encoder 的定义是输入整个 `V`，论文未声明 causal mask、online cache 或每个 emitted token 的可用观测终点，故不能把该 decoder 公式独立读成“在线只凭 `≤t` 预测未来”。

固定源码进一步给出发布 encoder 的实际时间边界：window length 8、emitted steps 5，并定义 `lookahead=window_length-emitted_steps=3`；每个 window 把八帧的所有 spatial tokens和五个时间 query 一起放进无因果遮罩的 self-attention。它返回按 window 起点收集的 `T-3` steps（[`what_moves/model.py:L302-L317`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/model.py#L302-L317)、[`L640-L724`](https://github.com/CompVis/WhatMoves/blob/5f89fea8dceb8463fb9c72a1ff478bf52ece0cb7/what_moves/model.py#L640-L724)）。`lookahead=3`是这个 window/output-count 的差，不是“每个前部 query 只看三帧未来”的 attention 上限：所有 query 都可 attend 同一八帧；按 query 的代码时间位置 `0…4` 读，前部 query 还可读取窗口的其余更晚帧。故当前 release 的原有 token 对齐应视为**有限离线 sliding-window encoding**，不能宣称无前瞻 online motion embedding。

论文 Appendix A.2 对 25-frame generation 写 `p=3` 时产生 21 embeddings，当前 fixed code 则按 `T-3` 产生 22；两者的 index/对齐说明不完全一致（论文的算术/时间索引未进一步解释）。本笔记不把 release 输出数组第 `j` 项等同于某一个精确原始帧 `t_j`，也不虚构 delay。

这不排除另一种**时间上合法但不同于发布 token 对齐**的读法：在当前 `t` 已收到完整窗口 `(x_{t-7},\ldots,x_t)` 后，只把该完整已见窗口的摘要/末帧 readout 用作 `t` 的 motion feature；没有 future observation，因而不必然需要 causal attention mask。代价是目标时刻、readout 规则和有效输入长度已改变，不能继承原论文/发布 release 的 action 或 generation 性能。若要让window内较早时刻也立即输出，则需限制该输出的全部输入路径只读取当时已见观测；causal mask或截断窗口是实现选择，不能只改一个mask就默认所有路径均因果。与本项目 `[t-2,t-1,t]→t` 的比较应先分别写清这两种时间锚点，再独立验证，不能直接继承 WhatMoves 的数字。

## 训练、下游证据、tiny-object 空白与成本

发表设置先以 DINOv2-B 初始化 frame embedder，另有 86M 的 3D motion encoder 和 86M masked tracks decoder；二者 joint train 600k steps。Appendix A 写训练视频约 17M clips（OpenVid-1M + 约16M internal YouTube clips），8 frames/clip、`256×256`、batch 64；正文 batch size 写 128，二者存在报告口径不一致，应分别保留，不能自行选一个当最终事实。motion-transfer generator 又是 CogVideoX-5B LoRA（训练 `25×480×720`、8 fps）；仓库后来提供的是 Wan2.2 I2V-A14B integration，README 给一条 25-frame `480×704` generation 约 67.6 GiB peak on A100 的应用数字。这个生成成本不等于 encoder-only 成本，更不等于定位速度。

论文没有 encoder FLOPs、吞吐、hardware-hours、decode、mask acquisition、TAPNext preprocessing、8-frame等待延迟或逐帧 latency。它也没有以下量化任务/分桶：tiny/dense point detection、few-pixel support、ball center error、frame-pair large displacement recall/rank、blur、球员/场线 false positive、camera pan/zoom，或在上述 `256→224→16×16` 路径上的局部probe。因此：

* A2D actor-mask action Accuracy/F1、global action retrieval、motion-transfer leakage 是其领域内证据，不能转换成高速小球中心定位或自动发现证据。
* training decoder 所采样的 region-inside points 与 TAPNext trajectory target，不能混为真实球 center-center displacement，也未提供候选覆盖率。
* 8-frame、256→224两次resize和16×16 grid 使“一个数像素球能否在 region/full scene token 中保留”成为未测问题；全局 context 可能帮消歧，也可能被下采样/大背景淹没。两种结果都须测，不能从没有 explicit candidate axis 推出不可读。

## 对研究主张的限制

| 可以引用 | 不能据此写出 |
|---|---|
| What Moves? 已做 full-video、mask-query 的 localized motion representation，并显示 crop-only 与 global-query 在 A2D action task 较弱。 | “保留 global context 的 local motion”、“mask-conditioned motion token”或“局部/全局 motion 解耦”本身新颖。 |
| region prompt 是定位条件，能让 encoder 为指定 entity 产生独立 latent。 | 该方法已自动发现球/实体，或 GT/SAM2 mask 下的收益代表 detector 的真实端到端收益。 |
| tracks decoder 对 masked-region points 训练 future trajectory prediction。 | 发布 encoder 输出 dense correspondence/flow、球 center，或在线未来预测。 |
| full frame 可提供 camera/scene reference，且在作者的 actor task/生成分析中有帮助。 | 已显式估计/消去 camera motion，或在体育 pan/zoom 下有效。 |
| 发布 encoder另有无 mask global mode。 | 无 mask 即可自动获得 entity-addressable representation，或“没有 candidate 输出”证明图像证据不可用。 |

可复用的窄结论是：**Fundel et al.（ECCV 2026，arXiv v1）提供了 region-addressable motion representation 的直接近邻：外部 mask/content query 指定“谁”，完整视频 token 提供场景/相机参考，输出为 compact latent 而非 spatial correspondence。它使“已给 region 后怎样避免裁剪丢掉 context”成为已有问题；本项目若没有可靠 candidate/region address，仍有自动发现和中心定位两项独立困难。它既不要求新增人工 mask 标注，也不能作为 tiny-ball 可读性或实时因果性的正面证据。**
