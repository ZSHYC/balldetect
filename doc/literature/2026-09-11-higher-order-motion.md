# MOSS 多阶时空自相似：关系保留与点定位边界补读

日期：2026-09-11。范围：只补读 MOSS 的官方 arXiv 全文及其可见的作者公开入口，供审查“局部相关之外，多阶关系编码是否已有最近邻”这一问题。本文不是复现，也不把 MOSS 纳入本项目基线清单。

## 结论先行

**MOSS 是这个方向的直接多阶先例。** 它先在每个特征格查询处枚举局部时空 offset 的一阶 STSS，再把该 relation tensor 编码为特征向量，递归产生二阶、三阶的“similarity of similarity”。因此，“多阶 self-similarity / 由关系模式而非单个 cosine 建模 motion”不能作为本项目的新颖性。

但 MOSS 不等于高速微小球的跨位置对应器：最终 MOSS 输出是以查询 `(t,h,w)` 对齐的 feature map，`(l,u,v)` 不再作为可直接读取的候选 offset 轴；论文没有输出候选点、匹配概率、中心位置或 no-match。这只说明**没有显式候选地址输出**，不证明编码后的 feature 不能隐式保留或被另一个读出器恢复相关信息。它也没有以 tiny object、自动点定位、真实中心对应或对应准确率检验。故它不能证明高阶关系会修复本项目已测试局部cost拼接的精细定位退步，更不能把二/三阶表示解释成物理速度、加速度或可靠对应。

## 版本、来源与可得性

Manjin Kim、Heeseung Kwon、Karteek Alahari、Minsu Cho，**“Exploring High-Order Self-Similarity for Video Understanding”**，arXiv:2604.20760v1，提交于 **2026-04-22**，cs.CV。[arXiv 记录](https://arxiv.org/abs/2604.20760v1)给出题名、作者、v1 时间及仅有 v1 的 submission history；[官方 HTML 全文](https://arxiv.org/html/2604.20760v1)与[PDF](https://arxiv.org/pdf/2604.20760v1)是本次方法证据。到本次读取日，arXiv 页面未列 conference/journal reference，故按**预印本 v1**记录，不推定已发表。

摘要只说“source code and checkpoints will be publicly available”。论文 HTML 没有作者仓库链接；以完整题名和 `Multi-Order Self-Similarity` 检索 GitHub 只得到第三方 arXiv 索引，未得到可核对的作者仓库，因而没有固定 SHA 源码证据，也不从猜测的实现细节外推。

此前 [modern_motion_evidence.md](modern_motion_evidence.md) 的一行表述得到补充：不是“高阶特征”这一泛称，而是下列明确的局部 STSS 张量、递归和编码融合，构成最近邻。

## 原论文实际定义的对象

设 backbone feature map 为 \(F\in\mathbb{R}^{T\times H\times W\times C}\)。论文式 (2) 先构造

\[
S_{t,h,w,l,u,v}=\phi(F_{t,h,w},F_{t+l,h+u,w+v}),
\]

其中 \((t,h,w)\) 是查询格，\((l,u,v)\) 是局部时空窗口中的候选偏移，\(\phi\) 可以是 cosine similarity。[§3.1、式 (2)](https://arxiv.org/html/2604.20760v1#S3.SS1)明确 \(S\) 的形状为 \(T\times H\times W\times L\times U\times V\)，并令每个候选落在具体的 `(t+l,h+u,w+v)` 格。因此一阶 STSS **确实显式枚举跨位置候选**；当 `l != 0` 时它是跨帧 spatial cross-similarity。它仍只是 appearance-feature similarity，而非由真值验证的 correspondence。

论文以编码函数 \(g:\mathbb{R}^{T\times H\times W\times L\times U\times V}\to\mathbb{R}^{T\times H\times W\times C}\) 避免递归后维度爆炸，式 (3) 为

\[
S^{(1)}=f(F),\qquad S^{(n)}=f\!\circ\!g(S^{(n-1)})\;(n\ge2).
\]

| 阶数 | 论文中实际计算的关系 | 作者的表征性解释 | 不能推出的量 |
|---|---|---|---|
| 一阶 | 原特征向量间、按 `(l,u,v)` 枚举的相似度 | appearance-based similarity；可显现 motion-flow / displacement-map 形态 | 真实中心位移、flow GT、匹配已正确 |
| 二阶 | 经 \(g(S^{(1)})\) 重新变回向量后，再做同样的局部 STSS；即一阶 relation pattern 之间的相似度 | coherent motion / motion segment | 物理速度导数、同一实例 ID、可靠匹配 |
| 三阶 | 对 \(g(S^{(2)})\) 再做 STSS；即 motion-segment pattern 之间的相似度 | segment layout；作者的 toy/可视化也观察到 motion boundary、(dis)occlusion 附近响应 | 物理加速度、跨位置轨迹、可靠性概率 |

上述二、三阶的解释来自[§3.2、式 (3) 与 Fig. 3](https://arxiv.org/html/2604.20760v1#S3.SS2)：是视频理解表征及可视化解释，不是基于运动学量或点对应标注的证明。论文还报告动作识别中四阶及以上没有显著收益；这不能改写成“三阶在小球任务必需”或“阶数越高越好”。

## relation tensor 怎样编码，以及候选轴是否显式输出

每一阶使用独立的 \(g^{(n)}\)。[§3.3、式 (4)–(5)](https://arxiv.org/html/2604.20760v1#S3.SS3)的实际顺序是：

1. 对每个 temporal offset 的 `(U,V)` 相似度图 flatten，并以 FC 映为 `D`，得到 \(T\times H\times W\times L\times D\)；
2. 对每个 offset 的 `(H,W)` 用若干 `Conv2d–BatchNorm–GeLU` block 精炼，保持 `D`；
3. 拼接 `L` 个结果，再以 FC 融合时间 offset，得到 \(M^{(n)}\in\mathbb{R}^{T\times H\times W\times C}\)；
4. 以 `FC(F) + Σ FC(M^(n))` 形成最终 MOSS feature map。

故有两层必须分开的事实：原始 \(S\) 既保留 query 格也保留候选 offset；**最终 MOSS feature map 只保持 query 格 `(t,h,w)`，不再显式给出 `(l,u,v)` 轴、候选点坐标或 match score。** 若 backbone 的格点能映回输入，输出仍可作 dense feature readout，但原论文没有把它解码为像素/点地址，也没有说明该输出是 correspondence field。

“不显式输出”不可偷换成“地址信息必然丢失”：默认动作配置 `(U,V)=(9,9)`，故每个 temporal offset 的原始 spatial relation 有 81 个量；[Table 7](https://arxiv.org/html/2604.20760v1#Pt0.A1.T7)列出的动作配置 `D` 为 64 或 96，随后还有可学习 FC、空间卷积和时间融合。`D=64` 小于 81，但 `D=96` 大于 81；仅凭这些维度和论文示意不能判定哪类候选信息可逆、不可逆或可由新头读出。严格说，MOSS 已证明的是这种**不暴露候选轴的编码**可服务语义任务；将其接入中心热图头后能否读出正确候选，须由新设计和实测建立，不能借 MOSS 自动成立。

## 输入时间、局部支持与成本条件

式 (2) 将每个 offset 定义在以零为中心的 \([ -\lfloor L/2\rfloor,\lfloor L/2\rfloor]\times[-\lfloor U/2\rfloor,\lfloor U/2\rfloor]\times[-\lfloor V/2\rfloor,\lfloor V/2\rfloor]\) 窗口。[§3.1](https://arxiv.org/html/2604.20760v1#S3.SS1)因此通常的奇数 `L` 同时触及过去与未来帧；例如动作识别默认 `(L,U,V)=(5,9,9)`，时间偏移为 `-2..2`。这是一种**双向、离线**视频窗口的定义，并非因果在线 guarantee。全文未说明序列边界如何 padding/mask，也没有给出面向目标帧的等待延迟协议。

动作识别把单个 MOSS 放在 Side4Video 的 spatial encoder 与 temporal encoder 间，默认结合**一、二阶** STSS；使用 8 或 16 帧，Diving48 也有 32 帧设置。[§4.1](https://arxiv.org/html/2604.20760v1#S4.SS1)；Video MLLM 用 `(7,11,11)`、短边 resize 到 224、2 FPS，[§4.3](https://arxiv.org/html/2604.20760v1#S4.SS3)。这些是各自语义任务的窗口，不能当作球的适宜帧率、最大可搜索位移或实时开销。

成本上，原始 relation 的体积按 \(T\,H\,W\,L\,U\,V\) 增长，多阶前还会重复构造；MOSS 通过先以 FC 编码 `(U,V)` 再做卷积来控制显式张量的内存。[§3.3](https://arxiv.org/html/2604.20760v1#S3.SS3)。在**其** Something-Something V1、MOSS-S、8 帧、单 GPU batch 32 的 Table 3 测量中，默认一+二阶为 151.5 GFLOPs、5.6M trainable parameters、9.9 GB memory；无 MOSS baseline 为 148.4 GFLOPs、4.5M、8.0 GB。[Table 3](https://arxiv.org/html/2604.20760v1#S4.T3)。

附录 D.4 给出更宽的、仍限于 Something-Something V2 的对照范围：[Table 11](https://arxiv.org/html/2604.20760v1#Pt0.A4.T11)在 ViT-B/16 下列 MOSS-S 为 453 GFLOPs、6M trainable parameters、9.9 GB，MOSS-B 为 538、22M、21.6 GB；在 ViT-L/14 下列 MOSS-M 为 2120、24M、17.9 GB，MOSS-L 为 2500、82M、36.5 GB。memory 的 batch 分别是 B/16 的 32、L/14 的 16；表中没有端到端 wall-clock latency，也不是高分辨率球候选或大位移搜索的成本。这些数值不能外推为本项目仍是 marginal。

## 任务、监督与现有证据到哪里为止

论文正文首先评估 Something-Something V1/V2、Diving48、FineGym 和 Kinetics-400 的视频**动作类别**；随后在 FAVOR-Bench/MotionBench 做动作相关 VQA，及在 MoveSense/PongPredict 做机器人动作成功率。[§4](https://arxiv.org/html/2604.20760v1#S4)还在附录 D.1 报告 THUMOS-14 temporal action detection：768 个 224×224 输入帧、以不同 temporal IoU 的 segment mAP 评价；[D.1](https://arxiv.org/html/2604.20760v1#Pt0.A4.SS1)。附录 D.2 在 Kinetics-GEBD 与 TAPOS 上报告 generic event boundary detection，以相对时间距离阈值的 F1 评价事件时刻；[D.2](https://arxiv.org/html/2604.20760v1#Pt0.A4.SS2)。两者增加的是**时间段/时间边界**监督和评价，并不产生空间点标签或跨帧点对应。

这些 MOSS 学习均由下游类别、问答、动作、时间段或时间边界任务驱动；方法段没有对 \(S\) 提供 GT flow、点轨迹、球中心、对应标签或 no-match 标签的直接监督。

PongPredict 的确出现“virtual ping-pong ball”，但评估是机器人选择球将撞到的三色墙区对应 cup 的成功率，不是找出球的像素中心、跨帧匹配球或报告轨迹误差。[任务定义](https://arxiv.org/html/2604.20760v1#S4.SS4)和[Table 5](https://arxiv.org/html/2604.20760v1#S4.T5)都只支持这个较弱的结论。全文的 STSS 图、feature norm 图以及动作/问答/成功率均没有 tiny-object 分桶、自动点定位指标、候选 coverage/rank、中心误差、flow EPE 或 correspondence accuracy。因此本次可明确说“**未读到这些证据**”，不能说“已证明多阶关系解决高速球”。

## 对本项目研究判断的改变

1. 若后续研究多阶局部relation，related work必须以MOSS为最近邻，贡献不能写成多阶self-similarity或similarity-of-similarity本身。MOSS的一阶也可用scalar cosine；本地cost方案也包含多个候选，二者区别不是“一个标量”与“一个关系张量”的简单对立。
2. 本项目已测试局部cost拼接方案的精细定位退步，仍是有价值的[本地失败事实](../experiments/2026-09-10-local-cost.md)，尚未隔离成cosine算子本身的损害；MOSS提供的是另一种关系编码先例，没有证明它能改善球中心。
3. 可被实证检验的关键问题是：在相同的球中心监督、时窗、feature budget 与定位 head 下，显式保留候选 `(l,u,v)` 的 relation，或从编码 feature 读出候选，是否提高真实中心候选 coverage/rank 和当前帧中心误差；若只提升全局动作式的语义分数或热图平滑度，不能归因于正确对应。此处是诊断条件，不是立即堆叠 MOSS 模块的建议。
4. 不能声明“二阶=速度、三阶=加速度”“高阶=可靠 motion/no-match”“MOSS 已验证 tiny ball correspondence”，也不能把它在双向视频理解窗口的结果包装为在线自动球发现。仍可争取的差异只会是有中心标签约束下、几像素/模糊/大相对位移的候选覆盖、排序与逐帧定位证据；这些尚未运行。

本次补读仅覆盖官方 arXiv v1 与作者声称将发布代码的入口，不是完整 STSS/动作识别综述。作者代码一旦公开，应以固定提交复核上述压缩、边界与时间处理；在此之前不将论文示意图补成源码事实。
