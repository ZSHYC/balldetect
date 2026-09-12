# 轻量时序路由：TSM、GSM、GSF 对本地 late fusion 的真实约束

日期：2026-09-12。问题：廉价 temporal routing 实际改变了什么，是否构成高速微小球的空间 correspondence，以及它和当前三帧特征拼接 probe 有何可证伪区别。

阅读范围：三篇一手全文的方法、关键消融与作者 PyTorch 实现；未复现、未测本项目速度或球定位结果。这里的“在线”只指论文是否给出可执行的在线协议，不从双向离线动作识别结果外推。

## 结论先行

**事实。** TSM、GSM、GSF 都在相同空间坐标 `(x,y)` 沿时间轴搬运或融合通道；三者均不显式提出位移候选、查询另一帧的不同坐标，或输出 correspondence score。因此其算子本身不是空间匹配机制；逐层路由是否仍能改善大位移球定位，须由任务实验决定。

**事实。** 它们与当前“逐帧独立、共享可训练 prefix 后一次性拼接三帧，再由小 head 融合”的主要区别，是把时间交互放进 backbone 的多个深度，在后续空间卷积/非线性之前持续改变特征形成。GSM/GSF 还以输入条件的门控决定多少特征移动或保留。

**推论。** TSM 可作为廉价、非 correspondence 的时序路由对照；GSM/GSF 只在研究问题明确变为“内容条件的逐层时间路由是否补足 late fusion”时才值得适配。GSM/GSF 的动作分类增益不能支持其对本项目有效；TSM 虽另有视频检测实验，亦未测几像素球中心定位、漏检或球类模糊条件。

**当前决定。** 不为固定 signed difference 或固定槽位置换重复开实验：在本地现有 GN 之后，固定可逆的三槽线性基变换可由第一个可学习线性映射重参数化。这个判断不适用于 GSM/GSF 的输入条件门控，也不等于已有 head 已涵盖逐层 temporal routing。

## 版本、来源与阅读边界

| 方法 | 已核对版本 / 正式发表 | 一手来源与阅读范围 |
|---|---|---|
| TSM | *TSM: Temporal Shift Module for Efficient Video Understanding*，arXiv:1811.08383 v3（2019-08-22）；ICCV 2019 | [CVF 正式论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf)：Sec. 3、4.4、4.5 与效率表；[arXiv v3](https://arxiv.org/abs/1811.08383)；作者代码。 |
| GSM | *Gate-Shift Networks for Video Action Recognition*，arXiv:1912.00381 v2（2020-03-21）；CVPR 2020 | [CVF 正式论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Sudhakaran_Gate-Shift_Networks_for_Video_Action_Recognition_CVPR_2020_paper.html)：Sec. 3、4.3；[arXiv v2](https://arxiv.org/abs/1912.00381)；作者代码。 |
| GSF | *Gate-Shift-Fuse for Video Action Recognition*，arXiv:2203.08897 v3（2023-04-15）；TPAMI 45(9), 2023，DOI [10.1109/TPAMI.2023.3268134](https://doi.org/10.1109/TPAMI.2023.3268134) | [arXiv v3 正文](https://arxiv.org/pdf/2203.08897)：Sec. III、IV；[作者仓库](https://github.com/swathikirans/GSF)。出版元正文未开放获取时，机制以开放预印本和作者实现为准。 |

## 先分清：temporal shift 不是 spatial correspondence

设特征为 `X[t,c,y,x]`。TSM 的操作是把某些通道换成 `X[t-1,c,y,x]` 或 `X[t+1,c,y,x]`；其索引中的 `(y,x)` 不变。GSM/GSF 先按输入生成门 `g[t,y,x]`，再搬运 `g·X` 的一部分并把残差留在原时刻；它们同样不生成 `X[t-Δ,c,y+δy,x+δx]` 的多个 `(δx,δy)` 候选。

这回答“某处是否存在变化、哪些通道借到邻帧特征”，不直接回答“当前球在历史帧的哪个远处位置有视觉对应”。若球跨过许多 feature cell，同坐标的历史球证据可能已不在读入位置；后层感受野或语义上下文可能仍帮助分类/定位，但这不是被验证的几何匹配。

## TSM：固定、同坐标的通道时间搬运

**事实（机制）。** TSM 将一小部分通道向过去/未来各平移一帧，其余保持；论文默认总共移动四分之一通道。该 shift 本身没有参数和 MAC，随后 2D convolution 读取已混入相邻时间的通道。作者[实现的 shift](https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py)明确以时间维切片赋值，空间下标不变。

**事实（放置）。** 论文和[作者实现](https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py)把 shift 包在 residual block 的 `conv1` 前，而非只在末层分类头拼帧；下一空间卷积、BN/ReLU、残差与下一层 shift 都作用于已时间混合的表示。多层堆叠扩大 temporal receptive field，却不直接产生显式位移搜索。

**事实（延迟）。** “zero FLOPs / zero parameters”不等于零成本。论文报告 naive all-channel shift 造成 CPU +13.7%、GPU +12.4% 延迟，partial shift 降低该搬运量；这只是其 ResNet/action 设置，不可移作本项目时延。[原文 Sec. 3.3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf)。

**事实（在线）。** TSM 给出单向 online 变体：每个 residual block 缓存需从过去送入的通道；作者报告 ResNet-50 约 0.9 MB 缓存。该在线定义仍须在 clip/rally 边界清空缓存，且历史可经多层累计，不能等同于固定三帧窗口。

## GSM：输入条件 gate，但仍为同坐标 shift

**事实（机制）。** GSM 将通道分为两组，在归一化/ReLU 后用一个分组 `3×3×3` 3D convolution 加 `tanh` 产生每组一个时空 gate；gate 乘以输入，剩余 `X-gX` 留在本时刻，门控部分分别前/后移一帧。作者[实现](https://github.com/swathikirans/GSM/blob/master/gsm.py)对应 `conv3D → tanh → gate*x → residual → zero-padded shift`。

**事实（不是固定差分）。** gate 是由当前三维局部输入算出的空间变化张量，并经 `tanh`，不是所有样本共用的固定矩阵。门为零时退化为不搬运，门为一时近似全搬运的 shift；该输入条件 gate 不受“固定可逆基变换 + 首层重参数化”的等价证明覆盖，却不产生远距空间候选或可解释的 match reliability。

**事实（消融边界）。** GSM 论文在 BN-Inception / 动作分类中报告从 1 到 10 个 GSM 的收益与小量参数/FLOPs 增加；该设置、8 帧双向输入与 video label 不能推导球中心误差。论文未给出可直接复用的单向在线 GSM 协议。

**未知（流式在线）。** GSM 的 `3×3×3` gate 在原定义中用对称时间 padding，论文未给出逐帧持续 state 的协议。若目标是每一时刻都输出该时刻并复用前层 state，则须规定单向 gate/shift、初始化缺历史、逐层缓存与边界 reset；门函数和训练分布都会变化，不能把这种改写称作论文已经验证的 online GSM。固定历史窗口的末帧定位则是另一回事，见下文。

## GSF：门控搬运后再按通道、时间自适应融合

**事实（机制）。** GSF 的 grouped `3×3×3` gate 与 GSM 相似；它将 gate 部分时间移动后，不直接同权加回残差，而是从 shifted/residual 的空间 pooled 描述，经 `2D conv + sigmoid` 得到每通道、每时间的融合权重。作者[实现](https://github.com/swathikirans/GSF/blob/main/gsf.py)包含 gate、shift、融合权重和其加权混合。

**事实（对照含义）。** GSF 的固定 `gate=1 / sum` 消融对应 TSM，`gate=-1 / sum` 与 TDN 型差分关联；但其完整 GSF 还含输入条件 gate 和 fusion。因此不能用“完整拼接可线性重参数化固定差分”否定 GSF，也不能把该动作分类消融当作球定位上的差分优越性。

**事实（成本）。** GSF 不是零计算：有 `3×3×3` 分组门卷积、空间池化、两个 `2D conv` 和 sigmoid，另有时序 activation shift。论文在特定 backbone 的 GFLOPs/参数表称额外量小，但没有本项目输入、硬件、解码边界下的端到端延迟证据。

**未知（流式在线）。** 本次全文和作者公开实现没有找到因果缓存或 online 推理协议；实现也默认 `num_segments=8`、双向左右 shift。若要在每帧持续复用 state，则须规定单向 gate/shift、缓存各插入层所需激活并在 clip 边界重置，之后重新训练与报告延迟。这是新适配，不是原论文结果；它不等同于固定历史窗口只预测末帧的因果语义。

## 与本地三帧 late fusion 的严格比较

**事实（本地实现，2026-09-12 已读）。** 当前完整 BlurBall/Tennis build 以 `train_backbone=True` 共同微调 stages 0–1；`prefix.eval()` 只固定训练模式随机性，并不表示 `requires_grad=False`。`src/ballmotion/backbone_probe.py` 对每帧独立运行同一个共享 prefix，随后按时间拼成 576 channel。`src/ballmotion/probe.py` 以 `GroupNorm(num_frames=3, affine=False)` 分别归一化三个完整帧组，再以 `1×1 Conv(576→32) → GELU → 3×3 Conv → pixel_shuffle` 产生位置输出；absence head 使用归一化 appearance 的空间均值。旧的冻结特征 probe 应作为另一种实验条件单独记录。

**事实（可重参数化条件）。** 若在这三个 slot 各自完成 GroupNorm **之后**施加固定可逆线性变换 `z'=A z`，并允许同时改写所有读取 `z` 的首个线性映射，则 location 的首层可取 `W'=WA^{-1}`，absence 的线性层同理。就表示能力而言，固定 signed difference 或槽位基变换不新增函数族；后面的 GELU 和 `3×3` 在该线性层之后，不能仅因“网络含非线性”否定这个等价。

**边界。** 该命题不涵盖在 GN **之前**跨时间混合后又重新计算各组均值/方差的情形；此时 `GN(Az)` 一般不等于 `A·GN(z)`。它也不涵盖输入条件 `A(z)` 的 GSM/GSF gates、将 shift 插进可训练 backbone 的中间层，或限制某些 head 参数不随变换重参数化的实验。

**推论。** 当前 late fusion 具备“末层三槽的可学习线性混合”，但不具备“多层 backbone 内、混合后再经空间非线性处理”的机制。要比较 TSM，问题应是这个逐层放置是否在相同 backbone/输入尺度/因果范围下改善球定位；不应把它写成“首次加入 motion”。

## 它们会改变的证据，和不会自动改变的证据

| 观察到的结果 | 可支持的最窄解释 | 仍不能推出 |
|---|---|---|
| TSM 优于当前 late fusion | 逐层同坐标 temporal transport 在该实现和数据上有用 | 它找到球的跨空间对应，或它优于所有更简单的时序先验。 |
| GSM/GSF 优于 TSM | 内容条件 gate/fuse 可能比固定路由有用 | gate 已测得“该球对应可靠”或已分离相机与球运动。 |
| 三者均无增益 | 此骨干、窗口和训练条件下该路由不足 | 帧间 motion 完全没有用，或 correspondence 路线无价值。 |
| 对高速或大位移分桶有增益 | 跨帧上下文与该困难条件相关 | 球的远距位移已被正确显式搜索；需另有位置/匹配证据。 |

**推论。** 这张区分表说明为何不把 action-recognition 论文的模块名直接接到定位 head 后当作论文动机。可接受的定位实验还须把候选召回、可见帧位置误差和 absence/误检分开，避免单一定位 accuracy 将时间平滑或类别先验误写成 ball evidence。

**未知。** 目前没有针对本地 DINOv3 特征的逐层插入诊断，因而不能判断 TSM 会在浅层保留还是在深层稀释 tiny-ball 证据；这应由未来一个有明确比较目的的实验决定，而非从 ResNet 动作分类消融猜测。

## 因果、时间语义与效率的最低报告要求

先固定预测语义，避免把两种“因果”混为一谈：

1. **固定历史窗口、只预测末帧。** 取 `W_t=[t-2,t-1,t]`，仅监督/输出 `t`。即使原有双向 shift 或 gate 使窗口中的 `t-2` slot 读取较晚的 `t-1/t` slot，计算仍未读取 `t` 之后的帧，故对末帧 `t` 是严格因果的。它是将原动作分类模块用于定位的新适配，原论文分数不可继承。
2. **流式逐帧输出、持久 state。** 每个 slot 都输出自身时刻，下一帧复用前一帧的层内 state 而不是重算完整窗口。此语义才要求不能从未来 slot 回流，并须定义单向 shift/gate、各层缓存、warm-up 与 clip/rally reset；TSM 给出了这种专门的单向实现，GSM/GSF 本次未找到。

固定窗口中的特征依赖整个 `W_t`；滑到 `W_{t+1}` 后不能把原动作分类模型的某个“独立逐帧输出”误作可直接复用缓存。除非另行验证等价的 state 递推，固定窗口应按窗口重算，并报告该计时边界。

| 方案 | 原论文时域 | 官方流式 online 证据 | 固定历史窗口末帧定位 | 流式 state 复用时的改写说明 |
|---|---|---|---|
| TSM | 离线双向 shift；也有单向在线版本 | 有，per-block cache | 可用 `W_t` 作末帧新定位适配；写清窗口、目标帧和整窗重算成本。 | 采用作者单向方案或等价实现时，报告每层 cache、边界 reset 和端到端 latency。 |
| GSM | 双向 8-frame action clip | 本次未找到 | 可用 `W_t` 作末帧新定位适配；内部双向交互不读取 `t` 后帧。 | 须规定单向 gate/shift、缓存、warm-up、重训和实际延迟。 |
| GSF | 双向 8/16-frame action clip | 本次未找到 | 同 GSM；不能继承原动作分类分数。 | 同 GSM，另说明 fusion 的历史可用性。 |

**事实。** GSM/GSF 的原评估是视频级动作分类。TSM 另报告了 ImageNet-VID 的在线视频目标检测：以单向 TSM 插入 R-FCN/ResNet-101 backbone，评估框级 mAP，并按目标速度分桶；其 fast 列相对 2D R-FCN 高 4.6 mAP。[原文 Sec. 5.5、Table 7](https://arxiv.org/html/1811.08383#S5.SS5) 。这证明 TSM 曾在常规视频检测上接受过在线测试，但没有逐帧几像素球中心、不可见状态或球类模糊标注，不能作为“球的 motion evidence 更可靠”的证据。

**推论。** 如果目标是流式逐帧在线，TSM 的单向协议是可查证的因果机制先例；如果目标是固定历史窗只预测末帧，应报告窗口与整窗重算，而不能把它描述为逐帧 state 缓存。若允许固定前瞻，则仍须把未来帧、输出延迟与连续 clip reset 写进公平协议。只在真正接入后计时，避免把 shift 的 MAC 为零误报为整体免费。

## 最小有区分力的后续判断

1. **先不实现 GSM/GSF。** 现有证据没有显示它们的动作分类门控会恢复几像素球的远距位置；若没有一个可区分问题（例如固定 late fusion 已控制 head 容量后，内容条件的逐层路由是否降低特定大位移分桶漏检），完整适配只会扩大变量。只有中心与球尺寸均有可信标签时，才可将该条件写为 `rho` 分桶。
2. **TSM 也不是必跑基线。** 只有当实验需要排除“简单逐层 temporal transport 已足够”这一竞争解释，才以明确的固定历史窗末帧或流式 state 版本、小模型和相同目标帧做一次机制对照；不可因三篇论文存在而跑全套历史网络。
3. **若结果无增益，结论应窄。** 可说“本设置下同坐标时间路由未改善定位”，不能说所有 temporal modeling 无效；若有增益，也仍须按可得且可信的尺寸、位移、模糊、背景竞争分桶，才能判断它是在补语义上下文、平滑还是在减少真漏球。只有尺寸定义和中心跨帧对应均可靠时才报告 `rho`。
4. **空间 correspondence 仍是独立问题。** TSM/GSM/GSF 限制“时序融合尚未研究”的叙述，却不能否定细位置与远距证据的其他路线；显式 correspondence 只是可检验路线之一，不是预先锁定的论文贡献。

## 证据充分性与未覆盖项

本次补齐了 `correspondence_evidence.md` 中 TSM/GSM/GSF 原先仅摘要的缺口，并以作者公开实现核对具体算子。没有复现，没有读取封存测试标签，没有比较当前 DINOv3 prefix 内插模块的实际工程难度，也没有针对 2026 年新论文重新进行题名穷尽检索；这些均不是本问题所需证据。

后续若选择任一方案，应重新核对目标 backbone 的具体 block/归一化结构和可插入位置；不能假设 ResNet/BN-Inception 的作者实现可无损搬到 ConvNeXt 或 ViT。
