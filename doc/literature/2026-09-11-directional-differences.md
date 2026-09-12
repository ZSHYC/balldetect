# COMET：方向敏感高阶差分是时间变化竞争解释，不是球位移对应证据

日期：2026-09-11。范围：只补读 COMET，回答其 Taylor 式差分、正反序优化和成本究竟建立了什么；不复现、不把视频 MLLM 改造成球定位方案。

2026-09-12 补充：[Taylor Videos/TDN 全文核对](2026-09-12-temporal-difference-representation.md)明确了原始 Taylor 算子与公开导出路径，并纠正本文末尾原先指向另一篇论文的 PMLR 链接。本次同时收窄下文“必要竞争解释”的适用条件；不重新安排已被完整特征线性混合涵盖的差分实验。

## 结论先行

COMET 的一手证据支持一个较窄、但应认真保留的竞争解释：**在不计算 flow 或跨空间候选匹配的条件下，带时间方向的像素变化和外观—时间融合可以改善视频语义的动作/时序推理。** 它不支持“已经得到球的跨帧位移”、不支持“高阶 Taylor 响应是物理速度/加速度”，也没有 tiny ball、逐帧中心、密集定位或在线延迟的实验。

因此，旧表格把“COMET-style 差分”写成“必须竞争的简单基线”需要收紧：

* 若候选机制的解释需要区分显式 correspondence 与普通时间变化，应选择能区分这两者的廉价时序对照。**是否需要另跑一个模型，取决于已有基线是否已经覆盖该函数与输入。** 在下述自由线性混合条件成立时，已有完整三帧特征已经涵盖固定 signed-difference 换基，不额外新增重复对照；只有前置非线性、受约束分支或交互层级等实际改变了待比较机制，才另行考虑。
* 这**不等于必须复刻完整 COMET**。其两套 ViT、四层 cross-attention、CoT video-QA 微调和 GRPO 都服务于视频语言推理，输入/监督/输出与球中心定位不同。也不等于每个实验都必须采用 Taylor 五阶响应或倒序损失：后者的原论文奖励构造依赖文字问答，不能从中心标签自动推出。
* 差分基线失败只说明该指定的变化读出未够用；成功也只说明时间变化可竞争，仍不能分辨同一球还是背景、球员、场线、相机摇摄所造成的变化。它应作为对应方法的对手，不能因不输出 displacement 就被排除。

与本项目已有的[三帧因果基线线性等价结论](2026-09-10-causal-baselines.md#本次采用与不做的归因)不能混用：在分别归一化的完整特征仍保留时，线性首层前直接追加 signed difference 可由权重吸收。COMET 不满足该前提——它在 RGB 灰度像素上递归构造一至五阶差分、作幂和时间聚合、另行 clipping/rescaling，再与灰度外观混合，并用独立 ViT-B 和非线性 cross-attention 融合。因此不能用“线性可吸收”否定 COMET 的函数类或归纳偏置；反过来也不能由 COMET 的 MLLM 消融证明一个线性差分 stack 有额外表达能力。

## 版本、来源和已读范围

* **Chenghua Zhu, Zhaolu Kang, Qifan Shi, Siyan Wu, Kehan Jiang, Lei Wei, Lianyu Hu, Guangyuan Dong, Mingbo Yang, Rui Lu, Guibo Luo**, *COMET: Contrastive Motion-Enhanced Temporal Reasoning for Video Multimodal Large Language Models*，[arXiv:2608.21030 v1](https://arxiv.org/abs/2608.21030)，提交日期 **2026-08-21**；[官方可读 HTML 全文](https://arxiv.org/html/2608.21030v1)。
* arXiv 的 comments 和论文页眉写有 “Accepted ACM MM 2026”/ACM MM 2026 会议信息；截至本次检索，未取得 ACM Digital Library 或官方 proceedings 的可核验条目。因此本笔记按 **arXiv v1 预印本**引用，不能把正式出版状态写成已核实。
* 已读：全文的 §3.2--3.5、§4、表 1--4、参考文献，以及 arXiv 的版本/会议信息。检索题名、arXiv 页和作者公开入口，**未取得作者代码仓库或固定 commit**；下列实现细节只以论文为证，未用猜测填补。

## 实际输入：不是 feature difference，也不是仅两帧差

论文对 RGB 帧 $I_t$ 先取灰度 $G_t$，在**像素域**递归构造有限差分：

$$
D_t^{(0)}=G_t,\qquad
D_t^{(k)}=D_{t+1}^{(k-1)}-D_t^{(k-1)}\quad(k\ge1).
$$

随后为三个输出通道 $c\in\{1,2,3\}$ 定义

$$
T_t^{(c)}=\frac1W\sum_{\delta=0}^{W-1}\sum_{b=0}^{2}
\frac{(G_{t+\delta}-G_t)^b}{b!}D_t^{(b+c)}.
$$

因 $b+c$ 的范围为 1--5，$[T_t^{(1)},T_t^{(2)},T_t^{(3)}]$ 会用到一至五阶递归差分；它不是通常意义的仅 $G_{t+1}-G_t$ 的三通道复制。式中保留差分的符号和幂，论文没有使用绝对值的表述；倒放时方向会翻转的论据来自图 4 的定性展示。作者再把三通道逐通道 clipping/rescaling 的 $\operatorname{Norm}$ 与当前灰度外观混合：

$$
\Delta_t=(1-\gamma)\operatorname{Norm}([T_t^{(1)},T_t^{(2)},T_t^{(3)}])+
\gamma G_t\mathbf1,\qquad\gamma=0.33.
$$

所以送入 temporal ViT-B 的 $\Delta_t$ 是“多阶变化 + 粗外观”，不是纯 motion map，也不是 backbone feature 的差。论文没有给出 $\operatorname{Norm}$ 的数值 clipping/rescaling 参数。§4.1报告两个训练阶段均以 **stride 6**采样帧、帧数随视频长度变化；没有独立交代评测采样的相同细则。可读正文还未给出式中的 $W$、原视频 FPS 或逐输出帧的时间锚点；真实时间间隔、延迟和边界处理均无法由文中推出。

从式(1)还可直接推得输入支持边界：完整五阶前向差分$D_t^{(5)}$依赖$G_t,\ldots,G_{t+5}$，至少需要六帧；当前三帧窗口不能原样计算完整一至五阶构造。降低阶数、重复/补帧或改用更长历史都会改变算子或可见输入，必须另列条件。由于正文未明确差分构造与stride采样的先后关系，不能进一步把这六帧直接换算成原视频的固定秒数或延迟。这是公式推论，不是COMET作者已做的三帧实验。

作者明确把该构造定位为不用外部 optical-flow estimator 的变化表征。它没有 flow/displacement label、位移回归、cost volume、跨位置 candidate axis 或输出的 correspondence 地址。有限差分只描述固定像素邻域的强度随采样时间的变化，不能被重命名为目标中心位移或物理导数。

## appearance/motion 怎样融合，和它没有怎样对齐

RGB 走冻结预训练 ViT-A；ViT-B 与 ViT-A 架构相同、从 ViT-A 权重初始化，处理 $\Delta_t$。Qwen3-VL 时在 DeepStack 层 $\{5,11,17,24\}$ 融合（InternVL2.5 只在主层融合）。每个空间位置 $s$ 上，RGB feature 作 query，Taylor branch 作 key/value，attention 沿**时间轴**计算：

$$
\mathbf A_s=\operatorname{softmax}(\mathbf Q_s\mathbf K_s^\top/\sqrt{d_k}+\mathbf B_{s,:}),
\qquad \mathbf z_s=\mathbf A_s\mathbf V_s,
$$

再以零初始化 gate 的残差写回 appearance feature：

$$
\hat{\mathbf F}_l=\mathbf F_l^A+\tanh(\alpha_l)\operatorname{RMSNorm}(\operatorname{ProjUp}(\mathbf z_l)).
$$

TAB 的 $B_{s,t}$ 是 Taylor key 的瞬时幅值、同位置时间方差和相对时间均值偏离三项的乘积。这建立了“显式 motion branch + 同一 token 地址跨时间 attention”，不是无 motion 方法；但 attention 没有搜索 $s'$ 来匹配移动对象，也不输出 $s\to s'$。对高速球，固定地址上的响应可能来自球经过、背景移动、球员、线条或相机运动；论文没有把这些来源分开测量。

## 训练、正反序条件与时间语义

Stage I（TPD）在 Video-R1 的约 120K CoT reasoning split 上最小化文本答案/推理链的自回归损失。此阶段 **ViT-A 冻结**；ViT-B、fusion 和 LLM 更新。Stage II（TC-GRPO）冻结整个视觉路径，**只更新 LLM**，使用约 40K 个被作者筛为 temporal-sensitive 的 RL 样本；全部训练报告为 8×A100 80 GB。

TC-GRPO 对同一个文字 prompt $(x,\mathcal V)$，从正序视频采 $N_f$ 个 response、从 $\mathcal V^{\rm rev}$ 采 $N_r$ 个 response，放进同一 GRPO group；论文的设定为 $N_f=6,N_r=1$，并明确说文字问题和reference answer保持固定，答案正确奖励为1、否则为0。这是[§3.5的明示定义](https://arxiv.org/html/2608.21030v1#S3.SS5)，不是一个未说明如何生成的倒序新答案目标；作者希望以正序占多数的group保留正序问答目标。

由此可作一个有限推论：若某个问题的正确答案随倒序改变，这个固定reference奖励不能被解释成同时监督正、倒序视频各自的正确答案。它也不是把正序球样本配上已重映射的倒序中心标签。本项目不能仅凭该RL收益，就假设球定位中的倒序标签可以保持不变。这里没有判定作者实现有bug；temporal-sensitive样本筛选及checker的完整代码仍未取得。

论文的输入是采样后的视频加语言 prompt，输出是答案文本。没有指定“预测第 $t$ 帧”的 target frame，也没有规定输入只到 $t$ 的因果窗口；因此不能把它写成在线/固定前瞻的定位证据。倒序样本是训练时的辅助条件，推理时是否额外输入倒序视频，正文没有作为部署协议报告。

## 预印本实验实际支持什么

论文在 STAR、SSv2、NExT-QA、CLEVRER、LLaVA-178K 子集和 PerceptionTest 各随机取 1,000 个 video-QA 样本，以accuracy测试动作、推理和一般感知。其中SSv2改为5选1问题（真实template加4个随机干扰项），不能与标准全类别动作识别分数直接混比。Qwen3-VL-8B 的表 1 中，BL-GRPO 平均 76.3，COMET TC-GRPO 为 78.8；作者说明相对于 BL-GRPO，Action +4.9 pp、Reasoning +2.1 pp、Perception -0.4 pp。它支持“该系统的增益更集中在方向/时序语义，而非均匀提升一般感知”，不支持小球或密集空间读出。

两项消融都存在，不能误写成没有：

* 表 2（Qwen3-VL-8B、Stage I）从 BL-SFT 74.5 到加 temporal branch + appearance-motion fusion、去 TAB 的 76.1（+1.6），再到 full COMET 76.5（+0.4）。这把 **branch、fusion 和其训练**作为一个组合加入，未隔离原始一阶差分、五阶 Taylor、归一化、灰度混合或空间定位能力。
* 表 4 固定同一个 full COMET Stage-I checkpoint：标准 GRPO 7+0 为 78.0，TC-GRPO 6+1 为 78.8，5+2 为 78.4，4+3 为 77.6。它支持该 video-QA 设置中小比例 reverse exploration 比无 reverse 更好，不能推出中心监督下的倒序增强或 displacement 正确性。

作者的最近邻实测比较是 Flow4Agent（SAMFlow motion prior）和 TempFlex，在其相同 Qwen3-VL SFT + 标准 GRPO pipeline 中与 COMET 比；这不是 cost-volume/correlation 在 tiny-target 任务上的对照。全文也引用 Two-Stream、TSM、Trajectory Attention 与 RAFT 等时间/对应背景，但没有和显式 correspondence 做同任务、同输出的定位消融。

## 成本和可比性边界

表 3 仅给参数附加量：在 Qwen3-VL-8B，Taylor ViT-B core attention +143.3M（+1.6%），四个 fusion module 各 +37.8M（总标注为 +1.7%）；InternVL2.5-8B 为 +100.7M（+1.2%）和一个 +28.3M（+0.4%）。表中 external SAMFlow/MegaFlow 分别列 +650M/+936M。作者特意称自身统计为保守的 core-attention accounting。

没有 FLOPs、wall-clock latency、逐视频吞吐、显存曲线、解码/采样成本、不同 $W$/帧数的 scaling，且双 ViT + LLM 的端到端成本显然不等于灰度差分本身。因此“Taylor primitive 很便宜”可作为算子层的推论；“完整 COMET 是球定位实时基线”没有证据。任何球实验若借此讨论效率，必须重新在本项目输入分辨率、窗口、目标帧和计时边界下测量，不能移用参数百分比。

## 对球研究可复用的严格判断

1. **可抽取的原语**：有符号、方向敏感的多帧变化可在不求 correspondence 的情况下提供时间证据；它值得作为同窗口的低算力竞争解释，尤其用于检验“复杂匹配的收益是否只是看到了变化”。原文的高阶公式还提示“简单差分”必须说明到底是一阶还是多阶、是否保留符号、是否混入当前外观以及实际帧间隔。
2. **不能随原语带走的结论**：COMET 没有证明这类响应能保存同一高速小球、跨大位移找到其新位置、抑制相机运动或提高几像素中心精度。大位移也不自动令固定像素变化失效：球在当前像素短暂经过仍可产生时间响应；但其与球员/线条/全局相机变化竞争的程度是待测假设，不是 COMET 实证。
3. **完整系统不构成强制复现项**：它是语义 video MLLM，而本项目是逐帧二维定位；不应把 CoT、LLM、Video-R1、GRPO 或倒序文字奖励作为“COMET 基线”硬塞进定位实验。需要公平比较的是可兼容的时间原语，在一致的帧窗口、目标帧、监督、backbone、定位头与计时边界下，分别报告定位、误报和速度。
4. **本笔记未取得的内容**：作者源码/commit、$W$与归一化精确实现、视频FPS/真实秒级采样间隔、temporal-sensitive样本筛选与checker完整实现、独立评测采样和推理是否使用reverse condition的部署说明，以及tiny-ball/dense localization的任何数据。这些缺口限制复现和迁移判断，不应由合理猜测补齐。

## 直接来源

1. [COMET arXiv 摘要与版本记录（v1）](https://arxiv.org/abs/2608.21030)。
2. [COMET 官方 arXiv HTML 全文：Taylor representation、fusion、TPD/TC-GRPO、实验和表 1--4](https://arxiv.org/html/2608.21030v1)。
3. [Taylor Videos for Action Recognition（COMET 引用的 Taylor 表示来源）](https://proceedings.mlr.press/v235/wang24ck.html)。该来源只解释先行 Taylor-video 概念；本笔记的 COMET 机制与数字均以上述 COMET v1 为准。
