# GMoT：软空间池化与微手势 motion token，不是小球候选或对应机制

日期：2026-09-11。范围：只补读 GMoT 的门控、token 化、训练与微手势实验，审查其对极小高速球研究能支持的边界；不复现、不引入模型或数据。

## 结论先行

GMoT 给出的直接先例是：在短微手势 clip 中，以**空间 softmax 加权池化**与相邻帧差分构造压缩 motion token，并经近关闭 residual gate 注入 Qwen3-VL 的视觉流，可提高微手势类别识别。正文描述两者的概念路线，却未给出它们作用张量和先后顺序的完整代数式。它支持“弱、局部、短暂变化可能被均匀聚合或静态外观淹没；空间加权与显式时间差是应竞争的解释”。

它没有输出球点、空间 heatmap、跨帧对应、flow 或物体 ID；motion token 不保留显式 patch-candidate axis，也未验证能否从它读出坐标，而原始视觉流 $Z_t$ 仍在。因此不能以它声称 gating 能自动发现并持续保留高速小球，或以微手势 Top-1 代替逐帧定位证据。反过来，空间 scorer 是 softmax 而不是明示 top-k/threshold 的硬删选，不能仅从架构就断言它必然在门控前“漏目标”；小球会否因弱权重被压低，须由球数据实测。

完整 GMoT 是 8B MLLM、半自动 CoT 标注和四阶段训练配方，不能被当作本项目必须复刻的轻量模块。门控/路由本身也不是新颖性空白；可研究的仅是特定 tiny-fast 定位条件下的损失、收益和失败证据。

## 版本、状态、来源和阅读缺口

* **Taorui Wang, Wei Xia, Hui Ma, Zijia Song, Jiayu Zhang, Zeheng Wang, Yong Xu, Zitong Yu**, *GMoT: Gated Motion-Aware Tokenization for Fine-Grained Micro-Gesture Video Reasoning with Multimodal LLMs*，[arXiv:2607.16322 v1](https://arxiv.org/abs/2607.16322)，2026-07-15；[官方 arXiv HTML 全文](https://arxiv.org/html/2607.16322v1)。
* arXiv comments 和 HTML 页眉写 “Accepted to ACM MM 2026”/ACM Multimedia 2026（2026-11-10--14）。本次没有取得 ACM Digital Library 或正式 proceedings 条目；故记录为 **arXiv v1，作者/稿件标注 ACM MM 2026 接收，正式刊载未核**。
* 已读主文 §3--5、表 1--7、limits 和 arXiv HTML；并实际读取 [arXiv PDF v1](https://arxiv.org/pdf/2607.16322v1) 的全部 **10 页**（至参考文献结束）。正文多次指向 Appendix A--D，但 HTML 与该 PDF 都没有附上 A--D；本次也未取得独立 supplemental。论文指向的[作者 GitHub 入口](https://github.com/timwang2001/GMoT-Motion-Aware-Tokenization-for-Fine-Grained-Micro-Gesture-Video-Reasoning-with-Multimodal-LLMs)当前返回 404；论文说 “will be publicly released”，未取得固定作者 commit。因此不猜 scorer/difference 的精确代码、训练 schedule 或完整成本。

## 实际门控信号与空间/时间轴

设 clip $V=\{I_t\}_{t=1}^{T}$。Qwen3-VL 的视觉 encoder 为每帧产生 patch tokens $X_t\in\mathbb R^{N\times d_v}$，经 aligner 变为语言空间 $Z_t\in\mathbb R^{N\times d}$。GMoT 的三个已披露部分是：

1. **Motion-prompted spatial saliency**：对每个 patch token 过线性层得到 relevance score，并在该帧的 **N 个空间 patch** 上作 softmax；用权重加权求和为 frame-level representation。它不是 hard top-k、候选框或候选位置列表。softmax 会让全部 patch 仍有非零权重，却把 N 个 patch 显式候选轴汇聚成一个向量；该新增分支未验证坐标读出。论文没给 linear score、softmax 温度、patch lattice、score 是否直接由时间差驱动或 score map 监督的公式。
2. **Kinematics-inspired temporal differencing**：对相邻帧差显式保留 signed difference，并计算 mean magnitude 量化 motion energy；二者 concat 后经 learnable projection 压为固定长度 dense motion token $m_t$。这里有实际的时间方向/符号信息，不是仅绝对帧差；但正文没有给出差分作用于 pooled frame feature、每个 patch feature 还是 aligner feature的精确公式，也没有多阶、长间隔或跨位置搜索。
3. **Semantic-preserving adaptive gate**：将原 aligned visual feature 与 motion token作

$$
\tilde z_t=z_t+\sigma(g)\odot m_t.
$$

其中 $g$ 为 learnable parameter，初始化在 near-closed 区使初始 $\tilde z_t\approx z_t$。这是为稳定预训练语义的残差注入；它不是以目标位置为条件的门控，也不输出 displacement。论文未说明 $g$ 的标量/通道/空间形状，或不同帧如何共享 $m_t$。

因此，论文的概念路线包含每帧 patch 的 soft weighted pooling、相邻差分和压缩 token 的语言前融合；但差分究竟作用在哪个张量、与 pooling 的严格顺序仍未知。它可降低静态背景的竞争，却没有显式令移动目标从 $s_t$ 关联到 $s_{t+1}$。即使最终视觉流仍包含原始 $Z_t$，新增 branch 也没有显式点候选轴或已验证的球坐标读出。

## token 选择会否漏掉目标

论文没有硬候选阈值或 top-k discard，不能声称“GMoT 先选 token 所以必漏球”。softmax 权重和随后的加权求和可能压低弱目标相对背景/人体区域的贡献，也可能提高它的权重；motion token 不带显式 patch-candidate axis，且论文未验证它的坐标读出。对此应作的严格表述是：

* **事实**：作者在微手势识别中观察到 spatial scorer 有用；去掉它，Stage-2 iMiGUE accuracy 从 61.19 降到 56.94。
* **推论**：这提示 sparse local change 可受益于非均匀空间汇聚，值得成为 tiny-ball 方法的竞争解释。
* **未知**：该 score 是否覆盖 2--5 px 球、是否被球员/线条/反光压制、在大位移时一帧的高分位置能否跟到下一帧、以及候选漏球率；论文均未测，不能预设适合或不适合。

## 任务、尺度与已发表实证

任务是 human micro-gesture **clip classification / reasoning**：iMiGUE 与 SMG，输出类名，后期输出含 `<think>`/`<answer>` 的文本；不是逐帧二维位置。论文描述的对象包括 finger twitch、shoulder shrug、lip press 等身体局部，在 few frames 内发生，且视频通常短于 3 秒。训练/评测采样为 4 FPS、至多 12 帧、单帧像素上限 786,432、visual token 上限 896。

这说明它确实面对“低振幅、局部、短时”的人类动作，不能被概括为纯全局事件分类。输入采样率已知为 4 FPS（相邻采样的名义间隔 0.25 s）；但论文没有逐帧原始时间映射/PTS、逐帧目标 $t$、手指/嘴唇像素尺寸、相对/绝对位移、motion blur、随机相机运动、无纹理小目标或遮挡。因此没有微手势尺度可与球的几像素支撑或多 ball-size 位移做定量对齐。

表 1：Qwen3-VL-8B 从 iMiGUE 60.52、SMG 70.00 到 +GMoT 的 67.32、73.11；Qwen2.5-VL-7B 也从 59.71、53.27 到 63.23、60.85。表 7 在 Stage 2 固定条件下，去 temporal diff 为 55.65（相对 full 61.19，-5.54），去 gate 为 56.28（-4.91）。这支持 spatial weighting、相邻 temporal difference、gate 和训练阶段的**组合**对两个微手势分类集有实际贡献，不能隔离为球定位的单项收益。

BRG Recall 是“预测正确样本中，CoT 是否含该类预设身体区域词”的 lexical proxy；它从 96.82 到 97.70 的同时，平均 rationale 长度从 45.08 到 64.84。论文自己指出长度是混杂因素，且 BRG 不证明因果 faithfulness。response map 可视化集中在头/躯干，只是定性 body-region grounding，不能替代点坐标或 patch coverage 的量化。

## 训练、推理和成本边界

四阶段实际不同：Stage 0 在 general video datasets 只更新 GMoT、冻结 LLM 和 vision encoder；Stage 1 在微手势 video-label pair 上以 LoRA 允许 MLLM 组件训练；Stage 2 冻结 vision encoder、微调 LLM 与 GMoT 作 CoT SFT；Stage 3 是 grouped rollout 的 reward-guided policy refinement，奖励包含标签正确、长尾懒猜惩罚、格式和背景词惩罚。半自动生成、筛选且部分人工复核的 anatomical CoT 也是训练条件，不能把 full GMoT 数字归因于门控 alone。

训练在 H100 上以 ms-swift、DeepSpeed ZeRO-3 实现。主文未报告 GPU 数、耗时、batch、参数增量、FLOPs、显存、解码/帧采样耗时、motion token branch 的单独延迟或 end-to-end fps；这些细节被指向不可访问的 Appendix B。因此“它是 lightweight”只能指文字所称模块相对 MLLM 的意图，不能作为完整部署速度结论。推理所见输入为采样 clip 加 text prompt，输出类别/理由；没有目标帧 $t$、因果 mask、前瞻或逐帧 latency 协议，不能称为在线球定位。

Stage-wise 表也要求谨慎：在 iMiGUE，原 Qwen3-VL SFT 60.52，单加 policy refinement 62.96，GMoT warm-up + video-label pairing 59.85，随后 CoT SFT 61.19，最后 policy refinement 67.32。主文的 +6.80 是完整训练路线相对 SFT，而非“gate 一项”的独立因果数字。

## 对球点研究的严格影响

1. **已有先例而非可直接采用结论**：localized soft pooling + adjacent temporal difference + conservative residual gate 已用于弱局部运动，不可把 gating、motion token 或“防 pooling 稀释”单独包装为创新。
2. **可保留的竞争问题**：在一致输入窗口、空间分辨率和定位 head 下，空间权重/时间差是否解释所有运动收益，是可证伪问题。必须测球的可见性、候选覆盖、误报及位置误差；不能只看分类或 CoT。
3. **不能继承的主张**：没有以球点为 query 的跨帧对应候选、可靠度校准、camera/object 分离、大位移覆盖、tiny-ball 点定位或实时成本证据。GMoT 对手势有效不证明体育球有效，也不证明无效。
4. **本轮未做的事**：没有创建 GMoT 分支、CoT 标注、RL 训练、额外数据或实验。作者固定代码与可访问附录缺失，精确实现/成本的复现判断须待上游发布。

## 直接一手来源

1. [arXiv 摘要、作者、版本和接收注记](https://arxiv.org/abs/2607.16322v1)。
2. [arXiv v1 HTML 主文：GMoT、训练、实验、消融与局限](https://arxiv.org/html/2607.16322v1)。
3. [论文声明的作者 GitHub 入口](https://github.com/timwang2001/GMoT-Motion-Aware-Tokenization-for-Fine-Grained-Micro-Gesture-Video-Reasoning-with-Multimodal-LLMs)（本次访问 404，未作为源码证据）。
