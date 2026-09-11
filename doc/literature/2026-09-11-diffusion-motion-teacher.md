# MotionEnhancer：扩散 attention 蒸馏的是语言条件的时空显著性，不是球的 correspondence

日期：2026-09-11。范围：只核验 MotionEnhancer 从 video diffusion model（VDM）取出的 attention、训练/推理边界及细粒度实验；不复现、不下载权重、不提出本项目的新教师或训练任务。

## 结论先行

MotionEnhancer 的正式论文通过实验支持一个较窄的结论：冻结的 CogVideoX-1.5-5B 对原视频作 DDIM inversion/reconstruction 后，其文本—时空 latent attention 可离线作为 auxiliary attention-alignment target，改善 VideoQA 的 motion-level accuracy。它**不是直接将扩散特征或 attention 注入推理 VLM**，也没有 flow、point、bbox 或中心监督。

因此它是“生成式视频模型的内部时空注意可做语义 motion teacher”的已有先例，不能支撑以下更强说法：attention 给出同一物体的跨帧对应、attention 值就是可靠度、VLM 的密集位置有像素级可读性，或该教师对几像素高速球有效。论文确实有较细的 motion QA 分项（地点相关运动、物体相关运动、相机运动、非主体运动等），不应因它是 MLLM 而抹掉这些实证；但每项仍是**视频问答 accuracy**，不是点定位。

## 版本、发表状态、来源与阅读范围

* **Yifan Xu, Chao Zhang, Ruifei Ma, Fei Gao, Zhifei Yang, Jiaxing Qi, Zhipeng Chen**, *MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models*。arXiv [2606.06853 v1，2026-06-05](https://arxiv.org/abs/2606.06853)；[完整 arXiv HTML（含 supplementary）](https://arxiv.org/html/2606.06853v1)。
* **CVPR 2026 已正式可核验**：CVF 的 [论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_MotionEnhancer_Leveraging_Video_Diffusion_for_Motion-Enhanced_Vision-Language_Models_CVPR_2026_paper.html) 列为 CVPR 2026, pp. 2778--2787；[CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_MotionEnhancer_Leveraging_Video_Diffusion_for_Motion-Enhanced_Vision-Language_Models_CVPR_2026_paper.pdf) 指明其为 accepted version，最终 proceedings 在 IEEE Xplore。
* 已读主文 §3--5、algorithm、全文可访问 supplementary 的 DDIM、模型、实验与可视化段。论文项目页为 [motion-enhancer.github.io](https://motion-enhancer.github.io/)，本次未发现作者公开代码仓库或可固定 commit；项目页也没有提供可审计的实现链接。因此，不把实现层的缺省项（attention hook、输入帧数/分辨率、具体 CFG/memory 配置）猜成事实。

## 教师究竟取什么 attention

教师是**冻结的 CogVideoX-1.5-5B**，一个 DiT video diffusion model；VLM 是 Qwen2.5-VL（3B/7B）或 InternVL3（2B/8B）。输入视频 $V$ 先由教师的 3D causal VAE 编成时空 latent，论文写作

$$z_{\rm vision}\in\mathbb R^{F\times H\times W},\qquad S=FHW,$$

algorithm 明确令教师输入 $\operatorname{VDM.TextEncoder}(I)$，其中 $I$ 是该 video-QA 的**问题**，而 VLM 的自回归 target 是答案 $s$；文中没有把答案、caption 或额外描述送入教师作为 condition。$z_{\rm text}$ 与 $z_{\rm vision}$ 串接为统一 multimodal sequence $z_{\rm mm}$。它不是 RGB-frame attention，也不是预训练 VLM feature 的差分。对该**完整送入教师的 sampled video**，先作 5-step DDIM inversion，再以 classifier-free guidance 和 parallel inverted-path cross-stream memory 作 5-step denoising/reconstruction；每一步计算

$$A_{\rm mm}=\operatorname{Softmax}(Q_{\rm mm}K_{\rm mm}^{\mathsf T}/\sqrt d).$$

作者说对 raw attention 作 layer-wise 与 denoising-timestep-wise average pooling。文本—视觉子块形成 $A_{\rm t2v}\in\mathbb R^{T\times S}$：每一行是文字 token 对 $F\times H\times W$ latent 位置的权重。其列确有显式空间/时间地址，能支持 text-to-video grounding；但它不是以球点为 query 的跨帧候选集合，不给 $(s,f)\to(s',f')$ 的物体对应、instance ID 或 flow vector。

### MHS：以 vision-to-vision attention 选 head

MHS 从每个 VDM transformer head 的 vision-to-vision 子块 $A_{\rm v2v}$ 计算三项无参数统计量，并各自标准化后相加，选分数 top 50% heads：

* Diagonal Focus Coefficient（DFC）：diagonal mask 内外 attention 平方和之比；
* Temporal Continuity Score（TCS）：在固定空间位置 $s$，抽出跨帧 $F\times F$ 子矩阵，阈值为平均 attention，量连续高值段的平均长度；
* Diagonal Saliency Ratio（DSR）：diagonal region 中高于该阈值的单元比例。

这三个量是对某类 attention 图“对角集中、同址跨帧连续、对角高值覆盖”的启发式选择准则，**没有真实 motion/flow/point label**。它们不能验证 selected head 对目标保持身份：对角结构偏向固定地址的时间连续；真正移动物体的对应可落在非对角位置。论文将它命名为 motion-sensitive，是其任务驱动的经验定义，不能将分数重解释为已校准的 motion reliability。

主文把 $S$ 定义为 $FHW$，并以固定 $s$ 的 $F\times F$ 子矩阵解释 TCS；但 supplementary algorithm 的注释又把 $A_{\rm v2v}$ 标为 $[S\times S],S=H\times W$。两处的空间/时间展平记法不一致，且没有源码，不能精确还原 diagonal mask 如何覆盖完整时空 token 序列。

### MTTI：选文字 token，不产生候选目标

对 MHS 聚合后的 $A_{\rm t2v}$，作者先对空间维平均成 $A_{\rm t2f}\in\mathbb R^{T\times F}$，为每个 token 计算

$$
MS(t)=\operatorname{Mean}_f(A_{\rm t2f}^{t})+
\frac1{F-1}\sum_{f=1}^{F-1}|A_{\rm t2f}^{t}(f+1)-A_{\rm t2f}^{t}(f)|,
$$

取 top 50% token。它偏向“总体 attention 强且平均后仍随时间波动”的词；用于选词的统计先作空间平均，本身不保留空间地址。随后从原 attention 中取出的 $A_{\rm VDM}\in\mathbb R^{T'\times S}$ 仍保留 latent-grid 的时空列，能作文字条件 grounding；这不需要从平均值恢复地址，但仍没有以球点为 query 的跨帧对应候选、物体 instance ID 或由球中心监督校准的读出。

论文对 pooling 叙述也有未解细节：主文说最终 attention 同时跨 layer/timestep 平均，algorithm 又说每个 denoising step 先跨 layer average、并“store all heads”。head 身份在何时相对 layer/timestep 聚合、MHS 是逐步还是对聚合张量算，正文未完全指定；没有作者代码时应保留该复现缺口。

## 是蒸馏/先训教师，不是推理注入

同一 video-question-answer sample 上，VLM 也抽自身 text-to-vision attention，跨其 head/layer average 得 $A_{\rm VLM}\in\mathbb R^{T'\times S_{\rm VLM}}$。作者把它插值到教师尺度，用一个三层 MLP aligner，并最小化：

$$
\mathcal L_{\rm total}=\mathcal L_{\rm AR}+λ\|\operatorname{Aligner}(A_{\rm VLM})-A_{\rm VDM}\|_2.
$$

这里必须保留论文的记法不一致：主文 Eq. (15) 将该项命名 $\mathcal L_{\rm MSE}$，但写为未平方的 $\|\cdot\|_2$；supplementary Eq. (20) 才写成 $\|\cdot\|_2^2$。没有代码不能断言实际优化的是哪一种，故本笔记只称 attention-alignment loss，不将它自行固定为“已知平方 MSE”。

教师用自己的 text encoder 对问题 $I$ 取 token，student 的 $A_{\rm VLM}$ 却被 algorithm 直接以同一选中集合 $\mathcal T_m$ 索引。正文没有说明 CogVideoX 与 Qwen2.5-VL/InternVL3 tokenizer 是否相同、若不同如何把教师选词映射到 student token、或如何处理分词边界。对无文本的球定位，这意味着原方法的 token-conditioned teacher target 没有已发表的“球”query 或 token 映射；不能假定它可不改动地删除文本后仍成立。

注意：MHS/MTTI 被称为 parameter-free selection module，但论文明确有三层 MLP aligner；“没有新增训练参数”的宣传不能按字面扩展成没有可训练对齐网络。正文只明确训练时 VLM 的 vision tower、merger 和 LLM backbone 都更新；教师 VDM 冻结。aligner 是否单列 optimizer/parameter count 没有说明，需源码才能固定。

教师 attention 在 VLM SFT **之前离线抽取**，论文说一次抽取每个样本在 A100 要 20--30 秒，priors 可供多个 VLM 或 ablation 复用。SFT/部署阶段不需要再用 VDM 作 online input；作者没有把 VDM attention作为 test-time feature injection。因而它是 expensive offline teacher/distillation，而不是参数无关、零成本的运行时 motion module。

## 时间窗口、因果性与成本边界

算法写的是完整输入 video $V$，$F$ 为送入 VDM 的帧数；VLM 的评测表写 Qwen 为 1 fps、InternVL 为 8 frames。论文**没有报告** VDM 实际 $F$、输入分辨率、clip 截断/采样规则、目标时刻或“只见历史”的因果协议，因此不能把它归为逐帧在线、固定前瞻或某一长度 temporal window。CogVideoX 的 causal VAE 不等于整个教师过程对未来帧不可见：后续 DiT attention 对完整 token sequence 的可见性仍须以实际 mask/代码为准。

“5-step”也不是廉价单 forward：一次先 inversion、再 reconstruction/denoising，且每个 denoising step保存多层多头 attention。作者只给 extraction 的 20--30 s/A100 和训练为 8×A100 80GB、25k QA sample、1 epoch、batch 8；没有报告 attention 存储、VAE encode/decode、DDIM、VLM SFT 与最终 VLM inference 的分项 FLOPs、wall-clock 或显存。其 10-step 消融比 5-step QA 更高，2-step 显著降分，说明教师抽取成本与质量有实际 tradeoff；不能把“frozen/offline”混同为免费，也不能把缓存后的 VLM 推理速度说成完整端到端成本。

## 已做的细粒度实验，及其不能证明的内容

训练用 MotionBench-Train 全部 5k QA 加 MotionVid-QA 抽取 20k，共 25k；评测 MotionBench 与 FAVOR-Bench。MotionBench 包含 Motion Recognition、Location-related Motion、Action Order、Repetition Count、Motion-related Objects、Camera Motion；FAVOR-Bench 包含 Action Sequence、单/多动作细节、Camera Motion 与 Non-Subject Motion。因此工作确有对位置相关动作、物体相关动作和相机/背景变化的量化，不宜笼统说“只做高层语义”。

但所有指标是问题答案 accuracy。示例：Qwen2.5-VL-7B 在 MotionBench Overall/Average 从 52.81/48.29 升至 57.04/52.92；FAVOR-Bench 从 42.61/42.58 升至 46.88/47.01。MHS-only 与 MTTI-only 也被消融（MotionBench Overall 56.60、55.80，相对二者皆无的 54.83；二者都有 57.04）。5-step 的 MotionBench/FAVOR Overall 为 57.04/46.88，2-step 为 52.76/24.01，10-step 为 57.51/49.02。这支持 **VDM attention selection + alignment 对这些语义 motion QA 有作用**。

它没有 tiny-object、球、dense segmentation、keypoint/center、box、tap/point tracking 或 optical flow 指标；没有比较预测 attention 的真实对应 rank、中心误差或大位移 coverage。可视化的“attends to correct locations”是定性 language grounding，不是空间标注评价。摄像机运动虽为 benchmark 类别，也不是相机—物体运动分解或补偿证明。

## 对本项目研究主张的影响

1. 不能把“用 diffusion attention 作 motion teacher / attention alignment”写成首次；MotionEnhancer 是直接近邻，且其 MHS/MTTI 已对 head/token motion selection 做了消融。
2. 不能以此声称 VDM attention 本身是 correspondence、球位移、可信度或稠密 motion supervision。若以后把注意力当这些量，仍须在连续球中心上另测 coverage、rank、误匹配与相机/背景竞争。
3. 本文没有证明它不适用于球，也没有正面证据支持适用。它的 semantic motion QA 改进仅足以说明此类教师值得被审计为竞争先例；是否引入、怎样控制离线成本和怎样用现有中心标签检验，都仍是未验证的研究选择。
4. 本轮没有新增教师、模块、数据或实验任务；该笔记只限定已有文献所能支持的表述。

## 直接来源

1. [CVPR 2026 官方 CVF 论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_MotionEnhancer_Leveraging_Video_Diffusion_for_Motion-Enhanced_Vision-Language_Models_CVPR_2026_paper.html)。
2. [arXiv v1 摘要/版本记录](https://arxiv.org/abs/2606.06853)。
3. [arXiv HTML 主文与 supplementary：attention extraction、MHS/MTTI、alignment、cost 和实验](https://arxiv.org/html/2606.06853v1)。
