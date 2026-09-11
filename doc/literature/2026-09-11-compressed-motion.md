# ReMoRa：统一 H.264 重编码的 compressed motion proxy，与微小球定位的边界

日期：2026-09-11。范围：补读 ReMoRa 的 [CVPR 2026 论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Yashima_ReMoRa_Multimodal_Large_Language_Model_based_on_Refined_Motion_Representation_CVPR_2026_paper.html)、[官方论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yashima_ReMoRa_Multimodal_Large_Language_Model_based_on_Refined_Motion_Representation_CVPR_2026_paper.pdf)、[官方 supplementary](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Yashima_ReMoRa_Multimodal_Large_CVPR_2026_supplemental.pdf)、arXiv v2 全文及作者公开代码的固定提交；未运行模型、未下载权重/数据。本文展开 [modern_motion_evidence.md](modern_motion_evidence.md) 中 ReMoRa 一行，只判断 codec motion proxy 是否适合本项目的数据条件。

## 结论先行

在 ReMoRa 的**论文实验重编码配方**中，motion vector 不是直接采用各输入文件保留下来的原生 bitstream。作者先把每个视频统一降至 $384\times384$、重采样到 16 fps，再以 FFmpeg **重编码为 H.264**，做 scene-adaptive I-frame insertion、最大 GOP 长度 32 帧；随后取这个新码流中 I-frame 的 RGB 和 P/B-frame 的 block motion vectors。它是低成本但有明确编码条件的 proxy，不是“免费得到的原视频物体运动”。后期公开代码另有直接读取传入视频 MV 的入口，不能将其与论文配方混同。[论文 §5.1](https://openaccess.thecvf.com/content/CVPR2026/papers/Yashima_ReMoRa_Multimodal_Large_Language_Model_based_on_Refined_Motion_Representation_CVPR_2026_paper.pdf) 明确该统一重编码步骤。

RMR（Refined Motion Representation）用 CoTracker3 产生的 dense optical-flow **teacher**进行 $L_2$ 预训练，目的是把粗 block MV 变成更适合长视频语言理解的 motion feature。作者实证的是 LongVideoBench、NExT-QA、MLVU、VideoMME、MotionBench 等 MLLM/QA 分数与模块消融；没有 RMR flow EPE、像素/点对应、自动检测、中心误差、tiny-object 尺寸分桶或大位移匹配准确率。因而“refinement”可说明该语义任务中有用，不能说明输出是真实球位移、已经恢复球点地址或可替代当前帧视觉证据。

这改变项目判断如下：ReMoRa 是“稀疏 appearance anchor + 更密 codec proxy”可作为**条件化效率假设**的近邻，不能作为运动可靠性或精细定位的直接证据。本项目 Tennis 主发布包为 JPG+CSV，缺少可提取的原生 MV；若试验 codec motion，必是对 JPEG/视频重新编码得到的新 proxy，并应记录 codec、fps、GOP、I/P/B、macroblock 与输入缩放。另三类本地数据虽保留 MP4，也只说明容器可得，不能保证其原码流符合 ReMoRa 的统一 H.264 条件。[数据入口](../../data/README.md)对此发布形态有明确记录。

## 版本、正式状态、阅读面与代码边界

Daichi Yashima、Shuhei Kurita、Yusuke Oda、Komei Sugiura，**“ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding”**，发表于 **CVPR 2026，pp. 31845--31855**；CVF 的 BibTeX 标为 2026 年 6 月。[arXiv:2602.16412v2](https://arxiv.org/abs/2602.16412v2) 的 v1 为 2026-02-18，当前 v2 为 **2026-02-21**，并标记 accepted to CVPR 2026。CVF PDF 在水印外称与 accepted version 一致；本笔记以它和 supplementary 为正式方法/成本证据，arXiv v2 用于版本日期与 HTML 锚点。

本次实际读到正文 §3--§6、supplementary A--G，特别是 B 的吞吐/预处理时间、D 的训练与模型配置、F 的 error analysis、G 的 scene-adaptive preprocessing。作者 [daichi1207/ReMoRa](https://github.com/daichi1207/ReMoRa) 可访问；为避免移动分支，源码只固定查阅至 `bdb699908ea0122f1724748c262fcba00fe765d3`（2026-07-19，仓库 `master` 当前提交）。该提交晚于 CVPR/arXiv 文本，README 未声明与论文精确同版；下述源码仅交叉核实公开 extraction/refiner 的操作面，不能以它补写论文没有披露的实验条件。

## 实际输入：重编码后的 GOP，而非原始物体轨迹

论文的抽象 codec 模型是每 GOP 一个 I-frame，后随 P/B-frame。对 inter-coded frame 中每个 block，编码器到某 reference frame 搜索近似块，保存 block displacement $\mathbf m^{(k,t)}(u,v)$；P-frame 由过去 reference 预测，B-frame 可由前后 reference 和 residual 重构。[§3，式 (1)--(2)](https://arxiv.org/html/2602.16412v2#S3) 同时给出了 residual $\mathbf R$，但 ReMoRa 的实际 GOP input 是

\[
\operatorname{GOP}^{(k)}=(\mathbf V^{(k,0)},\mathbf m^{(k,1)},\ldots,\mathbf m^{(k,T_g-1)}),
\]

即 I-frame RGB 加 P/B-frame motion-vector field；**没有 residual input branch**。[§4，式 (3)](https://arxiv.org/html/2602.16412v2#S4) 的后续 RMR 也只接 $\mathbf m$。所以压缩残差并未作为小球纹理、拖影或误匹配的补充证据进入作者模型。

实际统一预处理是：输入 clip 空间 downsample 至 $384\times384$、时间 resample 16 fps；FFmpeg H.264 scene-adaptive detection 按视觉不连续插 I-frame，GOP 不超过 32 帧。作者原文称 “All frames were encoded with $4\times4$ macroblocks”；这里按原词记录，不将其扩展为已确认的均匀 partition 或具体 encoder 命令。[CVPR §5.1](https://openaccess.thecvf.com/content/CVPR2026/papers/Yashima_ReMoRa_Multimodal_Large_Language_Model_based_on_Refined_Motion_Representation_CVPR_2026_paper.pdf)；supplementary G 的示例也明确 MV 从该 H.264 codec 提取。故可确认的是**论文配方中的统一重编码**，不是保留输入视频既有 GOP/MV。

论文没有报告 H.264 profile、FFmpeg/encoder version、量化/码率、具体编码命令或 partition 策略、B-frame 数、实际 P/B 比例、每个 MV 采用前向还是后向 reference、reference index，或 B-frame 的双参考 vector 如何保留/规约。这些不可由“P/B”或 $4\times4$ 原词泛称补齐，也不等于最终网络一定逐 $4\times4$ 格读取。更不能把一个 block 的 prediction vector 等同为球、手或任何物体的中心位移：它服务于编码率失真优化，可能由背景、遮挡、纹理与 reference 选择决定。

固定源码提供一个有用但版本受限的实现交叉检查：PyAV 打开 `export_mvs` side-data，记录 frame type 为 I/P/B，并取每条 MV 的 destination 坐标、块宽高、`motion_x/y` 与 `motion_scale`；随后在一个可配置 aggregation grid 内对覆盖同一 cell 的 vector 求均值。见 `daichi1207/ReMoRa@bdb699908ea0122f1724748c262fcba00fe765d3:infer_with_mv.py:L80-L209`。其 `block_size=16` 是 aggregation cell、在 $384\times384$ 时为 $24\times24$ grid，而非 codec partition size，源码注释已明确这一点：`daichi1207/ReMoRa@bdb699908ea0122f1724748c262fcba00fe765d3:infer_with_mv.py:L212-L243`。代码还可识别 P/B 帧型，却不读取 MV 的 source/reference index；这进一步说明固定代码不能确认论文实际的 reference-direction policy。

该固定仓库的 README/inference extractor 直接对传入视频以 PyAV 导出 MV，`scripts/extract_motion_vectors.py` 也直接调用 `extract_codec_data(video)`：`daichi1207/ReMoRa@bdb699908ea0122f1724748c262fcba00fe765d3:README.md:L38-L66`、`scripts/extract_motion_vectors.py:L75-L99`。它没有在此入口调用论文所述重编码器，故与正式论文的统一 H.264 preprocessing 不可直接视作同一 recipe；本笔记以论文为实验事实，以固定源码仅说明后来公开的 extraction surface。

block 粗、量化与平均只能说明需要测量重新编码 RGB 中小球是否仍可读；它们**不证明**微球信息必然已消失。若微球尺度小于编码块、背景主导 block matching 或重编码改变模糊/纹理，作者现有证据不能决定该 proxy 对本项目是否有用。

## RMR 的监督、输出与 “refinement” 的实际含义

RMR 的训练目标是把 block MV 映到 “fine-grained dense motion fields”。作者使用 **CoTracker3** 作为 off-the-shelf 生成器，对 RMR 预测与其 dense optical-flow target 作 $L_2$；下游 instruction tuning 时，RMR 作为 feature encoder，将每个 inter-frame 的 raw MV 转成 $\mathbf E_M^{(k,t)}\in\mathbb R^{N_m\times d_s}$，与 I-frame 的 image-encoder patch embedding 拼接，再进入双向 Mamba/HMSS 和 Qwen2 生成文本。[§4.1--§4.2](https://arxiv.org/html/2602.16412v2#S4.SS1)。下游训练的最终输出是词 token，损失为 standard cross-entropy，[§4.2](https://arxiv.org/html/2602.16412v2#S4.SS2)；没有输出 $(x,y)$、flow map、点轨迹、object ID 或 match/no-match 分数。

论文未披露 CoTracker3 checkpoint、输入 resolution/frames、由 point tracker 怎样栅格化为 flow、target 的时间 reference、visibility/occlusion 用法或 teacher EPE。也没有报告 RMR 自身对 CoTracker3 target 的 $L_2$/EPE，故 “approaching dense optical flow” 是方法目标/图示表述，不能改写为已验证的 dense point correspondence。

论文也未给 RMR 内部的 dense 化/插值结构、MV grid 到 $N_m$ motion patch 的映射。固定代码里的 GOP compressor 默认保留 native MV grid，并以逐 vector MLP 投影；只有显式设定 downsample factor 才做 adaptive average pooling，未构成论文 RMR 的可核验替代：`daichi1207/ReMoRa@bdb699908ea0122f1724748c262fcba00fe765d3:llava/model/multimodal_resampler/mamba_ssm/modules/gop_compressor.py:L142-L205`。因此不能补称作者已用某一种 bilinear upsampling 或已恢复逐像素 field。

固定源码的附加证据与论文方向一致，但版本/训练数据对应关系未声明：`train_mv_refiner.py` 将作者预先提供的 CoTracker3 flow pickle reshape 到 MV grid，支持 visibility-weighted loss，计算 $L_2$ 与 EPE，且可把 predicted residual 加回 raw MV：`daichi1207/ReMoRa@bdb699908ea0122f1724748c262fcba00fe765d3:llava/train/train_mv_refiner.py:L293-L358`、`L528-L542`、`L646-L688`。它没有生成 CoTracker3 target 的脚本，也没有一个固定 paper checkpoint 或复现数值；因此不能从这份后续代码断言作者论文的 RMR 具体 architecture、teacher 处理或评估结果。

## appearance/motion 的时间支持与因果边界

两路并非“appearance 低 fps、motion 原视频 fps”的无条件分工：作者先将**整个统一码流定为 16 fps**，I-frame RGB 是 scene-adaptive anchor，P/B MV 表示其间 inter-frame；模型最多取 64 个 I-frame、每个 I-frame 后最多 32 个 P/B-frame。paper 的 GOP 上限/64 I-frame 设置见 §5.1 与 supplementary B/Table 9。I-frame RGB 进冻结 SigLIP ViT-SO；Qwen2 用 LoRA 适配。[§5.1](https://arxiv.org/html/2602.16412v2#S5.SS1)。项目的真实帧号/中心标签映到这个 16 fps 重编码时间线的规则尚未定义，不能直接沿用当前 target-frame 集合。

local GOP Mamba 与跨 GOP Mamba 都是 bidirectional，且 MLLM 可聚合长视频全局上下文；这服务 offline long-video understanding，不给某个末帧球位置的在线/因果保证。即使本项目模型窗口截在 target frame，B-frame 的 future reference 和可解码时刻仍可能引入前瞻/等待；论文未披露实际 reference strategy，截断模型张量不能自动消除该问题。ReMoRa 也没有以逐帧检测 target 定义其可用时间范围，不能拿它替代该协议。

## 已有实证、成本口径与未覆盖的定位证据

作者的主证据是 long-video MLLM：LongVideoBench、NExT-QA、MLVU、VideoMME、Perception Test，以及 MSVD-QA/ActivityNet-QA。supplementary 还报告 MotionBench 的 motion recognition、location-of-motion、camera motion、motion order、attribute-of-object、rotation count 等**问答分数**；“location of motion”不是输出像素位置或点 correspondence。RMR 消融显示去掉 optical-flow pretraining 或整个 RMR 会降低 VideoMME/NExT-QA 分数。[论文 Table 4](https://openaccess.thecvf.com/content/CVPR2026/papers/Yashima_ReMoRa_Multimodal_Large_Language_Model_based_on_Refined_Motion_Representation_CVPR_2026_paper.pdf) 仅证明该语义评测中的辅助作用。

没有 tiny/small-object 尺寸分层、自动点/框检测、中心定位误差、光流 EPE、GT 初始化 tracking、真实运动对应、遮挡/absence 或大位移分桶。supplementary F 的作者 error analysis 反而记载：small object manipulation、subtle gesture 和 fine-grained transition 会漏/混淆，refined signal 对部分 fine-grained dynamics 仍不够信息。这是失败案例描述，不能外推为球定位量化，却直接反驳“RMR 已解决局部微小运动”。“bounce the ball”的问答定性例也只显示活动类别判断，非球中心或轨迹正确。

成本需分开读：supplementary B 的 H200 单 GPU decode-time 对比为 ReMoRa 0.40 samples/s、24.45 tokens/s、10.59 GB max memory；文字称该测量在相同 resolution/batch/max output length 下，且报告的是生成/decoding 阶段。它没有明确将统一重编码、MV extraction 或 CoTracker3 teacher 计入该吞吐。另一个单列的 preprocessing 表才计 scene-adaptive pipeline：LongVideoBench 100 视频、每视频 3 次平均，64 I-frame 加每 I-frame 最多 32 P/B-frame 为 **3.12 s/video**，uniform 64 RGB 为 2.05 s，等 temporal coverage 的 naive RGB（2,112 frames）为 58.51 s。外部 CoTracker3 是 RMR 预训练 teacher，不是作者宣称的推理输入；其产生 teacher target 的成本未报告。论文还报告约 7.6B 参数、约 2,900T multiply-add、16×H200 训练约 21 h，故其“linear temporal scale”不等于本项目逐帧球检测的端到端轻量成本。

## 对本项目低成本 proxy 的决定边界

1. ReMoRa 的可迁移主张应限为：在**统一重编码且有连续视频容器**时，I-frame RGB 加较密 block-MV 有机会以更少 RGB anchor 覆盖长时程。它不支持将 codec MV 当物体速度、可靠 correspondence、球中心监督或 absence 证据。
2. Tennis 当前发布入口是逐 JPEG 帧和 `Label.csv`，没有原始 bitstream/PTS；直接提取“原生 MV”不可行。重编码可建立可审计的新 H.264 proxy，但这改变编码参数、时间采样与小球像素形态；真实帧/标签到新码流的目标时刻映射必须先定义。Shuttlecock、BlurBall、OpenTTGames 的 MP4 可供未来检验，却不自动满足 ReMoRa 的 H.264/GOP/reference 条件。
3. 若以后考虑该 proxy，才定义并测量：重新编码 RGB 中的球可读性、I/P 时间覆盖，以及在明确 reference frame/方向后的中心邻域 MV 与真实位移关系。若这些条件下不能提供额外证据，停止把 codec proxy 当作主要解释。不要据此搭建 7.6B MLLM、HMSS 或 CoTracker3-refiner baseline。

本次结论只来自正式 CVPR 论文/附录、arXiv v2 与版本受限的作者代码；没有把编码器 block MV 误写为物体位移，也没有因块级/重编码而断言微球信息必然不可用。
