# TrackNetV2 / TrackNetV6：原始来源、可读范围与仍然缺失的协议

**范围。** 这是一份针对直接体育近邻的来源核验，不重述 TrackNet 系列综述。先读了 `sports_evidence.md` 与 `2026-09-10-causal-baselines.md` 中已经确认的 V2/V6 边界；只补会影响当前三帧因果 BlurBall 基线及后续 motion 决策的事实。检索和读取截至 **2026-09-12**。未读取本地数据或测试标签，未运行任何模型。

## 先给决定

1. **V2 是可引用的三帧 MIMO heatmap 基线，但还不是“explicit motion / correspondence”先例。** 正式摘要可证实其用多帧输入、多帧输出、U-Net skip、二维连续 heatmap 和 weighted cross-entropy；可证实的 3-in/3-out 速度报告为 31.84 FPS。正式正文当前无法从 IEEE 合法公开入口取得，所以 loss 权重、精确 target-frame 对齐、match split、visibility 处理、定位容差及 post-processing 都不能写成已复核事实。
2. **V6 不再只是“没有材料的 demo”。** ICMR 2026 正式发表可由 DOI、会议信息和出版页码确认；作者仓库的 10 页 Appendix 是可读的一手补充材料，说明其将 DDF（小波）、CSCP、PICC 与 nmODE/LMM cross-scale decoder 作为核心。然而 ACM 正文当前受访问限制，仓库 README 同时明确训练与 complete inference pipeline 未发布、demo 可能不同于论文。因此，V6 仍不能成为可公平重训基线，也不能据 demo/附录推定主文的时序因果性、监督、评价容差或完整模块组合。
3. **对当前研究的直接影响不变且更精确：** V2 是“多帧 dense heatmap + 空间解码/训练”的直接历史架构近邻；在其完整协议未核实前，不据摘要指标认定它是本地强基线或提高复现优先级。V6 可作为最新 TrackNet 相关工作和“跨尺度 decoder + frequency prior”近邻，但不能用于声称已经排除了最新方法，更不能把它当作 temporal correspondence 的已读反例。当前 `[t-2,t-1,t]→t` 末帧因果协议不应被改成 V2/V6 原协议的复现。

## 来源状态与阅读深度

| 方法 | 版本/发表事实 | 本次实际可读一手材料 | 阅读深度 | 不能越过的边界 |
|---|---|---|---|---|
| TrackNetV2: *Efficient Shuttlecock Tracking Network* | [IEEE 9302757](https://ieeexplore.ieee.org/document/9302757)，DOI [10.1109/ICPAI51961.2020.00023](https://doi.org/10.1109/ICPAI51961.2020.00023)，ICPAI 2020，2020-12-01 | IEEE/NYCU 的正式元数据和摘要；[NYCU institutional record](https://scholar.nycu.edu.tw/en/publications/tracknetv2-efficient-shuttlecock-tracking-network/)；作者数据说明 [HackMD](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw) | **B：官方摘要/数据说明，不是论文正文** | IEEE landing page 没提供可读正文；publisher PDF URL 返回拒绝，OpenAlex/Semantic Scholar 也没有 OA PDF。未把第三方镜像或复现源码冒充原文。 |
| TrackNetV6: *A Unified Framework for Lightweight and Robust Fast-Moving Tiny Ball Tracking* | [ACM DOI](https://doi.org/10.1145/3805622.3810690)，ICMR 2026，页 2616--2625；[ICMR technical program](https://icmr2026.org/technical-program.html) 和 [DBLP record](https://dblp.org/rec/conf/mir/YuBLXCL26) 可交叉确认 | DOI 元数据；作者 [GitHub repository](https://github.com/Gi-gigi/TrackNetV6) 的 README、[Appendix PDF](https://github.com/Gi-gigi/TrackNetV6/blob/main/assets/Supplementary/Appendix.pdf)、公开 demo/model 源 | **正文 U；官方补充材料 A；公开 demo 源 A（只对 demo）** | ACM PDF/landing page 访问被拒绝；检索不到作者自存主文。附录不包含主文训练/测试协议的全部公式和定义，demo README 还明确提示实现可能不同。 |

### 代码与版本更新核验

- **V2。** 正式摘要曾给出作者的 NCTU/NOL 代码/数据地址；截至本次访问，该 HTTPS 地址握手失败、HTTP 返回 gateway error。GitHub 搜索命中的 V2 项目均自述为 PyTorch implementation、reimplementation 或其他改造，未确认是论文作者发布，故不作为 V2 原始代码或“版本更新”证据。当前没有可核实的 V2 作者源码更新。
- **V6。** `Gi-gigi/TrackNetV6` 自称 official implementation；当前仓库只有 `main`、没有 release，最新实际 commit 为 **2026-05-20**，因而自上次 2026-09-09 审查以来没有新代码提交。README 的 “paper is not yet published” 与已发表 DOI 冲突；按其余内容（只释出 visualization demo / weights）应视为 README 未更新，不能反推论文未发表。

## TrackNetV2：可证实的机制、时间和监督

### 正式摘要能支持什么

- V2 将 V1 的 MISO 改为 **MIMO**，并称其 3-in/3-out 版本达到 **31.84 FPS**；同时报告输入尺寸缩小、VGG16 + upsampling + U-Net skip connections。[NYCU official abstract](https://scholar.nycu.edu.tw/en/publications/tracknetv2-efficient-shuttlecock-tracking-network/)
- 输出由高内存的 pixel-wise one-hot 3D heatmap 改成 real-valued 2D heatmap，损失由 RMSE 相关设计改为 **weighted cross-entropy**。正式摘要所描述的监督对象是每帧位置 heatmap；它没有报告 flow、offset、correspondence、trajectory ID 或给定 GT 初始化等额外监督。由于正文不可读，这应理解为“摘要未给出”，而不是对全文作更强的不存在断言。
- 正式摘要报告 18 个羽毛球 match、55,563 帧；并将“brand new match”作为测试，给出训练 Acc./P/R 96.3/97.0/98.7%，测试 Acc./P/R 85.2/97.2/85.4%。这些是作者报告的**检测式指标**，不等于像素中点误差、球在多大 displacement 下可见，亦不说明独立 match 数量如何划分。
- 作者数据说明确认发布格式是 rally 内的 `Frame, Visibility, X, Y`，30 FPS、1280×720，且不可见行为 `Visibility=0,X=Y=0`。它支持“窗口不能跨 rally”的数据规则；它不是 V2 论文的原始 18-match split 证明。[HackMD](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)

### 正式正文尚未证明什么

以下项目不能从摘要、数据页或后来论文的转述补成“V2 原文事实”：

- 三张输入和三张输出各自的 target-time 语义；例如第一个输出是否使用未来帧、边界如何 pad、是否重叠滑窗聚合。
- weighted BCE 的精确式、positive weight、Gaussian/heatmap target 形状和 visible=0 如何进入损失。
- 18 match 的训练/验证/brand-new-test manifest；与当前公开 23/26 match 数据页的对应关系。
- P/R/Acc 的 center tolerance、是否先 threshold，再取 peak/contour，以及是否有轨迹后处理。
- 参数量、精确计时边界、硬件、真实在线延迟，或对长位移 / 模糊 / 遮挡的分桶证据。

因此，V2 支持的最小比较是：**同一三帧、同一时序可见范围下的 MIMO dense heatmap / skip-connection 空间基线**。不能把“3-in/3-out”写成当前 `t` 的因果观测，也不能说 V2 已经解决大位移匹配。

## TrackNetV6：附录与 demo 实际支持什么

### 发表与材料的矛盾需要保留

- ICMR 2026 和 DOI 已确认正式论文；作者仓库 README 仍称论文未发表、训练和 complete inference 未发布，且 demo 可能不同于论文。[README](https://github.com/Gi-gigi/TrackNetV6/blob/main/README.md)
- 官方 Appendix（10 页，文件元数据创建于 2026-05-20）写“提供 additional implementation and architectural design details”，也写 full implementation will be available upon publication。两句话与当前已发表状态共同说明：**有可读补充材料和 demo，并不等于训练/完整复现实验已释出。** [Appendix](https://github.com/Gi-gigi/TrackNetV6/blob/main/assets/Supplementary/Appendix.pdf)

### 附录的可读机制事实

Appendix 将网络描述为：VGG-style multi-scale encoder，DDF（Discrete Wavelet Dynamic Fusion）给 decoder 初态；以 nmODE 为连续形式、以 Linear Multi-Step predictor--corrector 在**网络尺度层级**做 top-down decoding；CSCP 聚合多级 encoder features，PICC 进行轻量 residual/context correction。它对“decoder 如何跨空间尺度融合”给出了公式和更新阶数，但这不是跨**视频帧**的显式 correspondence、cost volume 或 optical flow。它也没有给出“无可靠匹配时拒绝输出”的变量/监督。

公开 demo/model 源进一步只证明其 demo 路径：首个卷积接收 **9 channels**，模型输出 **3 heatmaps**；demo 将连续视频切成**互不重叠的三帧组**，分别送入模型后把三张热图配回这三帧。([model](https://github.com/Gi-gigi/TrackNetV6/blob/main/TrackNet-main/models/TrackNetV6_Beta.py), [demo](https://github.com/Gi-gigi/TrackNetV6/blob/main/TrackNet-main/demo.py)) 这可以证实 demo 的 3-in/3-out frame stacking；不能证明主文训练也使用同一分组、同一边界策略或同一 causal latency。

Appendix 的补充表使用 Acc./Precision/Recall/F1 和 FPS，比较 V1--V4、WASB、TOTNet、BlurBall 等；它报告本方法与 V4 的作者设定差异。但没有在可读补充页定义用于本项目复现所必需的**中心匹配容差、heatmap 解码、可见性/遮挡标签、数据划分、损失公式、输入输出目标帧和计时边界**。这些继续标为未知，不能用附录数字排本项目的名次。

### 对当前三帧因果设置的含义

若只使用当前公开 demo，三张连续帧被一起取得：对组首/组中输出分别有两帧/一帧 future，组尾才没有 future；而且窗口不重叠。它与本项目每一 `t` 只用 `[t-2,t-1,t]` 预测末帧一次的协议不同。这个观察只说明 **demo 不可作为因果公平对照**，不应归因于 V6 论文主模型。

附录里所谓的 “memory/trajectory” 是 decoder cross-scale state `y_i`，不是跨时间帧传递的 tracker memory；不能把 nmODE/LMM 的尺度历史误写成已解决球的 long-range motion correspondence。

## 与本项目有关的明确研究边界

| 可能表述 | 本次证据后的正确写法 |
|---|---|
| “TrackNetV2/6 没有 motion。” | 不成立。两者至少都是多帧 frame-stack 的时序定位；V6 demo 还是 3-in/3-out。更窄的事实是：当前可读来源没有给出在高分辨率细球证据上做显式大位移 correspondence search 的机制或分桶验证。|
| “V6 的最新完整代码已覆盖，因而可排除。” | 不成立。当前只有 demo/weights/Appendix，训练和完整 inference 未释出；正文不可读。它只能作为相关工作边界，不能作为已重训或已完整审查的直接比较。|
| “V6 证明 wavelet / multiscale decoding 是 ball motion mechanism。” | 不成立。其可读材料的主要机制是 decoder 跨尺度语义融合和 frequency prior；没有可读证据把这等同为对象间帧匹配、物理位移或可靠性。|
| “三帧 3-in/3-out 就是因果。” | 不成立。MIMO 的每个 output 可能有不同 future exposure；V2 原文目标时序未知，V6 demo 对组首/中有 future。后续所有表格必须声明预测 slot 和前瞻。|

## 仍未知与下一步取证，不代替当前训练

1. **V2。** 只有在获得 IEEE 合法正文或作者公开 accepted manuscript 后，才补精确架构层数、WBCE 公式、训练/测试 match split、三输出对齐、可见性监督和 P/R 口径；不要从第三方 code 倒推。
2. **V6。** 只有获得 ACM 正文或作者版后，才把 Appendix 的 CSCP/PICC/DDF 与主文方法一一对应，并核对三帧时间顺序、训练监督、loss、所有 benchmark split、解码容差、速度计时边界及 demo 与正式模型差异。
3. **TOTNet。** 本次没有发现自 2026-09-09 后会改变既有“3D temporal + visibility/occlusion supervision、但遮挡点是时间推断标签”判断的一手更新，故不重复阅读或改写它。
