# DeepPro 的 temporal profile：固定像素长窗检测，不是跨位置 correspondence

日期：2026-09-11。范围：只补读 Ruojing Li 等的 *Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better*（DeepPro）；不复现、不下载数据、不把红外结果外推成体育球结论。本笔记修正[高速微小目标 motion 证据](tiny_motion_evidence.md)中 DeepPro 的摘要/README 级条目，并与[本项目的三帧完整微调对照](../experiments/2026-09-11-full-temporal-control.md)核对时间语义。

## 可直接使用的判断

DeepPro 是一个**固定空间坐标上的长时间信号异常检测器**：对每个 `(x,y)`，把该坐标在 `T` 帧中的特征向量作为 temporal profile，用可学习的 `T×T` 矩阵沿时间混合；它不估计 `u_t → v_{t-Δ}`，不作 warping、光流、offset、相机运动补偿或跨位置搜索。目标移动时，固定像素会经历一次强度的上升—下降脉冲；作者利用的是这个脉冲的形状/时序统计，而非“同一目标在不同像素处的外观对应”。论文明确把结果表述为每个输入帧的置信图，非一个指定目标帧的中心预测。

因此它是“高分辨率 correspondence 并非唯一效率路线”的有效**任务邻域反证**：若运动目标在固定像素留下可分辨的时变响应，便宜的 time-only profile 可以增强检测，复杂空间搜索未必必要。要分清两件事：大位移确实不会在同一像素保留“同一目标的连续外观”，所以 DeepPro 不能建立 object correspondence；但这不等于大位移妨碍**检测**。球快速穿过当前像素仍会留下短暂变化，论文的 toy analysis 恰称同尺寸下速度越高脉冲越窄（[§3.1.1，PDF p.4](https://arxiv.org/pdf/2506.12766v5)）。因此不能因它没有 displacement 输出就排除 fixed-pixel time-change/profile baseline。相机、球员、线条、高光和压缩伪影也会经过/改变该像素，可能产生竞争脉冲；论文只在红外数据上测到了 profile detection 的最终收益，并未在体育球上验证上述推论。它**不**反驳体育球在相机跟随/变焦、RGB 纹理背景或只有短因果历史时对跨位置证据的需求，也不能支持“DeepPro 已解决大位移对应”或“体育球会同样受益”。

## 来源、版本与阅读边界

* 正式书目信息已可核验：IEEE TPAMI **48(8):10157–10175 (2026)**，DOI [`10.1109/TPAMI.2026.3683258`](https://doi.org/10.1109/TPAMI.2026.3683258)；[PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/41973585/)标为 2026-08、`ppublish`。作者个人页也称 2026-04 accepted。这取代了旧笔记中“作者仓库给出 TPAMI citation，未复核”的保守状态。
* 已完整阅读的开放正文是 [arXiv:2506.12766v5 PDF（2026-03-27）](https://arxiv.org/pdf/2506.12766v5)，其 [arXiv 记录](https://arxiv.org/abs/2506.12766)列出 v1 2025-06-15 至 v5 2026-03-27。通过本轮通道未取得 IEEE 出版页全文；故下面的公式、表与段落以 arXiv v5 §3–4/附录为准，**未声称其逐字或逐表等于 TPAMI 排版本**。
* 已读作者公开仓库的固定提交 [`TinaLRJ/DeepPro@8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`](https://github.com/TinaLRJ/DeepPro/tree/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28)（2026-07-15；没有 release/tag）。这是源码行为证据，不能替代期刊实验结论。该提交 README 仍写“under peer review”，与正式 TPAMI 状态冲突；README 的该状态已过时。仓库的模型/日志文件名多含 2024 时间戳，且没有把每份提交与论文 v5/期刊版逐一对应的说明，故论文与当前源码分开记录。

## temporal profile 的精确定义与机制

论文 §3.1 将 temporal profile 定义为**固定检测单元/固定空间位置**随时间的灰度或特征变化。其 toy model 假设目标短时近似匀速，目标经过该位置时强度先升后降；脉冲幅值关联亮度、宽度关联尺寸和速度（[§3.1.1，PDF p.4](https://arxiv.org/pdf/2506.12766v5)）。这是一种对固定位置观测的统计描述：目标真中心在帧间换到哪里，并不是输入或输出变量。

对基础特征

\[
F_b\in\mathbb{R}^{C\times T\times H\times W},\qquad
F_b(x,y)\in\mathbb{R}^{C\times T\times1\times1},
\]

TPro 在每个 `(x,y)`、每个被分出的 channel group 上乘一个可学习的 statistics-based correlation matrix（SCorM）`W_i∈R^{T×T}`：

\[
F^{cor}_i(x,y)=F_b(x,y)W_i^{T\times T},\qquad
F_{tp}=\phi([F^{cor}_1,\ldots,F^{cor}_m]).
\]

其中 `φ` 是 `1×1×1` 卷积、BN、ReLU；论文的[§3.3、Eq. (8)–(9)，PDF pp.7–8](https://arxiv.org/pdf/2506.12766v5)说明了这一点。矩阵是“不同时间索引间的可学习线性混合/统计关系”，不是由两个像素特征相似度动态生成的 cost volume；没有输出 match probability、displacement 或 correspondence address。

当前源码与该解释一致：[`TPro.py:L29–41`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/networks/layers/TPro.py#L29-L41)把每个 `h,w` 的长度 `slen` 向量送进 `nn.Linear(seqlen,out_len)`，没有空间采样/warp；[`L18–26`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/networks/layers/TPro.py#L18-L26)显示多头线性层与 `1×1×1` 融合。这里的“correlation”应只读作作者学习的时间关系权重，不能升级为可验证的物理或对象对应。

## 空间路径、输出与监督

**基础 DeepPro 不是预训练 backbone。** 原始单通道序列先经 `5×1×1` temporal conv 与 temporal-difference residual blocks；三条 level 分别在原空间尺度、一次 max-pooling、两次 max-pooling处生成基础特征，分别作 TPro，再上采样、拼接、`1×1×1` 融合为每一帧的置信图。论文[§3.4，PDF pp.8–9](https://arxiv.org/pdf/2506.12766v5)明确最终为 `C∈R^{T×H×W}`，即每个输入帧一张 segmentation confidence map；作者源码的[完整 forward](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/networks/models/DeepPro.py#L17-L65)也显示 `pool → TPro → interpolate → concat → 1×1×1` 的顺序。

“only calculations in the time dimension”应按论文所说的卷积/乘加主张窄读：基础 DeepPro 没有空间卷积或显式空间对应，但源码确有空间 max-pooling 和双线性上采样，不能把它写成整个程序完全没有空间操作。`DeepPro-Plus` 是另一个版本：在 TPro 前改用 spatial-temporal difference/conv，只保留第一 level；论文[§4.5，PDF p.13](https://arxiv.org/pdf/2506.12766v5)和源码[`DeepPro-Plus.py:L16–36`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/networks/models/DeepPro-Plus.py#L16-L36)均可核对。它不能被混称为 time-only DeepPro。

训练目标是**逐像素 mask**：论文使用 Soft-IoU；对于 RGBT-Tiny，作者由 bbox 生成粗 Gaussian mask（[§4.1.3、§4.6，PDF pp.9、14](https://arxiv.org/pdf/2506.12766v5)）。当前源码将标签二值化为 mask，并在 `T` 张预测图和 `T` 张 mask 上算 Soft-IoU（[`TrainDataLoader.py:L51–60`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/data_utils/TrainDataLoader.py#L51-L60)，[`DeepPro.py:L113–130`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/networks/models/DeepPro.py#L113-L130)）。未见 GT center displacement、flow、轨迹 ID、bbox regression 或 correspondence loss。评测主指标是 `Pd/Fa/AUC`，不是逐帧中心误差（[§4.1.2，PDF p.9](https://arxiv.org/pdf/2506.12766v5)）。

## 时间窗口、未来帧与窗口边界

论文默认 `T=40`、`m=4`，输入为 40 个连续帧；末尾不足则补全零图。推理以 10% 重叠窗口，重叠帧的结果作 union（[§4.1.3 与脚注 5，PDF pp.9–10](https://arxiv.org/pdf/2506.12766v5)）。长度消融为 `T=5,10,20,40,60,80`，作者选择 40 作为性能/规模折中（[Table 5，PDF p.12](https://arxiv.org/pdf/2506.12766v5)）。

这不是已定义目标帧的因果在线协议。论文输出每个输入时刻的图，且 `T×T` SCorM 允许任一输出时刻读取同窗所有输入时刻；当前 TPro 源码的全长 `Linear(T,T)`也没有 causal mask。固定提交的测试 loader 以 `T=40`、重叠 `4` 形成一个 sequence 内的窗口（[`TestDataLoader.py:L154–164`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/data_utils/TestDataLoader.py#L154-L164)），测试脚本再以逐元素最大值实现论文所称 union（[`test.py:L129–162`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/test.py#L129-L162)）。因此：

* 对窗口内非末帧的输出，结构上可使用右侧未来帧；起始位置最多有 `T−1=39` 帧前瞻。重叠 union 还能使部分边界帧依赖后一个窗口的更远未来。论文/源码都没有报告“只取末帧、无未来帧”的结果。
* 它是有限 40 帧块处理，非跨调用递归状态或无限历史。当前 loader 在每个 `seq_name` 内建样本/窗口（训练：[L124–169](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/data_utils/TrainDataLoader.py#L124-L169)）；没有 scene-cut、rally 或相机运动检测逻辑。红外序列边界不能替代本项目必须执行的 clip/rally reset。
* 论文称训练随机选起始帧；固定源码训练样本实际上从每个序列的连续已排序帧构成并随机抽一个窗口，短窗口才零填充（同上）。两者都不定义体育视频稀疏标注、PTS、切镜或跨 rally 的语义，不能直接搬用。

## 相机/背景运动：观察到适应，不等于建模或对齐

论文在 RGBT-Tiny 的定性案例中说有 dynamic background/camera movement，并称一个示例中 DeepPro-Plus 与 DTUM 能适应背景变化（[§4.6，PDF pp.14–15](https://arxiv.org/pdf/2506.12766v5)）。这只支持“该数据/定性例中 final detection 仍可工作”；没有全局运动参数、光流、homography、背景 mask、对齐误差或 camera-motion bucket，也没有证明固定像素 profile 在任意移动相机下保持目标信号。相反，论文自己列出遮挡造成不完整 profile、长时间静止目标没有规律 profile 的漏检（[§4.7，PDF p.15](https://arxiv.org/pdf/2506.12766v5)）。

所以对于球赛，镜头跟球使球停在近似固定像素、平移使背景扫过固定像素、变焦改变球支撑和经时脉冲宽度，三者都直接改变其假设；这应作为待测 domain shift，而非“有 camera movement 案例”即可消除的风险。它也说明为何 fixed-pixel 变化/1D profile 仍应是对手：它可在**不声明位移对应**的前提下检验“当前像素的时间变化已足以发现球”这一竞争解释。这个判断目前是针对体育球的待验证假设，不是 DeepPro 的实证结论。

## 成本数字能说明与不能说明什么

论文在 `256×256` 输入上给 `GFLOPs per frame` 与 FPS；其 Table 2 里 DeepPro 是 `0.197M` 参数、`184.55 FPS`，DeepPro-Plus 为 `0.284M`、`3.89 GFLOPs`、`224.05 FPS`（[§4.1.2、Table 2/8，PDF pp.9、13](https://arxiv.org/pdf/2506.12766v5)）。这说明在该论文的**离线窗口吞吐**设置里，它远轻于所列 IR baselines；不说明逐帧因果延迟。

当前作者 README 给的是不同的 `480×720` 表和 DeepPro `155.40 FPS`，而固定测试脚本只对完整 `(1,1,T,200,300)` 输入 profile，并打印 `FLOPS for T frames`；其实际 wall-clock FPS 输出还被注释掉（[`README`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/README.md)，[`test.py:L196–210`](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/test.py#L196-L210)）。故不能把任一数字写成本项目端到端实时 FPS，也不能声称其计时已包含 40 帧等待、视频解码、窗口合并或目标帧的前瞻 latency；源码没有提供该计时分解。

## 论文内部对照实际隔离了什么

| 对照 | 已经隔离的因素 | 未隔离/不能据此宣称 |
|---|---|---|
| DeepPro `w/o TPro` | 同一模型位置将 TPro 替换为同组数 `1×1×T` conv，说明在该架构内 TPro 形式优于该长时间卷积替代（[§4.3.1，PDF pp.11–12](https://arxiv.org/pdf/2506.12766v5)）。 | 不是 no-temporal 对照；二者都用长窗。论文未给这个替代的参数/FLOPs 匹配表，不能将差异全归于“correlation”而排除容量/算子差异。 |
| `T=5…80` | 报告长窗增大与最终检测、参数、GFLOPs、FPS的联合变化。 | T 增大时 SCorM 同时从 `T×T` 改变，参数和 GFLOPs 也变（Table 5）；它不是固定特征改造/固定参数下仅改变时间长度的隔离实验。 |
| DeepPro vs DeepPro-Plus | 说明在该数据上加入少量空间信息的 Plus 版本可提升最终 mask 指标。 | Plus 同时把所有 temporal conv 换为 spatial-temporal conv、从三 level 留一 level；不是只加空间特征的单变量消融。 |
| 与既有 SF/MF 方法的表格比较 | 给出 IR mask detection 与论文规定 256²吞吐的系统结果。 | 未统一各法的目标帧、前瞻、窗口长度、训练重实现或逐帧等待；不能作为“等 temporal budget/同因果性/同 backbone”的特征改造归因。 |

## 对本项目短因果三帧完整微调的约束

当前实验是 `[t−2,t−1,t]→t` 的严格短因果、完整 DINO 微调，且重复当前帧 `[t,t,t]` 保持三槽位/读出/初始化不变。DeepPro 的证据不能直接与其性能或效率横比，至少有以下不同前提：

1. **时间可用性不同。** DeepPro 的默认 40 帧 all-output/全时间矩阵与 union 可有未来；本项目目标帧没有未来、只有两帧历史。不能把 DeepPro 的长窗结果当作三帧因果结果的速度或精度基准，更不能写“我们三帧已覆盖 DeepPro 式 long-term profile”。
2. **任务/监督不同。** DeepPro 学红外 mask 与 `Pd/Fa/AUC`；当前任务是 RGB 球中心/存在与位置指标。中心标签无法把 DeepPro 的固定位置脉冲解释为 flow 或跨像素真对应，也不应把其 mask 结果称为球中心定位证据。
3. **表示与计算不同。** DeepPro 没有预训练视觉 backbone、宽域 search 或 correspondence；当前系统的 DINO 三帧 feature stack 也还不是 profile probe。仅证明 real history 相对 repeat-current 的净增量，不能声称是“temporal profile”、“长时统计”或“替代 correspondence”的收益。
4. **效率口径不同。** 本项目须分别记录 decode/feature backbone/head、三帧历史及无前瞻的真实 latency；不可引用 DeepPro 的 per-frame FPS 或参数量来声称本系统高效。

目前不能宣称的新增 motion/效率表述包括：“首次用固定像素长时 motion profile”、“无需 spatial/correspondence 的高效运动定位”、“long-term temporal profile 天然适用于大位移球”、“与 DeepPro 同等实时”，以及“论文已证明相机运动下 profile 稳健”。若后续结果支持进入该路线，再用同一球数据、片段边界、目标帧、监督和计时口径检验固定像素变化是否已能解释收益。当前因果任务应先选择一个必要的短历史对照；只有研究问题实际需要未来帧时才另立固定前瞻协议，不能因本次补读同时启动多条baseline。某个短因果实现若不能超过current-only，只能否定其当前设置下的增量，不能否定所有temporal profile。固定前瞻若有效，则连同等待延迟报告为offline结果。

## 未取得与可复用结论

未取得 IEEE 版可访问全文，也未复现官方代码、核验作者数据、逐层比较 arXiv v5/TPAMI 排版或测量真实 GPU 速度；没有源码/论文证据证明训练、测试的每个窗口都遵守本项目所需的 clip 边界。仓库 README 的审稿状态与正式记录冲突已标出，其他论文—代码差异不猜补。

可在 related work 中准确复用的窄结论是：**Li et al.（TPAMI 2026）已把 moving IR small-target detection 作为固定空间位置的长时间 profile anomaly detection，并用全时间 `T×T` profile mixing 取得高离线窗口吞吐；这排除“复杂高分辨率 correspondence 是利用运动的唯一高效形式”。它没有建立跨位置对象对应、因果三帧球定位或体育相机运动适用性；这些仍需在本项目协议下实测。**
