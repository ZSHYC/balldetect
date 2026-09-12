# 体育高速微小球：方法、数据与协议核验笔记

**范围与截止日：** 仅核验项目蓝图中体育球定位/跟踪链（TrackNet 1--6、WASB、TTNet、TOTNet、BlurBall、LaTBT/`TrackFormer`、RacketVision，以及 TrackNet Tennis / Shuttlecock / OpenTTGames 数据）。检索与网页读取完成于 **2026-09-09**。证据优先级为已读原论文正文（P）、作者/官方数据页或仓库（R）；未下载数据、未运行代码、以及只有摘要的信息明确标为未核验（U）。这不是穷尽综述；额外以 `"sports ball tracking"`, `"tiny ball tracking"`, `"ball detection" + 2025/2026 + arXiv` 检索最新近邻，结果只保留五个与本题直接相关的条目。

**2026-09-12 补充：** [V2/V6 原始来源补证](2026-09-12-tracknet-source-gaps.md)已取得 V2 官方摘要和 V6 十页官方附录，并核对 demo。V6 附录的状态演化主要沿网络尺度，不能等同视频轨迹；其主文、完整训练协议仍未取得。以下表格保留首轮阅读范围，V2/V6 的当前证据边界以该补充为准。

## 可直接用于立项的结论

1. **不能用“TrackNet 没有 motion”立题。** TrackNet-1 已用连续帧学习轨迹模式；V3 用八帧、多帧热图、背景和轨迹修复；V4 明确使用帧差生成 motion attention；V5（预印本）进一步将无符号差分改为带极性的方向场；V6 已自称面向 fast-moving tiny ball。可成立的缺口只能是更窄的：这些方法没有直接证明“在高分辨率、真球仍可读的细特征上，以受控预算完成大位移、多假设 correspondence search”这一因果链。
2. **网球主实验必须同时保留两个名字不同的协议。** 原始 TrackNet 数据集没有标准 train/test split；`tennis_game_level_split` 与 `tennis_clip_level_split` 是 TrackNetV4 官方仓库后来定义的复现实验协议，不是原始论文的“官方划分”。前者按 game 隔离，适合作为主报告；后者只保证 clip 不跨集合，仍可让同一 game 的视觉域进入训练和测试，不能用来主张比赛/来源泛化。WASB 的 Table 1（PDF 第 7 页） 已明确其所用 tennis manifest 为 7 games/65 clips/14,160 train frames、3 games/30 clips/5,675 test frames，共 **19,835**。
3. **TrackNet Tennis 的标签语义本身会污染 motion 结论。** 原论文规定拖影帧坐标是球轨迹的“latest position”，并会借助邻帧为难见/遮挡球估计位置。因此该点不全是当前帧的直接视觉测量；评估应按 visibility class 分层，并把“当前视觉证据定位”和“与发布的时间推断标签一致”分开。
4. **OpenTTGames 不能作为全视频密集定位真值。** 官方页明确其是 event-centred：每个事件前 4、后 12 帧才有球坐标/分割标注。未出现的 frame key 是未知，不是负例；训练时可把它当 context，但不能计算球损失、召回或假阳性。真实帧号、视频文件边界和事件窗口去重必须进入 manifest。
5. **BlurBall 是最合适的“模糊运动”诊断集，但不能把其 `theta,l` 当帧间速度真值。** 它将位置定义为 blur centre，并令 `l` 为沿 PCA 主轴的**半**长度；该方向/长度描述的是曝光期内的 streak。论文的基线还明确说其 frame-difference blur 提取只适用于静态相机。它支持按 blur 分桶、辅助标签和标签定义消融，不支持未经快门标定的真实速度或跨帧 correspondence GT 声称。
6. **WASB 是必须先跑的强空间/训练/后处理对照。** 它同时包含 high-resolution extraction、position-aware training 和 temporal-consistency inference，不能把它缩写成“普通 CNN 空间基线”。任一新 motion 分支必须与 WASB 风格高分辨率、输入像素和后处理公平对齐，否则收益不能归因给 motion。
7. **TOTNet 已覆盖“保留时间维 + visibility/occlusion supervision”的叙事。** 正式 CVIU 版保留 temporal dimension、用 3D convolution、visibility-weighted BCE 和遮挡增强，并在完全遮挡帧上把标注坐标照样变成 Gaussian supervision。它是“轨迹补全/遮挡预测”强近邻，却不等于当前帧直接 correspondence；本项目若不做遮挡主线，应以可见帧和视觉对应为核心，单列或排除其推断标签。
8. **最新资源的可复现状态不均衡。** V5 仍是 arXiv 预印本；V6 的 ACM DOI 可证实 ICMR 2026 出版，但其官方仓库同时写“paper not yet published、仅 demo、无训练/完整推理”，显然是未更新 README，不能把仓库 demo 当论文实现。LaTBT/作者称 TrackFormer 的 README 写 code `coming soon`、数据不完整，不能列为首轮可运行基线。RacketVision 已有静态文件和显式 train/val/test manifest，但其球标注是每 clip 均匀抽取的 20%，不是稠密轨迹 GT。

## 已读正文的体育方法证据

| 方法 | 发表/状态（截至截止日） | 已核验机制 | 会改变本项目的审查结论 |
|---|---|---|---|
| [TrackNet (AVSS 2019)](https://arxiv.org/abs/1907.03698) | P，正式会议论文 | Sec. III--IV：VGG/Deconv heatmap；比较单帧与三连续帧输入；原文说连续帧用于学习 trajectory pattern。Sec. III 的 VC=0/1/2/3；Sec. III 明说模糊拖影取 latest position。 | 不得声称首创“连续帧中的球运动”。它是合理的 stacked-frame/heatmap 基线，且标签定义需原样保留。 |
| [TrackNetV2 (ICPAI 2020)](https://ieeexplore.ieee.org/document/9302757) | 正式论文入口；本轮直接读取的是作者数据说明 R | 后续 TrackNetV3 正文（Related Work）概括其 U-Net skip connection 与 weighted BCE；作者数据页给 rally、`Frame/Visibility/X/Y` 格式。 | V2 是定位网络/多帧监督而非 explicit correspondence。论文方法细节需在正式 PDF 二次摘录，不能从第三方同名 PyTorch 代码反推。 |
| [TrackNetV3 (MMAsia 2023 PDF)](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf) | P | Sec. 1--3、Fig. 3：输入连续帧和 estimated background；8-frame tracking heatmaps；mixup；以预测轨迹和 mask inpaint/rectify 失检位置。Sec. 3.1 利用 match/固定 court-view 的 median background。 | 它已把背景抑制和后验 trajectory repair 用于此任务。与本项目的可见帧 visual correspondence、相机运动条件不同；若引用其遮挡结果，要说明它含未来/双向修复的可能性，而非实时逐帧检测。 |
| [TrackNetV4 (ICASSP 2025)](https://arxiv.org/abs/2409.14543) / [官方代码](https://github.com/TrackNetV4/TrackNetV4) | P + R | Sec. II-B、Fig. 2--4：frame-differencing 经 motion-prompt layer 形成 motion-attention，再与 high-level visual feature element-wise multiply。论文明确讨论 original vs **absolute** difference；实验 Table I/II 对 V2/V3 和自建多球 tennis。 | V4 的 motion 是“哪里变了”的 attention，不是远距离 correspondence；但任何“显式差分/attention”都必须纳入 baseline。若新方法声称方向性，最少与 V4 无符号差分、V5 有符号极性同时比。 |
| [TrackNetV5 (arXiv:2512.02789)](https://arxiv.org/abs/2512.02789) | P（预印本 v4，2026-01-13；无正式 venue 已核验） | 摘要与 v4 正文说明：MDD 将 temporal dynamic 分解为 signed polarity fields；R-STR 用 factorized spatio-temporal context 预测 corrective residual。 | 是直接的“差分丢方向”近邻；不能把 signed motion cue 作为新颖点。因其尚是预印本，数值只能标“作者报告”，不能与已复现系统同等强度引用。 |
| [TrackNetV6 (ICMR 2026 DOI)](https://dl.acm.org/doi/10.1145/3805622.3810690) / [官方仓库](https://github.com/Gi-gigi/TrackNetV6) | 出版状态由 DOI/会议元数据支持；机制正文 U；仓库 R | 官方 README 称自己是 ICMR 2026；但同时明确仅 visualization demo + pretrained weights，训练和 complete inference pipeline 未释出，且称 demo 可能不同于论文。 | 可做 related-work 条目，不能列为可公平重训的主 baseline。应记录论文 PDF/补充材料版本；README 的“not published”与 DOI 矛盾，视为仓库过时而非否定发表。 |
| [WASB (BMVC 2023)](https://papers.bmvc2023.org/0310.pdf) / [官方代码](https://github.com/nttcom/WASB-SBDT) | P + R | 摘要明确三部分：high-resolution feature extraction、position-aware training、temporal-consistency inference；Table 1（PDF 第 7 页） 的 Tennis 为 train 7 games/65 clips/14,160 frames、test 3 games/30 clips/5,675 frames、total 19,835。它在五项运动数据、六种既有方法上做统一重实现比较。仓库说明已加入 DeepBall/DeepBall-Large/BallSeg/TrackNetV2/ResTrackNetV2/MonoTrack evaluation。 | 19,835 是 WASB 已确认的处理/协议统计，20,844 是 TrackNet 原文主实验统计；二者是不同论文/数据处理口径，不能互相判错。它也是最低强度空间控制组；主表要分“原作者报告”“WASB/BlurBall 工程复现”“本项目重训”。 |
| [TTNet (CVPRW 2020)](https://arxiv.org/html/2004.09927) | P | Fig. 2、Sec. 5/5.1：9 帧 downscaled full-HD 的全局 coarse ball heatmap，使用其位置回到原 full-HD 进行局部 crop/refinement；预测序列最后一帧的球。另有 temporal event spotting 与 ball detection/semantic segmentation；OpenTTGames test 报 2 px RMSE、97.5% ball accuracy。 | 蓝图“全局粗定位+局部精修、多任务”正确。它是直接的“低分辨率广域搜索 + 原分辨率局部细化”近邻；新方法必须说明为何其细特征远距 correspondence/multi-hypothesis 与 TTNet 的 coarse-to-fine 预算分配有可检验区别。 |
| [TOTNet (CVIU 264, 2026, 104657)](https://rbouadjenek.github.io/assets/pdf/YCVIU_104657.pdf) | P，正式期刊 PDF | Sec. 3.1/3.3--3.4、Fig. 4--5：fully occluded location 由前后可见观测的 trajectory-consistent interpolation 推断；2D spatial + shortcut 3D temporal convolution，temporal dimension 不按 channels collapse；visibility-weighted Gaussian heatmap；遮挡 augmentation。Sec. 4.3--4.5 明列在线/离线时间上下文和遮挡失败。 | 若论文评测 fully occluded position，必须对齐其发布的推断标签定义；该成绩可来自 temporal/contextual extrapolation，不能被归为“当前帧视觉检测更准”。本项目可选择 causal 或 offline protocol，但必须明示预测帧可见的时间上下文，且不与另一设定的数字横比。 |
| [BlurBall (CVPRW/CVSports 2026)](https://arxiv.org/html/2509.18387v2) / [官方代码与数据](https://github.com/cogsys-tuebingen/blurball) | P + R | Sec. 3.1/3.3、Fig. 1/5/6、Table 1--3：新标签把球定义为 blur centre；预测 ball location、orientation、extent；用 PCA 主轴，`l=(max projection-min projection)/2`；模型是 HRNet + SE、多帧输入。 | 要做两种 target convention 的标签消融：front-edge vs blur-centre。只训练定位时，`theta,l` 可作预注册困难分桶；一旦加入 blur auxiliary loss，必须用 2×2 证明收益不是新增监督。 |
| [LaTBT / 作者仓库称 TrackFormer](https://github.com/Gi-gigi/TrackFormer) | R；论文全文与正式出版 U | README 标题为 *Towards Highly Effective Moving Tiny Ball Tracking via Vision Transformer*，README 表示 code coming soon、仅提供部分数据。它不是 MOT 的通用 TrackFormer。 | 不可把它加入第一轮 reproducible baseline；可作为 ViT-tiny-ball related work 待论文/资源完整后重新核验。 |
| [RacketVision (AAAI-26 正式 PDF)](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324) / [作者数据卡](https://huggingface.co/datasets/linfeng302/RacketVision) | P + R | Table 1--2、Sec. Data Collection：435,179 video frames、64,042 ball annotations、24,621 racket annotations；942 games、1,672 clips；clip 为 5--10 s连续在 play 片段；每 clip 均匀人工标注 20%帧。数据卡有 raw clips、ball CSV 和 per-sport `train.json/val.json/test.json`，并分开 `interp_ball`。 | 是很有价值的跨三球种稀疏监督扩展，但不是密集 GT。`interp_ball` 是派生轨迹，不能当独立人工真值或和 CSV 原标注混算；正式论文 Table 1 报的是约 88k（ball+racket）而非 64k ball-only，避免统计口径错配。 |

## 数据、时间连续性、划分与许可

### 1. TrackNet Tennis

**已读证据。** 原文 Sec. III 写其主实验手工标了 **20,844** 个 2017 Summer Universiade 男单决赛帧，并另加 9 个视频的部分标签用于 10-fold CV。WASB Table 1（PDF 第 7 页） 则在其明确的 7-game/3-game protocol 下报告 **19,835** 帧（14,160/5,675）。两数是不同论文/处理口径，不能据此判定蓝图数字错误，仍应由本地 manifest 记录精确版本。同一 TrackNet 原文还明确每帧最多一个球，并给 visibility class：0 out of frame、1 easy visible、2 in-frame but hard-to-identify、3 occluded；难见和遮挡点会参考相邻帧估计，模糊拖影点取 latest trace position。

**切分事实。** [TrackNetV4 的 `DATASET.md`](https://raw.githubusercontent.com/TrackNetV4/TrackNetV4/main/docs/DATASET.md) 明说 “original dataset lacks a standardized training and test split”，随后定义：

* **game-level**：train `game2,3,5,6,7,8,10`，test `game1,4,9`，约 70.81/29.19；
* **clip-level**：按 cumulative frame count 把完整 clip 分至约 70/30，不重叠。

**必须写进 protocol 的约束。**

* 主结果：game-level split，验证集从训练 games 中按 game（或至少原始视频来源）另分，绝不从 game1/4/9 拿超参；报告随机种子。
* 辅结果：clip-level，只作为与 V4 复现可比的结果；禁止写成 domain/generalization evidence。
* 训练 window 必须在 `game/Clip` 内；任何 `Clip` 尾部样本丢弃或 pad，但不可从下一个 clip 填帧。
* 数字 **19,835（WASB Table 1（PDF 第 7 页） protocol）与 20,844（TrackNet 原论文主实验）** 已知为不同统计口径；下载后仍应以 manifest 的 image 数、CSV 行数、visibility 分布等元信息记录本项目所用版本，且不能把不同协议的 F1 直接排名。

### 2. Shuttlecock Trajectory Dataset / TrackNetV2

**作者数据页已读。** [作者 HackMD 数据说明](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw) 写 1280×720、30 fps、rally 为从 serve 到 score、总计 78,200 帧，并指定 `Frame, Visibility, X, Y`：不可见时 `Visibility=0`、`X=Y=0`。其目录也是逐 match/逐 rally 的 CSV、帧和 rally video，天然支持“不跨 rally 取窗口”。

**关键不一致，不能忽略。** 同一页面一处说 26 videos、23 professional（68,675 帧）+3 amateur（9,525 帧），但其下又列了 23 个 training professional match 和 3 个 test professional match；TrackNetV4 的作者数据适配文档又说数据“now contains 23 professional matches”，而 V2 paper “appears to use only 15”。这些陈述无法仅靠网页互相推出一个唯一 manifest。故：

* 不能把“26 个来源/78,200 帧/23+3 train-test”同时当已确认的不可变事实；
* 下载后按发行包的 `Professional/Amateur/Test`、match id、rally id 导出 frozen split；保留作者的 test matches，训练内按 **match**（不要按随机 frame）做 validation；
* 若要复现 V2，须复核 V2 PDF 的具体 15 professional matches；若要跨版本主结果，应该单独命名“current release match-level protocol”。

**许可：U。** 作者页面提供 SharePoint download，但本轮没有看见明确数据许可证；在公开代码/数据再分发前必须取得作者书面/页面许可，不能从“公开下载”推导可再发布。

### 3. OpenTTGames

**官方页已读：** [OSAI OpenTTGames](https://lab.osai.ai/) 是最强一手依据：5 个 10--25 分钟 train video、7 个 short test video；Full-HD industrial-camera 120 fps；4,271 个手工事件（bounce/net-hit/empty）。每个事件只有**前 4 + 后 12 帧**带 ball coordinate 与 segmentation mask，标注以 deep-learning-aided model 完成；ball markup 是 `frame_number -> {x,y}`，`(-1,-1)` 才明确表示 ball absence。

**因此应实施的最低数据规则：**

```text
decode_index = 原 MP4 帧号（绝不按可标注帧重编号）
supervise_position[t] = CSV/JSON 中 t 有合法坐标
known_absent[t] = CSV/JSON 中 t 明确为 (-1,-1)
unknown[t] = 没有 ball key（可作 temporal context，不作正/负 loss）
window = 同一 video 内的原始连续索引；跨 event window 重叠只去重评价样本，不改变时间
```

**许可：** 同一官方页写 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。对当前非商业研究可用的判断是合理的，但须 attribution、非商业和 ShareAlike 条款；数据、派生 annotations、代码、权重应分别维护许可表。不要只因 NC 排除它，也不要反过来声称代码/权重自动是 CC BY-NC-SA。

### 4. BlurBall

**已读证据。** [论文 Sec. 3.1--3.3、Fig. 1、Table 1](https://arxiv.org/html/2509.18387v2) 和作者仓库称其为约 64k real-game frames、带 ball position 与 blur length/orientation；数据经官方仓库链接的 Nextcloud 发布。论文以 26 个录制、64,119 帧、train 51,423/test 12,696 的统计出现于蓝图，但本轮网页正文未逐项复核该 split 表，故这些精确 split 数字暂标 **U，需下载 manifest 后核验**。

**标签解释：** ball centre 是新的目标定义；`theta` 是无符号主轴角还是带符号的真速度方向，取决于文件/论文约定，当前正文的 PCA eigenvector 只保证主轴，天然有 180° 歧义。`l` 是半长度。研究设计不应把它同相邻帧 displacement、光流或真实速度混为一项；分别报告 `blur-length bucket` 和 `frame-to-frame displacement bucket`。

**许可：U。** 代码仓库为 MIT，但这不自动覆盖数据视频/annotation；作者仓库没有在已读 README 明示数据许可证。下载/发表前记录 Nextcloud 的 LICENSE/README 和访问日期；不要再分发原视频，除非明确许可。

### 5. RacketVision（后续扩展）

**已读证据。** [AAAI-26 Table 2 / Sec. Data Collection](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324) 表明其 5--10 秒连续 play clip 符合 continuous-clip 边界，但只人工标 20% 均匀抽样帧；[数据卡](https://huggingface.co/datasets/linfeng302/RacketVision) 将原标注 CSV 与 `interp_ball` 明确分目录、并给出 train/val/test 文件。对本项目可做：连续 RGB window + 稀疏 target-frame loss；不可做：把 interpolation 视为人工 ball GT，或把无 CSV frame 视为无球。

**许可：** Hub dataset card 当前标作 MIT，仍建议在使用前下载并保存该版本的 README/commit（页面当前显示 `85157ca`），核对原 YouTube 视频与派生数据的再分发条件；该页面并非对每个源视频的权利链证明。

## 对蓝图现有表述的逐项修订建议

| 蓝图主张 | 审查结论 | 建议替换的严谨表述 |
|---|---|---|
| “TrackNet Tennis / WASB 协议 19,835 帧；WASB 14,160/5,675” | 已由 WASB Table 1（PDF 第 7 页） 确认：7 games/65 clips/14,160 train 和 3 games/30 clips/5,675 test。TrackNet 原文 20,844 是另一论文口径，不构成矛盾。 | “WASB tennis protocol: 19,835 frames（14,160/5,675）；同时在 manifest 写清与 TrackNet 原文 20,844 的口径不同。” |
| “TrackNetV4 的 game-level/clip-level split” | 方向正确，但来源性质需改。 | “TrackNetV4 作者仓库定义的两个复现协议；game-level 是主协议，clip-level 仅为历史可比。” |
| “TrackNet 没有 motion” | 已被所有版本反证，且 V1 也有 multi-frame trajectory pattern。 | “现有方法从 implicit stacked-frame 到 difference attention、polarity 和 temporal repair 都使用时序线索；尚未证明何时细粒度球证据在高分辨率 correspondence 前被破坏。” |
| “TTNet：全局粗定位+局部精修、多任务” | 已由 Fig. 2 和 Sec. 5/5.1 确认：9-frame downscaled global coarse localization，再对原 Full-HD crop 做 local refinement，并预测最后一帧球。 | “TTNet 是直接的 coarse-global / fine-local 近邻；本文需用候选覆盖、远距对应和相同预算实验说明差异。” |
| “BlurBall 的 `theta,l` 可测试高速 motion representation” | 可作分桶/辅助，但不能当对象帧间 velocity 真值。 | “按曝光期 blur 主轴与半长度分析；与实际标注相邻帧位移分开评测，并先固定 target convention。” |
| “OpenTTGames 未标注帧不能当无球；可用连续帧作输入” | 完全由官方页支持。 | 加入三态 mask（valid-pos / explicit-absence / unknown）及原帧号和 window boundary 单测。 |
| “TOTNet 完全遮挡点由前后观测和轨迹合理性推断” | 已由正式 PDF Sec. 3.1 确认：fully occluded location is inferred through trajectory-consistent interpolation from preceding/following observations。 | “完全遮挡评测衡量的是对发布推断轨迹的一致性，不能等同于当前帧的直接视觉测量。” |
| “TrackNetV6 完整训练/推理未发布” | 官方仓库支持；它的“paper not published”与 ACM DOI 冲突。 | “ICMR 2026 论文已可按 DOI 引用；截至核验日仓库仅 demo，不能做可重训基线。” |

## 最小、可审稿的实验协议补丁

1. **在模型前冻结 data manifest。** 每条样本记录 `dataset_release, source_video/match, clip_or_rally, original_frame_id, target_status, visibility_raw, coordinate_definition, split_id`。target status 至少是 `direct_visible / released_low_visibility / released_occluded_or_inferred / explicit_absent / unknown`；不要跨数据集硬把 visibility 数字同义化。
2. **三套不可替代的诊断。** (a) 可见帧定位；(b) 两帧均有可信位置标签时的 **像素 displacement** 分桶（同时报告原分辨率和网络输入分辨率）。当前核心数据没有已发布的逐帧 ball diameter，不能把它伪归一化成 `rho=displacement/ball-size`；BlurBall 的 `l` 是拖影半长度，也不是球直径；(c) BlurBall 的 blur-length 分桶。三者不能合成一个“motion difficulty”。只有获得已发布、可追溯的球大小标注时，才额外报告 rho。
3. **公平基线矩阵。** 同一输入分辨率、decoder、训练 token/window、后处理下，至少比较 single frame、stacked frames、absolute difference/V4-style attention、WASB-style high-resolution；新 correspondence 模块与“新监督/foreground loss”做 2×2。V5 和 TOTNet 作为外部近邻，不可复现时报告作者数值而非伪重训。
4. **在线/离线分开。** 每个结果注明预测帧能见到 `t-k...t` 还是也能见到 future；V3 rectification 和 TOTNet offline 都可能改变问题定义。研究可选择 causal 或 offline 主协议，但必须预先固定、全基线一致，并分别报告两种设定而不横比。
5. **不要提前扩大到相机运动。** 主证据可先限定固定/近固定机位的 TrackNet/BlurBall/OpenTTGames；V3 的 median background、BlurBall 的 static-camera difference baseline 都对连续相机运动脆弱。以后若加摇摄/变焦，应单独报告 camera-motion 标记或估计误差，不能把 fixed-camera 成绩外推。

## 2025--2026 补充近邻（最多五项，非穷尽）

| 条目 | 状态与一手链接 | 为什么需要纳入，而非直接作为主基线 |
|---|---|---|
| [TOTNet](https://arxiv.org/abs/2508.09650) → [CVIU 2026 正式 PDF](https://rbouadjenek.github.io/assets/pdf/YCVIU_104657.pdf) | 2025 arXiv，2026 CVIU；上述已详读。 | 已覆盖 temporal 3D feature、visibility loss、occlusion augmentation；是最近且能改变创新边界的主近邻。 |
| [BlurBall](https://arxiv.org/abs/2509.18387) → [CVPRW 2026 代码](https://github.com/cogsys-tuebingen/blurball) | arXiv v2 2026-02，CVSports/CVPRW 2026。 | 已覆盖显式 blur label 和 HRNet multi-frame; 其 workshop 身份不能被写成 CVPR 主会。 |
| [RacketVision](https://arxiv.org/abs/2511.17045) → [AAAI-26 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324) | 2025 arXiv，AAAI-26。 | 直接提供跨网球/羽毛球/乒乓球的连续片段与稀疏标签；它研究多模态球拍/轨迹预测，不替代密集检测评测。 |
| [TrackNetV5](https://arxiv.org/abs/2512.02789) | 2025-12 arXiv preprint（v4 2026-01）。 | 最近的 polarity/temporal-refinement 近邻；尚无本轮核验的正式 venue。 |
| [TrackNetV6](https://dl.acm.org/doi/10.1145/3805622.3810690) / [repository](https://github.com/Gi-gigi/TrackNetV6) | ICMR 2026；仓库仅 demo。 | 标题直接占据“lightweight robust fast-moving tiny ball tracking”空间，必须读论文/补充材料；当前不作为可复现主实验依赖。 |

**检索遗漏的边界：** 以上 Web/arXiv 检索没有发现另一篇截至 2026-09-09、同时满足 *RGB 单目、逐帧 tiny-ball localization、连续体育视频、公开可读且比上述更直接* 的 2025--2026 方法；这不证明不存在。投稿前应以 Semantic Scholar/Google Scholar、CVPR/ICCV/ECCV/AAAI/ACM DL 及关键词 `sports ball`, `racket`, `fast tiny object`, `video localization`, `motion blur` 复跑检索，并保存 query、日期和筛选理由。

## 未解决问题与下一轮取证清单

1. 下载每个数据包后核验：实际文件/CSV 行数、split、原始 frame index、缺失标注、坐标范围、visibility code、license 文本。此步骤会固定本项目的 TrackNet manifest（而非把 TrackNet 20,844 与 WASB 19,835 误当冲突），并消除 badminton 26/23/15、BlurBall 64,119 split 等不确定项。
2. 取得 TrackNetV2 原始 PDF，摘录它实际使用的 match split、输入/输出帧数、visibility target；不要让当前 HackMD 新版本替代历史论文协议。
3. 对 TTNet 的 Fig. 2/Sec. 5/5.1 已确认其 global coarse + full-HD local refinement；后续应完整摘录其 crop 尺寸、训练/推理条件和后处理，作为严格预算对照。
4. 读 TrackNetV6 PDF/supplement 和 TrackFormer/LaTBT 原论文（若公开），区分论文贡献、demo 代码和数据可用性。
5. TOTNet 已确认 fully-occluded coordinate 来自前后观测的 trajectory-consistent interpolation；仍须在数据使用时保存该 provenance，并且不让这些点充当当前帧 visual-correspondence ground truth。
