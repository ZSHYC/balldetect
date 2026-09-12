# KeepTrack 深读：候选关联、相似干扰物与持久状态的适用边界

**结论先行。** KeepTrack 是当前“若未来模型显式维护候选身份、跨帧状态或在线更新”时必须面对的直接近邻：它把同一局部搜索区内的多个目标候选（目标和干扰物）跨相邻帧关联，并用关联置信度控制在线分类器的记忆。它**不是**自动逐帧球检测/定位方法：输入含首帧目标 GT 框，随后在先前状态附近搜索；论文中的长时跟踪、重检和记忆去污染结论不能直接外推到当前 BlurBall 无跨窗口持久状态的三帧自动定位基线。

- 论文：Christoph Mayer *et al.*，**Learning Target Candidate Association to Keep Track of What Not to Track**，ICCV 2021，pp. 13444–13454。本轮采用 CVF 版本；arXiv:2103.16556 首次提交于 2021-03-30，版本页截至本次访问的最新版本为 v2（2021-08-18）。
- 一手论文：[CVF 官方 PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV_2021_paper.pdf)，[arXiv 摘要/版本页](https://arxiv.org/abs/2103.16556)，[作者论文页](https://martin-danelljan.github.io/publication/keeptrack/)。
- 作者代码（访问于 2026-09-12）：[visionml/pytracking](https://github.com/visionml/pytracking)，固定上游提交 `7eb9e74bd3d40e29dbcec444902237da13de247b`；核心为 [`keep_track.py`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py)、[`candidates.py`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/candidates.py)、[`target_candidate_matching.py`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/ltr/models/target_candidate_matching/target_candidate_matching.py)、[`superglue.py`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/ltr/models/target_candidate_matching/superglue.py)和[默认参数](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/parameter/keep_track/default.py)。当前版本不自动等于 ICCV 2021 实验版本，下面分开说明。
- 本地可复核缓存：[`outputs/literature/keeptrack-iccv2021.pdf`](../../outputs/literature/keeptrack-iccv2021.pdf)、[`outputs/literature/keeptrack-iccv2021.txt`](../../outputs/literature/keeptrack-iccv2021.txt)。
- 实际阅读范围：CVF 11 页版本的正文、图表和参考文献，以及上述作者实现的候选提取、关联、状态/记忆更新、匹配网络构造器和默认参数。训练数量与调度依据论文 §3.6；不声称已复核作者完整训练入口或独立 supplementary，也没有运行模型。

## 1. 任务语义：它解决的是已知实例的在线跟踪

论文的通用对象跟踪设定是首帧给定被跟踪对象的位置；其 SuperDiMP 基座带有 DiMP 分类器、边框回归器和在线内部记忆（论文 §3.1）。因此实际因果关系近似为：

\[
(I_0, b_0^{GT}), I_1,\ldots,I_t,\; \text{online state}_{<t}
\longrightarrow (b_t,\;\text{target-present confidence}) .
\]

这里的 `b0` 已经指出“这一段里哪一个实例是目标”。这与本项目当前目标

\[
I_{t-2:t}\longrightarrow (\text{ball position}_t,\text{visibility}_t)
\]

不同：后者需要自动从球场图像中产生球候选，不以 GT 初始框或跨窗口身份状态作为输入。即使 KeepTrack 在 OxUvA/VOT-LT 同时输出存在性和框，该存在性也属于**已初始化实例**的长时 tracker 协议，不能替代球数据集的 visibility/可定位标签。

这一差别同时限制性能比较：KeepTrack 的 LaSOT、NFS、UAV123、OxUvA 和 VOT-LT 表格不能作为 TrackNet/BlurBall 的球中心指标，更不能说它是“自动小球检测基线”。它只可作为一种机制先例或在未来另设“已知球初始位置的在线跟踪”协议时的可比方法。

## 2. 候选阶段先决定关联能否发生

### 2.1 候选是基座局部 score map 的峰，不是全图检测结果

论文 §3.2–3.3 先由基座 tracker 在当前搜索区产生 target score map `s`，取分数不低于 \(\tau=0.05\) 的局部极大值为候选。论文用 5×5 max-pooling 找峰；作者当前实现同样调用 `find_local_maxima(score_map, ks=5, th=0.05)`。每个候选的输入是：

\[
v_i=(s_i,c_i,f_i), \qquad z_i=f_i+\psi(s_i,c_i),
\]

其中 `f_i` 是分类特征图上经卷积抽取的外观描述符，`c_i` 是图像坐标，`s_i` 是基座分数（论文 §3.3–3.4）。作者构造器使用 ResNet-50 layer3 特征和 256 维描述符；默认推理搜索图像边长 480、`search_area_scale=8`。该参数在初始 `search_area=prod(target_sz*8)` 中作用于两维，不能翻译成“面积仅为目标的 8 倍”。实际 crop 还受尺度和边界处理影响。[构造器](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/ltr/models/target_candidate_matching/target_candidate_matching.py#L90)、[初始化尺度](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L71)

这带来可证伪的硬边界：球若没有进入基座的局部 search crop、在 score map 中不成峰或低于阈值，后续关联没有任何对象可关联。KeepTrack 的关联成功率不能抵消候选召回为零，也不能证明其可解决“大位移自动发现”。对任何把它借鉴到球定位的方案，都应单独报告真球是否进入候选集合（在给定容差和真实时间对齐下），再讨论条件关联正确率。

这是**候选关联子模块**的输入限制。完整 tracker 还含基座定位回退和 IoU-Net 框细化，不能把峰坐标的候选召回直接宣称为最终预测中心的严格上限。[完整 track 路径](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L117)

作者实现的 `track()` 又以持续的 `self.pos` 为中心裁局部区域；仅在 `not_found` 后才基于至多 60 个历史尺度调整搜索区。它不是每帧全图候选生成或全局重检测器。连续摇摄或球大位移使球落出该局部区时，论文并未提供自动全图发现保证。

### 2.2 相似干扰物的关联机制

在前、当前帧候选集合 \(V'\) 与 \(V\) 上，KeepTrack 采用 SuperGlue 风格 GNN：同帧 self-attention 与跨帧 cross-attention 交替，以邻近候选的相对格局辅助区分外观相似的峰（论文 §3.4）。经嵌入得到 `h` 后，以点积

\[
S_{ij}=\langle h'_i,h_j\rangle
\]

形成候选相似度，再以 Sinkhorn 做带约束的分配（论文 §3.5）。作者训练配置是两层 `self,cross` 交替 GNN、10 次 Sinkhorn；当前源码也将 `GNN_layers=['self','cross']*2` 和 `num_sinkhorn_iterations=10` 明确写在构造器中。

这是一种**候选图上的一对一关联**，不是高分辨率稠密 cost volume，更不是光流。它可在候选已经产生且候选数有限时比较远距离候选；实际覆盖范围仍由前述局部搜索 crop、score-map 候选和前一帧状态限制。它同时表明“为抗相似干扰必须做像素级全局相关”不是成立的必要命题；但并不反驳微小球的空间细节/远程自动候选问题，因为 KeepTrack 假定基座已给出了可关联峰。

## 3. dustbin、new candidate 与真正的 no-match 不是同一个概念

论文对 `V'×V` 分配增加一行一列 dustbin：候选可被配到虚拟槽，以表达“上一帧候选本帧不见”或“本帧新候选无上一帧对应”（论文 §3.2、§3.5）。当前 `superglue.py` 的 optimal transport 矩阵也确实追加了这两个 bin，并只保留互为最近且超过阈值的候选配对。

其语义应严格写为：

| KeepTrack 事件 | 它确实表示什么 | 它不表示什么 |
| --- | --- | --- |
| candidate → dustbin | 这个**已经产生的**候选在相邻候选集中未获配对 | 球在当前帧不可见，或整帧没有球 |
| dustbin → candidate | 当前出现的峰没有前帧候选匹配 | 一个可信的“新球”检测 |
| 低匹配概率/新 object-id | 对候选身份延续不确定，保守地切断旧身份 | 经校准的视觉 no-match 概率 |
| `not_found` | tracker 依据分数或候选状态没有找到已初始化目标 | 数据集标注意义上的球消失/未知标签 |

因此它是论文要求审查的“拒绝候选身份延续”的已有具体机制；任何新方法不能泛称首次做 no-match 或多假设可靠性。它也不等价于**候选产生之前**的自动发现可靠性，不能拿 dustbin 当成 visibility loss 或把没有标注的 OpenTTGames 帧当负样本。

## 4. 持久 object database、重检与错误持续

论文 §3.7 把每个候选（目标或干扰物）放入跨帧 object database \(\mathcal O\)。匹配到的候选继承 object-id 和分数历史；未匹配候选建新 object-id；旧 object 若当前没有候选匹配即删除。该节对关联模糊、概率低于 \(\omega=0.75\) 的情况删除旧 object 并新建 object，避免把模糊关联硬写进身份。

论文的重新检测和随后切换是两个阶段。原目标身份丢失后，先用超过 \(\eta=0.25\) 的当前最高分候选重新选定目标；随后若另一个候选的当前分数更高，还要求它超过**当前已选 object 的全部历史分数**，才改选该候选。不能把历史分数门槛误加到第一次重新检测上。当前源码对此也有实现差别：只对当前已选 ID 使用 `match_score<0.6` 或 `match_score<0.85 且 score<0.2` 的切新 ID 条件；目标仍在时的替换比较两个 object 的历史最高分，目标丢失后则以当前分数 `>0.25` 重选。这些是该上游提交的实际规则，不能当成论文阈值的逐字复现。[候选状态实现](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/candidates.py)

这确实给出了“错误可能如何持续”的机制链：错误候选若持续继承初始 object-id，则后续局部搜索与记忆更新可能继续强化它；反过来，dustbin/低匹配切 ID/重检门槛试图阻断这种链。可证伪条件也明确：若关联模块使目标身份延续错误率上升，或正确候选频繁在候选阶段漏掉，则 object database 反而会延续错误，并不能因存在 Sinkhorn 就宣称可靠。

**对当前基线的限制。** 当前 BlurBall 基线没有跨窗口持久 `CandidateCollection`、在线分类器样本记忆或 ID 状态。[已完成的状态连续段诊断](../experiments/2026-09-12-blurball-error-persistence.md)确实找到持续误选背景亮点、球网和运动员区域的样例，但不能将其命名为 memory contamination；模型可能在每个独立窗口反复受到相近干扰。只有未来主动加入跨窗口 ID/在线更新后，才有相应的错误传播路径可供归因。

## 5. 记忆样本置信度：有价值的防错模式，但带有强任务前提

KeepTrack 不只关联候选，也用身份与基座 score 控制 SuperDiMP 的在线分类器记忆（论文 §3.8）：

\[
J(\theta)=\lambda R(\theta)+\sum_{k=1}^{t}\alpha_k\beta_kQ(\theta;x_k,y_k),
\]

其中 \(\alpha_k\) 是年龄衰减，\(\beta_t=\sqrt{\sigma}\)（当前 object 与首帧目标 ID 相同）或 \(\sigma\)（否则），\(\sigma=\max_i s_i^t\)。论文为这个公式假设 \(\sigma\in[0,1]\)，不是已证明的概率校准。固定容量满后丢弃最小 \(\alpha_k\beta_k\) 样本；最终设置还抑制 \(\beta<0.5\) 的样本。当前代码先传入 score-map maximum，进入 `update_memory()` 时按所选 ID 做平方根增强，随后以 certainty×年龄权重管理样本，并在训练权重中抑制低 certainty；外层 `not_found/uncertain` 不进入该分类器更新。[记忆更新](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L628)、[在线权重](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L850)

这不是独立的 no-match 标定器：\(\beta\) 混合了 target classifier 分数和“是否仍是初始 ID”的内部判断；目标已丢失时又完全依赖 score。因此它可作为**未来有在线适配时**“不要无条件用当前预测更新记忆”的近邻，不能写作已证明的物理 correspondence reliability，也不能用于当前无该记忆的模型。

论文表 3（LaSOT AUC，GT 初始化单目标跟踪）逐行是：无 replacement/无 online confidence/无阈值 63.5；仅 replacement 64.1；再用 confidence 更新、阈值 0.0 为 64.6；阈值 0.5 为 65.0。它说明该特定基座的在线记忆选择有贡献，不能拆成“关联网络自身”或移植为球定位提升幅度。

## 6. 训练证据与不可直接照搬之处

候选间真实 distractor correspondence 在普通单目标数据中未标注。KeepTrack 的训练由两部分构成（论文 §3.6）：

1. **partial supervision：** 从冻结 base tracker 的连续输出中，为已标注 target 候选构造真实对；随机移除候选以模拟遮挡/重检。
2. **self-supervision：** 同帧候选复制到另一集合，对位置、分数、图像外观做扰动，并再模拟遮挡；已知复制前后的候选身份，训练 assignment NLL。

真实帧只进 \(L_{sup}\)，合成候选只进 \(L_{self}\)，最终 \(L_{tot}=L_{sup}+L_{self}\)。论文训练前将 LaSOT 基座 search region 与 score map 离线导出、冻结基座；15 epoch、每 epoch 6400 subsequence、真实/合成各半、Adam `1e-4`、每 6 epoch 乘 0.2。表 2 的 LaSOT AUC（均已包含论文所列 train setting）是：无 TCA 65.8，`Lsup` 66.0，`Lself` 66.9，二者无 data mining 66.8，二者加 data mining 67.1。

本项目有球中心/visibility 等既有标签，但没有“多个基座候选的跨帧 object identity”标签，也不应把稀疏未标注帧伪造为负例。若未来采用类似训练，必须预先定义候选来源、候选 GT 匹配准则、因果相邻帧、合成扰动是否保持 tiny-ball 外观，以及真实标签缺失的处理；这些都尚未被 KeepTrack 解决。

## 7. 性能应怎样读，不能怎样读

论文全部比较属于已初始化单目标跟踪。最相关的是表 1 最后两行：已有相同 memory/search 配置时，加入 TCA 后 LaSOT AUC **65.8→67.1**，NFS **65.2→66.4**，UAV123 **69.1→69.7**。这是该配方下关联有增量价值的实际证据，应当接受；不能只强调任务边界而漏掉这个对照。各数据集五次运行的平均用于该表，论文没有提供几像素球中心误差或当前三帧自动定位的直接结果。

长时存在性表 6 的列也需要分清：SuperDiMP 的 TPR/TNR/MaxGM 是 **79.7/70.2/74.8**，KeepTrack 为 **80.6/81.2/80.9**；**74.9/75.4/75.1 属于 LTMU**。此处按官方 PDF 图像复核，避免把邻列写成基座。该 presence 是已知初始实例的存在性，仍不同于本项目原始 visibility。

§4 报告 18.3/29.6 FPS（标准/Fast，作者 2080 Ti 环境）；两者同时改变搜索输入尺寸和框精修次数，不能据此反推出关联模块的独立成本。当前固定源码的标准 `keep_track.py` 也会在两帧各仅一个高分候选时跳过网络匹配，不能把该快捷路径专属归给 Fast。完整视频解码与本项目 DINO 自动定位的成本需要实际重新测量；本轮未运行这些模型。[快捷路径](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L271)

## 8. 对本项目的研究判断与最小可证伪迁移

KeepTrack 已覆盖了下列泛化说法，论文中不应再作首次性宣称：

- 对多个候选（含相似干扰物）做跨帧身份关联；
- 用 Sinkhorn/dustbin 表示候选新增、消失和不确定关联；
- 把候选身份不确定性用于在线记忆样本抑制；
- 在 tracker 生成的候选上以 partial/self-supervision 训练关联。

它**没有**证明下列命题：高分辨率自动小球候选怎样在大位移、运动模糊和连续相机运动下仍被发现；球中心几像素时 descriptor 是否可读；无需初始 GT 框的每帧定位；或其 object-id/记忆机制对短固定窗口自动球定位会带来收益。

本地持续错误还要求分清两个事件：**某个背景点与历史中的自身对应正确，和这个点是球，是不同的命题。** 一个错误的球候选可能有真实、稳定的背景对应；因此，给关联加 no-match 并不能逻辑上保证抑制它。这里是由任务定义得到的推论，本轮没有测量样例的实际特征匹配质量。KeepTrack 借首帧实例身份及干扰物历史来区分对象，不能把这种先验偷偷带入自动球发现。

若以后有充分理由检验短窗口候选关联，可以在同一 `[t−2,t−1,t]` 内比较关系利用方式，每次调用重置内部状态；这样仍可与三帧自动定位共享信息范围。若改成跨窗口持久 tracker，就已经使用更久的历史，需要另行定义允许的历史与初始化，并与获得同等历史信息的对照比较；不能同时声称“持久记忆”和“仅三帧输入完全等价”。候选召回、条件关联和最终定位需要分开，visibility 与 dustbin 也仍是不同事件。

当前不据此启动新记忆/ID 分支。现有错误段是探索性样例，也未按高速或长拖影筛选；若将来改进只来自单帧外观区分，仍不足以支撑 motion representation 的贡献。下一阶段是否采用关联，应由正在完成的真实历史对照和具体失败证据共同决定。

## 9. 有界后续检索（2024–2026）

本轮只做了三组直接机制检索，范围限定为 arXiv 上同时指向 *single-object tracking + candidate association + distractor/target candidate* 的题名/摘要检索（2026-09-12）：

1. `"candidate association" "visual tracking"`；
2. `"distractor" "single object tracking" "candidate"`；
3. `"target candidate" tracking association`。

结果仅稳定返回本论文的 [arXiv 页面](https://arxiv.org/abs/2103.16556)，没有发现 2024–2026 年、可由一手 arXiv/CVF/作者稿核实、且在机制上直接续写“tracker score-map multiple candidates → learned association/dustbin → persistent target/distractor object state”的论文。此为**有界未命中**，不是对全领域或所有修订版本的穷尽性否定；依任务要求，到此停止该支线，不用泛化 MOT、通用点跟踪或仅含“distractor”字样的论文凑近邻。

## 来源与证据边界

- **官方论文证据：** Mayer *et al.* ICCV 2021，尤其 §3.1–§3.8、§4、表 1–8；CVF PDF 链接见开头。
- **作者源码证据：** `visionml/pytracking` 的固定提交与实现路径见开头，关键文件缓存于 `outputs/literature/keeptrack-code/`。源码用于核对候选提取、局部搜索、阈值、状态/记忆和网络构造器；其后续改动不替代 ICCV 版本的实验事实。
- **适用范围：** 未下载 KeepTrack 权重、运行模型或作本地性能比较；文中本地错误证据引用独立完成的 BlurBall 诊断。
