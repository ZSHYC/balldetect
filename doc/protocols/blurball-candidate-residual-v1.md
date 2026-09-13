# BlurBall冻结候选残差读出 v1

日期：2026-09-13。状态：运行前锁定，三条件各30轮已完成；[结果与监督修订问题](../experiments/2026-09-13-blurball-candidate-residual.md)。

## 有限问题与已有先例

接续[历史硬筛失败](../experiments/2026-09-13-blurball-history-ball-evidence.md)。检验：在保留当前候选分数差的可学习读出中，历史全图软对应汇聚是否比当前外观、同地址历史提供更好的自动定位增量。本项只批准冻结特征的小型读出对照，不等于最终架构或端到端motion模型。

已有机制直接覆盖此类特征汇聚，不以其本身申报创新：

- [SELSA，ICCV 2019，§3.2](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Sequence_Level_Semantics_Aggregation_for_Video_Object_Detection_ICCV_2019_paper.pdf)：query条件化的跨帧proposal点积/softmax/value汇聚。这里改为固定微小球候选对历史native格、冻结cosine描述符和短因果输入，仍是受限关联读出。
- [FGFA，ICCV 2017，§3.2](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.pdf)：先光流warp，再在同位置跨帧以cosine权重聚合；与本项历史空间格上的softmax不同。
- [STSN，ECCV 2018，§4](https://openaccess.thecvf.com/content_ECCV_2018/papers/Gedas_Bertasius_Object_Detection_in_ECCV_2018_paper.pdf)：无光流的学习式deformable历史采样，再按跨帧cosine聚合。本项没有学习offset，也不称其全局替代已优于STSN。

本轮独立来源核查实际阅读上述公式段及SELSA/FGFA作者实现；没有运行这些方法。本项无需以新增arXiv清单证明新颖性，因为更早先例已经直接覆盖核心聚合。近期全文与最终贡献的审查仍按项目已有文献执行。

## 数据、时间与固定部分

train match00–17全部38,854目标，val match18–21全部14,192目标。数据标签、原帧号、PTS、rally及内部边界继承cross原协议。最终22–25不使用。输入始终真实合法`t−2:t→t`，不读取历史三帧模型的过去输出，不加入更早图像或递归状态。

cross best3的prefix、head均冻结；当前K16的局部坐标、peak logits及q复用已有训练/验证候选。不会恢复训练旧cross、same或repeat。本项只能改变候选排序，不生成新位置、修补候选漏球或学习q。主结果保持所有V1和V0目标，包括没有正确候选的帧。

stage0为96×72×128；每帧每通道减空间均值。当前query在当前候选原图坐标双线性采样后L2归一化。历史native格各自L2归一化。沿用`align_corners=False`及border采样，原图像素中心映射不变。

## 三个训练条件

每个候选的共同输入维度288，按`[当前query, near, far]`排列。

1. `current`：历史两向量置零，检验仅增加当前候选外观读出容量能带来的收益。
2. `stationary`：在两历史帧的当前候选同坐标采样，采样后L2归一化。它是同地址时序特征对照，不是球的真实零速度标签，也不是对应方法上界。
3. `correspondence`：分别对near/far全部9,216格计算当前query与key的cosine，以固定0.1温度按历史空间格softmax，再对unit key向量加权求和。两帧独立归一化，顺序不变，不取单个argmax。

`current`只限制新增残差支路的输入；原候选、原peak logits与q仍来自真实三帧cross，所以它不是重新训练的单帧模型，也不是整个系统完全没有历史的条件。

共同读出为`Linear(288,32)→GELU→Linear(32,1)`，输入统一乘`sqrt(96)`，末层weight/bias初始化为零。输出为原`peak_logits−peak_logits[:,0]`加残差；原坐标、当前q及同分优先原K顺序保持。初始epoch0逐目标精确退化为cross。current条件虽然分配相同参数，但历史输入权重不获得有效数据梯度，报告时不称三个条件的有效容量严格相同。

这是对**软聚合后的描述符**训练读出，没有学习相似度度量、历史任务头或匹配坐标。加权和丢弃空间坐标和多峰形状；其向量模长也混合了key方向抵消与权重集中度。因此本项不自称多假设运动表示，也不能以阴性否定全局对应所含信息。

## 监督、优化与选优

仅`当前V1 且 K16最小原图误差<4px`参与位置损失；否则位置loss为零，避免把漏球帧中最近的背景强制成正例。不增加none类；当前q固定。所有38,854目标仍按同一批序遍历，记录实际可监督数及无监督batch，不缩减训练/验证比赛。

可监督目标的候选软标签为`softmax(−||candidate−GT||²/(2×4²))`，训练交叉熵。4px取既有定位单位，固定不扫；这是候选偏好监督，GT不输入读出、对应或候选生成。无正例帧仍在全量评价和候选覆盖分母中。训练候选来自训练过这些图像的cross，存在训练内拟合偏差，不能当独立校准数据。

三条件seed0、相同初始化和NumPy批序、batch256、AdamW lr=3e−4/weight_decay=.01、固定30epoch、无scheduler。每个batch以可监督目标数归一化；全无监督batch跳过optimizer。每epoch使用全部验证目标选优，先最大TP4、再最大raw正确4、同分保留最早，epoch0参与。q固定使TP4选优与本地F1@4一致。保存各epoch实际train loss/val结果、best/last及optimizer/RNG，支持真实续训，不从观察超时推断任务退出。

不训练额外q头，不以候选置信度改变输出阈值。没有从训练候选覆盖中得出的GT条件在推理时充当开关。

## 计算与验证

先在冻结前缀上共享一次train/val逐帧提取：每个源帧stage0只编码一次，跨batch保留最后两帧。持久化每个目标的query、stationary与correspondence向量，float32总计约1.63GB；不保存整幅dense特征或完整cost。已有对应NPZ只有max分数/位置，不能重建本项软加权向量，故新提取必要；不因seed/读出训练重提取。

GPU缓存导出和缓存训练分开计时。三个MLP相同不代表完整计算量相同：correspondence额外全图相关与汇聚为真实成本，不能把读缓存速度称作在线模型速度。使用Conda `zshihyc`；prefix、对应float32，原候选坐标float64保留。此次导出记录共同前缀、两种聚合及写盘的合计耗时，没有单独测量correspondence相对stationary的算子耗时；如后续有阳性结果再为部署决策测量，不重跑整个缓存填补计时表。

先测试空间坐标/near-far采样、软汇聚能区别位移与同址、零描述符有限、残差零初始、软标签及梯度；复用已有candidate_costs回归。首真实验证batch匹配格/最高cosine对照上一项缓存，检测源权重或坐标漂移；生产编码数等于合法窗口唯一源帧数。训练前核对源身份及epoch0原候选选择，训练后用保存best复算相同验证输出。

代码职责：`correspondence.py`共享候选token和软聚合；`candidate_readout.py`负责残差头与候选软标签；`export_blurball_candidate_features.py`只生成紧凑冻结缓存；训练入口加载缓存、训练三条件、写预测和配对。结果分置`candidate_residual/features/`与各条件run，协议/记录与导航随实际进展更新。

## 判据与停止边界

必须报告原cross及三个训练条件，全部验证、四场、位移/拖影与固定202/74群体的raw位置、q下检测及救回/破坏；同时报告候选覆盖和训练可监督比例。不是只看GT有候选子集，也不能按固定失败群体选模型。

仅当correspondence相对原cross与两个训练对照都提高全体PCK4/F1@4、PCK16不降，且相对原cross四场PCK4均不降，才保留这条固定聚合读出作下一阶段候选。否则停止此固定配方，不追加温度/隐藏宽度/训练轮数扫描。单seed开发阳性仍需独立重复与进一步机制验证，不直接进入最终测试或论文创新声明。

执行顺序：定向函数验证→一次缓存导出→三条件完整训练及保存→全量配对→据结果决定下一结构问题。独立设计审查已促成本协议的可达正例loss和current控制，普通有界实验执行不启用OMX共识/运行时工作流。
