# DQAligner的自动query、窗口记忆与候选漏检边界

日期：2026-09-10。范围：补读一项最接近“当前弱证据与历史支持”的已检索先例；修订[高速微小目标笔记§1.3](tiny_motion_evidence.md)中尚未由实现支持的候选解释。未复现模型，没有下载其数据或权重，也没有改变本项目训练条件。

## 当前判断

DQAligner公开实现以学习得到的全局query读取整个窗口的视觉特征，输出稠密mask，**没有先由当前检测器筛出硬候选再匹配的必要阶段**。因此不能把“当前候选先漏球”当作它已被证明的结构缺陷，也不能将其和QueryDet的稀疏候选直接归成一类。它是自动全局查询与对齐的近邻，足以限制这类模块组合的宽泛首次性；是否能改善体育小球仍需共同任务实测。

## 来源与实际阅读范围

论文为Chuiyi Deng等，*Learning Global Dynamic Query for Large-Motion Infrared Small Target Detection*，IEEE TGRS 64，文章号5002016，DOI 10.1109/TGRS.2026.3657842，线上日期2026-01-26。[出版记录](https://ieeexplore.ieee.org/document/11363482/)

研究代理按准确题名、作者、PDF、arXiv和作者仓库定向查找，未找到作者公开可访问的完整论文；该结果不表示全文不存在。本轮论文证据仍限摘要，不能升级为全文阅读。机制证据来自作者[固定仓库版本c586f393](https://github.com/dengfa02/DQAligner_MIRSTD/tree/c586f3939f53e6af6572ea12d9f92903ee6f0755)，日期2026-03-08。研究代理核对训练/评价调用，根代理直接复读下列核心源码；源码行为与论文中的文字、消融结论分开。

## Query到底是什么

ObjectQueryModule从一个可学习的16维类向量开始，每次forward重新复制初态；逐帧以全空间key/value作attention，经GRU更新窗口内状态。它没有输入当前检测坐标或top-K列表。随后query与特征的点积产生sigmoid图，以残差乘法增强特征。这是类条件调制，不能直接当作球坐标、速度或校准后的匹配可靠度。[ObjectQuery.py：初始化、窗口更新与调制](https://github.com/dengfa02/DQAligner_MIRSTD/blob/c586f3939f53e6af6572ea12d9f92903ee6f0755/model/ObjectQuery.py#L177-L272)

这个区别对本项目有实际影响：单query访问全图，不等于每个细位置与历史全图做all-to-all匹配；它把历史信息压入状态，也仍保留空间特征路径。既不能按“global”一词假定其代价等于高分辨率全相关，也不能因query是一个向量就断言整个网络丢失全部位置。尚未测量它在球区域的精细定位与真实成本。

## 对齐、输出与跨窗口状态

公开模型以多尺度reference/key特征生成deformable offsets，逐级细化，浅层再接受query调制。默认最后一帧为key，其他帧对齐后与key一起由五帧Conv3D聚合为单通道输出。forward没有读取feat_prop和cat_flag，并原样返回feat_prop；结合query每次重置，当前路径没有显式跨调用的query记忆。[DQAligner.py：对齐和完整forward](https://github.com/dengfa02/DQAligner_MIRSTD/blob/c586f3939f53e6af6572ea12d9f92903ee6f0755/model/DQAligner.py#L297-L420)

因此，不能仅凭memory、iteration等名字宣称它实现跨rally或无限历史状态，也不能把deformable offsets自动当作已标定的物理光流。本项目若借鉴，首先要说明使用窗口内状态还是跨窗口状态，以及实际输出关系是什么，而不是重命名现成机制。

## 因果性需要看完整数据路径

IRDST默认last窗口在视频内部取过去四帧到当前帧，测试监督末帧；但开头索引按视频长度取模，会从末尾回绕。训练增强还以0.5概率同时反转图像和mask时间顺序。因此只能说未回绕的默认last推理窗口具有因果末帧形式，不能把整个发布训练协议称为全部遵循原始时间的严格因果采样。[IRDSTDataLoader.py](https://github.com/dengfa02/DQAligner_MIRSTD/blob/c586f3939f53e6af6572ea12d9f92903ee6f0755/IRDSTDataLoader.py#L96-L301)

这不否定其方法，也不把训练时反转增强误称为测试输入泄漏。它意味着本项目不能直接照搬loader：已锁定的真实帧号、禁止跨clip以及因果输入协议仍然适用。本轮没有导入或改写该loader。

## 评价能证明什么

公开评价包含最终mask的像素指标和PD/FA；PD通过预测与标签连通域中心距离小于3像素匹配，FA统计未匹配预测像素。该路径没有独立的candidate recall@K，这不是一项缺失的必需候选指标，因为模型本来没有硬候选阶段。全文不可访问，也不能据此宣称论文绝无额外分析。[metric_basic.py](https://github.com/dengfa02/DQAligner_MIRSTD/blob/c586f3939f53e6af6572ea12d9f92903ee6f0755/utils/metric_basic.py#L161-L205)

若未来本项目使用显式稀疏候选，候选覆盖是自己的必要诊断。与这种dense系统比较时应统一最终球定位与检测评价；若另从dense响应派生top-K，必须说明它是额外的事后读出规则，不能冒充原方法的内部候选。

## 对当前研究的具体影响

[适配后诊断](../experiments/2026-09-10-adapted-correspondence.md)测的是给定当前GT格之后的历史对应；DQAligner则从图像产生自动稠密输出。两者之间尚未解决的是完整任务中的证据使用，不能简单归纳成“加一个全局query就解决自动发现”。

本次修正三处论证：不再给DQAligner强加硬候选瓶颈；不将窗口内GRU状态称为已实现跨窗口记忆；不将源码的默认末帧模式与完整严格因果数据协议画等号。未来若采用全局query、历史状态和deformable alignment，必须将它作为近邻讨论，并用具体球定位失败与同任务对照区分贡献。当前继续既定全量HRNet/DINO，不新增DQAligner复现或模块。
