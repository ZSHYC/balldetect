# 局部对应基线的最近机制先例

日期：2026-09-10。范围：为[GT query诊断](../experiments/2026-09-10-correspondence-probe.md)之后的局部cost volume基线做定向核对。结合原始论文与作者代码，不是穷尽检索，也不形成新颖性保证。

## MotionSqueeze：局部对应、位移和置信度已有先例

MotionSqueeze在相邻帧每个特征位置的方形邻域计算点积相关；文中28×28特征上使用15×15候选邻域。随后用kernel-soft-argmax压成二维位移，并取相关峰值作为每位置confidence，再经卷积构造motion feature并融合appearance。因此局部cost volume、位移估计和对应置信度不能作为本项目的新动机。原设相邻方向含未来帧，本项目若严格因果必须显式改为查询历史。[原稿，Sec.3](https://arxiv.org/pdf/2007.09933)，[作者实现](https://github.com/arunos728/MotionSqueeze/blob/e94684d05347a75dd1b487fe426e00b19b29bc93/resnet_TSM.py#L146-L167)。

本项目可比较保留完整offset通道与早期压成单个位移，但不能仅凭这种差别声称创新；后者还须对照下述STSS。

## SELFY/STSS：保留时空候选分布同样已有先例

STSS定义逐位置、逐时间偏移和逐空间偏移的相似度张量，时间和空间均为有限邻域。论文比较了soft-argmax、MLP和在偏移轴上用3D卷积提取信息等方法；后者保留相似度分布供学习。因此dense query、多时距与未先压缩的offset volume已经被直接研究。其对称时间邻域包含前瞻，严格因果版本只能使用历史偏移。[ICCV 2021原文，Sec.3.1–3.2](https://openaccess.thecvf.com/content/ICCV2021/papers/Kwon_Learning_Self-Similarity_in_Space_and_Time_As_Generalized_Motion_for_ICCV_2021_paper.pdf)，[作者实现](https://github.com/arunos728/SELFY/blob/d6e1e002e78b01e1cc7dba5253692b812f3e9c3a/ops/selfy.py#L13-L109)。

拟议的Δ1/R2与Δ2/R4组合应被视为稀疏、因果、任务特定的STSS类基线。不能声称首次多间隔、首次保留多假设或首次在细位置搜索局部对应。L2归一化也已经是常见组成。

## TDN：差分依旧是必要竞争解释

TDN使用中心帧附近的RGB差分构造短时信息，并在稀疏片段之间用多尺度、双向差分作长时excitation。它不是correspondence/cost volume，也不输出位移候选分布，不能替代局部对应对照；但它否定了“变化线索尚未被建模”的动机。原始中心时间窗口包含未来帧，使用历史差分时同样要明确因果改动。[CVPR 2021原文，Sec.3.2–3.3](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.pdf)。

本项目暂不为“完整特征拼接”与其可逆有符号线性差分机械增加重复实验；有额外非线性、约束或门控的TDN等方法不属于这种简单等价。

## DMR：背景一致运动与微小目标局部动态也已有直接研究

《Decoupled Motion Representation Learning for Moving Infrared Small Target Detection》，arXiv:2606.15286 v1，2026-06-13，明确讨论平台/背景一致动态淹没稀疏微小目标的问题。其使用全局flow先验、局部可变形对齐和一致运动引导的局部推理。它不是本项目的简单cost volume，也不能证明空间均值处理能补偿相机，但直接限制了“首次全局/局部解耦或背景运动抑制”的叙述。[arXiv原稿，Sec.III-D–E](https://arxiv.org/pdf/2606.15286)。

本次核对中的“每帧、每通道减空间均值”只称描述子预处理。未在这几个特定先例中找到完全同样的公式，不等于该操作新颖，更不能改名为相机运动估计。

## 对当前设计的实际约束

局部cosine volume是一条合理的已有机制基线。当前实验要回答的是：在相同冻结特征、相同因果窗口和目标、相同细位置输出网格下，显式对应是否提供超出三帧拼接的定位收益。

保持隐藏宽度并不自动保持参数量或输入信息：只保留当前appearance再加cost，会丢掉历史appearance；拼接全部appearance再加cost，则增加通道与非线性计算。实现协议必须选定一种清楚的比较，不把这些差别藏在“相同head”四个字里。raw/空间去均值应在相同匹配与融合路径上比较。

先保留dense query和完整有限offset，避免在候选阶段先漏球；但这本身已有先例。几何覆盖、GT query检索与真实逐帧定位继续分开报告，GT query结果不能替代自动发现球的实验。无球、困难目标及其他比赛上的失败仍决定该机制是否值得继续。
