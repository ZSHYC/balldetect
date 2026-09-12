# 差分何时只是换基：Taylor Videos、TDN 与递归高通

日期：2026-09-12。范围：区分可被后续线性层吸收的差分基、Taylor/TDN 的前置运算，以及 MCATrack 的有状态高通响应；回答“时间变化表示”在哪些条件下真的改变输入或计算路径。

本文不设计新模块，不运行论文复现、球数据或训练。外部事实优先使用论文原文；作者代码只用于核对公开提取路径，不能替代论文实验协议。

## 阅读范围与版本

| 条目 | 实际阅读范围 | 版本/状态 |
|---|---|---|
| [Taylor Videos for Action Recognition](https://proceedings.mlr.press/v235/wang24ck.html) | PMLR ICML 2024 全文，重点 §3、§4.3 与 Appendix A--B（含 Algorithm 1） | PMLR 235，pp. 52117--52133；arXiv [2402.03019 v4](https://arxiv.org/abs/2402.03019)，2024-05-10 |
| [作者仓库 LeiWangR/video-ar](https://github.com/LeiWangR/video-ar) | README 与 `taylor-video.ipynb` 的全部代码单元 | 2026-09-12 可访问；仓库最后推送显示 2024-07-25；未运行 |
| [TDN: Temporal Difference Networks for Efficient Action Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.pdf) | CVPR 2021 全文 §3.1--3.4、Table 1 | 正式 CVPR 2021 |
| [TDN 作者仓库](https://github.com/MCG-NJU/TDN) | `ops/tdn_net.py` 的公开前向路径 | 2026-09-12 可访问；代码最后推送显示 2022-09-17；未运行 |
| MCATrack：Tracking Tiny Drones against Clutter | ICCV 2025 官方全文；研究代理阅读全文，根代理复读初始化、§4.2–4.3、图3、实验与表3，核对公式/表格原页 | ICCV 2025，pp.7361–7371；本文末附直接来源，作者所链仓库当前404，未复现 |

## 直接结论

Taylor Videos **不是**“完整当前帧 + 完整有符号一阶差分，随后自由线性混合”的可逆重参数化。

它把 RGB 投影为灰度，在一个时间块内取最高至 \(K+2\) 阶的有限差分，将这些差分与相对初帧的逐元素幂相乘、按阶数截断并跨时间平均，最后仅输出三个 motion map。这个变换含非线性和降维，不能由本项目已固定的“完整 normalized feature 拼接后自由 \(1×1\) 线性混合”简单吸收。

反过来，也不能把 Taylor 的图像亮度差分图叫作经验证的球位移、光流或 correspondence：原文没有搜索另一个位置、没有输出像素 \(u\to u+\delta\)，也没有对几像素球的中心误差做实验。

## Taylor Videos 的实际算子

原文先把每帧 RGB 转成灰度，令时间块为 \(F=[F_1,\ldots,F_T]\)，步长为 1 的滑窗逐块输出一个三通道 Taylor frame（§3.1）。

相邻一阶图像差分为 \(d(F_i)=F_{i+1}-F_i\)。二阶、三阶及更高阶差分由前一阶差分继续相减得到；原文将它们分别命名为 velocity、acceleration、jerk 等（Fig. 2、§3.2）。

三条输出通道对应作者命名的 displacement、velocity、acceleration map。对每一通道，使用初始差分导数与 \((F_\tau-F_1)^{\circ k}/k!\) 的逐元素乘积，再对 \(\tau=1\ldots T\) 平均（Eq. 3--6）。

这不是普通线性滤波：\(k>1\) 时含逐元素幂；不同阶差分与幂项相乘；最后还把整个时间块平均成三个通道。无穷级数在实际中截断。

Algorithm 1 明确：若使用 \(K\) 个 Taylor 项，先计算并保存 \(K+2\) 阶有限差分；三个通道分别从错开的差分阶次取项。最少需要 4 帧才能得到三通道输出；当 \(T>4\) 时作者建议取 \(K=T-3\) 以使用全部帧（§3.4、Appendix B）。

论文将符号解释为方向信息，但这里的“方向”是同一像素亮度有限差分的正负及三通道相对响应。它不提供二维几何方向、速度单位、对象身份或物理 displacement 的可识别性。

## 信息是否保留：论文公式与公开导出代码须分开

论文公式没有绝对值算子，有限差分及后续项在数学上可为正或负；Fig. 3 与脚注 2据此说明正负响应。

但公开 `taylor-video.ipynb` 的 `videoConvert` 导出路径做了更强的信息处理：

1. 用 `cv2.COLOR_BGR2GRAY` 并除以 255，RGB 颜色首先被丢弃。
2. `length=terms+3`，计算 \(K+2\) 阶差分；每个输出通道累加 `pow(xa_Tensor, incB)/factorial(incB)` 与相应差分图的乘积。
3. `preprocess_tensor` 将所有负值置零，而非取绝对值。
4. 余下正值按**该帧、该通道**的最大值缩放到 0--255、转换为 `uint8`，再写入 H.264 mp4。

所以该 notebook 的已写出视频既不保留负号，也不保留跨帧/跨通道的原始量纲；逐通道最大值为零时全置零。README 说可视化像素值为更好显示而缩放，但此 notebook 的写出函数本身确实调用该预处理。

这只证明公开导出实现的行为；原文没有把该 `uint8`/H.264 路径逐项与论文全部实验设置一一核对。不能反向声称论文所有内部训练输入都必然经过同一量化。

## 时间语义：相对哪个目标帧才谈“未来”

论文的公式以块首 \(F_1\) 展开到 \(F_\tau\)，但又把 motion map 定义为编码“ending at \(F\)”的时间块（§3.1）。因此不能脱离输出索引宣称它天生在线或天生使用未来帧。

若将一张 Taylor frame 对齐到块末 \(F_T\)，输入是 \([F_{T-(K+2)},\ldots,F_T]\)，形式上是有 \(K+2\) 帧历史的因果窗口。若对齐到块首 \(F_1\)，同一计算使用 \(F_2\ldots F_T\)，则是前瞻。

作者的 action-recognition 发布协议只说滑窗、步长 1；没有逐帧中心定位 target、延迟定义或在线部署评价。论文举例：生成 16 个 Taylor output，\(K=1\) 需 19 个 RGB 帧、\(K=2\) 需 20 个（§4.3）。这支持上述窗口长度，不能自动替本项目决定目标帧对齐。

## 与本地已定线性等价结论的精确关系

本地当前路径是：同一 prefix 的逐帧独立 feature，按 \(t\) 拼接为 576 通道，三组 GroupNorm（`affine=False`）后接自由 \(1×1\) Conv、GELU、空间头。

在**完整 normalized feature 已保留**、差分只是和当前 feature 一起进入该自由线性 \(1×1\) 混合的条件下，当前 feature 加完整有符号差分是可逆线性换基；任意下游 GELU 不会单独改变这个函数族结论。这个已定判断无需再跑等价实验。

Taylor 不满足这些条件：灰度/三通道汇聚已丢信息，幂乘与截断在自由线性层之前，且公开导出路径还会截负、归一化、量化。故“Taylor 不是可逆基”不能被泛化成“任意 signed-difference 路径都不等价”。

即使固定可逆变换可由首层吸收，初始化、优化和参数空间正则化仍未必等价；在上述相同宽度与自由首层条件下，有限宽度本身不推翻函数等价。Taylor 原文没有通过控制实验把这些因素与信息丢失或非线性预处理分离。因此不能把其动作识别提升解释为某一单独算子的必然优越。

## 计算与任务迁移边界

原文的效率结论限于预计算 action input：Taylor frame 的网络处理时间与 RGB frame 相近；生成时间随项数增加，并低于其比较的 TV-L1 flow（§4.3、Fig. 7）。这不等于三帧 DINO 球定位的端到端成本。

项数越多需要更多高阶差分和逐元素运算；作者自己报告较多项会把移动相机下的背景细节重新带入（Fig. 8）。这与“天然去除相机 motion”不同。

实验任务是 HMDB-51、MPII Cooking、CATER、Kinetics 等 action recognition；MPII 的细粒度人类动作和 CATER 的 moving-camera 设置不是微小球中心定位。没有球标签、visibility、blur、空帧、PCK/中心误差、跨大位移 correspondence 或自动发现球的实测。

论文还明确 Taylor frame 缺少静态纹理，必要时建议与 RGB/gray frame 堆叠（§3.4）。对 tiny ball，这既可能压制背景，也可能删除小球或其弱、短暂响应；这是待测迁移假设，不是本文证据。

与 [2026-09-11 COMET 笔记](2026-09-11-directional-differences.md)相比，二者都不是 physical correspondence。COMET 额外使用一至五阶前向差分、幂/归一化及 MLLM temporal branch；Taylor 是三通道 action 输入格式。两者均不构成 RGB 微小球定位效果证据，也不能互相替代为同一算子。

## TDN：非线性、分支和融合在何处

TDN 的 S-TDM 先围绕中心帧取四个 RGB 差分 \([D_{-2},D_{-1},D_1,D_2]\)，然后执行 `Downsample → CNN → Upsample` 得 \(H(I_i)\)，再与 appearance feature 相加（Eq. 1、3）。所以 CNN 分支在最终 appearance 融合之前。

正式论文还比较了 \(F\odot H\)、\(F+F\odot H\)、\(F+H\)；默认报告的 S-TDM 是后者（Table 1b）。这不抹除前置差分 CNN 的非线性与结构约束。

L-TDM 先对相邻 segment feature 作 \(C(F_i,F_{i+1})=F_i-\mathrm{Conv}(F_{i+1})\)，即学习卷积发生在减法之前（Eq. 4）。其后有三尺度支路：短接、\(3×3\) 卷积、以及 average pooling 后 \(3×3\) 卷积再双线性上采样（Eq. 5、§3.3）。

三支路汇合后再经卷积和 sigmoid，双向 \(M(F_i,F_{i+1})\)、\(M(F_{i+1},F_i)\) 平均，作为 gate 与 \(F_i\) 逐元素相乘，最后残差写回（Eq. 2、5、6）。因此完整 L-TDM 含“预减法卷积、多尺度、sigmoid、乘性门控”这些都位于最后 residual 之前。

作者公开 `ops/tdn_net.py` 的短时实际前向同样可见：四个 RGB 一阶差分先拼为 12 通道，经 avg-pool、`Conv+BatchNorm+ReLU`、max-pool/ResNet block，插值后才线性写回中心 RGB feature。它支持 S-TDM 的前置非线性事实；它不替代论文 Eq. 4--6 对完整 L-TDM 的定义。

这说明**完整 TDN**不受“完整 normalized feature 后附加线性差分再自由混合”的首层重参数化证明覆盖；不能仅凭其非线性就断言完整可训练网络之间的函数族严格不同或存在包含关系。它不推翻本地特定条件下的线性等价，也不说明任何差分模块都应优于当前基线。

## MCATrack：有状态的递归高通，本身仍然线性

### 原始机制与可用证据

Jiahao Zhang 等的 *Tracking Tiny Drones against Clutter: Large-Scale Infrared Benchmark with Motion-Centric Adaptive Algorithm* 是首帧框已知的红外单目标跟踪工作。它在图像进入骨干前加入 Magno-motion 响应，并以初始/动态模板产生和重评 proposals。时间高通的传递函数为：

\[
H(z)=\omega\frac{1-z^{-1}}{1-\omega z^{-1}},
\qquad \omega=e^{-\Delta t/\tau},\quad\Delta t=1,\;\tau=2.
\]

论文同时指出相机位移伪响应需要相邻帧配准，但没有完整说明配准算法、时域状态或额外成本。动态模板仅在连续帧置信度至少0.95、中心位移至多20 px时更新。以下表3为其 TDTIV 跟踪 SA，未做本地复现：

| Motion | 模板特征增强 | SA |
|---|---|---:|
| 无 | 无 | 46.98 |
| 有 | 无 | 54.11 |
| 无 | 有 | 47.60 |
| 有 | 有 | 54.97 |

出处：[ICCV官方全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Tracking_Tiny_Drones_against_Clutter_Large-Scale_Infrared_Benchmark_with_Motion-Centric_ICCV_2025_paper.pdf)，§3、式1、图3、§5.1及表3。精确题名的arXiv查询本轮未命中，不排除改名稿；作者所链[仓库](https://github.com/zhangjiahao02/MCATrack)不可访问，因此不补写源码事实。

### 从公式能推导什么

令 \(x_t\) 为进入这一固定系数滤波器的输入、\(y_t\) 为响应，则式1等价于：

\[
y_t=\omega(x_t-x_{t-1})+\omega y_{t-1}.
\]

这是对滤波器的代数展开，不是对未知完整预处理的实现复现。固定 \(\omega\) 并采用零初态时，运算对输入序列线性；非零固定初态额外贡献一个衰减项。**递归、具有记忆，不等于非线性。** 空间/时间配准若由输入图像决定，完整预处理可以是数据依赖的，但不能倒推这个固定滤波公式本身非线性。

展开 \(L\) 步可得：

\[
y_t=\omega^L y_{t-L}
 +\sum_{j=0}^{L-1}\omega^{j+1}(x_{t-j}-x_{t-j-1}).
\]

因此，相对三帧窗口，它可能真正改变的是更早历史的可用性、历史压缩方式，以及进入非线性骨干之前的变化强调。输出仍是亮度变化响应，没有二维对应地址；完整 tracker 后续可以建立模板匹配，不能因这个滤波器不输出位移就否认整个系统的匹配能力。

若输入只限一个固定有限窗、初态重置、完整输入仍保留，且这些组合随后进入同一个自由线性层，则滤波输出可被吸收为加权时间组合。当前已证明的等价仅限**逐帧规范化特征后的同一线性头**：图像级预处理先进入非线性骨干，不满足这个限定，不能未经分析从原结论外推。

这也说明在长视频里使用此类状态时，信息范围和边界重置属于方法本身；不能把访问了更早帧的系统写成与三帧输入相同。这里只限定解释，不重新运行已定论的线性换基检查。

### 消融与模板更新必须按各自条件理解

表3对两个因素都有四种组合。运动增强在无模板增强条件下增加7.13；模板增强单独增加0.62，在已有运动时增加0.86。**0.62和0.86来自不同条件，可以同时正确。** 正文的0.62与表中单独增益一致，不能仅因另一条件得到0.86就报告作者数值矛盾。

这支持该已初始化 tracker 的运动增强效果，却没有单独分离配准、历史长度与高通响应的贡献，也不能把 SA 的增量换算为球中心定位收益。缺少这些分离不等于整套方法无效。

20 px 条件控制的是**模板写入**，不是预测输出的搜索半径或允许位移。快速且正确的定位可以继续输出而不更新模板；不能由这个门限断言它无法跟踪超过20 px的运动。它也不是 no-match、无球概率或已经校准的可靠性。相较于[KeepTrack](2026-09-12-distractor-association.md)，这里同样有真正的持久模板；当前 BlurBall 窗口模型没有这样的状态，不能直接借此归因模板污染。

### 对当前研究的实际影响

MCATrack补上了“微小目标、背景杂波、便宜的递归变化增强”的直接先例。它增加的竞争解释是**在合适的配准和历史条件下，时间变化强调就可能有用**，而非所有任务都必须显式搜索位移。

当前不增加该滤波器、模板或配准分支，也不为此重跑三帧线性差分。只有既定时序对照完成后，具体失败需要区分历史范围、早期交互或背景变化时，才选择其中一个问题做必要对照。它没有让当前自动球发现与高分辨率大位移对应的问题自动消失。

## 链接勘误与可复用判断

旧蓝图的 Taylor 链接 `wang24ck` 正确：[PMLR 235 / wang24ck](https://proceedings.mlr.press/v235/wang24ck.html)。

`2026-09-11-directional-differences.md` 的 COMET 直接来源 3 原先写成 `wang24ac`，该链接实际是 *Learning with Complementary Labels Revisited: The Selected-Completely-at-Random Setting Is More Practical*，不是 Taylor Videos；本轮已改为 `wang24ck`。

Taylor部分改变决策的结论是：不能把 Taylor Videos 归入已被本地自由线性头吸收的“可逆 signed-difference 基”并据此排除。若未来将其作为竞争解释，应把它视为有明确时间对齐、预处理与信息丢失的独立表示条件；它目前没有提供直接支持球中心定位的实证。

原定义 $K\ge1$，至少需要 $K+3=4$ 帧，不能直接替换当前三帧窗口。未来若确需比较，应让 RGB 控制也看到相同的 $K+3$ 帧并固定同一末帧 target；改变阶数或补帧属于另一算子，不能继续使用原方法名和原输入条件而不说明。

## 一手来源

1. [PMLR 正式全文与元数据：Taylor Videos for Action Recognition](https://proceedings.mlr.press/v235/wang24ck.html)（§3--4、Appendix A--B）。
2. [Taylor Videos arXiv 版本记录](https://arxiv.org/abs/2402.03019)（v4，2024-05-10）。
3. [Taylor Videos 作者代码与说明](https://github.com/LeiWangR/video-ar)（`taylor-video.ipynb`；实际读取未运行）。
4. [TDN CVPR 2021 正式全文](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.pdf)（§3、Table 1）。
5. [TDN 作者代码](https://github.com/MCG-NJU/TDN/blob/main/ops/tdn_net.py)（公开前向路径；实际读取未运行）。
6. [MCATrack ICCV 2025 正式全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Tracking_Tiny_Drones_against_Clutter_Large-Scale_Infrared_Benchmark_with_Motion-Centric_ICCV_2025_paper.pdf)（§3–5、式1、图3、表3；实际阅读未复现）。
