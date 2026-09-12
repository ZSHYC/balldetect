# 异常门控有效，不等于已经识别因果：CHAL 的方法、源码与论证边界

日期：2026-09-12

性质：正式论文、补充材料和固定作者源码的研究审查；未运行CHAL，未读取新的球标签或改变正在进行的训练。

问题：背景正常性、时序异常与门控能否成为显式对应之外的竞争解释？它们的检测收益支持多强的机制结论？

## 一、应当保留的判断

CHAL是必须引用的近邻：它把微小目标发现组织为“背景特征预测—外观偏离—时序验证—特征重加权”，在红外检测上给出有效消融。不能预设每一种有用的motion表示都必须显式输出位移或构造cost volume。

但有三种不同命题：模型利用了过去帧；异常门控提高了检测；门控识别了某个真实因果效应。前两项有相应代码或实验支持，第三项需要额外论证。补充材料提供了理论推导，不能说作者没有尝试证明；逐式审查后，其中关键过渡仍不能由列出的假设推出。

公开实现中的低异常分数也不是硬删候选。把它误写成候选召回上限，会错误排除一种仍有当前帧特征可用的竞争方法。

## 二、来源、版本与阅读范围

Weiwei Duan等，*CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection*，CVPR 2026，pp.21357–21366。

- [正式主文](https://openaccess.thecvf.com/content/CVPR2026/papers/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.pdf)：阅读方法、监督、实验与成本，查看表2/3原页并核对设置。
- [官方补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Duan_CHAL_Causal-guided_Hierarchical_CVPR_2026_supplemental.zip)：重点复核C.1理论推导、E中的背景/异常实验及F失败分析。
- [作者代码固定版本](https://github.com/UESTC-nnLab/CHAL/tree/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6)：追踪数据读取、主干、背景场、异常提取、时空处理、门控、损失与训练/测试入口。下文的代码判断限定于该公开版本，不当作作者所有实验的运行记录。

本轮题名检索没有定位到可确认的同题arXiv版本；使用正式来源，不据此声称其他题名或未索引版本不存在。PDF、提取文本与只读源码保存在outputs/literature/下；本文是项目结论入口，不保留另一份并行维护的代理草稿。

## 三、从真实输入到最终输出

### 1. 末帧定位与因果推断是两种“causal”

默认数据读取以有标注的帧号t为目标，读取t至t−4再反转。因此通常窗口为：

\[
(I_{t-4},I_{t-3},I_{t-2},I_{t-1},I_t)\longrightarrow\widehat Y_t.
\]

起始位置使用第0帧重复填充；模型没有为跨窗口复用而维护轨迹状态。该实现不需要未来视频帧，但有标签文件的目录边界并不自动等于本项目的合法连续clip边界。若未来接入，仍由本项目时间索引构窗。[数据路径](https://github.com/UESTC-nnLab/CHAL/blob/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6/utils/dataloader_for_DAUB.py#L112)、[主网络](https://github.com/UESTC-nnLab/CHAL/blob/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6/nets/slowfastnet.py#L161)。

这支持“末帧预测不看未来”，不支持实时速度承诺。论文标题中的causal主要指后门调整动机，与输入是否因果是不同问题。

### 2. 它首先判断偏离背景，并不先寻找对应地址

| 环节 | 实际内容 | 对球问题的边界 |
|---|---|---|
| 特征与背景场SNF | 每帧共享主干，多尺度融合；序列编码与位置编码生成背景特征 | 输出在特征空间，不是已知真实无球背景或相机运动 |
| 外观异常 | 同位置真实特征与预测背景的余弦偏离，经卷积处理 | 背景重建误差和真实目标都可能产生异常 |
| 时序处理HAL | 多帧异常图的局部时空attention与重建 | 没有显式目标对应地址；仍可能隐式利用时间关系 |
| 门控CRG与检测 | 外观软掩码、异常残差门控、特征细化、YOLOX头 | 自动稠密检测，不需要GT点初始化或提前给定球候选 |

来源：[主文§3.2–3.4](https://openaccess.thecvf.com/content/CVPR2026/papers/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.pdf)。论文以关键帧检测标注联合训练，并未用独立真实背景、相机运动或稠密对应监督来识别中间表示。因此“background”是模型赋予的角色，不能仅由名字断言它恰好排除了全部目标信息。

### 3. 阈值后的残差不能误读成硬筛除

对公开实现的归一化异常分数a，门控可写成：

\[
H(a)=
\begin{cases}
0.8a,&a>0.3,\\
0.1a,&a\le0.3,
\end{cases}
\qquad
F_g=F_t\odot M(F_t)\odot(1+H(a)).
\]

M为sigmoid外观掩码；随后还有卷积与CBAM细化。[实际门控](https://github.com/UESTC-nnLab/CHAL/blob/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6/nets/causal_anomaly_neck.py#L247)。

在a∈[0,1]时，低分支残差因子为[1,1.03]，高分支为(1.24,1.8]。所以低分支不是把特征直接乘0.1，更不是删除位置。外观掩码可以削弱证据，学习也可能失败；但这不构成“候选已经永久消失”的结构性证明。相对强调不同区域，与绝对删除区域应分别表述。

## 四、为何目前不能把门控等同于因果识别

### 1. 问题不在于“没有真正动手干预世界”

若正确的因果图、调整集合和可识别条件成立，后门调整本来就可以用观察数据识别干预分布。不应以“作者没有拍摄物理干预视频”作为否定理由。

需要审查的是：这里定义的结果变量是什么，异常图为何足以代表混杂，实际特征运算为何等价于相应统计量。门控在普通检测监督下有效，并不会自动回答这些问题。

### 2. 标准公式与局部门控之间缺少推导

论文§3.4的目标是：

\[
P(Y\mid do(F=f))
=\int P(Y\mid F=f,Z=z)\,p(z)\,dz.
\]

而实际计算是在一份已观察特征上乘异常函数，再送入预测头。它没有显式构造不同z条件下的预测，也没有直接计算上述积分。某些网络当然可能学习等价估计器，不能因没有显式积分就一概排除；但等价性必须证明或由明确的估计设计支持。

补充C.1假设归一化异常代理 \(\widehat Z\) 随样本增加一致地趋于真实混杂Z，又在命题2中令 \(W(z)=1+H(z)\)，声称由代理收敛可得 \(W(z)\to1\)。这一过渡不能成立：

\[
\widehat Z\to Z
\quad\not\Rightarrow\quad
1+H(\widehat Z)\to1.
\]

即使代理从一开始就完全准确，取论文允许的高异常分支ηe=2、τ=0.3，在z=0.6处也仍有W=2.2；在一段高异常区间中同样不趋于1。这里不是要求实际参数一定取2，而是说明现有假设没有推出声称的极限。还需要限制H的极限行为，而该文列出的假设未提供这种限制。

此外，映射的平滑性本身并不能把“一次特征乘法后预测”变成“对混杂分布平均条件预测”。因此，后续关于损失最小化可保证真实因果效应的结论，也不能直接继承这一尚未建立的等价关系。[补充C.1，命题2式(6)–(8)及命题3](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Duan_CHAL_Causal-guided_Hierarchical_CVPR_2026_supplemental.zip)。

这是对具体论证步骤的判断，不是对全部背景建模或因果推断的否定。

### 3. 变量语义还需要一致

主文式(1)以Y表示关键帧的监督目标；因果图解释又用背景导致false alarm来说明Z→Y。误报通常属于模型预测 \(\widehat Y\)，不是标签Y。如果Y改为模型输出，应重新说明所估计的干预效应为何对应研究问题；如果Y保持真实标签，则要解释该路径的生成语义。两种选法都不能悄悄混用。[主文§3.1](https://openaccess.thecvf.com/content/CVPR2026/papers/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.pdf)。

同样，HAL试图得到目标样异常，补充却把该异常作为背景混杂的一致代理。这需要测量或模型假设支持。由图像特征构造代理并不天然禁止因果推断；但代理的名字、分布可视化和最终检测提升不能代替代理可识别条件。

因此目前合适的引用是：**受后门调整启发的异常门控，在作者检测任务上有收益。** 不将其转述为已经证明消除了背景混杂、识别真实物理运动或获得无偏因果特征。

## 五、正结果应当接受到哪一步

主文表2中，DAUB-H的mAP50/F1从无模块23.48/46.36，经完整SNF达到43.76/60.90，再加HAL达到48.63/66.68，完整系统为54.28/74.15。这支持整套背景与异常学习流程有效，但逐级增加容量的链不能单独识别每一个理论解释。

更有区分力的是表3：在已有SNF+HAL的系统上替换末端机制。

| 设置 | DAUB-H mAP50 | DAUB-H F1 | 参数量 |
|---|---:|---:|---:|
| 无CRG | 48.63 | 66.68 | 14.17M |
| Self-Attention | 49.85 | 67.74 | 16.28M |
| CBAM | 50.45 | 68.85 | 15.13M |
| CRG | 54.28 | 74.15 | 15.69M |

CRG优于这两个替代设计，且参数比该Self-Attention对照少。不能把这个结果简单说成“只因为模型更大”；同样，它不能单凭检测分数证明因果图或后门等价。表2/3数值均由[正式原页](https://openaccess.thecvf.com/content/CVPR2026/papers/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.pdf)核对。

补充中的背景模型替换、合成异常强度和类型测试也提供支持。合成测试利用GT mask与设定运动生成受控样本，这属于测试构造，不是正常推理输入，更不等于模型学到了真实球对应。

作者也报告亮点误检、运动云边误报、近电线目标被背景表示吸收等失败。关于傅里叶带宽和目标混入背景的解释是作者提出的机制解释，不能把几个图例升级为已被独立隔离的普遍原因。[补充E/F](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Duan_CHAL_Causal-guided_Hierarchical_CVPR_2026_supplemental.zip)。

## 六、公开实现与论文必须分开记录的行为

| 环节 | 固定源码实际行为 | 对复现或机制解释的影响 |
|---|---|---|
| 门控系数 | τ=0.3、异常倍率0.8/0.1为普通常数 | 不能称该版本已经学习论文所述ηe>1、ηs<1与阈值 |
| 时间坐标 | forward固定current_time=1；time_scale经.item()参与坐标构造 | 该路径没有可学习的时间梯度，也没有逐槽输入不同时间坐标 |
| 时空attention | neck显式num_layers=1、空间窗口4；唯一层shift为0 | 该attention层没有执行shifted roll；类的两层demo不能代替生产配置 |
| 时间读出 | 训练时对全部时刻重建异常做softmax加权，eval只用末时刻 | 两条读出不相同；两者输入仍可仅含过去与当前 |
| 定位损失 | NWD与CIoU各取0.5后组合，再进入总损失 | 不能把这个版本的实际回归损失简写成纯NWD |

对应来源：[neck构造与forward](https://github.com/UESTC-nnLab/CHAL/blob/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6/nets/causal_anomaly_neck.py#L168)、[背景场](https://github.com/UESTC-nnLab/CHAL/blob/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6/nets/neural_background_field.py#L176)、[时空层与读出](https://github.com/UESTC-nnLab/CHAL/blob/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6/nets/stst.py#L337)、[损失](https://github.com/UESTC-nnLab/CHAL/blob/34661770a7e61b1534de8ceb42a4c8e3d43ebdf6/nets/yolo_training.py#L198)。

已追踪下载的train/test/predict入口，未发现覆盖这些neck默认值的调用。这里的时间坐标判断不等于整网没有时间信息：场景编码器本身读取多帧；没有shift也不等于整网没有跨区域信息：上游场景编码与卷积仍会交互。它们只是限定哪些具体机制实际被执行。

这些差异应在未来复现时决定采用哪一版本；现在没有运行作者实验，不能由公开默认实现与文稿不同直接断言表格造假或结果无效。

主文报告五帧512×512、RTX4090条件下15.69M参数、137.04GFLOPs、12.96FPS，未拆分背景场/时序处理成本。它不是本地DINO模型计时，也不足以直接定义一个便宜的motion模块。[主文表1](https://openaccess.thecvf.com/content/CVPR2026/papers/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.pdf)。

## 七、回到本项目的研究判断

CHAL将一个重要竞争解释变得具体：历史帧可能用于估计“这里正常应当是什么”，再改变当前帧的判别，而非显式寻找球的过去地址。这与[DMR的背景运动条件化](2026-09-12-coherent-motion-conditioning.md)不同，却同样要求我们区分时间上下文收益与精确对应收益。

现有中心标签与整网检测监督允许训练有用表示，但不能自动辨认哪个latent是真背景、真位移或真实混杂。若未来借鉴异常分支，值得检验的是它在自动定位路径中是否减少球场线、反光点、球员细节等误选，以及收益是否需要时序；没有必要先给latent附加更强物理解释。

反过来，球运动可预测、背景也会运动，“更异常”并不等于“更像球”。也不能用CHAL的红外bbox mAP替代本项目的球中心、模糊和大位移评价。它保留了一个可竞争的方法族，尚未决定当前主架构。

当前先完成[锁定的BlurBall真实历史/重复当前帧对照](../protocols/blurball-full-temporal-control-v1.md)。本次新增的是机制证据与论证边界，没有增加背景场、因果模块、五帧训练或新的GPU任务。后续是否需要异常建模，取决于这个对照及自动定位的具体错误，而不取决于论文模块名。
