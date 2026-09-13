# BlurBall压缩后空间交互对照

状态：已锁定。版本：v1。锁定日期：2026-09-12，训练前。

依据：[时序输入控制与条件拆分](../experiments/2026-09-12-blurball-temporal-control.md)、[当前头的函数分析](../research/2026-09-12-motion-literature-reassessment.md)。本项是机制筛查，不预先宣称新的对应算法；不重训旧repeat_current组。

## 问题与唯一结构差异

已有结果支持真实历史在大位移条件下带来定位增量，但同时改变V0误出与V1拒绝。当前头把三帧的同地址特征先压缩并非线性处理，再线性汇合邻域。这不足以证明整网缺乏motion，值得检验的具体问题是：**在同样的32通道压缩后，让邻域汇合发生在非线性之前，能否改善细定位？**

令 `z=GN([f(t−2),f(t−1),f(t)])`，`g=GELU`。两组均以相同注册顺序构造下列带bias卷积，权重初始化逐张量相同：

| 层 | 配置 |
|---|---|
| P | 1×1，576→32 |
| A | 1×1，32→32 |
| S | 3×3，32→32，padding=1 |
| R | 1×1，32→64 |

控制 `same_address`：`R(S(g(A(g(P(z))))))`。

处理 `cross_address`：`R(g(S(A(g(P(z))))))`。

只移动第二个GELU，P/A/S/R的先后顺序相同。两组均输出36×64粗格的64通道，经原PixelShuffle(8)得到288×512位置格。位置头各30,880参数、70,778,880卷积MAC/三帧窗口；全模型各1,266,497参数。MAC不包括GN、激活、absence或前缀，也不等于实际延迟。

对自由归一化输入z，控制头可写成各地址函数的线性和，不同地址之间的混合二阶导数为零；处理头一般不为零。这一命题仅针对GN之后的子函数。前缀已有空间感受野，GN还引入全图统计耦合，不能把它推广为原图到logit的不可交互结论。两组参数预算一致不代表可实现函数族、优化难度或有效容量完全相同；相邻线性层的组合与因子化约束也随非线性位置改变，这是本次干预本身的组成部分。

本项没有显式相似度、位移候选、目标身份或current-query/history-key角色；阳性最多支持这套配方的压缩后邻域非线性方向，不能直接证明物理对应或新颖性。近邻边界沿用[TSM/GSM/GSF](../literature/2026-09-12-lightweight-temporal-routing.md)、[STSS与局部对应](../literature/2026-09-10-local-correspondence-baselines.md)、[MIST/DQAligner](../literature/tiny_motion_evidence.md)的已有审查，不把简单卷积顺序对照作为论文贡献。

## 数据、时间与训练

继承[BlurBall因果中点v2](blurball-causal-midpoint-v2.md)：原标签中点，match00–17训练、18–21验证；22–25保持封存。完整合法训练目标38,854、验证目标14,192；全部比赛与batch保留。内部边界、rally起点及真实帧号不变。两组输入均为真实 `[t−2,t−1,t]`，预测t，零未来帧；V0依原协议监督缺失类，不当作可定位中点。

两组从同一官方DINOv3 ConvNeXt-Tiny LVD权重的stages0–1及seed0新头开始，共同微调30轮，包含epoch0验证。batch8、float32、无AMP/增强、AdamW、head/prefix学习率3e−4/1e−5、weight decay0.01、固定学习率及前缀eval模式沿用原设置。相同seed控制模型初始化与NumPy打乱顺序，不承诺所有CUDA执行逐bit确定。

复用 `data/cache/blurball/rgb_512x288_all_h2/` 的逐帧RGB mmap，不重复解码或缓存会失效的微调特征。训练仍逐窗口显式三槽，验证沿用批内唯一帧encode与原顺序恢复。两组顺序运行，使用 `last.pt` 完整轮次恢复机制；正式训练不从smoke更新后的参数开始。

旧history best6仅作为实际效用参照。它的位置头为36,960参数，总参数1,272,577，结构和参数预算不同，不能用旧头直接代替新控制组。

## 选优、读出与评价

保留原argmax本地F1@4最大、其次F1@8、最早同分选优，包含epoch0；不按本次更有利的raw PCK另选checkpoint。完整保存30轮的验证记录，另查看固定epoch30的argmax raw PCK，判断方向是否只出现在各自F1选中的epoch。

两组最佳checkpoint各自保存原argmax预测，再应用共同的固定15×15、T=1局部重心，使用[局部读出v1](blurball-local-readout-v1.md)的公式及紧凑logit缓存。主机制指标是全部V1的固定重心raw PCK@4；并列报告argmax、raw PCK@16、本地F1@4、V0误出和V1拒绝。容差严格原图 `<4/<8/<16px`，不改变原q≥0.5输出规则。

absence仍为 `Linear(mean(GN(z))) + log(位置格数)`，但q是联合softmax得到的可见中点概率，位置logit和共同微调都可以改变q。因此相同absence公式不意味着相同q，不能把F1变化自动归为位置交互收益。

沿用已有全体、四个match、五个l组、历史可见状态及d1/d2分组，完整配对统计救回与破坏。主条件群体为当前与t−1均V1且d1≥16原像素，不用GT筛选模型输入或删训练样本。保留低位移组代价，不重新构造稀疏笛卡尔分桶或只挑有利比赛。

## 结果决定下一步

保留该结构方向需要：cross相对same的总体固定重心raw PCK@4提高，大位移组总体及四场各自的@4配对净增均为正，总体raw PCK@16不下降。若argmax或固定epoch30的方向相反，按依赖读出/选优的混合结果报告，不宣称稳健改进。

要实际替换现有头，还需达到旧history best6的固定重心raw PCK@4，且本地F1@4不下降；这只是效用门槛，不是与旧头同容量的机制归因。达到门槛后先保留这一简单结构，后续根据仍存在的失败决定下一项实验，不立即堆attention、DCN或可靠性模块。

若cross优于same但低于旧头，仅保留顺序差异的证据，不替换当前系统。若两组持平、反向或收益仅集中于单场，停止在当前32通道、3×3、30轮配方下继续扩该路线；这不否定压缩前交互、更广搜索或其他训练条件。若只改善F1/q而不改善raw位置，不支持细定位机制。单seed与四个开发match仅支持开发判断，不能代替论文级多seed、跨球种及backbone×motion归因。

## 最小验证与产物

训练前检查真实区别是否被实现：相同seed初始张量与参数/MAC一致；自由z的跨地址混合导数一零一非零；原缺失logit、PixelShuffle形状及V1/V0梯度正确。再用实际完整batch8检查全模型loss/梯度有限、初始化一致、目标数和显存可用；失败则修实现，不启动正式运行。只验证本次变化，不重跑数据发布审计。

实现、实际命令、进程与结果见[实验记录](../experiments/2026-09-12-blurball-spatial-interaction.md)。输出分别为 `outputs/blurball/dino_same_address_seed0/` 和 `outputs/blurball/dino_cross_address_seed0/`；共享一次性启动/验证记录放 `outputs/blurball/spatial_interaction/`。不添加依赖、哈希或实验管理框架。

## 2026-09-13用户执行调整

用户要求：若cross最佳轮次很靠前，则停止后续训练并直接推进。核实best为epoch3、epoch4–21连续18轮未超过后，结束剩余训练，使用原best3完成全部预测和固定读出。same实际30轮、cross实际完成21轮，原配置、日志及权重保留。

上述v1保留为原先锁定的方案；本次后续结果标为开发比较。另核对共同epoch0–21选优和固定epoch21的argmax：两组保存best是否仍是该共同范围内的原F1最优，由分析入口直接验证。共同范围是用户调整后明确的分析条件，不改称训练前预设；cross没有epoch30结果，不能填入原固定epoch30配对。其余全部比赛/目标、时间窗口、读出、q、容差和条件群体保持。实际取舍见实验记录。
