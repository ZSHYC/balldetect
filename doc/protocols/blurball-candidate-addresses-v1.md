# BlurBall候选对应地址 v1

日期：2026-09-13。状态：运行前锁定。

## 问题与依据

[集合监督实验](../experiments/2026-09-13-blurball-candidate-set-supervision.md)修复了共同训练定位退步，但全局对应池化仍未超过同址历史。池化向量对已经提取的历史key的空间重排不变，因此没有显式保存对应地址；这不等于真实图像平移不会改变卷积特征。

本实验检验：在相同当前候选与历史外观上，保留多个对应地址是否比只保留匹配分数更有用。不是首次提出位移表示或多假设；MotionSqueeze、SELFY/STSS、PDC-Net、RoMa和MOSS已限制这类创新主张，来源与已读范围见[局部对应](../literature/2026-09-10-local-correspondence-baselines.md)、[原生匹配](../literature/2026-09-12-native-matching-confidence.md)、[高阶关系](../literature/2026-09-11-higher-order-motion.md)。本轮是表示诊断。

## 固定条件与两组输入

沿用[集合监督v2](blurball-candidate-residual-v2.md)的合法三帧、完整train/val、cross best3、当前K16、q、原坐标、stage0和集合损失。冻结骨干，不读取最终测试。旧同址残差best5作为性能参照，不重训。

每个当前候选、每个历史帧从全图72×128原生格的cosine取前16格，分数降序，并列优先较低扁平地址。16格可能聚集在同一峰附近，不宣称16个不同目标或不同运动模式。near为t−1，far为t−2。

- `scores_only`：真实前16分数；每格二维位移置零。
- `addressed`：同样分数；保留其绑定的二维归一化位移。

位移为历史格中心相对当前候选：`(cell_xy+.5)/(gridW,gridH) − (current_xy+.5)/(imageW,imageH)`。按near/far、rank顺序展开每格`(cos,dx/W,dy/H)`，共96标量。

两组均用`[Q,near_same,far_same] * sqrt(96)`的288维描述符与上述96标量拼接，单个384→32→1 GELU MLP，12,353参数。**只有描述符乘sqrt(96)，分数和位移不缩放。** 末层零初始化，输出加原候选相对logit。不是多头叠加，q与候选位置均不变。

同seed0、初始化、batch256、AdamW lr3e−4/weight_decay .01、30轮、全部38,854训练目标；31,887个V1且K16有4px内位置的目标监督。每轮评价全部14,192验证目标；选TP4最大、其次raw4、再最早epoch，含epoch0。记录last30训练全体及可监督子集。

`scores_only`比旧同址MLP多了真实匹配分数，也改变首层维度与初始化，因此不是纯容量对照。**地址增量只由addressed对scores_only判断。** 对旧同址组的差值是整体配方差值。

## 可证伪判据

保留addressed进入后续竞争要求：相对scores_only和旧同址组均提高全体PCK4与F1@4、PCK16不降，且相对cross四场PCK4不降。记录配对救回/破坏、位移分组、固定match21的202/74群体。单seed开发阳性仍非论文结论；阴性只否定这个冻结、排序后MLP读出的具体配方，不证明所有位移信息无用。

另报告前16历史格覆盖：当前K16有4px内候选时，选离GT最近的当前候选进行纯诊断，计算各历史可见条件下历史原生GT格与16px覆盖；同时报告包含当前候选遗漏的联合分母。GT不进入提取或推理。

不因结果扫温度、候选数、隐藏宽度、学习率或轮数。若无地址增益，停止直接拼接这套地址配方，先依据对应覆盖和训练拟合区分搜索丢失与读出问题。

## 实现、成本与验证

输出`outputs/blurball/spatial_interaction/candidate_addresses/`。旧query/同址特征完全复用。仅追加一次原生匹配提取，存float32 scores与int16 cells、真实目标ID，总计约163 MB。源RGB缓存读入、每个需要的源帧stage0只提取一次；不存稠密cost或重复窗口，不解码视频。训练只读紧凑缓存。

定向测试已验证并列地址顺序、分数地址绑定、非方格坐标及near/far顺序、描述符/标量分别缩放；复用既有零残差、集合loss测试。提取首个验证batch对照旧maxcos/地址，阻止坐标或特征接线变化；训练验证epoch0保留原输出、完整样本和best复现。另用真实缓存batch验证两条件分数相同、坐标开关和梯度有限。独立有界设计审查指出并已修正标量缩放及“纯容量对照”措辞。
