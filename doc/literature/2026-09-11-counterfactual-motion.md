# CMRTrack 的目标擦除反事实运动：对真实历史控制的边界补读

日期：2026-09-11。状态：本次补读完成；未复现、未训练、未下载数据。本文只更新对一篇近邻工作的证据强度，不追改[2026-09-09微小目标运动审查](tiny_motion_evidence.md)当时的阅读深度。

## 结论

Yuehai Chen、Jian Lan、Yuan Wei 的 **“Counterfactual Motion Reliability Learning for Robust UAV Tracking”**（CMRTrack，arXiv:2607.23209 **v1**，2026-07-25）已可读取完整官方PDF，当前是预印本而非已证实会议论文。[arXiv记录](https://arxiv.org/abs/2607.23209)，[v1 PDF](https://arxiv.org/pdf/2607.23209v1)。论文直接覆盖了“从真实历史中擦去已知目标、用事实/反事实差训练运动可靠性”的基本思路，因此不能声称首次可靠motion、首次target-erased history，或首次以背景竞争压制不可靠时间差。

它不取代本项目的[真实历史/重复当前帧重训控制](../protocols/tennis-full-temporal-control-v1.md)。后者问的是：在同一DINO+SpatialProbe参数化和完整训练协议下，历史**真实像素**相对`[t,t,t]`是否带来定位增量；CMRTrack在另一个、已初始化单目标跟踪系统中，用带GT历史框的训练期像素干预来学习一个内部motion gate。两者的反事实对象、可用信息和可识别结论不同。

## 实际读到的任务与状态语义

论文开头将视觉跟踪定义为“给定初始状态，在后续帧定位任意目标”；方法输入是template `Z`、当前search region `X_t`和一帧历史search region `X_{t-1}`（§III-A、式1–3）。这属于**单目标、给定初始化的tracking**，不是自动球发现/逐帧检测。论文采用OSTrack的单流跟踪范式；作为解释该范式所需的原方法来源，OSTrack官方实现把`info['init_bbox']`存为初始状态，随后以当前预测`self.state`裁剪下一帧search region：[OSTrack固定源码L55–96](https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/test/tracker/ostrack.py#L55-L96)；原论文为[ECCV 2022 OSTrack PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820332.pdf)。CMRTrack自身没有在论文中给出可访问的作者代码链接；本次也未通过GitHub精确项目/代码检索发现作者仓库，故不把OSTrack源码误报为CMRTrack实现。

- **历史、未来和状态边界。** CMRTrack在推理只使用`X_t`与前一search region；处理一帧后把当前search region存为下一帧历史（§III-F）。没有未来帧，反事实支路也不在推理执行。论文没有明确说明多clip、切镜或视频边界如何重置历史；它的任务是连续单目标序列，不能把其隐含tracking state直接迁移到本项目的rally/clip边界。
- **GT的作用。** 标准tracking初始化给定目标状态；训练期又用历史target box `b_{t-1}`产生擦除区域，并从当前GT box生成target heatmap `G_t`（§III-C，式4–12）。论文未提供源码字段以核对数据管线，但其数学定义和“所有motion loss只在valid positive target samples计算”的说明足以表明：反事实训练依赖有标注的正目标历史，不能在本项目仅凭模型预测无代价构造同等监督。
- **输出对象。** 最终输出仍是预测bbox `B_t`和经融合的score map `S_t^final`（§III-E–F）。它不是自动检测器的presence/absence softmax，也没有把未初始化候选中“哪个是球”作为任务。

## CMRTrack 的反事实到底做了什么

### 像素层擦除与事实参考

实际motion输入首先是两张search image逐通道绝对差的平均：`D_t=(1/C)Σ_c |X_t^c-X_{t-1}^c|`。轻量卷积编码器（两个3×3 Conv–BN–ReLU块、一个1×1 Conv、sigmoid）把它变为当前目标score-map尺度的`M_t∈[0,1]^(1×H×W)`；`M_t`以由当前GT bbox生成的热图`G_t`作focal loss监督（§III-Ca，式2–4）。因此它是受当前目标位置监督的**learned temporal-difference response map**，不是flow、显式patch correspondence、中心位移向量，亦不是轨迹先验本身。

反事实不是擦feature、擦轨迹或遮掉当前帧，而是在**训练期历史像素**中把以`γ b_{t-1}`缩放的历史目标框替换为该历史search region的全局均值：

```text
X̃_(t−1) = E(X_(t−1), γ b_(t−1))
M̃_t     = Φ(X_t, X̃_(t−1))
```

（§III-Cb，式5–6）。当前`X_t`保持不变；同一个`Φ`生成`M̃_t`。默认参数为`γ=1.0`，并额外比较0.8/1.2/1.5（§IV-G，表VI）。因此“擦除依据GT还是预测”的准确答案是：**训练期由历史标注目标框定义，不是由tracker预测框定义；推理没有擦除。** 它只能检验“令同一训练样本的已标注历史target像素被均值取代”这一干预，不能直接证明物理运动因果，也不能在无可靠bbox标注的帧上执行同一监督。

### reliability target、损失与实际部署信息

论文不把`M̃_t`强压为零。它在当前GT前景/背景热图上汇总四种平均响应：事实前景`sf=Avg(M_t⊙G_t)`、反事实前景`ŝf=Avg(M̃_t⊙G_t)`、事实背景`sb=Avg(M_t⊙Ḡ_t)`、反事实背景`ŝb=Avg(M̃_t⊙Ḡ_t)`，以

```text
L_cf    = max(0, m − sf + ŝf)
L_cf-bg = max(0, m − sf + max(sb, ŝb))
y_r     = sigmoid((sf − max(ŝf, sb)) / τ)
L_rel   = BCE(r_t, y_r)
L_MEE   = L_motion + L_cf + L_cf-bg + L_rel
```

进行训练（§III-Cc，式7–13）。`y_r`是由**GT区域上的事实/反事实响应差**构成的soft训练目标，不是“历史中心位于某search support内”的标注概率，也没有概率校准、coverage、risk–coverage或no-match的报告。

部署时，`M_t`一方面残差调制search tokens `Ẽ_X=E_X⊙(1+αM̂_t)`；另一方面和appearance score map `S_t`经卷积残差融合得到`S_t^m`。内部gate `r_t=MLP([max(S_t), max(M_t), ||SoftArgmax(S_t)-SoftArgmax(M_t)||/√2])`，最后

```text
S_t^final = S_t + β r_t (S_t^m − S_t).
```

（§III-D–F，式14–24）。因此它的“reliability”是一个由appearance/motion峰值和两峰中心距离产生、经上述训练target监督的**连续融合权重**；不是拒绝输出、不匹配分类或已校准的correspondence置信度。所有motion相关损失只用于valid positive target samples；论文虽评价state accuracy并展示out-of-view/occlusion属性，仍未给出“历史无有效视觉对应 vs 候选漏失 vs 目标不存在”三类no-match语义和监督的拆分。

## 实验证据覆盖到哪里，以及还没有覆盖什么

论文在Anti-UAV/Anti-UAV410红外单目标跟踪上报告了整体、属性、擦除策略与模块消融。Anti-UAV410上，OSTrack为53.7 AUC，完整CMRTrack为67.3；去counterfactual learning为63.0，去reliability supervision为60.3（§IV-E，表IV）。同一骨干的motion基线表中，raw temporal difference为56.4、supervised motion map为60.8、无reliability的motion fusion为62.7、完整模型为67.3 AUC（§IV-F，表V）。这提供了**该跟踪协议内**“仅加差分不够，带GT擦除的训练约束和gate与更高tracking分数相关”的消融证据；不能分解成独立的自动发现增益、真实对应准确率，或球定位F1增益。

论文也正面检查了常见的擦除伪影顾虑：相同训练/评估下，zero fill、random-noise、Gaussian blur、local mean、global mean的AUC依次为65.8、65.4、66.2、66.7、67.3（§IV-D，表III），作者据此选择global mean。这个对照支持“结果不只来自这两种最明显的人工边缘/噪声填充”，但不能排除所有mask统计改变、crop边界、均值填充或训练数据相关性的解释；论文没有对自然遮挡、随机同尺度背景patch替换、或完全不依赖GT框的反事实做独立判别。

对于目标尺度与背景竞争，属性分析包含`tiny-size`、fast motion、dynamic background clutter和thermal crossover（§IV-C），且`L_cf-bg`明确压低事实/反事实背景响应。故“作者完全没有测背景/小目标”是不成立的。其局限是：正文未给出可供球任务换算的tiny像素尺寸或只含中心标签时的擦除几何，也未把固定背景误峰、候选覆盖、自动detector漏检和tracking drift分别计数。已有证据不能排除这些机制在体育RGB几像素球上的不同作用，也不能把attribute平均曲线替代本项目的逐clip错误审计。

### 空间支撑、大位移与ROI坐标：全文能说与不能说的

**全文明确的事实。** CMRTrack先在`X_t,X_{t-1}`上计算像素差，`Φ`由两个3×3 Conv–BN–ReLU块和一个1×1预测层构成；预测响应被resize到score-map尺度，之后还会resize/flatten到token布局（§III-Ca、D）。训练时template/search分别以factor 2.0/6.0裁剪并resize为128×128/256×256（§IV-A）。全文没有写出`Φ`的stride、padding、dilation、pooling、差分与score map之间的精确resize算子，也没有披露`X_t`与`X_{t-1}`的ROI中心和坐标变换如何对齐；作者代码未公开，不能从OSTrack的实现替CMRTrack补这些细节。实验只按fast-motion等属性报告tracking曲线，没有将历史/当前GT区域的crop坐标距离、`Φ`的实际感受野或二者相对关系分桶/消融。

**可条件推导的边界，不是已发现bug。** 若两张search crop已经处在同一坐标系，且把未披露的全局空间混合排除、以推理态固定BN统计考察局部卷积，那么旧GT擦除只改变`γb_{t-1}`内的差分输入。局部`Φ`的某个输出仅在其感受野（以及后续resize的支撑）接触该被改区域时才会改变；而`sf=Avg(M_t⊙G_t)`按当前GT热图加权。因此若`G_t`的非零取样支撑与旧擦除区域在该crop坐标中相距超过这种有效支撑，`sf`与`ŝf`不会因这次擦除自然产生差异。这里的取样支撑由实际热图决定，不能未经核对就等同于GT矩形框。训练态BN的跨空间/批统计、以及论文未说明的padding/resize细节会破坏这个简化结论，故不能把它升级为对论文实现的断言。

原视频中的大位移不能直接代入上述条件：ROI若随初始/预测tracking state重心裁剪，两个目标可在各自crop中仍靠近或重叠；反之drift也会改变相对坐标。CMRTrack只报告crop factor与连续历史，不公开该对齐实现，因而本次无法计算其样本是否满足或违反局部支撑条件。结论只是，**target-erased GT区域的响应差不能天然等同跨位置correspondence或大位移可靠性**；它证明的是在该训练ROI、该监督和实际有效支撑下，擦去历史目标像素是否改变当前受监督motion response。

## 与本项目两个已锁定诊断的因果差异

| 问题 | CMRTrack | 本项目的真实历史/重复当前帧重训 | 本项目的空间读出诊断 |
|---|---|---|---|
| 干预对象 | 训练期把**GT历史框内像素**改为全局均值 | 从同一官方前缀和同seed随机新头分别训练`[t−2,t−1,t]`与`[t,t,t]` | 不干预输入或训练；读取既有空间logits |
| 主问题 | 用target-erased reference能否训练一个更有用的motion gate | 真实历史视觉内容是否给当前完整定位系统带来增量 | 高`q`错位是否来自分散位置分布还是集中背景峰 |
| 推理可用信息 | template、当前和一帧历史search region；无反事实支路 | 三个真实历史/当前输入槽或重复当前帧 | 固定DINO历史组的logits与已有`q` |
| GT特权 | 初始tracking state；训练擦除框、当前热图和前景/背景统计 | 位置/absence监督；没有GT历史框像素擦除 | 只在事后评价使用GT，统计本身以预测argmax为中心 |
| 可支持的结论 | 在Anti-UAV tracking中，反事实约束+gate与更好跟踪关联 | 是否存在真实历史视觉增量 | 现有空间分布是否含超出`q`的排序信息 |

所以，重复当前帧重训即使得出“历史有增量”，也**不会**识别该增量是目标诱发差分、背景变化、窗口内运动先验，还是独立历史输入带来的函数类及优化变化；相同参数量的边界见[控制实验中的代数检查](../experiments/2026-09-11-full-temporal-control.md)。CMRTrack式擦除若要回答这些细因果问题，还需一个有GT历史框的训练干预及匹配的控制。反过来，CMRTrack的消融没有对同一参数化模型重训`[t,t]`，没有自动检测起点，也没有与本项目72×128位置/absence协议相同的任务，因而不能预先回答重复当前帧控制的结果。

这也澄清[读出集中度实验](../experiments/2026-09-11-readout-concentration.md)的边界：`max(c)`、熵和`q*m16`只诊断当前位置分布是否区分正确/错位，既不实现CMRTrack的target-erased训练，也不能当作其`r_t`或可靠motion的证据。

## 会改变的研究判断与不可迁移前提

**应更新的判断：**

1. “counterfactual target-erased history训练可靠motion、以背景竞争约束事实/反事实响应、按可靠度融合motion”已有极近直接先例。单凭这些基本机制不能声称创新，若未来做同类干预必须以CMRTrack为近邻并做明确差异实验。
2. 继续执行重复当前帧重训。它仍是最小、无GT擦除特权的输入内容控制；结果若无历史增量，应暂停新motion gate。即使有增量，也不能据此称可靠motion或correspondence已得到证实。
3. 当前不应立即堆CMRTrack式模块。它需要bbox历史标注、连续tracking state、当前GT热图和额外融合/gate；这些改变会把“真实历史在固定训练规则下是否有用”的问题与新容量、训练特权和tracking先验混在一起。

**不能直接迁移到体育几像素球的前提：**

- 只有中心标签时没有历史bbox，无法定义`γb_{t-1}`的擦除面积、前景/背景heatmap区域，亦不能把一格硬位置标签伪装成框面积监督；任何球尺度代理都须另行验证。
- 单目标tracking的初始框/template及上帧state已经给出“目标是谁、search在哪里”的先验。它的tracking成功不等于从全画面自动发现球，也不等于候选覆盖或presence/absence判断成功。
- `M_t`来自相邻search crop的像素差，最终由当前目标热图直接监督；它没有建立跨帧dense correspondence、中心位移真值或可拒绝的no-match类别。不能将其效果称为“验证了球的物理位移/匹配可靠性”。
- 论文没有明说片段边界的状态重置；本项目仍必须按连续clip/rally重置所有窗口、缓存和轨迹状态，不能把tracker的长序列状态跨边界沿用。

## 检索与证据边界

本次只补读CMRTrack官方v1 PDF、arXiv元数据，以及为解释GT初始化tracking所必需的OSTrack官方论文/固定源码；没有新增泛综述。CMRTrack PDF无作者代码链接，精确GitHub项目/代码检索亦未找到作者实现，因此源码级crop、模板更新、dataset parser和边界重置均保持“未证实”。论文所有数字都属于其Anti-UAV协议与作者报告，未复现；它们不构成对网球、羽毛球、乒乓球或本项目数据协议的性能预期。
