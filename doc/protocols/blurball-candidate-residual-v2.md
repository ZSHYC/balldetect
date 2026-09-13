# BlurBall候选残差读出 v2：可接受位置集合监督

日期：2026-09-13。状态：运行前锁定，三条件各30轮已完成；[结果与表示约束](../experiments/2026-09-13-blurball-candidate-set-supervision.md)。

## 只改变一个问题

[v1训练结果](../experiments/2026-09-13-blurball-candidate-residual.md)显示三条件soft CE下降，但第30轮训练PCK4从93.94%降至约89.3%；可监督目标平均9.78%软标签质量在4px外。这个事实提示检验监督目标，尚未证明软标签是唯一原因。

本项沿用[v1](blurball-candidate-residual-v1.md)全部数据、原帧/内部连续边界、固定cross best3、K16、stage0缓存、模型、三个输入条件、q/xy、seed0、相同初始化、AdamW lr3e−4与weight_decay .01、batch256、30epoch及验证选优。**唯一训练改动是位置损失。** 三组都从零残差初始化独立训练，不从v1的best/last接训。旧run与旧预测保持。

当前分支、同地址历史、全图softmax对应的机制与近邻来源继续引用v1，不新增网络或创新主张。current只关闭新增残差的历史特征，原cross候选/分数仍含真实三帧信息。

## 集合概率目标

令当前候选原图位置为`x_k`，GT为`g`，可接受集合`A={k: ||x_k−g||<4px}`。只对当前V1且A非空计算loss，与v1可监督mask一致。原双精度坐标先计算严格距离，再形成布尔集合；不因转float32改变边缘样本。

\[
L= -\log\sum_{k\in A}\operatorname{softmax}(z)_k
 =\operatorname{LSE}(z)-\operatorname{LSE}(z_A).
\]

这里`z`为原peak logit差加残差。多个合格候选不要求平分质量，也不强制选择其中离GT最近的一个；集合外无正标签质量。计算用稳定logsumexp，先减每行最大值，随后正集合外mask为−inf。调用方只传入有正例的已监督行，V0或漏候选行沿用无位置loss处理。

该损失是集合事件的负对数概率，不是新的motion损失。概率质量在集合中增加不保证argmax一定合格，例如两个合格候选各0.3、一个不合格候选0.4时仍输出错误。故不以loss下降替代实际PCK。

全部38,854训练目标仍每轮遍历；31,887个可监督目标负责位置loss。全部14,192验证目标保留，含V0、不可达4px候选和原q拒绝。GT集合仅训练使用，不作为推理候选筛选或门控。

## 测量与判据

三个条件均完整报告epoch0–30、best复算、验证TP4/raw4/PCK16/F1@4与原cross及彼此的配对；预定固定202/74群体照常报告，不能按它们选模型。选择仍为验证TP4、raw4、最早epoch，含epoch0。

在加载best前，明确从第30轮`last.pt`评价全部训练目标，同时报告：全部训练V1的PCK4，以及V1且K16可达4px子集的PCK4。前者保留候选遗漏上限，后者直接检验loss是否学会把实际argmax放入可接受集合。不能用best训练位置代替该终点诊断。

先判断v2是否改变v1训练集精位置退步，再讨论对应增量。要保留correspondence作为后续模型候选，仍要求其相对cross/current/stationary均提高全体PCK4/F1@4、PCK16不降，且相对cross四场PCK4不降。单seed开发阳性也不能证明最终motion贡献，不读取最终测试。

两个loss使用相同固定优化超参，但没有校准梯度尺度。因此v1/v2差值归于“监督目标在这套固定训练条件下的变化”，不是控制了所有优化效应的纯因果分解。不扫lr、阈值、温度、隐藏宽度或epoch预算。若训练位置仍下降，停止直接以当前读出训练阴性评价历史信息，先定位剩余共同失败；若训练改善而验证不改善，转向候选拟合与跨比赛泛化证据，不继续堆池化模块。

## 实现与运行

复用`candidate_residual/features/`，无骨干forward或视频解码。`candidate_readout.py`增加集合loss；已有训练入口以实际变化参数`--loss set`选择v2，默认soft保留v1复现；恢复时loss字段必须相同。比较入口从run记录读取共同协议，拒绝混合不同协议的三条件。

输出位于`outputs/blurball/spatial_interaction/candidate_residual/set_supervision/{current,stationary,correspondence}/`，保留配置、best/last、历史日志、完整预测、固定last训练选择及比较结果。旧缓存和旧run不移动。

定向验证：手算集合概率与梯度、集合内质量重分配不改变loss、稳定分数平移；复用零残差/near-far/原软标签回归。真实缓存batch验证集合loss有限且可反传；三个run继续验证epoch0与原坐标一致、best复现和完整目标数。已有缓存身份、对应接线和恢复下一步验证继续有效，不重复提取或重跑与loss无关的检查。

独立数学/归因审查确认该比较可执行，并要求同时保留固定last的训练全体与可监督子集位置指标；本协议已纳入。
