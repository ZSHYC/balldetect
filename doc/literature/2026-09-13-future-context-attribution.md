# 前瞻、输入长度与MIMO滑窗步长的归因

日期：2026-09-13。范围：补读与当前五帧实验直接有关的TrackNetV5和BlurBall时序配置；不重复全部TrackNet版本审查。用户已允许未来帧，旧文献笔记中的“三帧因果”是当时本地实验设置，不再是项目总边界。

## 两项一手证据

**TrackNetV5**，*Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking*，arXiv:2512.02789v4，2026-01-13。方法与实现使用三张连续RGB图，MDD与R-STR消融保持该输入范围。Table4–5是模块消融，不能证明五帧优于三帧；三输出MIMO还需分别说明每个输出的未来信息范围。[方法与输入](https://arxiv.org/html/2512.02789v4#S3)、[消融](https://arxiv.org/html/2512.02789v4#S4.SS3)。

**BlurBall**，*Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking*，arXiv:2509.18387v3。Table2–3比较 `steps=3` 与 `steps=1`；例如Mid标签WASB的F1为95.77与96.00。这个表不能仅凭数字“1/3”解释为单帧/三帧输入。[论文实验](https://arxiv.org/html/2509.18387v3#S4.SS3)。

作者仓库的固定版本 `2f0f5496f7ba4b5b1a36790749935121b2ce972d` 进一步明确：WASB与BlurBall均配置 `frames_in: 3`、`frames_out: 3`，而 `step` 是独立的detector参数。数据读取用连续三帧，按 `i % step == 0` 选择窗口；评估把同一图像路径的不同输出收集成候选，再交给tracker选择。[WASB配置](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/configs/model/wasb.yaml)、[数据读取](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/datasets/tabletennis.py#L282-L307)、[评估收集](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/runners/eval.py#L67-L79)、[候选选择](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/trackers/online.py#L53-L74)。

## 从源码推得的时间语义

对远离clip边界的内部目标帧t，step=1可从三个窗口收集输出：

| 产生当前候选的输入 | 目标在输出中的槽位 | 相对t的前瞻 |
| --- | --- | --- |
| [t−2,t−1,t] | 末槽 | 0帧 |
| [t−1,t,t+1] | 中槽 | 1帧 |
| [t,t+1,t+2] | 首槽 | 2帧 |

因此每个网络窗口仍是三输入，但同一目标的候选集合可以间接利用五张不同源图。step=3的非重叠窗口则令每个目标只属于一个窗口，其前瞻取决于输出槽位。这是依据源码作出的时间依赖推论，不能写成作者进行了一次五输入网络消融。

step=1与step=3同时改变了重复预测次数、可用候选及其时间信息；tracker选取后的F1不能单独归给更多输入帧、某个motion模块或更长前瞻。候选融合的输入并集与逐窗口输入长度需分别记录，速度也应覆盖实际重复窗口计算。已有GPU特征复用不能被计成新增视觉观测。

## 对本项目的具体决定

现有直接近邻没有在本次查阅范围内提供“固定其他条件时五帧优于三帧”的证据。这个结论是有界检索结果，不代表所有文献都没有此类实验；V6完整实现的已知证据缺口仍见[旧记录](2026-09-12-tracknet-source-gaps.md)。

继续[中心五帧与因果五帧实验](../protocols/blurball-five-frame-context-v1.md)，固定单目标输出、共同目标、模型及预算，不混入MIMO投票的收益。这一比较测量“同五帧预算下中心上下文替换因果上下文”的效用：中心化还会缩短支撑相对目标的绝对时间距离，因此不能宣称已单独隔离时间方向符号或证明motion创新。

已有BIRD、STSN的双向信息与任务监督采样机制仍是近邻，详见[全文笔记](2026-09-12-task-supervised-alignment.md)。允许未来帧使其原有时间语义更接近当前任务，但原数据、计算、框监督和逐帧传播差异仍在；不因为解除因果约束就直接移植整套模型。
