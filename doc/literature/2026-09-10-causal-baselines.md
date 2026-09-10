# 三帧因果基线：相关方法的输入与输出边界

检索/源码访问：2026-09-10。性质：本次实现的定向证据，不重新声称全面综述。核对官方原稿/代码中输入、差分路径与输出目标；版本访问范围如下。

## 直接改变本次设计的事实

1. **TrackNetV4**：三帧 MIMO 同时输出三张热图。相邻灰度差经绝对值、PN 非线性再乘性门控热图，不能简化成线性输入拼接。当前项目 `[t−2,t−1,t]→t` 的 latest-slot 选择不等于复现其全部 MIMO 评价。[原稿](https://arxiv.org/html/2409.14543)、[固定源码](https://github.com/TrackNetV4/TrackNetV4/blob/cb7eea7988474771ceac7e880bbffc35bfa87bca/src/models/TrackNetV4.py)。
2. **TrackNetV5**：核对 arXiv v4（2026-01-13），使用三帧 MIMO，MDD 将 signed RGB difference 作正/负 ReLU 分解，再配合 attention 与 R-STR residual refinement。中心帧的输入有未来帧；不能用该成绩直接横比本项目无前瞻目标。[原稿 Sec.3.1–3.3](https://arxiv.org/html/2512.02789v4)。
3. **TrackNetV6**：固定公开代码 `9a6e4b0443abe11edb52cf3a0297f49dafc17f11` 的 demo 是三张 RGB 拼接9通道、输出3张热图。README 说明完整训练/推理尚未发布，demo 可能与论文不同；不能据 demo 假定其完整论文 causal 口径，也不能把其解码尺度操作自动解释为显式帧间 correspondence。[官方说明](https://github.com/Gi-gigi/TrackNetV6/blob/9a6e4b0443abe11edb52cf3a0297f49dafc17f11/README.md#L100-L105)、[模型代码](https://github.com/Gi-gigi/TrackNetV6/blob/9a6e4b0443abe11edb52cf3a0297f49dafc17f11/TrackNet-main/models/TrackNetV6_Beta.py)。
4. **BlurBall**：核对 arXiv v2（2026-02-15）及官方 `2f0f5496f7ba4b5b1a36790749935121b2ce972d`。三帧输入/输出，9通道 RGB stack + HRNet/SE；默认不等于显式匹配。重叠窗口若聚合同一帧的多次输出会改变前瞻范围，必须另行定义。[原稿](https://arxiv.org/html/2509.18387v2)、[官方模型代码](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/models/blurball.py)。

## 本次采用与不做的归因

先采用严格因果 frozen feature stack，对每个目标只作一次预测。它检验真实历史视觉信息的增量，不声称 correspondence 或新颖性。后续正式 TrackNet 复现仍要沿各方法的原生输入/输出协议单列，不能把本轮自定义 head 写成 TrackNetV4/5/6 的复现结果。

对已经分别归一化的特征 `z0,z1,z2`，若在任意线性首层前追加 `z1-z0,z2-z1`，这些差分都可由首层权重吸收；在保留完整原特征时没有新增观测或表达能力。若在差分后另作归一化、绝对值、正负分解或乘法门控，函数类/归纳偏置发生变化，不能再作上述线性等价推断。V4/V5 属于后一种，不能把本项目的线性论证用来否定它们。

重复当前帧是相同三帧头的参数化/优化对照，并不是去除现实运动的反事实。结果报告必须保留 current→stack 的净收益和反向损失，不能只对单帧错误集合展示救回。
