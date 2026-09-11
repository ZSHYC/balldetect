# 原始逐帧目标上的历史年龄与可用运动证据

日期：2026-09-11。状态：协议锁定，合成检查与独立只读审查通过；实际元数据诊断尚未运行。

执行[历史年龄协议v1](../protocols/tennis-history-age-v1.md)。这是[固定细节读出联合收益未通过](2026-09-11-frozen-detail-readout.md)之后的独立研究问题：历史更早时，旧局部支持不足是否仍伴随可用可见证据？本轮不修改模型，也不把时间间隔造成的压力作为新方法或原生帧率失败的证明。

## 为什么现有统计还没有回答

独立审查确认，[原Tennis位移统计](2026-09-10-displacement.md)比较Δ1/2/4各自的双端合法对；[完整窗口几何](2026-09-11-search-support.md)只处理原`s=1`，没有共同当前目标上的Δ8/16、长历史visibility与两支联合覆盖。[Shuttle训练侧](2026-09-10-shuttlecock-displacement.md)同样已有Δ1/2/4，不需要为这个Tennis问题再次扫描。

比较固定`s=1/2/4/8`的`[t−2s,t−s,t]`，先记录各自自然合法窗口，再固定四者共有的当前目标。标签不决定历史帧是否可输入。近/远旧半径始终为R2/R4；原几何代码中的`2*delta`只适用于旧Δ1/2，不能照搬成随s扩大的半径。

当前预测节奏保持原逐帧，只改变历史年龄；不把较早历史的选择叫视频降帧率，也不把更多帧累计位移叫更高物理球速。较近帧原本可用，未来若提出旧历史方法，必须解释其计算、记忆或观测需求。

## 最小实现与验证

[analyze_tennis_history_age.py](../../scripts/analyze_tennis_history_age.py)只读完整RGB缓存的metadata。该缓存14,160源帧，标签过滤数0；仅读这一数量是为确认历史可用性没有在输入前被删标签改变，不是重新做数据完整性扫描。复用原像素中心到格的`grid_targets`，以原始身份索引构造窗口，不读图像、预测或模型。

[一个合成检查](../../tests/test_tennis_history_age.py)先在入口不存在时失败，随后通过：缺失Clip2/frame8不会从Clip1借帧，不按行号重编号；无球历史仍保留；共同目标交集正确；s=8时近端3格仍超出R2；当前VC2仅进入合法位置对而不混入当前VC1支持分母；空分组保持明确。

实际诊断将在提交实现后运行：

```bash
conda activate zshihyc
python tests/test_tennis_history_age.py
python scripts/analyze_tennis_history_age.py \
  --output outputs/full_heatmap/history_age
```

输出`summary.json`与共同窗口`windows.csv`。实际运行还会核对s=1自然窗口等于原窗口，以及三类支持状态正确分割当前VC1目标。未运行前不填覆盖数、选择s或提出motion结构。
