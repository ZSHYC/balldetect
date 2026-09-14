# 双向三帧是否已经足够

日期：2026-09-14。状态：实现与真实batch验证完成，按锁定协议启动开发训练。

依据：[五帧最终结果](2026-09-14-blurball-five-frame-results.md)、[长度协议v1](../protocols/blurball-centered-length-v1.md)。五帧双向确实改善位置，但外侧两帧的必要性尚未成立。独立审查支持先检验center3，不立即增加身份分支、留出源模型或校准任务。

## 实际改动

训练入口新增`--window center3`，从共同center5窗口取中间三槽，目标slot1。仍为原来的37,590/13,912目标，`five_frame_cohort`记录描述的是目标集合来源，实际输入由`input_slots`和`num_frames`记录，不把三帧写成五帧。新协议名为`blurball-centered-length-v1`；默认三帧与原两五帧行为保留。

读出入口按实际window恢复时间槽与模型帧数。没有添加模型模块、依赖、增强或新的缓存机制。正式用现有三帧cross-address模型形状，参数1,266,497；center5为1,279,169。少两帧减少的是逐帧编码与输入通道，参数减少不能直接等同于端到端提速。

## 验证证据

9个针对性CPU测试通过，覆盖原三/五帧、center3共同目标/近邻、缺帧与片段边界、读出和配对状态。真实metadata验证center3整个输入数组等于共同center5窗口的中间三列，目标行完全一致。

真实batch4通过147,457类输出、有限CE、所有可训练参数有限梯度和一步AdamW；目标slot1，loss9.656664，峰值分配约2,659.16MiB。这是随机初始化单批通路检查，不是模型成绩或吞吐benchmark。脚本和结果为`outputs/blurball/centered_length/preflight.py`、`preflight.json`；没有重新解码。

## 正式运行

项目根目录运行，环境`zshihyc`：

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/centered_length/center3_seed0 \
  --window center3 --interaction cross_address --temporal-input history \
  --epochs 12 --batch-size 4 --seed 0
```

顺序脚本`outputs/blurball/centered_length/run_center3.sh`在训练成功后运行固定局部读出。日志、启动PID/代码版本和退出码在同一实验目录，权重及预测在`center3_seed0/`。启动状态以该处进程和日志为准，完成后读取保存预测与center5作共同目标比较。

当前没有center3正式结果；不以单批通路或未完成曲线宣称三帧足够。已有五帧结果和可复用缓存保持不变。

## 保存预测比较入口

[compare_blurball_centered_length.py](../../scripts/compare_blurball_centered_length.py)已准备，待训练和固定局部读出完成后运行。方向统一为center5→center3；复用既有逐比赛、半拖影、位移、前后V1及旧困难群体定义，分别报告raw救回/破坏、正确且发出的救回/破坏和输出状态矩阵。

入口只允许协议中声明的时间槽、头形状、参数量、路径及版本差异；实际两配置已通过核对，故意改变seed会被拒绝。没有用未完成结果冒充正式比较。两个既有比较测试通过，新增断言验证更换模型名称后矩阵方向与救回计数仍一致；原五帧默认名称保持不变。

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python scripts/compare_blurball_centered_length.py
```

输出为`outputs/blurball/centered_length/comparison.json`。该命令只处理元数据与保存预测，不读取RGB或运行模型。当前尚未执行完整center3比较，因为它仍在训练。

## 实际未来等待：训练与验证不能共用一个典型值

从共同目标的真实PTS计算目标至最后输入帧的时间，单位毫秒：

| 集合/模型 | 目标数 | 中位数 | P95 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| train / center3 | 37,590 | 40.000 | 40.000 | 80.000 |
| train / center5 | 37,590 | 80.000 | 80.000 | 250.250 |
| val / center3 | 13,912 | 16.667 | 33.367 | 33.367 |
| val / center5 | 13,912 | 33.333 | 66.733 | 200.200 |

这里仅包含源时间上的前瞻，不包含计算、排队或解码延迟。源容器末帧PTS异常沿用既有数据说明，不把最大等待静默替换成名义FPS推算值。脚本和结果为`outputs/blurball/centered_length/time_support.py`、`time_support.json`；只读一次元数据，未读RGB。

重要限制是当前模型按帧槽融合，并未显式输入Δt。训练与验证的典型物理时间跨度不同，说明不能把相同帧数自动理解为相同秒数；它可能影响跨比赛运动泛化，但目前尚无干预证据证明它是主要失败原因。center3与center5采用相同目标和来源，长度对照依然成立；后续运动机制若依赖速度，应明确处理真实时间，并与现有无Δt输入条件区分。不同比赛同时存在外观等变化，不能用这张表直接归因于帧率。
