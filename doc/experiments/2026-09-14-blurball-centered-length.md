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
