# BlurBall中点定位是否从真实历史取得增量

日期：2026-09-12。状态：协议与输入改动完成，正式训练待启动。
协议：[完整时序输入控制v1](../protocols/blurball-full-temporal-control-v1.md)。

## 为什么现在需要这项训练

固定局部重心已经改善当前三帧模型的位置，不能继续仅凭沿轴偏差设计blur模块。但该模型的logits本身包含三帧输入，读出阳性没有回答历史的作用。Tennis上的单帧/真实历史对照也不能直接代替BlurBall的自然拖影中点任务。

因此只增加一次从头完整训练的repeat_current控制，保持官方预训练初始化、三槽参数化、标签、共同目标、损失和优化预算。双方继续依argmax选best，事后共同使用固定15×15、T=1重心；既有history best6和局部logit缓存直接复用。完整协议已明确强阳性、混合/阴性以及相应架构决定，不根据中途指标更改分组或训练。

## 实施范围与验证

[原训练入口](../../scripts/train_blurball_midpoint.py)仅增加实际输入选择。在v2连续窗口与6个内部边界目标排除完成后，repeat将模型输入的三个索引替换为当前帧；所有目标身份保持。配置明确`temporal_input=repeat_current`及`t,t,t`，原history默认路径保留。没有新模型类或新依赖。

[局部读出入口](../../scripts/analyze_blurball_readout.py)同步按run真实输入模式提取验证特征，并从该run的results核对其最佳epoch；旧history缺少输入模式字段时遵循其原来使用真实三帧的事实。历史局部读出产物不会重算，新repeat结束后再生成自己的局部缓存与原q复现证据。

真实GPU单batch检查已通过：7个V1与1个V0、原frame2–8及57，模型三槽RGB与各自当前帧完全相同。全部合法目标38,854/14,192及6个内部边界排除保持。输出`[8,147457]`，loss11.15012，prefix/head梯度范数71.8869/9.1974，均有限且非零；AdamW一步及原验证唯一帧路径通过。参数1,272,577不变；检查耗时1.07秒、峰值allocated5,277.22MiB，不能当正式性能。

该检查发生在8e05230之后的本次工作区修改，脚本及结果保留在`outputs/blurball/repeat_current_smoke/`。它验证索引/标签与实际训练通路，没有重做数据下载、完整源内容审查或已拒绝的单encode训练优化。

## 正式运行

拟执行命令（项目根目录，Conda zshihyc）：

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/dino_repeat_current_seed0 \
  --temporal-input repeat_current --epochs 30 --batch-size 8 --seed 0 \
  > outputs/blurball/dino_repeat_current_seed0.log 2>&1
```

输出为独立run目录。正式版本从启动后config记录；没有继承smoke的一步参数。预计计算规模与上一约10小时全量训练相近，实际耗时以后续记录为准。

## 当前结果

正式训练尚未启动，不填性能数值。预先统计的历史可见性与位移分组定义见协议；它们只为分析，不作为任何模型输入。
