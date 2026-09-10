# 显式局部对应能否改善真实三帧定位？

日期：2026-09-10。状态：实现和8目标集成smoke完成，完整训练尚未运行。
协议：[Tennis dense局部对应基线](../protocols/tennis-local-cost-probe-v1.md)。

## 假设与判别依据

[三帧输入](2026-09-10-causal-stack.md)已经在三个seed稳定优于当前帧及重复当前帧；[GT query诊断](2026-09-10-correspondence-probe.md)说明局部去均值描述子具有一定匹配能力。本轮检验该证据能否转化为自动定位增益，而非只在给定球位置时有效。

新增raw、centered和self_centered三个cost对照，均保留真实三帧appearance。self只将额外cost的两个历史key替换为当前帧，控制新增相关非线性与参数。局部volume具有明确先例，见[近邻文献](../literature/2026-09-10-local-correspondence-baselines.md)，不作为创新声明。

## 实际实现与smoke

在`08b8244`基础上的本轮修改，新增`ballmotion.correspondence.cost_volume`与缓存入口；复用原训练/评价代码。每个目标在全部36×64格生成106个offset通道，不读取GT query或以visibility筛选。每帧appearance单独GN，cost维持cosine及越界-2的尺度；absence线性层只读appearance。

继续使用Conda `zshihyc`、现有PyTorch及RTX5070Ti Laptop，不安装新依赖。真实smoke源为`data/cache/tennis/temporal_smoke_512`，8个目标包含24个唯一输入帧；直接复用已提取特征。新的cost smoke保存于`data/cache/tennis/local_cosine_smoke/{raw,centered,self_centered}`，每组shape为8×106×36×64。

三种cost一次生成合计1.95秒、峰值已分配显存1152.66MiB；包括源mmap读取、H2D、计算、有限值检查、float16写回与flush。每种元信息中的elapsed为三组共同总时长，不可相加。首批初始化计入第一个raw计算，不能据此认为raw计算比centered本质更慢。每组首次批次转换最大绝对误差为0.000244140625。

已知192维向量测试覆盖历史帧/offset通道、边界哨兵、去均值scalar cosine及self控制。新增head断言证明改变cost不改变appearance归一化或直接改变absence logit；时间拼接测试补充了cost位于完整appearance之后的断言。5项相关时间测试通过，原有8项空间测试及新的1项对应算子测试也已通过。

三个head都完成2epoch、batch4、train4/val4训练和预测保存，参数量均为23,589。输出为`outputs/correspondence_probe/smoke_{raw,centered,self_centered}`；这些极小样本运行只验证通路，位置指标不进入性能比较。

`outputs/correspondence_probe/precision_smoke.json`在4个验证目标上固定同一float16源appearance，比较float32重算cost与实际float16 cost缓存。三组最大cost差均为0.000244140625；位置argmax和0.5存在判断均无变化，缓存分支精确重现保存的位置指标。需要12个唯一源特征，但**重新解码/提取帧数为0**。这只隔离cost存储误差，不证明去均值描述子在float32 backbone与float16 backbone缓存间等价。

精度脚本旧分支因本轮条件执行改动，另用现有三组时序head smoke进行4目标回归：仅重新提取12个输入帧，缓存指标均重现，位置/存在判断无变化。结果为`precision_appearance_regression.json`；没有重复运行完整验证集的旧精度检查。

```bash
python scripts/cache_tennis_correlations.py --source-cache data/cache/tennis/temporal_smoke_512 --output data/cache/tennis/local_cosine_smoke --batch-size 4
# 下列训练对raw/centered/self_centered分别实际执行，使用同样参数。
python scripts/train_spatial_probe.py --cache data/cache/tennis/temporal_smoke_512 --cost-cache data/cache/tennis/local_cosine_smoke/centered --output outputs/correspondence_probe/smoke_centered --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input stack --epochs 2 --batch-size 4
python scripts/check_probe_precision.py --cache data/cache/tennis/temporal_smoke_512 --runs outputs/correspondence_probe/smoke_raw outputs/correspondence_probe/smoke_centered outputs/correspondence_probe/smoke_self_centered --output outputs/correspondence_probe/precision_smoke.json
```

## 正式结果与下一步

完整1,733目标cost缓存和三组seed0训练尚未运行。先提交已验证实现，再生成一次共享cost并顺序训练；与既有stack seed0比较相同230目标上的救回和新增错误。centered同时超过stack及self后才扩展seed1、2；若raw更好，补匹配的raw self控制再判断跨帧贡献。

无球误报、8个困难/4个遮挡目标的结果独立报告，不以easy类总体改善掩盖这些问题。完整固定头的cost精度检查待正式训练完成后执行。
