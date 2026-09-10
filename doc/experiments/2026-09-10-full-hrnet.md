# 全量因果HRNet能否建立更强的竞争系统？

日期：2026-09-10。状态：全量RGB缓存、监督单测、真实batch8预检与完整入口smoke通过；正式seed0在运行。
协议：[Tennis全量因果定位v1](../protocols/tennis-full-causal-v1.md)。设计：[全量强基线计划](../research/2026-09-10-full-baseline-plan.md)。

## 当前问题

旧step8冻结探针与prefix适配不能代表完整模型的能力。当前检验高分辨率HRNet、热图QFL与全量监督构成的系统是否能缓解错误；这些因素同时改变，不能用结果单独归因空间分辨率，更不能声称HRNet没有时序信息。

本地系统只监督因果三帧末帧，保留VC3位置目标，使用项目坐标映射和原始热图峰值。它与作者WASB的MIMO、VC3负例、后处理和未公开训练细节不同，不称作者训练复现。直接模型源与MIT许可证位于third_party/wasb，固定来源见[上游代码说明](../../third_party/README.md)。

## 输入与实际数据量

缓存data/cache/tennis/rgb_512x288_all_h2包含14,160张唯一uint8 RGB帧，shape=[14160,3,288,512]，约5.83GiB。旧step8缓存的5,199帧按真实身份复用，新增8,961帧各自解码/resize一次；窗口未复制成重复图像。实际复制、缺失图像读取/resize及flush共49.85秒，不含frames.csv选择与metadata最终写入。

```bash
python scripts/cache_tennis_full_rgb.py --output data/cache/tennis/rgb_512x288_all_h2 --reuse-cache data/cache/tennis/rgb_512x288_step8_h2
```

| 范围 | 合法目标 | VC1 | VC2 | VC3 | VC0 |
|---|---:|---:|---:|---:|---:|
| games1–6训练 | 12,167 | 10,941 | 784 | 37 | 405 |
| game7验证 | 1,863 | 1,628 | 83 | 35 | 117 |

共130个目标因clip前两帧缺少足够历史而排除；本开发范围非法标签行0。最终games8–10未进入缓存。此处验证目标数和旧230目标不同，分数不能直接相减作结构收益。

## 模型、监督与验证

作者HRNet配置只将frames_out改为1，参数1,481,393。保留两层stride1 stem、四stage及高分辨率分支，三帧RGB按时间合成9通道。使用现有OmegaConf 2.3.1直接读取配置，未新安装依赖；未增加模型工厂或训练框架。

一项热图单测先因模块缺失失败，再验证：半径2.5的中心盘有21格、角点有8格、无球全零；零logits概率QFL为log(2)/4，正负目标梯度符号正确且有限；已知热图峰值通过项目网格中心映射回原像素，低峰值仍保留位置但存在判断为负。另一项缓存测试验证跨clip边界、VC0历史、复用帧原图不存在时仍可使用、缺失帧像素与时间索引。

真实预检outputs/full_heatmap/preflight_batch8.json使用全量缓存的8个三帧训练窗口，float32输出[8,1,288,512]，loss=0.121063、首层梯度范数0.535969，完成一次Adam更新。参数与梯度均有限；峰值已分配7584.15MiB，峰值reserved9278MiB。一次预检总计2.37秒，含首次运行条件，不据此推算稳定吞吐。模型为一次性预检对象，不作为正式初始化。

完整入口smoke使用全量缓存复制出的20个唯一帧，构成8个训练与8个验证目标，另存outputs/full_heatmap/smoke_rgb；metadata明确smoke_targets=16，没有新解码或扩大实际数据。运行1epoch，history保存epoch0/1，训练loss=0.121065，best_epoch=0；恢复最佳checkpoint后保存8/8行预测。总计2.59秒，峰值已分配7548.15MiB。它仅证明训练/评价/保存链路，不能评价模型强度。

```bash
python -m unittest discover -s tests -p 'test_heatmap.py' -v
python -m unittest discover -s tests -p 'test_tennis_full_rgb.py' -v
python scripts/train_tennis_heatmap.py --rgb-cache outputs/full_heatmap/smoke_rgb --output outputs/full_heatmap/smoke_seed0 --batch-size 8 --epochs 1 --seed 0
```

以上缓存、预检和smoke发生在正式实现提交前，记录的父提交6400e06不包含这些新代码；这里明确其工作区实现范围，正式训练在提交后另保存实际revision。

独立审阅发现旧step8缓存也能满足T=3和512×288，可能静默污染全量比较。已修正为在模型构造前要求games1–7、target_step1、history2、PIL512×288；用现存旧缓存实际调用，得到明确ValueError且没有生成run config。审阅复核后无剩余问题。正常输入保持uint8传输后GPU转float32，沿用已有传输结论，不再重复benchmark。

## 正式运行

固定seed0、物理batch8、float32、30epoch；Adam lr1e−3/no weight decay，epoch10/20完成后乘0.1；不开AMP、增强、梯度累积、HLSM或tracker。每个epoch验证，按detection F1@16、再F1@8选优，完全相同取最早，epoch0可选。该选优包含存在判断，与旧conditional PCK选优分开。

```bash
python scripts/train_tennis_heatmap.py --rgb-cache data/cache/tennis/rgb_512x288_all_h2 --output outputs/full_heatmap/hrnet_seed0 --batch-size 8 --epochs 30 --seed 0
```

一次启动因提交前格式检查发现本地配置末尾多余空行而在提交完成前误触发；已主动中断并完整保留于outputs/full_heatmap/precommit_start_aborted，没有作为正式运行或选优证据。修正格式并提交实现后从头启动正式seed0，不续用该进程参数。

正式运行已从提交32d37d4启动，config实际记录12,167/1,863目标、batch8、30epoch与1,481,393参数。输出保留完整config、history、最佳模型、train/val预测、耗时和显存。主要查看精细定位、困难/遮挡、无球误报与有球漏报是否同步改善；全量DINO共同目标对照完成前，不写架构胜负结论。

首次训练进度记录：epoch1完成，训练loss=0.00045746，验证PCK8/16/32为59.85%/62.77%/63.80%，F1@16为51.45%；presence为TP639/FP2/FN1107。单epoch含验证701.06秒，从计时开始累计732.40秒。这是早期优化状态，正式30epoch尚未结束，不用于结构胜负或最终错误归因。后续完整history是训练过程的唯一逐epoch记录，本文不逐轮抄日志。
