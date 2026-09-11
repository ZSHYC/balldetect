# BlurBall自然模糊下的因果中点定位基线

日期：2026-09-11。状态：协议、实现、小规模检查与独立代码审查完成；全量缓存及正式训练尚未开始。

执行[原生因果中点协议v1](../protocols/blurball-causal-midpoint-v1.md)。在Tennis固定细节读出停止、历史年龄诊断未建立远搜索动机后，转向已经有真实blur标注、尚无定位失败读数的BlurBall。独立审查支持这一取舍；逐样本交换历史槽位只改变显式帧龄信息，尚不能有效决定对应结构，故本轮不训练该控制。

## 这一轮能改变什么判断

先测一个没有theta/l输入或辅助的定位器，错误是否随自然拖影半长度增加，以及变化来自原始位置错误还是拒绝输出。若只在一场比赛成立或不存在此趋势，就停止把自然拖影当作当前首要失败源。既有[轴一致性](2026-09-10-blurball-axis.md)只是标签间事实，不能替代这个自动定位证据。

验证match18–21全部1280×720，训练则有三种分辨率。按既有l=0、(0,2]、(2,5]、(5,10]、>10桶和match交叉报告，避免总体趋势被来源组成掩盖；不会宣称已有跨验证分辨率结果。

## 协议核对改变了实现

一手代码核对确认，作者V0使用空热图、评估TN/FP2，没有合法可见中心；这不证明物理上无球。原生基线只定位V1中点，V0不回归原CSV占位坐标。

作者以原图距离严格<4px判断正确；其FP1只进入FP，不同时进入FN。本地检测F1保留错位同时贡献FP/FN的定义，另列作者F1及五种帧状态，不混用。PCK以所有V1空间argmax为分母，另报有输出V1的条件误差。来源、确切公式和作者MIMO/仿射预处理差异见[协议](../protocols/blurball-causal-midpoint-v1.md#评价严格原图容差与完整分母)。本项不是作者结果复现。

因此新头输出288×512而不是沿用Tennis的72×128。后者在1280×720上间距10px，会预先限制4px评价；前者间距2.5px，仍只是在现有192通道stage1上提高位置读出密度，没有增加新的空间证据或motion分支。

## 实现与实际小检查

[缓存入口](../../scripts/cache_blurball_rgb.py)按原rally一次顺序解码，保留真实Frame和同次解码PTS；不去重原重复帧。showinfo显式`checksum=0`，无哈希、指纹、数据拷贝校验或逐帧PNG。RGB缓存为uint8唯一帧mmap，窗口只存索引。

[训练入口](../../scripts/train_blurball_midpoint.py)复用既有官方DINO构建、三帧forward、验证批内去重与预测CSV写出；公共构建函数只新增有实际用途的`upscale`参数，默认2保持Tennis调用。BlurBall传8；新模型共1,272,577参数，头37,537参数。[评价实现](../../src/ballmotion/blurball.py)保留原图坐标，分别计算两种F1与固定blur分组，不套用Tennis visibility含义。

两个小检查均先在入口不存在时失败，随后通过：

- [缓存测试](../../tests/test_blurball_cache.py)：不同rally的索引隔离、V0仍进输入和目标、原Frame不重编号、theta/l原样、Test禁止进入。
- [定位测试](../../tests/test_blurball_localization.py)：严格4px边界、TP/FP1/FP2/拒绝/TN分割、两种F1及PCK分母、空blur组、不同原图尺寸的像素中心映射。

真实缓存smoke使用match00/rally001：193源帧、191三帧目标、边界排除2，RGB形状[193,3,288,512]，解码/resize/mmap写入0.590秒。time_base=1/12800，前三帧PTS=0/512/1024，即0/.04/.08秒。同配置复用时日志字节数未变，没有重新解码。产物`outputs/blurball/cache_smoke/`。

随后真实GPU单batch选择同rally当前帧2–8和57，包含7个V1与1个V0；输出[8,147457]，初始CE=11.15007019，前缀/头梯度L2分别71.3995/9.2090，均有限非零。一次AdamW更新后所有参数有限，复用验证路径成功输出并评价。耗时1.564秒、峰值5268.09MiB；仅证明训练通路，不提供性能结论。产物`outputs/blurball/midpoint_smoke/result.json`，检查脚本同目录保留。

这些smoke在实现提交前运行，不能据其父提交6266767声称已含实现。正式运行在提交后记录实际代码版本，不将训练过的smoke模型作为初始化。

独立只读代码审查未发现阻断问题，确认数据/时间/坐标契约、两种F1与训练选优一致；改动文档本地链接及diff格式检查通过。新增的FFmpeg版本记录由正式缓存实际读取，不补写成smoke已经记录的字段。

## 待执行的固定命令

```bash
conda activate zshihyc
python scripts/cache_blurball_rgb.py \
  --output data/cache/blurball/rgb_512x288_all_h2
python scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/dino_midpoint_seed0 \
  --epochs 30 --batch-size 8 --seed 0
```

原manifest预期53,818帧，RGB约22.17GiB；当前文件系统有578GiB可用，缓存规模可容纳。正式实验的38860/14192目标数待全量源标签与解码对齐后确认；尚未运行时不填验证指标。
