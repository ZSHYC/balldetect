# BlurBall：邻域汇合与非线性的先后顺序

日期：2026-09-12。状态：same_address训练中，cross_address顺序接续。
协议：[压缩后空间交互v1](../protocols/blurball-spatial-interaction-v1.md)。

本组要检验32通道压缩之后的邻域非线性是否改善细定位。两组四个卷积的顺序、形状、初始化和训练预算相同，只移动第二个GELU。完整函数、最近邻限制和预定判断见协议；旧repeat_current不重训。

## 实现与验证

复用 `SpatialProbe` 的按帧GN、absence及PixelShuffle，新增 `SpatialInteractionReadout`；训练入口记录interaction并调用共同builder，局部读出按配置重建。默认baseline保持原模型初始化顺序和配置字段，已有训练checkpoint与读出可继续使用。未新增数据读取或缓存路径。

`tests/test_spatial_interaction.py` 已验证相同seed初始化逐张量相同、真实位置头30,880参数及70,778,880卷积MAC、自由归一化特征上不同地址的混合二阶导差异、缺失logit相同、输出尺寸与V1/V0梯度。测试先确认类缺失时失败，再通过实现；这些是函数和运行通路证据，不是定位性能证据。

真实batch验证脚本与原始结果保存在 `outputs/blurball/spatial_interaction/smoke.py`、`smoke.json`。每组采用同一个4个V0、4个V1的训练batch；全模型初始state逐张量一致，总参数均1,266,497，输出均为8×147,457，全部可训练参数的梯度有限。same/cross的交叉熵分别为6.631264/6.630963，峰值allocated显存分别5,268.05/5,267.05 MiB。单次包含RGB/H2D、forward、backward与AdamW的耗时为1.070/0.230秒；顺序首跑有冷启动且其他项目共享GPU，不可据此比较两组速度。

定向独立代码审查未发现实际问题，确认baseline默认调用及旧配置/续训保持、两臂构造顺序一致、新checkpoint按配置重建；新增测试、Python编译及Ruff F检查通过。审查时OMX LSP传输服务不可用，其结果不作为已通过证据，也不因编辑器服务缺失重做训练路径。

## 运行设置

Conda `zshihyc`，RTX5070Ti Laptop，float32；两组真实三帧，seed0，各30轮、batch8，其余设置见协议。训练38,854、验证14,192个目标，未缩减任何比赛或batch。训练代码提交 `45b61ba`；北京时间19:16以独立session启动 `outputs/blurball/spatial_interaction/run_pair.sh`，启动记录 `launch.json` 保存实际命令与时间。启动时父进程PID3341730、same训练PID3341733已核实存活，正式config的目标数、30轮、batch8、interaction与参数数一致。

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/train_blurball_midpoint.py --rgb-cache data/cache/blurball/rgb_512x288_all_h2 --output outputs/blurball/dino_same_address_seed0 --interaction same_address --epochs 30 --batch-size 8 --seed 0
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/train_blurball_midpoint.py --rgb-cache data/cache/blurball/rgb_512x288_all_h2 --output outputs/blurball/dino_cross_address_seed0 --interaction cross_address --epochs 30 --batch-size 8 --seed 0
```

本机另一个项目同时使用GPU，保留其进程。共享资源下的实际训练时间不能作为模型独占GPU的吞吐基准；本次仅据可用显存和运行证据判断是否可执行。

启动脚本依次完成每组训练及固定局部读出；任一步返回非零即停止接续，错误保留在对应run的 `train.log` / `readout.log`，不从best权重重置优化器伪装续训。关闭交互终端不会向独立session自动传递挂断；WSL整体关闭等情况则依原 `last.pt` 完整轮次恢复。上述PID仅是启动事实，当前运行状态须以实际进程和日志为准。

配对入口在 `9f0bf85` 提交后，已追加到启动脚本的两组循环之后，自动生成 `spatial_interaction/paired_seed0.json` 与 `paired.log`；只追加后处理，不改变正在运行的训练命令。脚本语法检查通过，正式配对尚待两组完成。

19:17左右已完成全部14,192个目标的epoch0验证并保存 `last.pt`，随后第1轮已有3,200/38,854目标的训练记录。配置、进程及实际日志均已核实；未据此填入最终性能结果。

## 保存预测的配对入口

`scripts/compare_blurball_spatial.py` 复用已有BlurBall指标、真实历史分组、坐标身份检查和状态矩阵，明确以same→cross为方向；只额外加入协议要求的各场d1≥16组。运行前读取两组共同配置、完整逐轮记录和两种读出的原q，若目标错位、共同条件不同或局部读出改变q则停止配对，避免产生不可比数字。不重新解码或模型forward。

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python scripts/compare_blurball_spatial.py --same outputs/blurball/dino_same_address_seed0 --cross outputs/blurball/dino_cross_address_seed0 --output outputs/blurball/spatial_interaction/paired_seed0.json
```

最小合成例验证了方向、位置救回但仍拒绝、V0误出被抑制及完整V1分母；已有指标公式没有重写。正式配对要等两组实际完成，合成例不作为项目定位结果。

另用临时目录中的小型metadata、两种读出CSV、配置及逐轮记录完成CLI全流程，确认输出目标数、same→cross救回/破坏与各场d1≥16分组；随后仅改变cross局部读出的q，入口正确报错且不生成结果。临时文件已随测试删除，没有新增解码或GPU计算。

## 结果与下一步

正式结果尚未产生。完成后按协议复用两组最佳预测和固定局部读出，完整报告位置救回/破坏及输出拒绝代价，决定是否保留该结构方向。单独参数预算相同不证明它发现了球对应。
