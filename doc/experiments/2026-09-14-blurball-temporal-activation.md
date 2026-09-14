# 目标/支持独立激活：直接检验第一次时间合并

日期：2026-09-14。状态：正式12轮运行已启动，训练后的结果未产生。
协议：[目标证据独立激活v1](../protocols/blurball-temporal-activation-v1.md)。

## 为什么现在做

五帧输入和同步水平翻转已分别得到实测支持；[残余候选及邻帧位置诊断](2026-09-14-blurball-centered-hflip.md)不支持把多数错误简单归为固定候选排序或输出邻帧球的位置。当前测试目标与支持在第一次GELU前后相加的区别，不同时改搜索、backbone、分辨率或训练配方。

独立架构审查将五路逐帧激活收窄为目标/支持两路，减少归因混淆。实现位于[SpatialInteractionReadout](../../src/ballmotion/probe.py)，保留原参数名和形状；[训练入口](../../scripts/train_blurball_midpoint.py)、[局部读出](../../scripts/analyze_blurball_readout.py)及[候选导出](../../scripts/export_blurball_candidates.py)传递实际目标槽，避免恢复模型时默认为最后帧。

## 实验产物与命令

参考：`outputs/blurball/centered_hflip/center5_seed0/`，已完成，不重训。
新臂：`outputs/blurball/temporal_activation/target_seed0/`。

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/temporal_activation/target_seed0 \
  --window center5 --interaction target_activation --temporal-input history \
  --augmentation hflip --epochs 12 --batch-size 4 --seed 0
```

比较固定初始权重、参数量、目标集与训练预算；两种网络初始输出本来就可以不同，不能要求epoch0指标相等。原位置、候选覆盖和输出状态分开归因，完整接受/否定条件在协议中训练前锁定。

## 运行前验证已完成

[目标激活测试](../../tests/test_temporal_activation.py)与原同步翻转、真实批内帧复用测试共11项通过。新测试覆盖全部参数初值相同、去掉GELU后的输出/梯度等价、目标槽0/2/4与显式逐帧贡献一致、两种非线性函数确实不同、absence logit公式不变以及V1/V0反向。既有`test_spatial_interaction.py`同预算及混合导数自检也通过。

真实预检位于`outputs/blurball/temporal_activation/preflight.py`，结果同目录`preflight.json`。使用第一个V0与前三个V1训练目标（match00/001，原帧57、4、5、6），同步翻转其中两个窗口。新旧seed0初始state_dict逐项完全相同，总参数1,279,169；旧增强best8的前4个验证目标argmax完全复现，q最大差异0。

新网络真实B4输出`[4,147457]`，loss9.6386013，所有参数梯度有限，五个时间槽的投影梯度范数均非零，完成一次AdamW更新。峰值已分配显存4396.18MiB，设备RTX5070Ti Laptop；此值是功能预检资源记录，不是独占性能比较。预检不保存新模型训练权重，也不产生正式指标。

[保存预测比较入口](../../scripts/compare_blurball_temporal_activation.py)已实现：核对两臂唯一机制差异、局部读出与K16来源，统计原位置/输出及固定参考群体的候选救回和破坏。对应新测试与已有比较回归共21项通过；真实基线13912×16候选契约也已读取验证。挑战臂尚未训练完成时不生成正式比较。

独立实现复核未发现实质问题，确认偏置分配、目标槽、旧路径和训练/读出/候选入口一致。现有验证已覆盖本次改动，不追加无关测试或其他结构扫描。

## 正式启动

启动UTC时间2026-09-14 15:37:07，源码版本524dbf0；顺序脚本PID1061134，训练子进程PID1061137，均已确认存活。实际生成配置已通过与参考的完整条件比较，只有声明的机制/协议/输出/源码字段不同，训练目标37590、验证13912、目标槽2和1,279,169参数符合锁定条件。

`outputs/blurball/temporal_activation/launch.json`保存命令入口与时间，`run_target.sh`顺序执行训练、固定局部读出、K16候选及保存预测比较。退出状态写同目录`exit_status.txt`；日志与best/last位于`target_seed0/`。正式训练仍复用原RGB mmap，没有重新解码或新建重复窗口缓存。

## 公式进一步限定了“保留”的含义

令a、b为某个投影通道的目标与支持贡献。原第一激活为G(a+b)，新式为G(a)+G(b)。两者始终输出同样的32通道，故本实验不是把融合后的信息容量从32扩到64，更没有数学保证保留目标信息。

对精确GELU，G(x)=xΦ(x)，φ为标准正态密度。原式对a、b的混合二阶导数为：

```text
∂² G(a+b) / ∂a∂b = [2 − (a+b)²] φ(a+b)
∂² [G(a)+G(b)] / ∂a∂b = 0
```

新式取消的是第一处非线性的直接目标/支持耦合；后续空间卷积与第二个GELU仍能重新形成交互。因此它也可能移除了有益的motion相关计算，而非只去掉干扰。这一点使负结果同样有解释价值。

在a与b互为相反数时，原式为0，新式为`a·erf(a/√2) ≥ 0`；在两者都为较大正值时，两种GELU近似线性相加。由此只能推导：新式改变投影贡献相互抵消时的响应。投影正负没有物理运动方向语义，也不能默认正响应属于球、负响应属于背景。

据此，训练后必须同时看新增候选与破坏的正确位置，而不能把GELU拆分命名为“证据无损保留”。如果没有位置/候选增益，优先接受此处理顺序不成立的结果；不把推导当作必须成功的保证。
