# 同步水平翻转：先验证训练干预，再决定结构变化

日期：2026-09-14。状态：实现、针对性测试与真实小批量验证通过；正式增强训练尚未启动。
协议：[中心窗口同步翻转v1](../protocols/blurball-centered-hflip-v1.md)。基线已按[完整长度结果](2026-09-14-blurball-centered-length.md)选择center5，原无增强模型不重训。

## 为什么做这项控制

原center5最佳权重的原argmax训练PCK@4为96.1163%，验证为82.7488%；均直接读取`outputs/blurball/five_frame_context/center5_seed0/results.json`。这里比较相同读出，没有把训练argmax与验证局部重心混用。训练与验证比赛不同，差距只能提出泛化问题，不能把原因直接归给某个模块。

[现代检测器审查](../literature/2026-09-13-modern-detectors-transfer.md)提出训练配方也是需要控制的解释。同步水平翻转能够改变场景左右布局，并保留真实窗口的时间连续性；它不能修复视觉证据缺失，也不等于新运动表征。当前选择单项对照，不引入完整检测框架或增强组合搜索。

## 实现与验证证据

改动入口：[train_blurball_midpoint.py](../../scripts/train_blurball_midpoint.py)。新增显式`--augmentation hflip`，仅用于center3/center5共同目标协议；原默认无增强，验证不变。

先写[针对性测试](../../tests/test_blurball_hflip.py)，观察到缺少辅助函数的预期失败，再实现连续坐标反射与窗口同步翻转。与现有多帧、预取、五帧比较测试共同运行，10项通过。覆盖量化边界、无球类别、时间槽不乱、未选窗口不变和恢复后的翻转mask/样本顺序。

真实缓存预检脚本为`outputs/blurball/centered_hflip/preflight.py`，结果保存同目录`preflight.json`。使用第一个V1及第一个V0训练窗口，不按图像或预测选择；center3只用于通路验证，不代表已经决定正式基线。检查源缓存未变、所有槽同步反射、原目标对齐、147457类输出、有限loss/梯度及一次AdamW更新。预检不保存权重，不作为正式实验指标。

实测B2输出`[2,147457]`，loss为6.7502875，全部梯度有限并完成一次更新，峰值已分配显存1348.53MiB。两个目标来自match00/001的原帧4与57；原类别83192/147456变为83207/147456。设备RTX5070Ti Laptop、Torch2.10.0+cu130；共享GPU的小批量通路验证不作为吞吐基准。

## 比较工具准备

新增[保存预测比较入口](../../scripts/compare_blurball_hflip.py)，复用现有`load_arm`与成对位置/输出统计，不调用模型或RGB解码。允许的干预仅为同步翻转，其他模型、数据与训练条件不同会直接报错。位移按原始身份查询上一帧，只有两端均V1才进入位移分组，不把不可定位状态解释成静止。

[比较测试](../../tests/test_blurball_hflip_comparison.py)与既有成对计数测试共13项通过：覆盖center3/center5合法配对、额外帧数/模型/学习率/batch/seed/样本集合差异、错误增强概率和4/16px分组边界。正式增强结果尚不存在，未执行两组完整产物比较，也未生成模拟正式指标。

## 后续条件

长度控制已完成并选择center5，协议已锁定；启动一个12轮正式增强运行。结果仍为空，不宣称增强有效。若有效，后续架构研究需要固定训练配方，避免把增强收益归给motion模块；若无效，保留负结果并返回已定位的视觉/对应问题。

## 正式运行设置

采用center5（前后各两帧、目标slot2），参数1,279,169；seed0、batch4、FP32、12轮，配置与[锁定协议](../protocols/blurball-centered-hflip-v1.md)一致。启动命令：

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/centered_hflip/center5_seed0 \
  --window center5 --interaction cross_address --temporal-input history \
  --augmentation hflip --epochs 12 --batch-size 4 --seed 0
```

输出目录保存配置、日志、best/last和预测；顺序脚本`outputs/blurball/centered_hflip/run_center5.sh`在训练成功后执行固定局部读出与保存预测比较。启动版本、PID、时间在同目录`launch.json`，退出码为`exit_status.txt`。进程启动后再记录正式运行状态，不能将准备完成当作已运行。
