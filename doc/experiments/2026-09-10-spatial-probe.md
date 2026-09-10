# DINOv3 四级冻结特征的球位置读出

日期：2026-09-10。状态：首轮四层比较已完成。
协议：[Tennis 空间探针 v1](../protocols/tennis-spatial-probe-v1.md)。

## 问题与判别

比较同样输入下四个 stage 的独立读出，区分该读出条件下的表征适配与输出网格限制。训练 game 1–6、验证 game 7，每 clip 原帧号模 8 取样；game 8–10 暂不参与。若深层差但其网格量化上限也低，不能直接判定信息丢失；若所有层差，优先检查非线性读出、分辨率与适度微调。

## 设置与真实命令

Conda `zshihyc`，Python 3.11.15，PyTorch 2.10.0+cu130，NumPy 2.2.6，Pillow 12.1.1。RTX 5070 Ti Laptop 12 GB；DINOv3 官方版本与权重路径见 [third_party](../../third_party/README.md)。全图 512×288，backbone 冻结、float32 forward、float16 缓存。四层独立归一化线性头，AdamW，lr=0.003，weight_decay=0.01，batch=16，seed=0，30 epochs；按验证 PCK@16、再 PCK@8 选最佳 epoch，相同分数保留首次。

```bash
python scripts/cache_tennis_features.py --output data/cache/tennis/dinov3_convnext_tiny_512x288_step8
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8 --output outputs/spatial_probe/linear_stage0_seed0 --stage 0
```

stage 1/2/3 使用相同参数与对应输出名。每个运行的 `config.json` 记录实际代码提交，`history.jsonl` 保存逐 epoch 损失和验证成绩，`results.json` 保存完整指标，CSV 保存原始身份与预测。下方结果只在相应运行结束后填写。

## 已执行的实现验证

- 5 个可运行测试通过：半像素/网格映射、无球和非法标签区分、发布 CSV 空坐标与原始帧身份、错位 FP/FN、合成特征小样本拟合。
- 32 帧真实 smoke 一次提取四层，shape 分别为 `[32,96,72,128]`、`[32,192,36,64]`、`[32,384,18,32]`、`[32,768,9,16]`；无缺失/多余权重键。float16 保存的相对 L2 误差为 0.000205–0.000217，无溢出。这不证明 float16 与 float32 的最终定位完全相同。
- 16 train + 16 val、3 epoch 的训练/保存/加载/预测流程已通过；它不构成性能实验。
- 首次 smoke 在训练后加载 checkpoint 时失败：重复保存的配置包含 `torch.torch_version.TorchVersion` 对象。删除 checkpoint 的重复配置后恢复；配置保留在 JSON。失败运行位于 `outputs/spatial_probe/smoke_stage0/`，修正验证为 `smoke_stage0_fixed/`。首轮后已删除三个 smoke 运行的无后续用途 checkpoint，保留小型日志/结果；32 帧特征缓存已用于子格头验证，继续保留以免下一次 smoke 重复解码。临时提取/精度控制台日志在统计写入 JSON 后删除。

## 首轮结果

缓存共 1,798 帧：1,559 train、239 val；其中定位分母为 1,509 train、224 val，val 的 visibility 1/2/3 分别为 210/9/5。总缓存 5,965,332,480 bytes（约 5.56 GiB），一次提取 33.42 秒，其中 GPU backbone forward 9.50 秒；峰值已分配显存 457 MiB。缓存训练耗时不计作端到端速度。

下表 PCK 均为**不依赖存在阈值的条件定位**，单位为原图像素；模型输出原生 cell 中心。

| stage / stride | 头参数 | 最佳 epoch | train PCK@16 | val PCK@8 | val PCK@16 | val PCK@32 | val 中位误差 | 训练及末次评价秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 / 4 | 194 | 9 | 51.89% | 45.09% | 49.55% | 50.89% | 16.90 | 252.37 |
| 1 / 8 | 386 | 18 | 63.75% | 33.93% | 61.61% | 63.39% | 10.12 | 77.76 |
| 2 / 16 | 770 | 20 | 35.65% | 8.48% | 40.18% | 73.66% | 18.43 | 47.13 |
| 3 / 32 | 1,538 | 11 | 7.02% | 1.79% | 6.70% | 26.79% | 53.62 | 27.79 |

**网格上限改变解释。** GT 总能选对 cell 的 oracle 中，四层 val PCK@8 上限分别为 100%、52.23%、10.71%、3.13%；PCK@16 上限分别为 100%、100%、53.13%、13.84%。因此不能拿上表直接断言“stage 0/1 的表示比 stage 2 好”。stage 2 的 PCK@32 和平均误差反而更好，值得检验细网格读出。

**存在分类尚未有效。** 四个头均对全部 239 验证帧预测有球，存在 F1 均为 96.76%，正好是始终预测有球的结果。带 16 px 定位条件的检测 F1 分别只有 47.95%、59.61%、38.88%、6.48%。现阶段不能把存在 F1 当作模型成功证据，最终定位系统仍需解决无球判别。

**困难标签仍差。** 四个头在 9 个 visibility=2 验证目标上 PCK@16 均为 0；visibility=3 只有 5 个样本，不据此比较细小差异。stage 1 在 visibility=1 上 PCK@16 为 65.71%。这些结果提示时序潜力，但尚不能证明加入时序能救回。

**失败不是只有量化。** stage 0 最佳 epoch=9 时 train/val PCK@16 也仅为 51.89%/49.55%；之后训练损失继续下降但验证退化，30 epoch val PCK@16=43.75%。固定规则抽取的最差可见球样例中，模型落到记分牌星点和背景反光，而 GT 附近仍能看见拖影。[样例图](../../outputs/spatial_probe/linear_stage0_seed0/error_examples.png)展示两个最小误差和四个最大误差例子，不能据此估计错误类型总体比例。

训练位置中位数 `(641,196)` 的固定坐标基线在 val PCK@8/@16 均为 0，@32 为 0.89%；这只能排除这个极弱位置基线，不能排除所有场景先验。诊断数字保存在 `outputs/spatial_probe/data_diagnostics.json`。

## 精度复核与审查处理

代码审查指出：首 batch 全图 L2 小不能证明 tiny-ball 预测不受量化影响。已在全部 239 验证帧上重新取得 float32 特征，对同一组已训练头分别输入原值与 float16 往返值。四层各自 **0 个空间 argmax 改变、0 个 0.5 存在判断改变**，PCK/位置误差完全相同；最大存在概率差为 0.0000641。见 `outputs/spatial_probe/precision_check.json`。

这支持本轮固定头推理结论不由缓存量化翻转；没有比较从 float32 缓存重新训练后的优化轨迹，不外推为所有后续模型都可无差异压缩。

```bash
python scripts/check_probe_precision.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8 --runs outputs/spatial_probe/linear_stage0_seed0 outputs/spatial_probe/linear_stage1_seed0 outputs/spatial_probe/linear_stage2_seed0 outputs/spatial_probe/linear_stage3_seed0 --output outputs/spatial_probe/precision_check.json
python scripts/plot_spatial_errors.py --run outputs/spatial_probe/linear_stage0_seed0
```

主训练源自 `ad38982`；新增位移辅助分析提交 `fd5e664` 不改变训练逻辑，实际各 run 版本保存在 config。本节精度与绘图脚本在后续读出修订提交中保存。

## 下一轮决定

先将 stage 1/2 的线性输出改为每个原生 cell 输出多个子格类别，统一到 stride-4 网格；不加入 RGB 分支或时序，保留同一缓存与交叉熵。这样可以测量更细位置是否能从已有通道读出。参数量会增加，必须报告，不能把收益归因于新增视觉证据。若仍不足，再做固定宽度非线性头及更高输入分辨率对照。具体设置见[子格读出实验](2026-09-10-subcell-readout.md)。

## 判断边界

这是冻结特征、单一验证比赛、稀疏目标帧的开发诊断，不是最终 benchmark。各层通道不同、头参数数不同；浅深层差异不能单独证明预训练因果作用。无球和定位指标分开，容易辨认、难辨认与遮挡标签分开。尚未验证 motion 增益、微调表现、跨球种和论文创新。
