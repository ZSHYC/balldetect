# DINOv3 四级冻结特征的球位置读出

日期：2026-09-10。状态：运行中。
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
- 首次 smoke 在训练后加载 checkpoint 时失败：重复保存的配置包含 `torch.torch_version.TorchVersion` 对象。删除 checkpoint 的重复配置后恢复；配置保留在 JSON。失败运行位于 `outputs/spatial_probe/smoke_stage0/`，修正验证为 `smoke_stage0_fixed/`。完成首轮后删除无复用价值的 smoke 特征，保留本记录。

## 首轮结果

尚未完成四层训练，不报告虚构分数。

## 判断边界

这是冻结特征、单一验证比赛、稀疏目标帧的开发诊断，不是最终 benchmark。各层通道不同、头参数数不同；浅深层差异不能单独证明预训练因果作用。无球和定位指标分开，容易辨认、难辨认与遮挡标签分开。尚未验证 motion 增益、微调表现、跨球种和论文创新。
