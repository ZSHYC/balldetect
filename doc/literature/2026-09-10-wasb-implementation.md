# WASB-SBDT：网球强基线的可运行性与本地受控适配

**核对日期：** 2026-09-10。本文只补充 [体育方法、数据与协议核验](sports_evidence.md) 中已确认的 WASB 事实，目标是为下一阶段建立可追溯的网球强基线；没有下载权重或数据、没有安装环境、没有运行代码。

## 结论先行

WASB 应作为强的**空间分辨率、位置监督与后处理**对照，而不能被称作无时序基线。它的公开论文足以固定核心网络、三帧输入、热图监督和原作者报告的训练超参数；但截至本次读取，作者仓库没有公开完整可训练工程。因此：

1. 不能把在本仓库重建的结果称为“作者代码复现”。
2. 可以建立明确标作 **WASB-local controlled adaptation** 的基线：保留论文可核验的 HRNet、分辨率、三帧 MIMO、损失和训练日程，同时把本项目的 game 1--6 / game 7 开发划分、设备和未公开细节逐项写出。
3. 在 12 GiB 单卡上不能从论文的“四张 V100、全局 batch 8”推断可用 batch 或速度。先用一个完整前反向 smoke 找出实际峰值显存，再选最大可运行 batch；记录实际值即可。

## 原论文已经固定的事实

| 项目 | 证据与本地含义 |
|---|---|
| 主网络 | 小型 HRNet，高分辨率模块四个 stage；论文去掉 stem 的两次下采样，默认采用其 Fig. 3(c)。输入与输出热图同为 288x512。作者报告约 1.5M 参数。见[论文 Sec. 3.1、Fig. 2--3](https://papers.bmvc2023.org/0310.pdf)。 |
| 时序输入与目标 | 连续 `N=3` 帧沿通道拼接，网络一次输出对应的三张同分辨率热图。它不是显式 correspondence 或 flow。见[论文 Sec. 3.1](https://papers.bmvc2023.org/0310.pdf)及[作者模型配置](https://github.com/nttcom/WASB-SBDT/blob/main/src/configs/model/wasb.yaml)。 |
| 网球处理与作者划分 | 原作者论文用 game 1--7 训练、game 8--10 测试；Table 1 为 7 games / 65 clips / 14,160 帧训练与 3 games / 30 clips / 5,675 帧测试。此处的本地开发划分 1--6 / 7 与其不同，不能混称。见[论文 Sec. 4.1、Table 1](https://papers.bmvc2023.org/0310.pdf)。 |
| 位置监督 | 普通样本为半径 `d=2.5` 的二值盘。困难样本改为截断实值图，以 Quality Focal Loss（`beta=2`）训练；论文指定实值图最小非零值 `cmin=0.7`。见[论文 Sec. 3.2](https://papers.bmvc2023.org/0310.pdf)。 |
| 作者训练日程 | 从头训练，Adam，30 epochs，论文写全局 batch 8；学习率等细节由[作者 Adam 配置](https://github.com/nttcom/WASB-SBDT/blob/main/src/configs/optimizer/adam_multistep.yaml)给出：`lr=1e-3`、无 weight decay、epoch 10/20 乘 0.1。HLSM 在 epoch 20 开始一次。作者报告使用 4 张 V100。见[论文 Sec. 5.2](https://papers.bmvc2023.org/0310.pdf)。 |
| 论文测试指标 | 作者以原始像素距离阈值 `tau=4` 计算 F1、Accuracy、AP；这与本项目当前 PCK8/16 不是同一指标。见[论文 Table 2 与 Sec. 5.2](https://papers.bmvc2023.org/0310.pdf)。 |

## 作者代码实际包含什么

作者公开仓库最后一次更新于 2023-11-23，README 的 Training 段仍为 “TBA”。该仓库可作为结构和评估后处理的直接证据，但不构成完整训练发布。

### 可直接核对的实现细节

- [模型配置](https://github.com/nttcom/WASB-SBDT/blob/main/src/configs/model/wasb.yaml)固定 `frames_in=frames_out=3`、`288x512`、stem strides `[1,1]`、四个 HRNet stage；[HRNet 实现](https://github.com/nttcom/WASB-SBDT/blob/main/src/models/hrnet.py)的第一层输入是 9 通道，最后为 3 通道热图。
- [Tennis 配置](https://github.com/nttcom/WASB-SBDT/blob/main/src/configs/dataset/tennis.yaml)把 visibility 1、2 视为可见；[CSV 读取器](https://github.com/nttcom/WASB-SBDT/blob/main/src/utils/file.py)据此把 0、3 转为 `is_visible=False`。数据加载器为不可见帧生成全零图，而不是利用其坐标回归。这个语义与本项目需单列的 VC3“邻帧辅助估计位置”并不等价。
- [数据加载器](https://github.com/nttcom/WASB-SBDT/blob/main/src/dataloaders/dataset_loader.py)按 clip 构造滑动的三个真实相邻帧序列，并对三帧都监督；原 1280x720 网球图被仿射缩放到 512x288，使用 ImageNet 归一化。训练增强的默认配置全为零；仓库另有 `full.yaml`，但没有公开训练入口说明作者权重究竟使用哪一份增强配置。
- [热图生成器](https://github.com/nttcom/WASB-SBDT/blob/main/src/dataloaders/heatmaps/heatmaps.py)和[底层定义](https://github.com/nttcom/WASB-SBDT/blob/main/src/utils/heatmap.py)实现二值盘及实值图，[QFL 配置](https://github.com/nttcom/WASB-SBDT/blob/main/src/configs/loss/hm_qfl.yaml)为 `beta=2`。仓库默认 `min_value=0.6`，与论文的 `cmin=0.7` 不一致；本地受控适配应优先采用论文值，并把这项差异写出。
- [评估配置](https://github.com/nttcom/WASB-SBDT/blob/main/src/configs/eval.yaml)和[README](https://github.com/nttcom/WASB-SBDT/blob/main/README.md)给出了可加载官方权重的评估命令；模型园仅提供数据集特化的已训练 checkpoint，未提供 ImageNet/HRNet 特征预训练来源。论文也明确写从头训练。

### 后处理、时间范围与未来信息

论文先把每张热图以 0.5 二值化、取连通分量候选；CoH 用热图值加权中心和分量热图和作为分数。在线轨迹器用历史三个已输出位置构造二阶运动预测，只筛/选当前候选，不插值缺失位置。`step=3` 是不重叠的三帧组；`step=1` 对不同三帧组合过采样，再合并同一图像的候选，因此论文也报告两种速度/精度设置。

这里必须把时间语义写清：公开数据构造的每个 `[t,t+1,t+2]` 输入同时预测三帧。因而其第一个和第二个输出都可见未来帧；step=1 对一个输出帧更会收集来自多种相邻窗口的候选。它不能被标成严格在线逐帧模型。若后续需要因果对照，应固定“只报告三帧窗口最后一张输出”的单独协议，不能把这个改动和原 WASB 结果混在一起。

[在线 tracker 代码](https://github.com/nttcom/WASB-SBDT/blob/main/src/trackers/online.py)的 `max_disp=300` 是原图坐标中的硬筛选；[postprocessor](https://github.com/nttcom/WASB-SBDT/blob/main/src/detectors/postprocessor.py)实施 0.5 阈值、连通分量、加权 CoH。代码的候选质量写法与论文公式并非逐字同一实现，所以本项目应报告“按公开代码适配的后处理”，而非声称精确复现论文内部实现。

## 为什么公开仓库不足以直接训练

这不是环境问题，而是发布缺口：

- [README](https://github.com/nttcom/WASB-SBDT/blob/main/README.md)的训练章节仅写 “TBA”；作者对“何时发布训练代码”的 [Issue #6](https://github.com/nttcom/WASB-SBDT/issues/6)截至本次读取仍 open 且无作者答复。
- [runner 工厂](https://github.com/nttcom/WASB-SBDT/blob/main/src/runners/__init__.py)将 `Trainer` import 和 `train` 注册都注释掉；没有 train 根配置。虽然仓库残留 [train_and_test.py](https://github.com/nttcom/WASB-SBDT/blob/main/src/runners/train_and_test.py)，它引用仓库中不存在的 `inference_videos` 模块，且训练路径不能由公开 runner 选择。
- [BMVC 官方页](https://proceedings.bmvc2023.org/310/)所列补充材料链接目前返回 404。因此 HLSM 的“错误样本”精确筛选阈值、训练时是否使用 `full.yaml`、随机性和 checkpoint 选择规则均无法由官方可访问材料固定。

因此，修补该仓库使其能跑是**本地新实现**，不是把官方训练工程“打开”。不要为此安装旧版 Docker 环境或以新旧 PyTorch 版本兼容问题掩盖上述缺失。

## 12 GiB 单卡上的最小、可追溯建立路径

| 阶段 | 要做的最小工作 | 产物名称与可作出的结论 |
|---|---|---|
| 1. WASB-core | 在本项目数据路径实现上述 HRNet、`[t,t+1,t+2] -> 3` 热图、论文的二值盘/QFL/Adam/30 epoch。窗口严格在 rally/clip 内；只读 game 1--6 训练、game 7 验证，完全不读 8--10。首轮先不启用 HLSM 和 tracker。 | `WASB-local-core`：强高分辨率空间 + 短窗口热图基线，不能称完整原作者 WASB。 |
| 2. 资源定标 | 在真实三帧 batch 上做一次前反向 smoke，记录单卡实际峰值显存和 batch。若 batch 8 不适合 12 GiB，只减 batch，并按实际 batch 报告；不猜速度或等效 batch。 | 一份实验配置与日志；只证明通路和资源边界。 |
| 3. 指标对齐 | 在 game 7 同时输出项目的 presence/conditional PCK8/16 与 WASB 风格 `tau=4` F1、Accuracy、AP，后者仅对 VC1/2 为正、VC0/3 为负。 | 不把 PCK 数字与论文 Table 2 排名；能审计标签语义。 |
| 4. 后处理隔离 | 固定 core checkpoint，单独比较 raw heatmap/CoH、`step=3` 与 `step=1`、以及公开代码风格 online tracker。记录每一项是否引入未来窗口。 | 把空间模型收益与轨迹先验、过采样收益分开。 |
| 5. HLSM（仅在必要时） | 先预注册本地“困难样本”判据，再加实值图；它不能伪称作者的 HLSM 复现，因为官方阈值和完整训练路径未公开。 | `WASB-local-core+local-HLSM`，是独立训练策略消融。 |

最初的正式比较应是 `WASB-local-core` 对当前 DINOv3 冻结/时序基线，且保持同一 1--6/7、同一原图坐标和同一可见性口径。待这条链稳定后，再把 correspondence 分支加入同一训练/评估协议；否则高分辨率、标签策略和后处理会成为混杂因素。

这里记录的是可重建范围，尚未锁定下一模型结构。只统一评价口径仍不足以完成严格归因：源WASB将VC3作为负例、采用三帧监督及从头训练，当前DINO探针将VC3视为有位置、监督末帧且冻结骨干。这两种原生配置的比较只能是系统参照。用于old/new backbone × motion主归因时，需要另行固定共同的训练标签、监督目标帧、数据量与训练条件，并保留各自原生协议的独立报告。

## 环境与依赖边界

[作者 Dockerfile](https://github.com/nttcom/WASB-SBDT/blob/main/Dockerfile)钉住了 PyTorch 1.11 + CUDA 11.3、torchvision 0.12、Hydra、OpenCV、pandas、scikit-image 等旧环境。当前项目的 `zshihyc` 环境版本不同，且本轮没有验证作者工程的运行性。第一步本地受控实现先复用项目已有的 PyTorch、NumPy、Pillow与数据读取能力；是否需要OpenCV等由实际实现决定，实际缺哪个才在 `zshihyc` 安装，不预装或降级整套旧容器。

## 使用时的表述

可写：“我们在公开论文和作者可读模型/评估代码的约束下，重建 WASB-local-core，并在未接触 game 8--10 的 1--6/7 开发协议中训练。”

不可写：“我们复现了作者的 WASB 训练结果”或“与论文 Table 2 的 1--7/8--10 数字直接公平比较”。前者缺完整训练发布，后者还改变了开发划分、设备和可能的 HLSM/增强细节。
