# 高速微小球运动表征与定位

本项目研究连续体育视频中的高速微小球定位，主要面向网球、羽毛球和乒乓球。目标是在现代预训练视觉 backbone 上，以有限计算保留微小目标的空间证据，并有效利用较大帧间位移下的运动信息。

这是长期论文研究项目。核心贡献需要来自 motion representation 的机制与实验依据；更换 backbone 本身不等于研究创新。

## 从这里开始

| 你要了解什么 | 阅读入口 |
|---|---|
| 文档怎么分类、已有材料怎么读、新文档怎么写 | [文档导航与组织规范](doc/README.md) |
| 研究问题、反例与实验思路 | [第二轮研究总纲](doc/research/高速微小球运动表征_第二轮第一性原理与对抗性研究总纲_2026-09-10.md) |
| 数据存在哪里、标注如何理解、有哪些已知异常 | [数据说明](data/README.md) |
| DINOv3 权重放在哪里、首轮用哪些、怎样匹配架构 | [预训练权重说明](models/README.md) |
| 实验前需要确定什么协议 | [实验协议说明](doc/protocols/README.md) |
| 实验计划和实际结果怎样记录 | [实验记录说明](doc/experiments/README.md) |
| 已经完成什么、哪些结论尚未建立 | [阶段进展](doc/README.md#阶段进展) |
| 与本项目协作时遵守什么规则 | [AGENTS.md](AGENTS.md) |

## 研究范围

- 每个输入窗口位于同一连续 clip/rally，相机可以连续运动；窗口和时序状态不跨片段边界。
- 主要输出为协议定义的二维球位置及相关可见性状态；第一阶段不研究切镜、跨镜轨迹 ID、SLAM 或三维重建。
- 使用现有公开标注。未标注帧不等于无球帧，稀疏标签必须保留原始时间索引。
- 优先研究微小空间支撑、大位移搜索与计算效率之间的矛盾，同时允许简单时序基线推翻复杂设计。
- 通过 old/new backbone × no/new motion 等对照区分收益来源；候选结构和论文贡献都需要实验验证。

## 仓库结构

```text
.
├── README.md             # 项目入口
├── AGENTS.md             # 长期研究与协作约束
├── doc/                  # 研究论证、证据、协议、实验记录与进展
│   ├── README.md         # 完整导航、命名规则与文档格式
│   ├── research/         # 立项蓝图、综合审查、方法论证
│   ├── literature/       # 有来源和阅读边界的专题文献笔记
│   ├── protocols/        # 实验采用的数据、输入与评价约定
│   ├── experiments/      # 实验假设、运行条件、结果及判断
│   └── progress/         # 按日期记录阶段变化
├── data/                 # 本地数据、源标注、索引和数据说明
├── models/               # 权重说明与本地外部预训练权重
├── src/ballmotion/       # 数据/时间语义、特征读出与定位评价
├── configs/             # 实际基线使用的模型配置
├── scripts/              # 特征提取和实验入口
├── tests/                # 几何、标签、评价与拟合验证
├── third_party/          # 上游 DINOv3、WASB 实现与来源说明
└── outputs/              # 本地运行日志、预测、训练 checkpoint
```

实际运行参数保存在每次实验的 `config.json`，WASB模型配置位于`configs/`。视频、权重、缓存和完整训练输出不放进 `doc/`，也不默认提交版本库。

## 数据与运行

Python 默认使用 Conda 环境 `zshihyc`：

```bash
conda activate zshihyc
```

后续命令均在此环境运行。用户已授权按实际需要直接安装依赖；安装到当前环境并同步更新实际依赖说明，不预装尚未用到的包。

数据范围为 TrackNet Tennis、Shuttlecock Trajectory Dataset、BlurBall 和 OpenTTGames。原始副本、版本差异和使用入口以 [data/README.md](data/README.md) 为准；不要为开始阅读项目而重跑下载或全量校验。

用户提供的 DINOv3 权重已整理到 `models/pretrained/dinov3/`，见 [权重说明](models/README.md)。首轮使用 ConvNeXt-Tiny，按 [上游代码说明](third_party/README.md) 准备 DINOv3；当前环境所用直接依赖见 [requirements.txt](requirements.txt)，无需重新安装已有依赖。

Tennis的全量基线、历史控制与多项表示诊断已经完成，见[阶段记录](doc/progress/2026-09-10-full-baselines.md)。BlurBall已推进到允许未来帧的上下文比较与候选失败诊断；当前状态统一见[阶段进展](doc/README.md#阶段进展)，尚未确定最终motion模型。以下保留[早期Tennis单帧冻结探针](doc/protocols/tennis-spatial-probe-v1.md)的运行示例：

```bash
PYTHONPATH=src python tests/test_spatial_probe.py
python scripts/cache_tennis_features.py --output data/cache/tennis/dinov3_convnext_tiny_512x288_step8
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8 --output outputs/spatial_probe/linear_stage0_seed0 --stage 0
```

`--stage 0/1/2/3` 选择独立层位，每个新实验使用独立输出目录；完整缓存可以复用，不重新解码或提取特征。训练/验证为 game 1–6 / 7，最终测试 game 8–10 本阶段不运行。结果以实际实验记录为准。

因果三帧使用真实 `t−2,t−1,t`，通过唯一帧缓存和窗口索引避免重复提取；当前帧、真实堆叠、重复当前帧三个对照共享目标集合。实际命令和状态见[因果对照实验记录](doc/experiments/2026-09-10-causal-stack.md)，不要用单帧缓存中相邻的稀疏行替代连续视频帧。

## 研究记录原则

原始想法、文献事实、待验证假设和实测结果分开表述。数据已下载不代表训练协议已确定，模型可运行不代表论文结论成立。历史审查按其记录日期阅读，采用其中的方案前仍需核对当前证据。

默认不增加哈希、校验和、指纹或防御性框架。每次检查必须对应具体失败和后续处理动作，已确认结果不无理由重复检查。完整规则见 [AGENTS.md](AGENTS.md)。
