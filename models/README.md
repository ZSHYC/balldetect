# 预训练模型权重

这里保存项目使用的外部预训练权重，原始文件不覆盖、不转换。当前保留 9 份 Web 预训练的 DINOv3 `.pth`，合计 6,607,050,241 bytes，约 6.61 GB（6.15 GiB）。来源目录原为根目录 `DINOv3/`，现已移动到下述位置，没有复制出第二份权重。

## 对本项目的用途

**这批权重有用，首轮需要的三个 backbone 都已具备：DINOv3 ConvNeXt-Tiny、ConvNeXt-Small 和 ViT-S/16。** 它们提供预训练视觉表征，后续仍需实现定位读出和训练；这些文件本身不是已经训练好的球检测器，也不直接输出球坐标。

首轮建议以 Web 预训练的 ConvNeXt-Tiny 为轻量起点，ConvNeXt-Small 用于同家族规模对照，ViT-S/16 用于另一类表示对照。是否保留微小球的空间证据、是否需要高分辨率分支，要通过实验判断。

ViT-S+/16 是后续可选对照，不与 S/16 混称同一模型。官方实现中两者都采用 16×16 patch、384 维表示、12 层，但 FFN 分别使用 MLP 与 SwiGLU，配置也不同。更大的 B/L/H+ 权重保留供规模或特征探针研究，不自动加入首轮实验。

本项目不开展遥感实验，SAT-493M 权重已按用户要求删除，不再保留遥感预训练对照。以上模型用途是研究建议，不是已经测得的性能排序。

## 文件夹与命名

```text
models/
├── README.md
└── pretrained/
    └── dinov3/
        └── lvd1689m/      # Web 预训练：9 个文件
```

以“外部预训练 / 模型家族 / 预训练数据来源”组织目录，文件名继续使用原始的 `dinov3_<架构>_pretrain_<数据来源>-<发布标识>.pth`。无需再为每个仅有一个权重的架构创建一层文件夹。

**不要把这些原始文件改名为 `best.pth`、`small.pth` 或 `model.pth`。** 文件名包含架构和预训练来源；尤其官方 ViT-L 本地加载逻辑会读取文件尾部发布标识，选择对应模型结构。保留已有标识不表示本项目新增或计算哈希。[官方加载实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py#L316-L340)

以后本项目训练出的 checkpoint 放在实际实验输出目录，例如 `outputs/<实验主题>/<运行名>/checkpoints/`，与这里的外部预训练权重分开。只有真实运行时才创建输出目录，原始预训练权重不被训练结果覆盖。

## 本地文件清单

下表链接直接指向已整理的文件。MiB 为文件大小，不是运行显存，也不等于模型实际训练开销。

| 架构 | 原始权重文件 | MiB | 用途 |
|---|---|---:|---|
| ConvNeXt-Tiny | [dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth](pretrained/dinov3/lvd1689m/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth) | 107.04 | 首轮起点 |
| ConvNeXt-Small | [dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth](pretrained/dinov3/lvd1689m/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth) | 192.20 | 首轮同家族对照 |
| ViT-S/16 | [dinov3_vits16_pretrain_lvd1689m-08c60483.pth](pretrained/dinov3/lvd1689m/dinov3_vits16_pretrain_lvd1689m-08c60483.pth) | 82.52 | 首轮 ViT 对照 |
| ViT-S+/16 | [dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth](pretrained/dinov3/lvd1689m/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth) | 109.60 | 后续可选结构对照 |
| ConvNeXt-Base | [dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth](pretrained/dinov3/lvd1689m/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth) | 337.61 | 后续规模对照 |
| ConvNeXt-Large | [dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth](pretrained/dinov3/lvd1689m/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth) | 780.74 | 后续规模对照 |
| ViT-B/16 | [dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth](pretrained/dinov3/lvd1689m/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth) | 326.98 | 后续规模对照 |
| ViT-L/16 · Web | [dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth](pretrained/dinov3/lvd1689m/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth) | 1156.86 | 后续规模或探针对照 |
| ViT-H+/16 | [dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth](pretrained/dinov3/lvd1689m/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth) | 3207.43 | 资源允许且研究需要时使用 |

## 已确认的本地结构

2026-09-10 使用当前环境中的 PyTorch 2.10.0，以 `torch.load(..., map_location="meta", weights_only=True, mmap=True)` 读取张量元信息。当前保留的 9 个文件均已确认可解析为由张量组成的 state dict，没有将整个模型权重加载到 GPU，也没有计算哈希。

- ConvNeXt 权重具有分层 `downsample_layers`、`stages` 和 `norms`；T/S 的 stem 权重形状为 `[96,3,4,4]`，B 为 `[128,3,4,4]`，L 为 `[192,3,4,4]`。
- ViT 权重的 patch embedding 为 16×16；S/S+、B、L、H+ 的通道数分别为 384、768、1024、1280，并包含 4 个 storage/register tokens。

上述检查当时只确认文件可解析及关键结构，没有执行 forward，也没有验证与发布方逐字节相同。2026-09-10 实施阶段已对 ConvNeXt-Tiny 完成官方构造器严格加载、四级真实 forward 和冻结空间读出训练，详见[实验记录](../doc/experiments/2026-09-10-spatial-probe.md)。其他架构尚未执行真实 forward；首次接入时再检查，不无理由重扫权重。

## 后续如何接入

使用 DINOv3 官方实现中与文件匹配的构造器。首轮三个名称为 `dinov3_convnext_tiny`、`dinov3_convnext_small`、`dinov3_vits16`；S+ 对应 `dinov3_vits16plus`。不要直接把它们当作普通 torchvision ConvNeXt/ViT 或另一个实现的权重文件。

官方支持本地源码与本地权重路径加载，具体接口见 [README 的本地加载示例](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md) 和 [backbones.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py)。本次只整理权重，没有下载源码或建立新的加载包装层；正式实现时使用项目配置传入模型名和权重路径。

输入尺寸、归一化与预训练来源一起记录。ViT 的 16 像素 patch 对微小球定位是需要研究的条件，不应通过未记录的裁边或 resize 改变标签坐标；也不要原样照搬分类演示中的中心裁剪到球定位任务。[官方模型卡](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/MODEL_CARD.md)

## 来源与保存约定

文件由用户提供，名称与结构按官方资料核对；本次没有重新下载，也未改写或转换文件。适用 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md)，不将其误写成 MIT/Apache。原始获取入口为 [Meta DINOv3 下载页](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)。

`models/pretrained/` 已加入项目 `.gitignore`，README 可以纳入版本管理，权重默认不提交或再分发。模型规格以外部来源为依据，球定位效果以本项目实验为依据。
