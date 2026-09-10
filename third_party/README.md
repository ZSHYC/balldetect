# 上游代码

## DINOv3

`dinov3/` 是官方代码的本地只读 checkout，未修改、未复制进本项目 Git 历史。DINOv3 使用其自定义许可证，见 checkout 的 LICENSE.md。

实际版本：`6876159a11b4df116f30f667f8c9888617df0751`。

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
git -C third_party/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
```

通过官方 `dinov3.models.convnext.ConvNeXt` 构造 Tiny，`torch.load` 直接读取已有权重并 strict load，避免 Torch Hub 再复制权重。调用 `get_intermediate_layers(n=4, reshape=True, norm=False)`；保持 `patch_size=None`，不把四级特征重采样成相同网格。

## WASB的HRNet

`wasb/hrnet.py`保存作者[固定版本的原始模型定义](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/models/hrnet.py)，版本为`923462cacdeb3353b84ddebdedb3f4b7a8553b0f`，获取于2026-09-10；文件未修改，原版权头与[MIT许可证](wasb/LICENSE.md)保留。本项目只引入实际使用的模型文件，不复制整个训练/runner工程。

[本地配置](../configs/wasb_hrnet_causal.yaml)来自该版本的`src/configs/model/wasb.yaml`，仅将frames_out从3改为1，以监督因果三帧窗口末帧。使用已安装的OmegaConf 2.3.1读取，直接`HRNet(cfg)`构造；不使用作者模型工厂、Hub、预训练下载或自定义配置兼容层。该受控适配不是作者完整训练复现，时间/标签/损失差别见[全量因果协议](../doc/protocols/tennis-full-causal-v1.md)。
