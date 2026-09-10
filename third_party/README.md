# 上游代码

`dinov3/` 是官方代码的本地只读 checkout，未修改、未复制进本项目 Git 历史。DINOv3 使用其自定义许可证，见 checkout 的 LICENSE.md。

实际版本：`6876159a11b4df116f30f667f8c9888617df0751`。

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
git -C third_party/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
```

通过官方 `dinov3.models.convnext.ConvNeXt` 构造 Tiny，`torch.load` 直接读取已有权重并 strict load，避免 Torch Hub 再复制权重。调用 `get_intermediate_layers(n=4, reshape=True, norm=False)`；保持 `patch_size=None`，不把四级特征重采样成相同网格。
