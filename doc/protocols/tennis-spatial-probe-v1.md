# Tennis 空间探针协议

状态：已锁定（探索性开发协议，不是最终 benchmark）
版本：v1；锁定日期：2026-09-10。
来源：[数据说明](../../data/tracknet_tennis/README.md)、[第二轮总纲](../research/高速微小球运动表征_第二轮第一性原理与对抗性研究总纲_2026-09-10.md)。

## 数据与隔离

采用本地原作者 Tennis JPG 和原始中心标签。训练 game 1–6；验证 game 7；game 8–10 保留为最终测试，本阶段不提取其特征、不读取其预测成绩。该验证划分是本项目开发决定，不声称是 WASB 官方验证划分。将来正式 WASB 比较需在规则固定后以 game 1–7 重训。

首轮每个 clip 按原帧号 `frame_id % 8 == 0` 取目标帧；这只是单帧探针的确定性稀疏采样，不把这些帧当成相邻时序输入。原 game、clip、帧号、visibility、坐标全部随预测保存。下一阶段时序输入必须从原连续索引取帧。

visibility=0 作为源协议无球；非零且合法的原始中心作定位监督。逐项保留 visibility，并分组报告，不能把遮挡位置解释为直接可见证据。非零 visibility 的非法坐标不回归也不转成负样本，记录排除数。已知 game8 边界异常本阶段不会进入 train/val。原文件不改动。

[TrackNet 原稿](https://arxiv.org/abs/1907.03698)定义：0 不在画面；1 易辨认；2 难辨认、标注可参考邻帧；3 被遮挡、位置可由邻帧估计。拖影标注倾向轨迹最新位置，不等同于模糊中点。发布索引中无球的 x/y 为空，保留为空，不转换为真实 `(0,0)` 位置。

## 输入与空间读出

单帧 RGB、无未来信息；全图 1280×720 等比例缩放到 512×288，PIL bilinear，无 crop、padding 或增强。使用 Web DINOv3 ImageNet mean/std。下一尺度比较需独立缓存且保持同一目标帧集合。

像素中心坐标变换：`x_input=(x_original+0.5)*512/1280-0.5`，y 同理。原生 stride s 的 cell i 对应原图 `(i+0.5)*s/scale-0.5`。训练目标是包含原中心的网格 cell；解码输出该 cell 中心，不偷偷加 GT 偏移或轨迹平滑。

分别读取 ConvNeXt-Tiny 四个 stage，stride 4/8/16/32，raw intermediate features（官方 `norm=False, reshape=True, patch_size=None`）。每帧缓存一次 float32 forward 后的 float16 特征；先测 float16 保存引入的数值误差。缓存不包含由测试标签生成的内容。

## 头、监督与评价

统一使用无仿射 GroupNorm(1,C) 后的 1×1 空间分类头及全局池化无球分类头，在 H×W+1 类上使用交叉熵。无球 logit 加 log(HW) 抵消初始类别数差异。该探针称为“归一化线性读出”；不同 stage 通道数不同，参数量不同，报告参数量，不能称完全等容量。

主要诊断不依赖存在阈值：对所有合法非零标签，使用空间最大响应计算原图像素误差中位数、PCK@8、PCK@16、PCK@32，分别报告 visibility 与 clip 汇总。它测条件位置读出，不等于自动检测 F1。

状态概率为各空间类别概率和；以固定 0.5 给出辅助存在 precision/recall/F1，定位成功还需在指定容差内。错位的预测在定位 F1 中同时产生 FP 与 FN。所有阈值明确，保存预测以便以后重新评价而不重复 forward。

同时给出 GT cell 中心的量化误差与 PCK 上限，以区分网格限制和识别失败。层选择依据验证数据；保留所有运行，不只报告最佳值。单一验证比赛与单 seed 的结果仅用于下一轮决策，不能支持跨赛事性能主张。

## 成本与缓存

特征提取单独计时；缓存训练吞吐不称端到端速度。缓存根 `data/cache/tennis/dinov3_convnext_tiny_512x288_step8/`，包含帧索引、实际配置、各 stage 的 `.npy`。读取配置防止误用不同输入/权重缓存；不增加内容哈希。完整写入后才保存完成元信息。模型输出、日志和预测在 `outputs/spatial_probe/<run>/`；保留 best checkpoint，不保存每个 epoch 的大副本。
