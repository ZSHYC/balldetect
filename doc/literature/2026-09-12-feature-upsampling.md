# 高分辨率 VFM 特征上采样与微小球空间证据

日期：2026-09-12。状态：四个直接近邻已核对；AnyUp 与 RaysUp 已读截至本日可得的最新 arXiv 正文、附录和作者代码。本文件只判断**单帧空间证据读出**，不把 feature upsampling 写成 motion representation，也不替代正在进行的 BlurBall 时序干预。

## 结论先行

高分辨率特征上采样值得作为一个严格受控的**空间基线/adapter 候选**，不值得预先成为论文的贡献主线。它回答的是“给定当前 RGB 和低分辨率 VFM feature，能否更好地把 feature 分给高分辨率位置”；它没有做跨帧对应、大位移搜索、球的自动发现，或相机/物体运动分解。

因此它可能改善小球中心读出，但那一改善有两种不能混淆的来源：

1. 原低分辨率 feature 已含有球相关语义，adapter 只把它更准确地分配到球的像素附近；
2. adapter 从当前高分辨率 RGB 的边缘、亮点、颜色或局部纹理获得了新增观测，再用 VFM feature 给这些观测赋予语义。

第二种仍是合法、可能有效的视觉模型，但不叫作“从被下采样 feature 恢复了丢失的球信息”。若球在当前 RGB 也不可分辨（完全遮挡、强拖影与背景混合），输出变锐不能证明新增了直接球像素证据；模型仍可能依靠上下文或先验作出较准的位置估计，但这是另一种解释。若方法使线条、反光或衣物也得到更锐利的 feature，它还可能增加误报；通用分割/深度成绩不区分这一风险。

对当前项目最实际的建议是：先完成已经锁定的 BlurBall 真历史对照；随后仅在 DINOv3 ViT-S 路线已有可用冻结空间基线时，考虑用一个预训练 adapter 作受控 probe，AnyUp 是可选的首个候选。RaysUp 的接口包含内参与射线位置编码，但论文主 ablation 使用 identity pose；按其官方默认几何约定运行不等于必须另做真实相机估计。两者都尚未完成本项目接入、显存或速度实测，也没有解决本项目的时序问题；这里不是批准顺序或唯一技术路线。

## 证据范围、版本与读取状态

| 工作 | 本次核对版本与发表状态（截至 2026-09-12） | 本次读取深度 | 结论用途 |
|---|---|---|---|
| [FeatUp](https://proceedings.iclr.cc/paper_files/paper/2024/file/c5601d99ed028448f29d1dae2e4a926d-Paper-Conference.pdf) | ICLR 2024；[arXiv:2403.10516](https://arxiv.org/abs/2403.10516) | 正式正文与补充 HTML 的方法、成本和任务段落；作者仓库已存在 | 高分辨率 RGB-guided 读出的基础先例 |
| [LoftUp](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_LoftUp_Learning_a_Coordinate-Based_Feature_Upsampler_for_Vision_Foundation_Models_ICCV_2025_paper.html) | ICCV 2025，pp. 9913--9923；[arXiv:2504.14032](https://arxiv.org/abs/2504.14032) | 正文与 supplementary 的 architecture、pseudo-GT、video protocol、效率段落；[作者代码](https://github.com/andrehuang/loftup) | 坐标条件、RGB 条件、全局 cross-attention 与 mask pseudo-GT 的直接先例 |
| [AnyUp](https://arxiv.org/html/2510.12764v2) | arXiv **v2，2026-02-16**；[ICLR 2026 正式版](https://openreview.net/pdf/e03a66822acec855114dde73a6fe2226ad23ad6b.pdf) | 完整正文、附录 B/C；[作者代码](https://github.com/wimmerth/anyup) 的当前固定提交 | 本项目最小可试的 encoder-agnostic adapter 候选 |
| [RaysUp](https://arxiv.org/html/2606.22749v1) | arXiv **v1，2026-06-22**；[ECCV 2026 官方 accepted-papers 页面](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)已列名 | 完整正文、附录；[作者代码](https://github.com/MAP-RaysUp/RaysUp) 的当前固定提交 | AnyUp 后的效率近邻；射线先验带来额外假设 |

正式会议状态与 arXiv 版本须分开写：AnyUp 的 v2 是已发表稿对应的公开版本；RaysUp 在官方录用名单中，但本次不把仓库 README 的口号当作 proceedings 版或本项目复现证据。下面关于代码的陈述只对应读取时的固定公开提交，不倒推为论文实验的精确实现：`wimmerth/anyup@351807a9c4287368732cc247f26c7c81c9139af4` 与 `MAP-RaysUp/RaysUp@a320000e5d89ef771a89b49c965a90312fbe8eab`。

## 先把“更密的 feature”拆成不同信息来源

令当前 RGB 为 $I_t$，冻结 VFM 的低分辨率 feature 为 $P_t=e(I_t)$，上采样输出为 $Q_t=u(P_t,I_t,z)$，其中 $z$ 可以是像素坐标或相机几何。输出网格更细本身不是信息论结论。项目中应按下面的来源写归因。

| 来源 | 它实际新增/保留什么 | 能支持的结论 | 不能支持的结论 |
|---|---|---|---|
| 低分辨率 VFM feature $P_t$ | 粗网格的预训练语义、上下文和可能存在的球响应 | 该表示在 coarse support 中仍可供 head 使用 | 球的像素级位置必然可读；高分辨率输出一定是原 feature 的真实逆变换 |
| 当前高分辨率 RGB $I_t$ | 球边缘、亮度、拖影、场线、反光和衣物等原始图像证据 | adapter 使用了当前帧的额外空间观测 | adapter 仅改善了 VFM readout；RGB 的锐化一定对应球而非干扰物 |
| 坐标/位置编码 | 像素位置、相对尺度、可能的 ray/intrinsics | 输出可在任意网格 queried，或对给定视角进行空间偏置 | 得到了帧间运动、相机运动估计或真实三维重建 |
| 外部训练监督 | multi-view augmentation、image crop feature、class-agnostic mask、EMA teacher 等 | 在该监督的结构偏好下，输出有更好 dense-task 可用性 | 只有中心点标签时同样适配；在球、模糊、遮挡或高速条件下仍正确 |

这是本问题最重要的边界。所有四个学习式方法都在当前帧利用的东西超过 $P_t$：FeatUp-JBU 显式以原图为 guidance；LoftUp 用 RGB+coordinate 生成 query；AnyUp 用 RGB feature 生成 query/key；RaysUp 用 RGB guidance、坐标和 ray pose 生成 query/key。AnyUp 的作者把该任务直接定义为 $q=f(p,I_{hr})$，而非 $q=f(p)$，[其任务定义](https://arxiv.org/html/2510.12764v2#S3)已经明确这一点。RaysUp 更明确地令 VFM output 只作 value，而高分辨率 guidance 作 query、key，[式 (4)--(6)](https://arxiv.org/html/2606.22749v1#S3.SS3)。

**推论。** 对球点定位，若 adapter 相对 bilinear 有增益，正确表述是“在固定原始图像与 VFM feature 的条件下，整个 adapter 提高了该 head 的位置可读性”。两臂共享源图像，但 bilinear 不消费独立的高分辨率 RGB guidance，故这不是信息路径相同的纯 interpolation 对照；增益可来自 adapter 对 RGB guidance 与 coarse VFM 的联合使用。只有再做信息来源对照，才能进一步说增益主要来自 coarse VFM 还是 high-resolution RGB guidance；两者都不能自动外推为跨帧 correspondence。

## 四个方法的机制与证据边界

### FeatUp：多视图一致性与 RGB-guided JBU

**论文事实。** FeatUp 有两条实现路径。feed-forward JBU 版将低分辨率 feature 和原图 $x$ 输入一串可学习 joint bilateral upsampler；论文明确说原图高频细节直接进入上采样过程，[§3.2](https://arxiv.org/html/2403.10516#S3.SS2)。每一个高分辨率输出由邻域低分辨率 feature 的加权组合得到，权重同时依赖空间距离和高分辨率 guidance 的相似性，[式 (4)--(6)](https://arxiv.org/html/2403.10516#S3.SS2)。Implicit 版则为**每张图像**过拟合一个隐式网络，可以产生极锐的任意分辨率输出，但不是可直接逐帧部署的正常视频通路。

训练不是中心点监督。它令高分辨率 output 经几何变换和 learned downsampler 后与多个 jittered 图像的 backbone feature 一致；原文称为 multi-view reconstruction，并使用不确定性处理不能上采样的 outlier，[§3](https://arxiv.org/html/2403.10516#S3)。其正式评测是 CAM、segmentation、depth 与 end-to-end segmentation，不含视频时序、球点、模糊或 frame-to-frame repeatability。[表 1](https://arxiv.org/html/2403.10516#S4) 的 dense-task 数值因此只能证明这些任务的 readout 改善。

**对 tiny ball 的解释。** JBU 可以让球边缘或拖影方向影响粗 feature 的空间落点，所以它不是纯 bilinear；同一机制同样可能让白色场线或局部高光得到更尖锐响应。Implicit 版的 per-image 优化还允许单帧看起来很锐，却没有共享的时间约束。FeatUp 正文没有任何相邻帧一致性、因果延迟、球中心或小目标尺度实验，不能把论文中“multiple views”误读为视频多帧 motion。

**成本。** JBU 是可前向部署的轻量基线；但其快速 CUDA JBU 是论文工程的一部分，[实现动机](https://arxiv.org/html/2403.10516#S3.SS2)。Implicit 路线不适合本项目在线/逐帧预算：LoftUp 的复现式比较报告它在 A100 上约 54.302 s/image，而 FeatUp-JBU 为 0.1213 s/image；该数字是对方论文的统一比较，不应当作本机端到端速度，[LoftUp Table 7](https://arxiv.org/html/2504.14032#S5.SS1)。本项目不应为了“更锐”部署 implicit FeatUp。

### LoftUp：坐标 + RGB query，低分辨率 VFM key/value

**论文事实。** LoftUp 将全分辨率 $(x,y)$ 正弦位置编码与 RGB 拼接，经卷积得到 query；低分辨率 VFM features 是 cross-attention 的 key/value，[§3](https://arxiv.org/html/2504.14032#S3)。它不是局部插值：论文的 attention 可从全局低分辨率 feature map 取信息，[attention 可视化说明](https://arxiv.org/html/2504.14032#S5.SS1)。这正是“坐标条件 high-resolution feature adapter”已经有强先例的原因。

其训练目标更加说明 output 的空间边界是有偏好的。Stage 1 先以 class-agnostic masks 让 bicubic feature 在每个 mask 内趋同，Stage 2 再用 EMA teacher self-distillation；mask-refined pseudo-target 和 $L_2$ loss 的定义见[§4](https://arxiv.org/html/2504.14032#S4)。训练使用 SA-1B 的 1M 子集、两块 cross-attention、两阶段一共不是仅用球中心标签；补充材料给出了 Stage 1/2 optimizer、EMA 和 1M-image 设置，[Appendix A.1](https://arxiv.org/html/2504.14032#A1)。

**对 tiny ball 的解释。** mask 内平滑对大物体 segmentation 很合理，却是小球的双刃剑：一个很小的圆球、它的拖影、邻近场线和反光是否落入同一个 mask，决定了 pseudo-GT 是否把它们同化。因而 LoftUp 的成功不能说明它保留了球的亚像素中心或独立的模糊方向。它确有 DAVIS 视频对象分割结果，[Table 4](https://arxiv.org/html/2504.14032#S4.SS1)，但评测对象是第一帧给出 mask 的半监督 VOS，非无初始化 tiny-ball 自动定位；该表没有报告中心误差、相邻帧 descriptor repeatability 或大位移重捕获。

**复杂度与成本。** 每个高分辨率 query 可以对所有低分辨率 token 做 cross-attention，理论工作量为 $O(HW\cdot hw)$（再乘 head/channel 的常数）；它避开在高分辨率上堆叠 key/value，所以不等价于 full high-res self-attention。论文在 A100 的统一设置下报 LoftUp 4.3M 附加参数、0.0893 s/image；此比较包含 backbone 参数显示为 26.4M，不能直接拿来扣除本项目 DINO forward 的时间，[Table 7](https://arxiv.org/html/2504.14032#S5.SS1)。在 448px 时，后续 AnyUp 的复现实验报告 LoftUp 201.2ms、1065.3 GFLOPs、7.9GB forward memory，[AnyUp Table 10](https://arxiv.org/html/2510.12764v2#A2.SS1)，说明“只对 coarse key/value attention”在高分辨率并不自动便宜。

### AnyUp：通用 feature adapter，但本质仍是 RGB-guided 局部路由

**论文事实。** 当前最新公开版本为 arXiv v2。AnyUp 的目标是一次训练、推理时适配未见 encoder/feature dimension；先用 feature-agnostic layer 把任意通道数的 coarse feature 变成 canonical representation，再以高分辨率图像 feature 形成 query、图像 feature 与 coarse feature 共同形成 key，value 则保留输入 patch feature，[方法 §4](https://arxiv.org/html/2510.12764v2#S4)。作者代码的 `AnyUp.upsample` 也可直接核对此数据流：query 来自 image encoding，key 拼接 image encoding 和 feature encoding，而 `v = feats`；见 `wimmerth/anyup@351807a9c4287368732cc247f26c7c81c9139af4:anyup/model.py:L64-L93`。

它刻意将每个高分辨率 query 的 attention 限在对应 coarse feature 的局部 window，而不全局匹配。论文的复杂度写作：global 为 $HW\cdot hw$，window 为 $HW\cdot\sigma h\cdot\sigma w$，实验使用 NATTEN CUDA kernels 且 $\sigma=0.2$，[Appendix B.1](https://arxiv.org/html/2510.12764v2#A2.SS1)。这解决的是**单帧 feature routing 的计算量**，不是球在 $t-\Delta$ 与 $t$ 之间的大位移搜索范围；不能把局部 attention 当作 temporal local matching。

训练从 ImageNet 的原图随机抽 local crop：full image 下采样后产生 coarse feature，crop 的 encoder feature 是相应高分辨率 target，再以 cosine+$L_2$、self-consistency 与 input-consistency 训练，[§4.3](https://arxiv.org/html/2510.12764v2#S4.SS3)。这使其 output 学到“同一张图不同缩放/crop 下应保留的 feature”，但没有球中心、visibility、blur orientation、video identity 或 motion label。训练作者报 100k steps、batch 4、每图 4 crops，单 H100 约 5 h，[Appendix C](https://arxiv.org/html/2510.12764v2#A3)。

**DINOv3 证据到底是什么。** AnyUp 表 6 的 DINOv2 ViT-S trained model 在 DINOv3 ViT-S+ 上有 semantic segmentation linear-probe 与 depth probe 数字（DINOv2-trained 62.96/81.82 mIoU/accuracy，DINOv3-trained 62.99/81.84），[Table 6](https://arxiv.org/html/2510.12764v2#S5.SS2)。这是“该 adapter 不必为 DINOv3 ViT-S+ 重训、在这两类 dense probe 上仍可用”的实测，不是 DINOv3 ConvNeXt、球、运动模糊、时序一致性或 correspondence 的实测。论文自己的 ablation 也显示移除 coarse feature 到 key 的路径仍能接近既有 upsampler 的 depth performance，[§5.3](https://arxiv.org/html/2510.12764v2#S5.SS3)；这进一步要求本项目不要把输出变锐一概归给 VFM feature 本身。

**成本。** 在单 A100、只计 upsampler 的比较中，AnyUp 在 224px 是 0.8M params、20.6ms、20.6 GFLOPs、0.8GB forward；448px 是 92.4ms、82.4 GFLOPs、3.3GB forward，训练 backward 12.9GB，[Table 10](https://arxiv.org/html/2510.12764v2#A2.SS1)。这是有用的量级，不是本项目 RTX 5070Ti 的端到端结果；还未包含 DINO encode、视频 decode、定位 head 和多帧 motion。作者公开配置当前默认 `window_ratio: 0.1`，而论文性能段写 $σ=0.2$，所以真正复现前必须以选定 checkpoint/config 为准，不能混用两者的速度宣称。NATTEN 的稀疏 CUDA attention 是实际代码路径的性能前提，不应在当前训练期间安装或假设已可用。

### RaysUp：更便宜的局部 routing，加了视角射线先验

**论文事实。** RaysUp 也是单帧 upsampler。它将 RGB guidance encoder 输出 adaptive-pool 到任意输出网格得到 query、pool 到 coarse grid 得到 key，VFM coarse feature 是 value，[§3.3](https://arxiv.org/html/2606.22749v1#S3.SS3)。随后用 $k=6$ 的 neighborhood attention 聚合，附录明确列出该 kernel、$D_g=256$ 与 ray frequency 配置，[Appendix C.1](https://arxiv.org/html/2606.22749v1#A3.SS1)。当前作者代码也显示 `features` 直接做 value、attention 固定局部 6x6，见 `MAP-RaysUp/RaysUp@a320000e5d89ef771a89b49c965a90312fbe8eab:src/upsampler/raysup.py:L165-L195`。

不同于 AnyUp 的附加输入是 RayPE。它以 camera-to-world 与 $(f_x,f_y,c_x,c_y)$ 构造每个像素的 ray origin/direction，再给 query/key 加旋转式 ray position encoding；这在代码中不是抽象口号，而是 `forward` 的 `poses` 参数和 ray construction，见 `MAP-RaysUp/RaysUp@a320000e5d89ef771a89b49c965a90312fbe8eab:src/upsampler/raysup.py:L165-L225`。论文 ablation 的默认是 **identity pose**，也比较了 Depth Anything 3 估计 pose，[Table 4](https://arxiv.org/html/2606.22749v1#S4.SS5)。它的“geometry-aware”是单视图/给定视角下的 feature routing prior，绝不是连续体育视频中的 camera motion estimation 或 object/camera motion disentanglement。

训练仍是 ImageNet feature reconstruction：448px target、224px guidance、100k iteration、batch 4、cosine+$L_2$，单 A100 约 1 h，[§4.1](https://arxiv.org/html/2606.22749v1#S4.SS1)。所以它不需要本项目另画中心标签来预训练，也没有证明该通用预训练与球中心点 supervision 的相容性。RaysUp 在 DINOv3 ViT-S/M 的 segmentation/depth linear probe 中优于 AnyUp；例如 ViT-S DINOv3 的 mIoU/accuracy 为 82.60/95.97、depth abs/rel RMSE 0.502/0.342，[Table 2](https://arxiv.org/html/2606.22749v1#S4.SS3)。这仍仅是 dense-task feature probe。

**效率证据与缺口。** 作者表报 224px 0.14M、10.17 GFLOPs、1.26GB、55FPS；448px 40.67GFLOPs、2.69GB、27FPS，较 AnyUp 更低，[Table 3](https://arxiv.org/html/2606.22749v1#S4.SS4)。但正文只明确其**训练**在单 A100，效率表没有交代测 FPS 的 GPU、batch、precision、warm-up、是否计 image encoder/pose construction。故可以把它视为作者报告的相对比较，不能把 55FPS/27FPS 写成本项目设备上的端到端承诺。

RaysUp 唯一的“video”评测是 DAVIS semi-supervised VOS：首帧 mask 用 cross-frame feature similarity 向后传播，memory queue 有首帧和前七帧，并把每个目标像素匹配限制在 12px local neighborhood、保留 top 5 affinity，[Appendix C.2](https://arxiv.org/html/2606.22749v1#A3.SS2)。这说明作者测试了 upsampled feature 能否帮助一个**另行定义**的半监督传播 pipeline；它不报告无初始 mask 自动发现、球点、强模糊、任意大位移、因果定位准确度，亦未证明 upsampler 输出本身跨帧一致。

## “能否恢复 tiny-ball 证据”的可证伪判断

以下不是论文事实，而是由四类机制共同限定的研究推论。

### 可以成立的弱命题

在一张固定 RGB 中，低分辨率 VFM feature 也许已表达“该 coarse cell 包含球/球样上下文”，而 RGB-guided routing 可让定位 head 把该响应从一个 patch 的中心移到更接近球中心的位置。这会使严格 PCK 改善，即使 coarse feature 没有新的 channels。该命题完全值得通过中心点监督 head 检验。

### 不能直接成立的强命题

1. **“上采样恢复了丢失的像素证据。”** 若仅 bilinear/native resampling 改善，不能恢复；若 RGB-guided adapter 改善，新增依据是当前 RGB。只有一个严格的 source ablation 才能判断哪一项在起作用。
2. **“更锐的 PCA/segment boundary 等于更准小球中心。”** 小球在数据里是极小、可能模糊或低对比的对象；物体边界 mIoU 与中心误差不是同一统计目标。
3. **“DINOv3 adapter success 等于 DINOv3 ConvNeXt success。”** 现有 AnyUp/RaysUp 的 DINOv3 数字是 ViT-S/M dense probe；不能以论文标题的 VFM-agnostic 取代实际架构验证。
4. **“DAVIS video segmentation 等于 ball tracking。”** DAVIS 使用首帧 object mask，且局部传播的空间半径为 12px；本项目要求是未以 GT 初始点启动的逐帧定位，并研究可能远大于球尺寸的位移。
5. **“RayPE 解决移动相机。”** identity pose 是 RaysUp 的主 ablation 条件；该模块没有估计连续视频的 pan/zoom 或分离球和相机的二维运动。

### 对当前固定浅层残差负结果的关系

[冻结 detail readout](../experiments/2026-09-11-frozen-detail-readout.md) 已经完成 native-stage0 对低通 pooled 残差的三 seed 对照：native 的 PCK@8 描述均值高 0.4582pp，但 F1@16 低 0.06492pp，预设联合条件失败，因此**停止该固定残差配方**。这不证明“一切高分辨率路径无用”，却明确禁止把它重新命名成“浅层细节不够，所以加 AnyUp”而跳过新的、可区分的证据。

AnyUp/RaysUp 与该已结束 recipe 的关键差别是：它们在低分辨率 VFM output 上再读当前 RGB，并以外部 generic dense reconstruction 预训练；原 recipe 只向固定原模型当前 stage0 添加一个很小残差。两者不相同，因而一个全新的 DINOv3 ViT-S adapter probe 可以是合理问题；但它必须保持 **同一 encoder layer、同一源图像、同一 locator head、无 motion operator**。native/bilinear 臂并不读取 adapter 的 high-resolution guidance，故该实验估计的是完整 RGB-guided adapter 的效果，不能伪称为输入信息路径完全一致的纯上采样对照，也不能被称为对已失败 recipe 的超参续跑。

## 对“只有中心标签”的适配性

公开球数据给的是中心、visibility/status，BlurBall 还给 blur geometry；它们没有每像素 VFM feature target、class-agnostic mask 或相机标定。对应关系如下。

| 路线 | 原训练监督 | 直接拿中心标签重新训练是否自然 | 对本项目最小使用法 |
|---|---|---|---|
| FeatUp | jittered-view feature consistency | 不需要中心标签，但每图 implicit 版本不适合逐帧；JBU 重训需其 multi-view pipeline | 只作历史近邻，不投入新工程 |
| LoftUp | mask-refined pseudo feature + EMA self-distillation | 不自然：中心点不能替代 class-agnostic masks；若同时训练 locator，adapter/head 的收益混在一起 | 不作为第一试验 |
| AnyUp | ImageNet crop encoder feature + consistency | 可直接使用公开 pretrained weight；只用中心 heatmap 端到端微调会失去“通用 adapter”归因 | 冻结 adapter，训练/复用同一个中心定位 head |
| RaysUp | ImageNet high/low resolution feature target | 同样可冻结 weight；从中心点直接训会同时学习 RGB routing 和 locator | 如后续需要第二个冻结 adapter 的效率反例，可作为候选；不是既定下一步 |

这里“冻结 adapter”检验的是预训练 adapter 的迁移，不是结构的能力上限。若先让 adapter 与 head 一起在球中心标签上训练，收益还混有新参数、优化与 RGB detector 的作用，不能直接归因于 generic upsampling prior。冻结版本失败只否定该冻结迁移配方；它不足以否定端到端 RGB-guided adapter，也没有单独构成追加端到端训练的理由。

## 最小、可区分的后续实验（建议，尚未执行）

这不是当前训练完成前的并行任务，也不是批准的新模型。当前 BlurBall 用的是 ConvNeXt-Tiny；以下方案仅在后续已有可用、独立建立的 ViT-S 无 adapter 空间基线时，才是一个单因素后续问题：**预训练 RGB-guided feature routing 是否给该 ViT-S 基线带来空间增益？** 若尚无此基线，它涉及先建立新骨干基线再测试 adapter 的两阶段工作，不能称为当前 ConvNeXt 瓶颈的最小诊断或一次 GPU 空档即可完成的实验。

1. 只选 DINOv3 ViT-S，因为 AnyUp 和 RaysUp 的“跨 DINOv3”实测只覆盖这一族。固定一个可合法的开发划分、原始连续 clip 语义和图像尺度；不读最终测试标签。
2. 对每个目标帧一次 DINO encode。比较同一低分辨率 layer 的 `native/bilinear` 与**冻结的**预训练 AnyUp。两臂从同一原 RGB 得到 VFM feature；AnyUp 额外消费全分辨率 RGB guidance，native/bilinear 不消费该路径，因此控制的是源图像、DINO layer 与 locator 条件，差异是完整 RGB-guided adapter 的信息路径。定位 head、训练步数、随机种子、presence/visibility 规则和输出坐标完全相同，且不接历史帧、差分、光流或 motion module。
3. 只训练同样容量的中心 heatmap/readout head。按可见/不可见状态报 PCK 与检测误报；再按数据真实可得的 blur length 或可定义 displacement 分桶。没有可靠球直径时不造 `rho` 或“球尺寸”分桶。
4. 保存同一预测样本的 paired rescue/break：严格位置救回是否伴随 V0 误报、场线/反光/衣物竞争增多。输出图更锐、平均 dense score 更好都不是通过条件。
5. 若 AnyUp 仅赢得整体平均，却没有严格 PCK 的 paired net-positive，或 F1/误报恶化，则停止这条 adapter 试验；不要扫描 window、输出尺度、head 宽度或重训目标来追正数。若未来需要一个第二 adapter 来检验效率或归因，预训练 RaysUp 是候选之一；是否比较、何时比较由第一轮结果与当时的实际接入成本决定。

一个可选的来源诊断是将 adapter 的 high-resolution guidance 降采样再恢复，而粗 VFM feature 保持不变；若收益消失，说明高频 RGB routing 是必要条件。该干预会把 adapter 放在其训练分布之外，所以只能作机制提示，不能代替主比较，更不能称为“精确贡献分解”。

## 何时不值得继续做

下列任一条件成立时，不应把时间从 motion 表征主线转去做 upsampler：

- DINOv3 ViT-S 的无 adapter 单帧空间基线尚未达到可用、可复现的定位水平；此时 adapter 的失败/成功都会被 backbone/head 不稳定混淆。
- 可见球严格位置没有 paired 改善，或改善由 V0/背景误报抵消；这表示它没有解决本项目关心的“可定位而非仅更锐”问题。
- 增益仅在提高输入分辨率后出现，且同一总计算下没有相对 native/bilinear 的收益；此时更合理的解释是原始采样增加，而不是 feature upsampling 的机制。
- 只有为了适配 RaysUp 而新引入真实 pose/intrinsics 估计、全局几何网络或专门视频缓存才可运行。其论文默认 identity-pose 几何并不自动要求这一步；在未做实际接入测量前，也不能用标定缺失直接排除该路线。若确实需要新增这些系统，才会把空间基线扩展成相机系统，违背当前边界。
- 主要剩余错误来自真实历史帧缺失、候选漏球或大位移对应，而不是当前单帧中心 readout；上采样无法替代 temporal evidence。

## 可以写入论文、不能写入论文的表述

**可以（有相应实验后）**：

> 在固定 DINOv3 ViT-S feature 与当前 RGB 下，预训练的 RGB-guided feature adapter 改善/未改善了单帧 tiny-ball 中心的严格读出；该结论按可见性和误报配对报告。

**不可以**：

> 我们首次以坐标/RGB 引导恢复 VFM 的高分辨率空间细节。

FeatUp、LoftUp、AnyUp、RaysUp 已覆盖该路线的主要形式；LoftUp 还直接覆盖 coordinate-based RGB cross-attention，AnyUp/RaysUp 已覆盖 VFM-agnostic any-resolution adapter。

**不可以**：

> 上采样解决了高速小球运动。

这些论文没有建立球的跨帧对应或大位移 search；即使 DAVIS 有 temporal label propagation，初始化和任务条件也不同。若此项目以后出现有意义的贡献，较可能来自经实验验证的残留失败及其机制性解释；细位置证据与远距离、多假设、可拒绝的 temporal correspondence 协同只是其中一条可能路线，显式 correspondence 也不是预设必需条件。无论采用何种路线，单独给 feature map 加一个 upsampler 都不足以构成核心贡献。

## 来源

1. Stephanie Fu et al. [“FeatUp: A Model-Agnostic Framework for Features at Any Resolution.”](https://proceedings.iclr.cc/paper_files/paper/2024/file/c5601d99ed028448f29d1dae2e4a926d-Paper-Conference.pdf) ICLR 2024；[HTML 正文](https://arxiv.org/html/2403.10516)。
2. Haiwen Huang et al. [“LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models.”](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_LoftUp_Learning_a_Coordinate-Based_Feature_Upsampler_for_Vision_Foundation_Models_ICCV_2025_paper.html) ICCV 2025；[arXiv HTML 与补充](https://arxiv.org/html/2504.14032)。
3. Thomas Wimmer et al. [“AnyUp: Universal Feature Upsampling.”](https://arxiv.org/html/2510.12764v2) arXiv:2510.12764v2, 2026-02-16；[ICLR 2026 正式版](https://openreview.net/pdf/e03a66822acec855114dde73a6fe2226ad23ad6b.pdf)；[作者代码](https://github.com/wimmerth/anyup)。
4. Yuchuan Ding et al. [“RaysUp: Ultra-light Universal Feature Upsampling via Geometry-Aware Ray Representation.”](https://arxiv.org/html/2606.22749v1) arXiv:2606.22749v1, 2026-06-22；[ECCV 2026 官方录用页](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)；[作者代码](https://github.com/MAP-RaysUp/RaysUp)。

本轮检索足以判断这四个直接近邻怎样限制“空间 adapter”主张，也足以决定当前最小对照应该如何归因；它不是对所有 feature upsampling、新近 arXiv 或 tiny-object 方法的穷尽证明。没有检到在高速、数像素、模糊球的自动逐帧定位与跨帧 pixel correspondence 上验证 DINOv3 feature upsampling 的一手论文；这是一项待测问题，不是新颖性声明。
