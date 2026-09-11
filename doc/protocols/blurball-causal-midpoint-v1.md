# BlurBall原生因果三帧中点定位基线

状态：已锁定；版本：v1；锁定日期：2026-09-11。

依据：[训练侧拖影轴诊断](../experiments/2026-09-10-blurball-axis.md)、[标注语义核对](../literature/second_pass_measurement.md)、[数据来源](../../data/blurball/README.md)。本协议检验自然拖影条件下的自动定位失败，不是新blur或motion结构；不延长历史间隔制造压力。

## 问题与数据

一个不读取theta/l输入或辅助监督的自动定位基线，其原图位置误差、漏出与错位输出是否随自然拖影半长度l变化？已有轴标签一致性不代表像素中可读，也不代表模糊是当前首要失败源。

保留已固定的本项目开发划分：match00–17训练，18–21验证，22–25最终保留不读取。训练13、14各出现一次，不称为作者划分复现。中心版CSV为唯一位置标签；不混入端点版，不继承Tennis上按其他位置约定训练的checkpoint。作者权重训练范围覆盖18–21，不用于本项验证。

manifest沿用`outputs/blurball/development_axis/development_rallies.csv`的383个真实rally。原视频53818帧，训练39486、验证14332；每rally前两帧只作为上下文，预期训练38860、验证14192个末帧目标，最终以标签与解码后的真实索引记录为准。验证70个rally全部1280×720；训练含1266×720、1280×720、1920×1080，不能声称已经有跨验证分辨率证据。

## 时间、图像与标签

输入同rally真实`[t−2,t−1,t]`，只监督t，原生相邻帧、逐帧输出、无前瞻。保留发布视频中的重复帧，不依据像素相似性去重或重新编号。各rally的源帧序号与CSV Frame保持一一对应；不跨rally构造窗口。

每视频只顺序解码一次，用FFmpeg bilinear独立缩放宽高到512×288、RGB uint8。缓存为唯一帧mmap及窗口索引，不展开成数万PNG或重复窗口。`-copyts`和`-vsync 0`保留源PTS和帧数量；同次解码记录整数PTS及有理数time base，showinfo显式关闭checksum。不从平均FPS生成时间戳；本项不将帧间位置差换算为物理速度。

本地宽高独立resize有明确的像素中心坐标映射：`x'=(x+0.5)W'/W−0.5`，y同理。它不同于作者在1266×720输入上的uniform-max-side中心仿射；因此不将本项称为作者预处理复现。原始x/y、theta/l、尺寸、match/rally、帧号与visibility在metadata保留，不改源CSV。坐标落格与逆映射按每帧实际宽高计算。

Visibility=1表示有合法可见streak中点，监督位置；Visibility=0使用独立“无合法可见中心输出”类别，不回归其原始CSV坐标，也不从此推断物理上没有球。历史V0图像仍是合法输入。异常visibility、非法可见坐标、帧号缺口或图像标签数量不一致时报告并停止该构建，不静默删标签、补帧或跨边界借帧。theta/l原样保留，分桶只使用V1的合法非负半长度；未定义轴不当作0°。

## 固定模型与训练

主基线使用官方DINOv3 ConvNeXt-Tiny前两阶段与随机新三帧SpatialProbe，从官方预训练重新初始化，不继承任何Tennis适配或辅助checkpoint。192通道逐帧特征、三帧576通道拼接、每帧独立GroupNorm与hidden32保持现有实现；不加入cost、细节分支、theta/l辅助或GT查询。

输出改为288×512位置格和一个无合法可见中心类别，SpatialProbe的pixel shuffle倍率为8。这个改动只避免评价被粗位置量化预先限制：Tennis原72×128格在1280×720上间距10px，不能直接用来判断原4px容差下的模糊定位；288×512格间距2.5px。它不增加新的空间证据，也不构成motion贡献。

固定seed0、batch8、float32、无AMP、无数据增强，30epoch。AdamW，head学习率3e−4、前缀1e−5、weight decay0.01，主位置/可见输出交叉熵。前缀参与梯度更新，但沿用已有前缀eval行为；不缓存正在更新的特征。验证可做批内唯一帧提取，训练保持原显式三帧forward，不重开此前未通过的重复特征梯度优化。

按下述本地detection F1@4最大、其次F1@8、再取较早epoch选优，包含epoch0。保存配置、种子、设备、代码与权重版本、实际输入条件、预测、checkpoint和每epoch训练/验证日志。耗时包括缓存读取、H2D、在线前缀与head的训练/评价；独立报告一次视频解码缓存成本，不称为端到端部署速度。

## 评价：严格原图容差与完整分母

模型给出位置argmax与有效可见中心概率q；q≥0.5输出位置，否则拒绝。阈值固定，不在验证集扫描。主容差使用原图Euclidean距离严格`<4px`，补充`<8px/<16px`。原图尺寸不同不会静默映射到统一坐标后当原像素误差。

主定位PCK以全部V1为分母，比较其空间argmax，即使q导致拒绝也保留该原始位置诊断；另报有输出V1的条件误差和分母，以及真正输出且定位正确的比例。不得仅对成功输出报告误差而隐藏漏出。

对每个容差记录TP（V1且输出且位置正确）、FP1（V1输出但位置错误）、FP2（V0仍输出）、FN_visible（V1拒绝）、TN（V0拒绝）。本地检测F1将错位输出同时作为一次FP和一次未定位目标：`FP=FP1+FP2, FN=FN_visible+FP1`。这个定义沿用本项目定位检测语义，选优使用它。

另报告作者发布F1：其`FP=FP1+FP2, FN=FN_visible`，FP1不同时计FN；作者RMSE仅在V1且有输出的样本计算。来源为[固定BlurEvaluator](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/utils/blur_evaluator.py#L35-L129)和[4px配置](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/configs/runner/eval_blurball.yaml)。两种F1分别命名，不能混成一张比较表。作者未定义本项目PCK。

本项因果三输入一输出、开发划分与后处理不同于作者三帧MIMO三输出模型；即使采用其4px容差，也不与论文数字直接排名。[官方模型配置](https://github.com/cogsys-tuebingen/BlurBall/blob/2f0f5496f7ba4b5b1a36790749935121b2ce972d/src/configs/model/blurball.yaml)与[论文](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Gossard_BlurBall_Joint_Ball_and_Motion_Blur_Estimation_for_Table_Tennis_CVPRW_2026_paper.pdf)保留为来源。

## 分组判断与停止条件

固定V1半长度组为l=0、(0,2]、(2,5]、(5,10]、>10原像素，复用此前轴诊断边界，不看模型误差后重新分桶。报告全体、match、分辨率以及match×半长度、分辨率×半长度的数量与错误/拒绝；训练和验证分开，不将相关相邻帧当独立来源。

只有较长l的失败在多个验证match成立，且不是拒绝筛选或单一来源造成，才支持下一项blur视觉证据实验；仍不能把关联当作模糊因果。若错误不随l增加，或仅单一match贡献，则结束“自然拖影是当前首要定位失败源”的默认动机，不为救它扫描新blur loss。首个单seed基线只决定下一项实验，不证明泛化或新颖性。

## 最小验证

数据检查针对真实风险：rally边界/原帧号、V0保留、V1坐标、原theta/l保留、Test排除；一个小训练rally实解码确认RGB与PTS数量/序号和标签对齐，错误则修读取，不全量重验下载。指标检查用手造TP、错位FP1、V0误出FP2、可见拒绝和恰好4px，区分两种F1与条件误差分母，并检查不同尺寸的像素中心变换。一个真实训练batch检查dense输出、有限loss/梯度与目标帧，正式训练完成后复算保存预测和选优记录；不为了纯文档再训练。
