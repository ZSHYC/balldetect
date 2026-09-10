# Shuttlecock开发范围与标签位移诊断

状态：2026-09-10锁定本轮数据诊断范围；本文件尚不是完整模型训练协议。
数据版本与发布异常以[data说明](../../data/shuttlecock/README.md)为准。

## 划分与本轮读取边界

训练开发集合为原目录`Professional/match1…20`与`Amateur/match1…3`；验证保留`Professional/match21…23`；原`Test/match1…3`保留为最终测试。使用完整match作划分，不按相邻帧或rally随机打散，也不重命名原目录。这是本项目的开发划分，不称为作者官方train/val划分。

本轮只读取训练开发集合的原始CSV位置标签，生成训练侧位移统计；验证集合仅从已有`verification.tsv`记录rally路径/元数据，不读其位置标签。Test标签及TrackNetV3修正版测试标签均不读取。划分先于本轮位移统计确定，不依据位移或当前模型结果选验证比赛。

## 时间和标签

一个原始MP4为一个发布rally，保留来源类别、match、rally和原始`Frame`。帧对只能在同一rally内按原始索引差Δ=1/2/4组成；不能删除不可见行后把前后可见行重编号为相邻帧。

`Visibility=1`且坐标有限并在原图内才作为located端点。`Visibility=0`保留为not_visible，不等同于物理上没有球；原始X/Y仍保留。非法可见坐标单独记录，不能裁剪或静默修正。CSV之外的真实视频尾帧为未标注，记录数量，不补零坐标/不可见标签。

复用已有视频尺寸和`avg_frame_rate`，不重复ffprobe或视频解码。平均帧率不等同于逐帧PTS；本轮报告原图像素位移及明确的帧索引间隔，并按平均帧率分组，不伪造精确秒级运动或曝光长度。

## 要回答的问题

统计训练集合双端合法位置帧对的L2位移分位数与L∞方形搜索覆盖，沿用Tennis诊断的20/40/80/160原图像素半径，并报告36×64 native格上R2/R4/R8覆盖。保留Professional/Amateur及帧率条件，避免把不同采样时间条件混成同一物理速度。

这一阶段判断现有羽毛球数据能提供何种位移压力，不训练新网络，不推断球尺寸/rho，不把几何覆盖当真实匹配成功率。产物放在`outputs/shuttlecock/development_motion/`：开发rally manifest、训练侧统计与实际发现的非法标签记录。原数据保持不变。
