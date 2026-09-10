# 实验协议

这里保存正式实验采用的数据与评价约定，回答“这组结果到底是在什么条件下得到的”。现有研究总纲中的实验建议尚未转化为已锁定协议；本文件也不是一个默认协议。

先为第一组实际要运行的比较写一份协议，不提前为所有数据集创建空文件。协议应能被多份实验记录引用；不要把每个学习率或训练 seed 都做成协议版本。

## 文件格式

文件名为 `<范围>-v1.md`，例如 `tennis-game-level-v1.md`。初稿标为“草案”，实际用于正式实验时记录锁定日期；只在划分、标签、输入信息或评价定义等影响可比性的条件改变时创建下一版，保留旧版。

```markdown
# 协议名称

状态：草案 / 已锁定 / 已由某协议替代
版本：v1
锁定日期：尚未锁定，或实际日期
来源：相关研究论证与数据说明的相对链接

## 任务与数据
预测什么；使用哪个发布版本、原始/修正标签与数据范围。
train/val/test 的具体划分文件或明确分组；clip/rally 边界来自哪里。

## 时间、坐标与监督
目标帧、输入帧、时间间隔、是否允许未来帧、窗口边缘策略。
原始帧号/PTS、尺寸与坐标映射；未标注、无球、遮挡及异常坐标的处理。

## 评价与比较条件
指标、误差单位、匹配容差、visibility 规则、后处理和阈值选择数据。
哪些条件保持一致；哪些作为实验自变量；效率统计的硬件和计时边界。

## 尚未确定的条件
只列真正阻止协议锁定的问题；已锁定时删去无内容的栏目。
```

协议中的具体值必须来自数据或研究决定，不照抄模板猜测。边界或标签语义尚不明确时，记录缺口并先解决相关问题，不能静默补成可运行默认值。

数据发布异常引用 [data/README.md](../../data/README.md)，不复制维护。训练配置和模型结构在具体实验记录中给出；只有影响整个比较口径的约定才提升为协议内容。

## 已有协议

- [Tennis 空间探针 v1](tennis-spatial-probe-v1.md)：冻结单帧开发诊断，独立保留最终测试比赛。
- [Tennis 因果三帧 v1](tennis-temporal-probe-v1.md)：真实连续上下文与统一目标集上的时间增量诊断。
- [Tennis GT query 对应 v1](tennis-correspondence-probe-v1.md)：区分特征匹配、搜索覆盖与同格背景自相似，仅作oracle诊断。
- [Tennis dense局部对应定位 v1](tennis-local-cost-probe-v1.md)：保留三帧appearance，以self-cost控制检验显式对应增量。
- [Shuttlecock开发位移 v1](shuttlecock-development-motion-v1.md)：固定比赛级开发范围，仅用训练标签诊断位移与搜索覆盖。
- [Tennis前缀适配 v1](tennis-prefix-adaptation-v1.md)：相同三帧读出初始化下，对比冻结前缀继续训练与共同微调，包含epoch0控制。
- [Tennis适配后GT-query对应 v1](tennis-adapted-correspondence-v1.md)：固定已选checkpoint，以相同CPU float32输入补测双端VC1下的匹配变化。
- [BlurBall开发轴向诊断 v1](blurball-development-axis-v1.md)：以真实过去到当前位移比较当前无向拖影轴，分别保留不可见与未定义方向。
- [Tennis全量因果定位 v1](tennis-full-causal-v1.md)：全量合法末帧监督与检测选优，建立共同任务的HRNet竞争系统。
