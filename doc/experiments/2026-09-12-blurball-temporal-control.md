# BlurBall中点定位是否从真实历史取得增量

日期：2026-09-12。状态：正式30epoch已从36bdd7c启动，训练运行中。
协议：[完整时序输入控制v1](../protocols/blurball-full-temporal-control-v1.md)。

## 为什么现在需要这项训练

固定局部重心已经改善当前三帧模型的位置，不能继续仅凭沿轴偏差设计blur模块。但该模型的logits本身包含三帧输入，读出阳性没有回答历史的作用。Tennis上的单帧/真实历史对照也不能直接代替BlurBall的自然拖影中点任务。

因此只增加一次从头完整训练的repeat_current控制，保持官方预训练初始化、三槽参数化、标签、共同目标、损失和优化预算。双方继续依argmax选best，事后共同使用固定15×15、T=1重心；既有history best6和局部logit缓存直接复用。完整协议已明确强阳性、混合/阴性以及相应架构决定，不根据中途指标更改分组或训练。

## 实施范围与验证

[原训练入口](../../scripts/train_blurball_midpoint.py)仅增加实际输入选择。在v2连续窗口与6个内部边界目标排除完成后，repeat将模型输入的三个索引替换为当前帧；所有目标身份保持。配置明确`temporal_input=repeat_current`及`t,t,t`，原history默认路径保留。没有新模型类或新依赖。

[局部读出入口](../../scripts/analyze_blurball_readout.py)同步按run真实输入模式提取验证特征，并从该run的results核对其最佳epoch；旧history缺少输入模式字段时遵循其原来使用真实三帧的事实。历史局部读出产物不会重算，新repeat结束后再生成自己的局部缓存与原q复现证据。

真实GPU单batch检查已通过：7个V1与1个V0、原frame2–8及57，模型三槽RGB与各自当前帧完全相同。全部合法目标38,854/14,192及6个内部边界排除保持。输出`[8,147457]`，loss11.15012，prefix/head梯度范数71.8869/9.1974，均有限且非零；AdamW一步及原验证唯一帧路径通过。参数1,272,577不变；检查耗时1.07秒、峰值allocated5,277.22MiB，不能当正式性能。

该检查发生在8e05230之后的本次工作区修改，脚本及结果保留在`outputs/blurball/repeat_current_smoke/`。它验证索引/标签与实际训练通路，没有重做数据下载、完整源内容审查或已拒绝的单encode训练优化。

## 正式运行

实际执行命令（项目根目录，Conda zshihyc）：

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/train_blurball_midpoint.py \
  --rgb-cache data/cache/blurball/rgb_512x288_all_h2 \
  --output outputs/blurball/dino_repeat_current_seed0 \
  --temporal-input repeat_current --epochs 30 --batch-size 8 --seed 0 \
  > outputs/blurball/dino_repeat_current_seed0.log 2>&1
```

输出为独立run目录。实际config记录代码版本36bdd7cd7213f8a9050124e8b9f0ba7d57878fc4、repeat_current、t/t/t、seed0、30epoch、38,854/14,192目标及1,272,577参数，均符合已锁定设置；没有继承smoke的一步参数。预计计算规模与上一约10小时全量训练相近，实际耗时以后续记录为准。

## 当前结果

正式进程已完成epoch0验证并进入训练，初始化PCK@4为0、本地F1@4为0，完整验证耗时49.08秒。输入已改变，不能要求其随机初始化指标与history完全一样；共同模型与初始化规则保持。正式训练能力、选优和配对尚未完成。预先统计的历史可见性与位移分组定义见协议，只为分析，不作为模型输入。

## 训练期间的有限探索：远错位是否停留在历史中心

这里仅使用**已完成history模型**的保存预测与原v2真实历史标签，尚未涉及正在训练的repeat结果，也不改变主要判定。问题是：当前错误位置与两帧可见历史的GT中点有多接近？若多数错误只是落在旧中心，才有理由把直接历史位置滞留作为主要失效动机；否则不应默认用运输旧坐标解释或修补多数错误。

对每目标，只使用原V1历史中点，计算`min_delta ||pred_t−GT_(t−delta)||`，没有可见历史时该量未定义；按已有严格4/16px容差统计。当前V0的占位坐标不参与距离，当前V1误差只对其合法当前中点计算。空间邻近仅与复制解释相容，不证明模型因果上复制历史；反之远离旧坐标也不排除模型使用了外推轨迹先验。

| history最佳模型的当前群体 | 原argmax：数量 / 有可见历史 / 距历史<4 / <16 | 固定重心：数量 / 有可见历史 / 距历史<4 / <16 |
|---|---:|---:|
| V1、已输出、当前误差≥16px | 712 / 692 / 0 / 5 | 709 / 689 / 0 / 4 |
| V1、拒绝输出的原空间预测 | 2,139 / 2,069 / 71 / 379 | 2,139 / 2,069 / 74 / 376 |
| V0、仍输出位置 | 432 / 101 / 1 / 3 | 432 / 101 / 1 / 3 |

多数已输出远错位不是“直接落在旧GT中心附近”。重心的709例只有4例在任一可见旧中心16px内；另外331/432个V0误出没有可见历史标签。不能据此断言历史毫无作用或所有错误都是纯背景，因为可能有历史外推、当前视觉混淆等不同机制，本项没有区分它们；当前应停止将简单旧中心滞留当作默认主因，继续等待完整输入控制。

产物`outputs/blurball/dino_midpoint_seed0/history_proximity.json`保留两种解码的全体及四match计数。CPU从原CSV按match/rally/原Frame对齐源窗口，复用原float坐标，没有视频解码、GPU计算、新标签或阈值扫描。本项为运行期间的描述性探索，不能冒充事前主要终点。

## 已准备的完成后比较

[CPU配对入口](../../scripts/compare_blurball_temporal.py)现已完成；它只读取两run的保存预测和history run保存的原窗口，分别比较argmax与局部重心，并保留各模型自己的q。按原定历史四状态、双端V1位移三档、l、match与长l×match报告完整救回/破坏。当前V0只进入FP2/TN等检测计数，位置配对分母为0。

一项合成测试通过：真实近/远历史顺序、恰好4/16px的位移边界、当前V0不参与几何、两模型可有不同q但同模型两解码q不变、错位目标身份拒绝。当前repeat还没有最终results.json，实际提前调用已明确拒绝且没有创建比较结果，未把未完成运行写成性能证据。

正式训练及repeat局部解码完成后才执行：

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python scripts/compare_blurball_temporal.py   --history outputs/blurball/dino_midpoint_seed0   --repeat outputs/blurball/dino_repeat_current_seed0   --output outputs/blurball/repeat_vs_history_seed0.json
```

首次完整训练轮已完成：epoch1平均训练loss3.55638，验证PCK@4为71.9448%、本地F1@4为69.5839%，该轮含验证耗时1,096.61秒。训练继续按既定30epoch和选优执行；这只确认基本拟合已经发生，不与历史组最终best6提前作结论性比较，也不改变预设判定。
