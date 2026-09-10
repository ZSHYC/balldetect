# 从稀疏探针转向全量因果强基线的实施计划

日期：2026-09-10。性质：本阶段设计与执行计划。采用已授权的直接执行，不等待常规确认。

目标：得到可追溯的WASB-HRNet共同任务基线，回答当前失败能否由更完整的训练与高分辨率热图系统缓解。架构保持作者HRNet，只改为末帧一张热图。技术栈为现有PyTorch、NumPy、Pillow、OmegaConf；不引入训练框架。执行依据：[Tennis全量因果协议](../protocols/tennis-full-causal-v1.md)。

## 为什么现在做这个

已有冻结current/stack/repeat对照支持历史视觉有价值；多seed局部cost与宽度控制暴露精细定位代价；prefix适配seed0有收益、seed1的16px持平且漏报增加。因此目前不继续扩展cost，也暂不增加prefix-current配对。更强的竞争解释是：训练目标稀疏、空间读出以及位置/存在监督共同限制了探针。

受控HRNet会同时改变结构、输出分辨率与损失，所得结果是系统参照，不能单独指认其中一个因果因素。与旧step8数字直接相减还会混入监督密度；正式架构比较必须等待同全量目标的DINO控制。

12个VC2/3输入图已从缓存生成并查看，保留于outputs/adaptation_probe/hard_inputs_page1–3.png：场线重叠、球员/球拍邻域和弱成像均存在。部分VC3窗口中前两帧也不可见或只有估计位置；不能把所有失败理解为半径扩大即可找回。它们是定性证据，不改标签，不声称几帧缩放图证明物理不可观测。

## 最小文件与执行顺序

1. `scripts/cache_tennis_full_rgb.py`、`tests/test_tennis_full_rgb.py`：复用现有read_frames、select_cache_rows、frame_key、Images。输出data/cache/tennis/rgb_512x288_all_h2，按唯一帧存储；复用旧step8 RGB，只有缺失帧才解码。一次临时数据测试覆盖真实连续窗口、边界、VC0可作历史、旧缓存复用与新帧像素/索引，随后生成实际缓存。metadata记录输入条件、frames/windows、计数和实际耗时，不伪装存在DINO特征。
2. `third_party/wasb/hrnet.py`与`LICENSE.md`：保存已读取的固定上游原文件；`configs/wasb_hrnet_causal.yaml`只将作者frames_out改为1，标明原来源。用现有OmegaConf直接读取，无自定义配置包装类；requirements记录实际已有版本。
3. `src/ballmotion/heatmap.py`、`tests/test_heatmap.py`：只有实际需要的disk、概率QFL与峰值解码。已知中心disk半径2.5应有21格，角点应有8格，无球为0；零logits的QFL为log(2)/4，正/负位置梯度符号相反；已知峰值经现有grid_to_original回原坐标。先跑失败测试，再实现并通过。
4. `scripts/train_tennis_heatmap.py`：复用RGB窗口、项目evaluate_predictions、保存预测格式。单个HRNet实际模型路径，不建factory、trainer基类、插件或resume框架。先真实前反向记录batch/显存，再一次短入口smoke；不会据smoke调整学习率/宽度。资源失败只降低实际batch。
5. 正式运行30epoch seed0，保存config/history/checkpoint/train与val预测；每个epoch报告detection与PCK。独立审阅时间/标签/坐标/损失，检查失败先修正，再写入doc/experiments。实际代码先提交、正式训练后启动，配置中的revision应包含运行实现。

各步只验证会改变下一步的真实失败；缓存已经通过后不反复解码核对。最后保留有复用价值的RGB、原始数据和权重。更完整的DINO共同协议、跨球种与motion新设计依据结果继续制定，不在这里预建空接口或全部模型。

## 前缀三seed完成后的DINO实施补充

根据[最终适配结果](../experiments/2026-09-10-prefix-adaptation.md)，全量现代对照采用官方预训练前缀与随机新head，不继承旧验证选优。具体参数已写入共同协议。

复用实际已有的全量训练入口，给scripts/train_tennis_heatmap.py增加两个实际模型选项hrnet/dino（默认仍hrnet）。只用明确的模型构造、输入、损失和读出分支，复用共同的数据、训练循环、F1选优和保存，不新增trainer/factory类。HRNet路径保持不变；DINO复用BackboneProbe和SpatialProbe，不新造adapter或motion层。增加一项CPU输入/读出综合测试，覆盖两种模型的时间通道顺序和预处理边界；真实DINO入口smoke与正式运行等待当前HRNet释放GPU后顺序执行。

正在运行的HRNet来自32d37d4，其Python进程已加载该版本；后续源文件修改不重新解释它的运行。DINO运行前提交新实现，再让config记录新版本。正式结果分别保存在outputs/full_heatmap/hrnet_seed0和dino_prefix_seed0，不能把两个模型的超参差异藏进共用入口。

## 全量DINO验证中的重复前缀计算

全量相邻三帧窗口在同一验证batch中反复包含相同RGB帧，现有DINO predict会对每次出现都执行同一个固定前缀。按实际帧索引取unique并用inverse恢复窗口，可复用当前batch内的特征，不需持久化缓存或跨clip状态。训练保留原forward，不复用已更新参数前的特征。

最小修改在BackboneProbe提取已有逐帧归一化/前缀计算为encode方法，由原forward和DINO predict两处调用。predict仅在DINO分支对索引去重，再按原时间/批次顺序拼接给原head；HRNet路径不变。CPU测试用重叠且索引顺序不平凡的窗口，对照原完整forward的输出并验证实际前缀输入帧数下降，原梯度测试仍需通过。

这种数学复用不自动保证不同CUDA batch形状逐位相同。GPU释放后，正式DINO启动前，另用实际已训练前缀/头对全量验证比较直接窗口与批内复用的位置、0.5存在判断和分数误差；若关键预测改变，先定位并决定是否保留该计算改动，不直接进入正式训练。不用理论帧数减少宣称实测吞吐倍数。
