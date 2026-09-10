# 从适配诊断转入全量竞争系统

日期：2026-09-10。状态：前缀适配诊断完成，HRNet全量训练中，DINO顺序等待。

## 本阶段改变了什么判断

[六组前缀适配](../experiments/2026-09-10-prefix-adaptation.md)全部结束。微调相对只继续训练head，三seed的严格PCK8均改善，但PCK16和检测F1没有一致提高，困难/遮挡也未稳定恢复。因此保留“表示可适配”的结论，结束固定条件诊断，不围绕该结果扩展超参。

[实证修订](../research/2026-09-10-empirical-reframing.md)将当前优先级从扩大搜索范围转为核对强基线、当前视觉证据、自动查询、背景竞争和位置/存在读出。它明确区分VC2困难标签与VC3遮挡位置目标；有坐标不等于有当前视觉对应。没有据此锁定新架构或宣称新颖性。

## 已完成的实现与验证

全量RGB缓存复用旧5,199帧，只新增解码8,961帧；共14,160唯一帧、12,167训练目标和1,863验证目标。真实时间顺序、clip边界和原visibility保留，最终games8–10未进入本轮。缓存、标签监督与真实HRNet batch8训练预检及完整入口smoke已通过，见[HRNet记录](../experiments/2026-09-10-full-hrnet.md)。

HRNet正式运行来自32d37d4，固定30epoch、batch8、float32。新增[DINO全量控制](../experiments/2026-09-10-full-dino.md)复用同一入口，官方前缀与随机新头共同训练，固定head/prefix两个学习率和交叉熵；按与HRNet相同的detection F1选优。新增CPU输入/解码测试与独立审阅通过，代码提交3ea19eb。新DINO入口的GPU smoke仍待执行，不能把CPU通过写成已完成全量训练。

环境继续统一为Conda zshihyc。OmegaConf 2.3.1本已安装，已记录为实际依赖；本阶段没有新装包或升级环境。

## 新的标签条件与文献证据

从同一全量metadata统计源visibility上下文，没有模型forward。验证中“当前VC2、前两帧至少一帧VC1”有64个目标，半数来自Clip4；VC3训练中35/37有这样的历史，验证只有8/35。当前VC0但历史有VC1的25个验证目标，也需要用于观察历史持续输出造成的误报。完整结果存于outputs/full_heatmap/visibility_context.json及CSV，解释见HRNet记录的输入部分。

[拖影轴近邻审查](../literature/2026-09-10-blur-guided-correspondence.md)核对五项直接先例，包含较新的MoTDiff预印本。blur-aware匹配、单帧曝光轨迹及fast-moving-object拖影恢复均有前史；本地BlurBall轴相关性只支持保留候选，不能替代图像预测轴和完整自动定位的证据。

[DQAligner源码补读](../literature/2026-09-10-dqaligner-query.md)修正了旧候选解释：该实现从学习初态用全局query读取窗口特征，并不先依赖当前硬候选；状态限窗口内，默认last模式也不能掩盖loader的开头回绕及训练时间反转。作者公开全文仍未找到，源码结论与论文摘要分开记录；未新增该模型训练。

## 正在执行与后续依据

一次性脚本outputs/full_heatmap/run_dino_after_hrnet.sh等待当前HRNet完整结束，再依次检查DINO批内帧复用的CUDA预测等价、验证DINO smoke、运行固定30epoch，最后复用compare_predictions.py比较相同验证目标。帧复用实现来自1185875，CPU顺序/复用与梯度检查已通过，真实CUDA检查仍待执行。日志位于outputs/full_heatmap/dino_queue.log；任一阶段失败就停止后续命令，由实际错误决定修复。脚本和训练产物留在outputs，不增加调度框架。

两系统共享任务，但结构、输出尺度、预训练、损失和优化器不同。全量结果是强竞争系统参照，不能单独归因某个因素，也不替代最终backbone × motion的2×2。完成后先依据保存预测分析精细位置、困难条件、无球误报和有球漏报，再决定是否存在需要新motion机制解决的残留问题。

[前缀适配后的对应测量](../experiments/2026-09-10-adapted-correspondence.md)现已完成。双端VC1、固定GT query下，Δ2/R4局部PCK16由70.53%升至77.89%/75.26%/74.21%，但净收益主要来自Clip4；Δ1全局精确格R@1的seed2反而下降。它补充了条件性对应变化的实际证据，没有改变原检测收益不一致的结论，也不能单独归因为motion学习。已保存同格/跨格和clip配对变化，本项诊断结束，不增加训练或重开cost搜索。
