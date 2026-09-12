# BlurBall基线完成与读出解释

日期：2026-09-12。阶段：自然模糊条件下的实际失败归因。

[因果中点基线](../experiments/2026-09-12-blurball-midpoint-results.md)完成30epoch并选中epoch6，保存预测与选优复算通过。原验证PCK@4为76.72%。长拖影显示多比赛的细中点偏差，但无拖影组更多拒绝，match20不支持整体错误随拖影增长的单调解释。

追加GT轴分解发现，242个长拖影近位置错误中93.39%以沿轴偏移为主。有限源帧与独立视觉复核还看到背景边缘、弱证据和遮挡，未确认统一的亮端点机制。

随后按[固定协议](../protocols/blurball-local-readout-v1.md)完成一次62.06秒的验证forward，原argmax和q完全复现；只改坐标读出的[局部重心](../experiments/2026-09-12-blurball-local-readout.md)把PCK@4提高到78.18%、本地F1@4提高到80.22%，PCK@16不降，长拖影四场均有净增。13.17MB局部logit缓存保留，无新训练、依赖安装或最终测试读取。

研究决定：接受已有logits的读出解释，保留固定局部重心作为强基线候选，暂缓新的blur几何模块。补充的[解码文献](../literature/2026-09-12-midpoint-decoding.md)表明这种算子已有直接先例。下一问是剩余失败需要哪种时间证据，尚未确立新的motion机制或完成论文归因；实证修订见[研究总判断第18节](../research/2026-09-10-empirical-reframing.md#18-2026-09-12自然拖影与读出细定位困难成立不能直接归给缺失的motion模块)。

同日推进：[完整时序输入控制](../experiments/2026-09-12-blurball-temporal-control.md)已从36bdd7c启动30epoch repeat_current训练，以同argmax选优及共同局部读出估计真实历史增量；GPU真实单batch与正式配置已确认。等待其结果期间，原history保存预测的几何邻近诊断不支持多数远错位停留在旧GT中心，暂不将简单历史位置复制作为主失效机制。

期间补齐[BIRD/STSN任务监督采样](../literature/2026-09-12-task-supervised-alignment.md)的完整方法边界：混合特征重采样与逐support对齐不同，原协议都含未来信息；BIRD还使用STF特征辅助，检得公开仓库尚无可复现网络。不能把这些工作概括为固定小范围、仅检测loss或现成因果三帧基线。当前继续等待输入控制，不预选DCN；若后续采用，须将采样作用与额外容量区分。
