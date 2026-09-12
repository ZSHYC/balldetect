# 局部置信度、整体对应误差与几何后验：2026 年 8 月新稿深读

日期：2026-09-12。问题是“局部位置分布很集中时，为什么仍可能发生大错位，已有方法怎样估计这种风险”。本文补读 *Semi-Dense Matching Uncertainty Is Not Just Local Confidence*，不把它扩写成通用不确定性综述，也不因此启动新的质量头或校准实验。

## 结论先行

这篇工作提供了应当接受的正证据：在静态图像匹配中，结合 coarse 分布与 fine 分布的误差模型，比仅看局部 refinement 更能描述大误差尾部；利用这些信息做几何重拟合，也优于若干简单重拟合对照。它的残差校准、错误排序和 coarse-success 检验分别有实验，不能笼统说“没有可靠性验证”。

但它估计的是**已经输出的 correspondence 的位置误差**，不是球类别、visibility 或无对应状态。CoRe 后处理还需要一个可提供点到点投影的初始几何模型。对独立运动的小球，背景模型即使完全正确，也不能自动成为球的正确对应代理。另有一处公开代码与 v2 论文的后验公式差异，必须在复用前区分。

## 一、来源和阅读范围

Khoa Hoang、Hoang-Tuan Nguyen、Huong Ninh、Hai Tran、Long Q. Tran，[*Semi-Dense Matching Uncertainty Is Not Just Local Confidence*](https://arxiv.org/abs/2608.08685v2)。arXiv v1 为 2026-08-09，v2 为 2026-08-29；本次官方记录未列正式会议版本，按预印本引用。

已读 [v2 正文与补充 A–C](https://arxiv.org/html/2608.08685v2)，核对主文 Table 3–5、补充 B.2/B.4，以及作者的校准、权重和 homography/PnP 调用路径。作者仓库固定为 [`khoavpt/Probabilistic-matching@111b831`](https://github.com/khoavpt/Probabilistic-matching/tree/111b8311842892ede53077aaf16904c8483ded47)，提交时间 2026-08-11，**早于 v2**。

只读缓存为 `outputs/literature/matching-uncertainty-2608.08685v2.{pdf,txt,html}` 与 `matching-uncertainty-source/`。本地只运行了第六节的 CPU 公式例子，未运行 matcher、拟合参数、读入体育评测数据或检验论文性能。

## 二、它建模的究竟是什么随机量

给定 source point `x_i`、matcher 输出 `y_hat_i` 和静态几何 GT 对应 `y_i`，目标残差为 `r_i = y_hat_i - y_i`。论文把误差解释为局部细化误差与粗匹配失败两个来源：coarse 选对时通常是窄误差，选错时可能出现窗口外的大误差。[§3.1–3.2](https://arxiv.org/html/2608.08685v2#S3)

这是一个近似建模解释，不是直接计算所有 coarse hypothesis 的完整条件分布。复用的输入只有：coarse row-wise softmax 的横纵位置标准差、选中匹配的 confidence，以及 matcher 原生 fine refinement 分布的标准差。没有重新搜索远处高分辨率地址，没有增加视频历史。

主文先用二维两分量 Laplace mixture 说明思想；实际 v2 的附录 A 则明确采用**两个坐标轴各自的两分量 mixture，再将两轴密度相乘**：

`p_d(r_d) = (1-alpha_d) Lap(r_d;0,s_f,d) + alpha_d Lap(r_d;0,s_c,d)`，

`p(r_x,r_y) = p_x(r_x) p_y(r_y)`。

两轴相乘后包含 fine/fine、fine/coarse、coarse/fine、coarse/coarse 四种组合，不能与共享一个二元隐变量的二维两分量模型混为同一个公式。两类尺度、gate 共九个参数；coarse spread 按图像宽高归一化，gate 同时依赖 spread 与 selected confidence。[附录 A](https://arxiv.org/html/2608.08685v2#A1)

两种尺度都以同一零残差为中心，所以“粗误差分量”并不是第二个具体目标地址，也没有给出纠正错位的方向。轴向尺度可区分横纵宽度；它们也不自动表示任意方向拖影的相关结构。论文的水平/垂直 Gaussian blur 示例只说明对这类方向性图像退化有响应，不能代替 BlurBall 曝光轴/半长度或帧间位移实验。

## 三、校准与推理分别需要什么信息

**误差校准。** 冻结 base matcher，用 200 对 held-out MegaDepth 图像，通过 depth、内参与相对位姿获得 GT warp residual，以 Powell 优化九个参数。默认实验在 1280×960 校准；§4.4 的分辨率迁移实验另在 960×640 校准，再去其他分辨率测试。两个设置应分别报告，并非相互矛盾。

它使用真实几何监督，不是无标签校准。补充 B.3 还说明每对图像大约提供 2,000 个匹配残差。因此“200 对即可拟合”不能直接翻译为“200 个单球帧足够”，大量空间对应与同一比赛相邻球帧的样本结构不同。[§4.1、§4.4 与补充 B.3](https://arxiv.org/html/2608.08685v2)

**CoRe 推理。** 先以 RANSAC 等获得初始几何 `theta_0`，再用 `r_i^(0)=y_hat_i-Pi_theta0(x_i)` 近似真实匹配误差，计算 coarse-success 后验权重，做一次加权几何重拟合。它输出新的 homography/pose；不是移动 matcher 的每个 target point，也不新增球候选。[§3.3](https://arxiv.org/html/2608.08685v2#S3.SS3)

这个残差代理有精确分解：

`r_i^(0) = (y_hat_i-y_i) + (y_i-Pi_theta0(x_i))`。

第二项既可能来自几何估计不准，也可能来自模型假设本来不适用于该目标。即使背景 homography 很准，球仍可能相对背景独立运动；此时完美的球视觉对应也能有很大的背景投影残差。**“不符合背景几何”不等于“球匹配错误”。** 这是对体育迁移的逻辑推论，本文没有在本地测出相应发生率。

作者自己也给出重要边界：essential matrix 只把 source point 映到 target 的一条极线，不能提供这里需要的完整二维点残差。补充 B.4 的直接 epipolar 替代使多数阈值变差；辅助 depth 的版本还使用 GT source depth 和 GT translation scale，是受控研究条件，不是普通无 GT 的两图推理。该阴性说明残差的语义会实际影响效果。[补充 B.4](https://arxiv.org/html/2608.08685v2#A2.SS4)

## 四、应该接受哪些实证，哪些增益不能混算

主文 Table 3 固定 Efficient LoFTR、HPatches，比较相同匹配输出的几何重拟合：

| 方法 | Homography AUC@1px | AUC@3px |
|---|---:|---:|
| 原 RANSAC | 29.14 | 52.59 |
| 一步 Huber refit | 32.80 | 59.41 |
| raw fine std 权重 | 33.65 | 59.47 |
| CoRe | 32.85 | 60.50 |

这支持 CoRe 在 @3px 的额外收益，也显示普通重拟合已经贡献大部分增量，且最严格 @1px 并非它最好。不能只拿 52.59→60.50 来归因于 coarse uncertainty；也不应因 @1px 不是最好而否定其他阈值的收益。[Table 3](https://arxiv.org/html/2608.08685v2#S4.SS3)

Table 5 的位置误差建模同样要按评价目标解释：fine-only 的 NLL / ECE_err / Spearman 为 2.897 / 7.756 / 0.240，完整模型为 1.546 / 1.612 / 0.360，说明整体误差建模有明显收益。但未校准 mixture 的 Spearman 为 0.368，高于校准后；去 confidence gate 的 ECE_err 为 1.575，低于完整模型。8–64px 条件下 fine-only NLL 4.373 也低于完整模型的 5.532。**更好似然、误差排序与分桶校准是不同目标，不能概括为每项单调改善。**

这里 `ECE_err` 比较预测期望误差与实际 `e=(|r_x|+|r_y|)/2`，不是本项目“是否在 4px/16px 内”的二元概率校准，也不等于每个条件输入的完整分布都已正确。

补充 B.2 进一步直接检查了 coarse-success：在 HPatches 上，后验分组与实际成功率有单调关系，报告 AUROC 0.760、AUPRC 0.951。作者也承认概率没有完全对齐。应接受它支持该任务上的判别信息，同时不把它升级为跨球种、跨候选预算或物理对应存在性的已校准概率。

成本方面，RTX A5000 的后处理增加 11.4–34.6 ms，包含 uncertainty extraction 与 CoRe，未包含 base matcher 全过程。输入分辨率迁移有实验；top-k 预算、球尺寸、曝光模糊和体育相邻帧条件没有相应验证。

## 五、与本地集中度负结果的关系

此前 [Tennis 位置质量审查](2026-09-11-localization-quality.md)与其后诊断考察的是现有全图位置分布的集中度，不能直接等同这里的 coarse-match/fine-refinement 两阶段结构。BlurBall 的固定局部重心也是一种读出规则，不是训练了这篇论文的两阶段匹配器。

因此新稿不推翻“仅给现有读出加集中度未带来足够排序收益”的本地结论；它说明另一件事：如果以后模型真的引入了候选截断和局部 refinement，必须考虑细化条件之外的错误尾部。分布拟合得更好也不自动意味着更多球被正确定位，CoRe 改善 homography 更不等于纠正了每个球点。

候选数量、temperature、feature scale 或 selected threshold 都会影响输入 spread/confidence。图像宽高归一化不保证这些变化下仍可共用校准参数。与此同时，[CasP 补读](2026-09-12-efficient-matching-budgets.md)已经指出候选选择本身也可能携带粗层视觉信息，不能将所有分数变化只归因于 softmax 分母。

## 六、公开后验实现与 v2 公式的差异

令 `F_d=(1-alpha_d)Lap_f(r_d)`，`C_d=alpha_d Lap_c(r_d)`。v2 §4.1 说 CoRe 权重为两轴 coarse-success posterior 的乘积，对应：

`w_paper = F_x F_y / [(F_x+C_x)(F_y+C_y)]`。

固定版本 [`core/utils.py`](https://github.com/khoavpt/Probabilistic-matching/blob/111b8311842892ede53077aaf16904c8483ded47/core/utils.py) 的 `fine_posterior_weights_from_params` 则分别累加两轴的 fine/coarse log joint，最后只对两种 joint 做归一化：

`w_code = F_x F_y / (F_x F_y + C_x C_y)`。

差别是分母没有 `F_x C_y` 与 `C_x F_y` 两个混合状态。对正分量，这个值大于论文的轴后验乘积；它对应另一种耦合假设，不能视为计算同一后验的等价改写。

沿实际路径核对：[`calibration/fit.py`](https://github.com/khoavpt/Probabilistic-matching/blob/111b8311842892ede53077aaf16904c8483ded47/calibration/fit.py) 的 NLL 部分逐轴计算 mixture、相加，与附录轴分解一致；[`homography_ransac.py`](https://github.com/khoavpt/Probabilistic-matching/blob/111b8311842892ede53077aaf16904c8483ded47/core/homography_ransac.py) 与 [`pnp_ransac.py`](https://github.com/khoavpt/Probabilistic-matching/blob/111b8311842892ede53077aaf16904c8483ded47/core/pnp_ransac.py) 直接调用上述权重，没有补回交叉项。这不是未使用的示例函数。

本地 CPU 小例设 `a_x=a_y=b_x=b_y=1`、`k_s=k_m=0`、所有阈值参数为零，两轴 raw fine/coarse std 均为 1、残差为零。两轴各自 posterior 为 0.5，论文乘积为 **0.25**；直接调用固定 helper 得 **0.49999999999999994**。结果与输入保存在 `outputs/literature/core-posterior-formula-check.json`，断言通过。这个例子仅核对公式，不是拟合出的实际参数、真实数据误差或两种后验谁更好。

**版本判断：** 当前只能确认 2026-08-11 的公开代码与 2026-08-29 的论文 v2 不同。它不证明作者所有版本都错误，也不证明文中指标由哪一条实现产生。若以后复用，先明确按 v2 公式还是按该源码实验；当前不修改未接入项目的上游代码，也不拿该差异代替球任务研究。

## 对下一步的实际影响

保留它作为“局部条件不确定性不足以代表整体匹配误差”的直接近邻。未来若进入可靠性研究，先明确目标是定位误差、粗候选成功、球身份还是无对应，再确定监督和可用输入。错误尾部建模、错误排序、几何模型重拟合和自动逐帧定位各自需要证据。

CoRe 的初始几何要求不能泛化给全部对应风险方法。同日[PDC、RoMa 与 PWarpC 补读](2026-09-12-native-matching-confidence.md)已确认图像对直接预测密度、共视及 null 的先例；需要另外几何估计的是相应后处理路径。是否适合球任务，应继续按预测事件、监督和最终决策判断。

当前没有增加 Laplace head、CoRe、相机估计或新的校准分支。先完成既定 BlurBall 时序对照，再从真实结果决定是否存在值得研究的可靠性问题。
