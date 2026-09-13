# 球类运动物理先验：何时可用于二维自动定位，何时只能用于三维重建

**结论先行。** 物理约束用于球定位、候选关联和训练已有先例，不能概括成“现有工作只做后处理”。截至 2026-09-13，本轮证据支持优先研究可观测的曝光几何与有适用条件的运动先验，但尚不足以支持在当前 BlurBall 三帧二维读出中直接加入完整三维动力学。下列六篇重建近邻主要使用已有二维轨迹、相机／场地几何或事件分段；后面的 CVPR 2016 与 2026 联合训练论文则说明物理也能进入关联和训练，只是输入、时域及评价前提不同。

本页是对“是否现在加入球类运动物理特征”的有界审查，覆盖六篇轨迹重建近邻、物理关联／联合训练先例、后续 arXiv 与 BlurBall v1/v3 的相关段落；不是“物理+视觉”所有论文的穷尽检索。重点是核对真实输入、初始化／事件依赖、相机几何、物理使用位置及其是否被证明能提升二维自动定位。

用户随后明确允许增加帧数：本文“三帧”专指现有实验，**不限制下一阶段时域**。多帧可能改善轨迹可检验性；更新后的选择见[多帧可行性分析](../research/2026-09-13-ball-physics-feasibility.md)。已有三帧负结果不能否定长历史上的条件物理方法。

## 结论与当前研究边界

不应把下列命题混为一谈：

1. 球在三维空间受重力、阻力、旋转和碰撞约束；
2. 单目图像中的二维轨迹可用二次函数近似；
3. 模糊条纹表示曝光时间内的位移；
4. 物理后验能提高逐帧二维自动检测／定位。

第 1 项不自动推出第 2--4 项。即使三维运动具有常加速度，透视投影含深度除法，图像坐标通常不是二次多项式；相机摇摄、跟拍或变焦会进一步破坏固定投影。球的模糊条纹对应一次曝光期间的轨迹，不等于帧间位移；若端点没有经时间信息标定前后，条纹本身还是无正负的方向轴。羽毛球更不能照搬无阻力抛体：其阻力显著，MonoTrack 明确将每个 shot 建模为带空气阻力的 ODE。

因此，当前项目应保持两层分离：

- **二维视觉定位层：** 用当前 RGB 外观与历史视觉证据，在没有 GT 初始化、没有已知击球／落点、没有可靠相机矩阵的条件下输出球中心与可见性。
- **可选的轨迹级物理层：** 只在已有连续二维轨迹、明确的场地／相机几何和被验证的事件语义后，单独研究三维重建、落点或旋转。它不能把历史预测包装成当前帧视觉证据，也不能改写二维 detector 的主指标。

当前 BlurBall `data/blurball/` 本地发布内容中，按 `calib`、`camera`、`intrinsic`、`extrinsic`、`rotation`、`translation`、`homograph` 搜索，未发现可直接核验的相机矩阵或说明；`data/README.md` 也没有该矩阵条目。BlurBall 论文 v1 §3.1 则称每场提供相对世界坐标的外参 `R,T`。这两件事并不矛盾：只能记录为**当前本地包尚无可核验矩阵**，不能断言上游数据集没有标定。将来若研究三维，先核对发布页、版本与该字段的实际位置，不能猜测或重建后称为官方输入。

## 六篇轨迹重建近邻

| 工作 | 实际输入与前提 | 物理如何进入 | 是否证明提升二维自动定位 | 对本项目的约束 |
|---|---|---|---|---|
| Liu & Wang, *MonoTrack*，CVPRW 2022 | 静态转播视角；court、player pose、逐帧 shuttle 位置；HitNet 用三类 hit（无／近端／远端）把 rally 分成 shots。 | 每 shot 解带空气阻力 ODE 的约束非线性优化，并以场地、击球／接球玩家位置和落点合法性作先验。 | 否。物理优化消费已获得的 2D track；论文没有把物理残差反传为当前帧 shuttle detector 的监督。 | 说明羽毛球需要 drag，且 shot 分段、场地标定和 pose 都是额外问题；不能直接迁入三帧球中心模型。 |
| Ertner et al., *SynthNet*，MMSports 2024 | TrackNet Tennis 的球、court corners、player pose；WASB 与 GRU Hit-Bounce Net 先分出 shots；又依赖手工标注 net-pole 以校准相机。 | 用 `x¨=g-(D/m)||v||v` 合成轨迹；FNN 从已分段 2D trajectory 与 court corners 回归 3D 初始条件。物理主要是合成训练生成，不是 detector loss。 | 否。2D ball 由 WASB 获得，hit/bounce 独立预测；论文评价 3D 重投影与落点。 | 网球也不能只凭三个 pixel center 做物理；其最难项之一正是远端深度／异常高速和旋转。 |
| Gossard et al., *TT3D*，CVPRW 2025 | 先以 BlurBall 取得整个 rally 的 2D ball positions；table segmentation + PnP/focal-length 优化得到镜头；分段后须精确区分 table bounce 与 racket hit。 | 空中段用 drag + Magnus ODE，bounce 用 Coulomb friction；以 bounce 作为初始状态，优化 bounce 前速度与旋转以最小化全段重投影误差。 | 否。物理在检测、分段、相机标定之后；它未以物理训练二维 ball detector。 | 很接近乒乓球，但依赖标定、bounce 状态和长段优化；适合未来三维轨迹项目，不是当前局部 correspondence 修复。 |
| Kienzle et al., *Towards Ball Spin and Trajectory Analysis*，CVPRW 2025 | 以完整 2D ball trajectory 和 13 个 table keypoints 为抽象输入；实际测试的 table points 与 camera matrix 需要人工标注／估计。 | MuJoCo 合成具有 bounce／spin 的轨迹，Transformer 从 2D 轨迹预测 3D trajectory 与初始 spin。物理是数据生成及结构动机，不是运行时 physics residual。 | 否。论文明确不把 raw video 输入后端，前端二维球检测不在其范围。 | “physically correct synthetic data”不等于“物理约束能补当前帧球”；若采用，须另备表格 keypoint、连续轨迹及三维任务评估。 |
| Ponglertnapakorn & Suwajanakorn, *Where Is The Ball*，CVPRW 2025 | 输入完整 2D tracking sequence、已知 camera extrinsics，以及 ground / net vertical plane；真实 TrackNet 评估还要以 court detection 做 PnP。 | Unity/PhysX 生成带冲量的合成轨迹，BiLSTM 预测轨迹终点、height 与 refinement；它学习 simulated motion prior，而非显式拟合重力／阻力方程。 | 否。输入就是 2D tracked positions；整段双向 LSTM 也不是三帧因果模型。 | 是重要的“从已知 2D track uplift 到 3D”竞争路线，但不构成自动球发现或物理辅助二维检测的证据。 |
| Chiha et al., *Physics-informed neural networks and monocular vision for kinematics analysis*，SPIC 2026 | 56 段双高帧率同步相机序列用于 stereo 3D reference；YOLOv11+SAHI 做二维检测，ResNet50 以 bbox 辅助估计球的图像尺寸，再由单目尺寸／校准恢复深度。 | PINN 预测 3D trajectory 与运动学参数，并明确建模 Magnus effect；实数据使用 stereo 产生的训练／参考量。 | 未见其将 PINN 物理损失反馈到 YOLO 检测器的证据。SAHI 用八个切片解决小球检测，物理位于其后。 | 这是截至本轮最接近的 2026 PINN 例子，但它恰好显示真实单目物理需要额外可观测量（球大小、标定、stereo 参考）。 |

### 1. MonoTrack：羽毛球物理不是抛体模板

[MonoTrack 论文](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf)实际发表于 CVPRW 2022；arXiv [v1 2022-04-04、v2 2022-05-18](https://arxiv.org/abs/2204.01899)。作者以 TrackNetV2 数据的逐帧 shuttle 标记为基础，并额外使用 court、pose 与 hit detection。其 HitNet 对连续帧输出 `no hit / near-player hit / far-player hit`，再以最小 hit 间隔和双方交替等规则把 rally 分成 shot。

每一个 shot 的物理式不是无阻力抛物线，而是作者所称的带空气阻力二阶 ODE，并通过相机投影的 2D reprojection error 优化初值。作者自己说明旋转也可能影响运动而未纳入这个简化 model。该工作还用 4 个 court corners 和 2 个 net-pole tips 得到 DLT 所需的六个三维参考点，并给出接近击球／接球者、落在场内等约束。它证明“羽毛球 3D reconstruction 可以使用物理+强额外几何”，并不证明这种目标函数能在一帧像素中分辨 shuttle、白线和高光。

### 2. SynthNet：网球需要 event 分段与额外标定

[SynthNet 全文](https://stellagrasshof.com/assets/pdf/2024_ertner_mmisport.pdf)发表于 MMSports 2024，DOI [10.1145/3689061.3689073](https://doi.org/10.1145/3689061.3689073)。其 pipeline 先从视频提取 WASB ball coordinates、court coordinates 与 RTMO player poses；21 帧 GRU 的 HBNet 预测 hit / bounce / non-hit，并据此将序列分成 shots。每个 shot 才进入由合成数据训练的 FNN。

论文的空中模型为 `x¨=g-(D/m)|v|v`，以 synthetic initial conditions 产生三维与投影后的二维序列，再训练 FNN 回归六个初始条件。相机参数来自 court corners 和**手工标注的 net-pole coordinates**。作者报告在远端、怪角度、高速、短轨迹和带旋转的 shot 上会失败；这些正是当前高 `rho` 球场景而非一个简单 physics loss 能绕过的成像歧义。

### 3. TT3D：乒乓球物理从已知 bounce 开始

[TT3D 论文](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Gossard_TT3D_Table_Tennis_3D_Reconstruction_CVPRW_2025_paper.pdf)发表于 CVPRW 2025，arXiv [2504.10035v1，2025-04-14](https://arxiv.org/abs/2504.10035)。论文先用 BlurBall 检测全 rally 球点，用 piecewise quadratic segmentation 找候选 bounce／racket hit，并由 table geometry 自动标定相机。它特别利用 bounce 时球在已知 table plane 上，把该点作为轨迹初始化；随后优化 bounce 前的速度和 spin，而不是从任意第一帧直接优化深度。

其空中段为含 drag 和 Magnus 的 ODE，bounce 则使用 rolling/sliding 区分的 Coulomb friction model。物理模型的输出以整个段的 2D reprojection error 优化。这个机制不能在当前 fixed K16 candidate 的三个输入帧中成立：没有已验证 bounce、table pose 或完整 event segment 时，未知参数多于直接视觉约束；将其输出回投为“当前帧 evidence”也会循环使用历史预测。

### 4. Kienzle et al.：物理正确合成，不是 physics loss

[论文](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Kienzle_Towards_Ball_Spin_and_Trajectory_Analysis_in_Table_Tennis_Broadcast_CVPRW_2025_paper.pdf)发表于 CVPRW 2025，arXiv [2504.19863v1，2025-04-28](https://arxiv.org/abs/2504.19863)。原文的输入不是 RGB 特征，而是每时刻球的二维坐标与 13 个 table keypoints；后端预测整条 3D 轨迹和初始 spin。作者用 MuJoCo 生成 50,000 条有效轨迹，随机化初始位置、速度、spin 与 camera parameters，并在合成三维 GT 上直接监督位置与 spin。

所以它的 “physically grounded” 指：合成训练样本遵守乒乓球碰撞与旋转规律，且 architecture 以 trajectory→spin 的 bottleneck 表达物理依赖；不指在真实 RGB detector 上加入一个实时可微 physics residual。其真实数据的 2D reprojection 还需由 table annotations 估计 camera matrix。项目若引用它，必须准确写作“物理仿真训练的二维到三维 uplift”，不能写成“物理帮助自动二维球检测”。

### 5. Where Is The Ball：学习模拟运动先验也依赖整条轨迹

[论文](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Ponglertnapakorn_Where_Is_The_Ball_3D_Ball_Trajectory_Estimation_From_2D_CVPRW_2025_paper.pdf)发表于 CVPRW 2025，arXiv [2506.05763v1，2025-06-06](https://arxiv.org/abs/2506.05763)。它用 camera extrinsics 将每个二维球点变成 ground-plane 与 net vertical-plane 的交点表征，EoT network 预测一段运动结束／改变方向的概率，再由前、后向 LSTM 预测高度和三维 refinement。训练数据来自 Unity PhysX 模拟；损失由 3D GT、EoT 与地面以下惩罚组成。

这避免了“先硬切每个抛体段”的一部分脆弱性，却没有消除输入前提：camera parameters 已知、完整 2D track 已有，且主 height network 使用双向序列。真实 TrackNet comparison 的 contact point 也直接由标签得到、再投影到 calibrated court。它可作为显式物理拟合之外的学习式对照，但仍属于**轨迹后端**而不是自动定位网络。

### 6. Chiha et al.：PINN 在真实单目条件下仍借助 stereo 与尺寸深度

[期刊原文](https://doi.org/10.1016/j.image.2026.117587)为 *Signal Processing: Image Communication* 147, 117587（2026-09）；出版社页确认开放获取。主文可读范围确认：56 条 real table-tennis sequences 由两台同步高帧率相机构成；YOLOv11 的快速球检测存在间断，作者改用 SAHI 八切片。stereo triangulation 与场景标定既充当 3D reference，也构成 26,000 个球图像尺寸标签；单目端再用 ResNet50（加 YOLO bbox 尺寸）估计球大小，借此获得深度。PINN 以 projectile motion 和 Magnus effect 推断三维运动与运动学参数。

这篇工作应进入相关工作，因为它确实是 table tennis 的 physics-informed 2026 近邻。但它不支持“只用已有 BlurBall centers 训练 PINN 即可”：其可观测性来自高帧率 stereo、已标定环境和图像球尺寸。已读段落也没有显示 PINN loss 回传改进 YOLO 的二维检测；检测增强是 SAHI，物理用于其后的运动学重建。

## 物理确实进入候选关联与训练的先例

### Maksai et al.：状态条件化的物理关联早已有之

[*What Players Do With the Ball: A Physically Constrained Interaction Modeling*](https://arxiv.org/abs/1511.06181)发表于 CVPR 2016；本轮读 arXiv v2（2015-12-01）§3、§6，[全文](https://arxiv.org/pdf/1511.06181)。它在多台已标定、视野重叠的相机条件下，将球候选、球状态及球员轨迹纳入混合整数优化。物理约束参与路径选择，不仅修正一条已确定的球轨迹。飞行、滚动、持球等状态使用不同约束；状态不连续或位于可能碰撞的区域时，相关约束可以关闭。实验包含去除物理、减少状态的对照。

这是“状态适用时施加物理、接触时放松”的直接先例；不能宣称首次提出。其主要实验处理至少500帧序列，且使用多视角三维候选及球员信息，不能把结果直接迁作我们单目因果三帧的证据。它支持研究方向有依据，同时提高了我们必须证明的差异要求。

### Chen 2026：联合物理损失已有论文声明，输入与协议需分开

[*Trajectory and landing point analysis of tennis based on improved real-time object detection algorithm*](https://link.springer.com/article/10.1007/s44163-026-01127-0)，Discover Artificial Intelligence 6, 484，2026-04-19。本轮读取出版社正文§3–5：其输入融合 RGB 与事件数据，预测三维轨迹和落点；式(5)将检测、跟踪与动力学残差放入同一损失，动力学含阻力与旋转。

因此“首次将物理损失用于网球检测／跟踪训练”也不成立。该文 TrackNet 部分使用合成事件，另有 DAVIS 实验采集，且涉及标定；§5.1 写随机70:15:15划分但未明确比赛／clip独立性。它还分别写30fps源数据与200fps输入标准化，真实时间支持需要进一步澄清。故不将其 headline 指标当作本项目 RGB、game-level、三帧条件下的可比实证，也不因输入不同而否认其机制先例。

### 较新 arXiv：Uplifting Table Tennis

[*Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation*](https://arxiv.org/abs/2511.20250)，v1 2025-11-25，是 Kienzle 路线的后续应用。本轮摘要级核对确认其前端球／球台检测与后端 uplifting 分开训练；后端使用物理合成数据，并面向漏检和变化帧率。不能仅根据较早 CVPRW 稿断言后续没有完整 RGB 应用；同样，完整应用不等于已经证明物理损失改善前端检测。未在本轮逐项复核其全文实验，也未下载新数据。

## BlurBall v1/v3：可用的 blur 物理线索与不能外推的地方

[BlurBall arXiv v1](https://arxiv.org/pdf/2509.18387v1)发布于 2025-09-22；目前 arXiv 有 v3（2026-03-28），数据出处保留 v1；本轮另核对 [v3 §4.4–4.6](https://arxiv.org/html/2509.18387v3)，上述三点拟合、95段评价与核心数字仍保留。没有据新论文版本静默替换本地标注。v1 §3.1 说明每场具有 `R,T` 外参，球标签为模糊线段中点 `p_b`、半长 `l` 和 orientation `θ`，且以

\[
p_{1,2}=p_b\pm l(\cos\theta,\sin\theta)
\]

定义两端点。该公式只由集合定义线段；若没有额外时间语义，交换 `p_1,p_2` 会使方向相差 `\pi`，所以不应自行把 orientation 当作有符号的帧间速度方向。

§4.5 的“trajectory prediction”是一个严格有限的后处理验证：作者从**手工分好的 95 段 airborne trajectory**中取前 3 个观测，拟合二维二次 `P_x(t),P_y(t)`；position+blur 版本再同时拟合一个 exposure time `t_exp`，加入导数与 `l_k(\cos\theta_k,\sin\theta_k)/t_exp` 的差，blur loss 权重为 0.2。其报告 position-only 预测 MAE `84.4±136.6 px`，position+blur 为 `53.0±87.1 px`。

这支持一个很窄的结论：在静态镜头、已知 airborne segment、前 3 个有效点和该标注约定下，blur 观测可改善随后图像轨迹的二次拟合。它不证明：

- 该二次图像多项式是任意单目体育视频的物理定律；
- `t_exp` 可由本地元数据获得或跨视频共享；
- 条纹方向可直接替代跨帧 displacement；
- 这种后验拟合会提升当前帧自动球候选召回、可见性 F1 或 PCK；
- 在相机运动中仍有效。v1 §4.6 也明确记录了非线性模糊（bounce 恰在曝光内）和非静态 camera 时误检更多的局限。

## 对当前实施的决定

1. **不在当前 frozen-DINO / K16 / 三帧 readout 中加入抛体、drag、Magnus、bounce 或 PINN loss。** 这会把未观测的深度、相机和事件变量硬编码为二维候选排序，且无法从一次 PCK 变化归因于 visual correspondence。
2. **不把 BlurBall blur angle / half-length 解释为帧间 velocity GT。** 它们可在未来作为单帧曝光内 shape evidence 的辅助任务或诊断分组，但方向极性、曝光时间与相机条件必须分别核实。
3. **若未来扩展为三维／落点研究，先建立独立任务协议。** 至少应有可审计的 `R,T` 或可验证 scene calibration、真实 PTS、明确 hit/bounce/airborne 定义、长于三帧的连续轨迹，以及把“输入是自动二维预测”与“输入是 GT 轨迹”分开的评测。那个任务的训练选择不能用最终三维重投影来调当前二维 detector。
4. **当前更有区分力的问题不是恢复既有融合配方。** 已完成的 v1／v2／address 融合对照未得到稳定的 motion 增益；固定描述符的全局 cosine 会偏向静态背景，而硬化的历史 ball evidence 又造成大量 break。当前应优先核验可预测的成像轴／有条件观测先验（例如 blur 的曝光内几何及其明确适用条件）；由新证据决定是否提出新的、可区分的融合机制，不把旧的 current-unary 融合重新包装为物理分支。

## 检索充分性与未读边界

- 本页完整阅读 MonoTrack、SynthNet、TT3D、Where Is The Ball、BlurBall v1 的论文方法相关段；Kienzle et al. 与 Chiha et al. 的方法和输入事实来自其开放作者／出版社正文。未运行任何作者代码、未下载数据或预训练模型。
- 多相机工作的输入不适合直接复用，但其机制仍限制新颖性，因此纳入 Maksai 的条件物理关联；没有扩展为工业系统或机器人控制综述。Chen 2026 按已读正文列作联合损失先例，Uplifting Table Tennis 明确保留摘要级阅读边界。
- 这不是“所有物理球追踪文献已经覆盖”的声明。若正式提出 3D reconstruction、spin、bounce 或 camera calibration，需按该具体机制再做一次文献与数据可用性审查。

## 来源

1. Paul Liu, Jui-Hsien Wang. [*MonoTrack: Shuttle Trajectory Reconstruction From Monocular Badminton Video*](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf), CVPRW 2022；[arXiv:2204.01899v2](https://arxiv.org/abs/2204.01899)。
2. Morten Holck Ertner et al. [*SynthNet: Leveraging Synthetic Data for 3D Trajectory Estimation from Monocular Video*](https://stellagrasshof.com/assets/pdf/2024_ertner_mmisport.pdf), MMSports 2024，DOI [10.1145/3689061.3689073](https://doi.org/10.1145/3689061.3689073)。
3. Thomas Gossard et al. [*TT3D: Table Tennis 3D Reconstruction*](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Gossard_TT3D_Table_Tennis_3D_Reconstruction_CVPRW_2025_paper.pdf), CVPRW 2025；[arXiv:2504.10035v1](https://arxiv.org/abs/2504.10035)。
4. Daniel Kienzle et al. [*Towards Ball Spin and Trajectory Analysis in Table Tennis Broadcast Videos via Physically Grounded Synthetic-to-Real Transfer*](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Kienzle_Towards_Ball_Spin_and_Trajectory_Analysis_in_Table_Tennis_Broadcast_CVPRW_2025_paper.pdf), CVPRW 2025；[arXiv:2504.19863v1](https://arxiv.org/abs/2504.19863)。
5. Puntawat Ponglertnapakorn, Supasorn Suwajanakorn. [*Where Is The Ball: 3D Ball Trajectory Estimation From 2D Monocular Tracking*](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Ponglertnapakorn_Where_Is_The_Ball_3D_Ball_Trajectory_Estimation_From_2D_CVPRW_2025_paper.pdf), CVPRW 2025；[arXiv:2506.05763v1](https://arxiv.org/abs/2506.05763)。
6. Zaineb Chiha, Renaud Péteri, Laurent Mascarilla. [*Physics-informed neural networks and monocular vision for kinematics analysis: Application to table tennis*（实际读取的开放全文入口）](https://www.sciencedirect.com/science/article/pii/S0923596526001104), *Signal Processing: Image Communication* 147, 117587, 2026；[DOI](https://doi.org/10.1016/j.image.2026.117587)。该页标为 CC BY；本轮成功通过该公开网页读取正文，未在仓库或 `/tmp` 保留可再分发的本地全文副本。
7. Thomas Gossard et al. [*BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking*](https://arxiv.org/pdf/2509.18387v1), arXiv:2509.18387v1，2025-09-22；[当前版本页](https://arxiv.org/abs/2509.18387)。
