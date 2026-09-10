# 精细定位的适配收益是否伴随匹配排序改善？

日期：2026-09-10。状态：CPU入口与真实单查询smoke通过；四个固定前缀的正式诊断尚未运行。
协议：[前缀适配后GT-query诊断v1](../protocols/tennis-adapted-correspondence-v1.md)。

## 问题与范围

[已完成的前缀适配](2026-09-10-prefix-adaptation.md)中，微调在三seed都提高严格位置PCK8，但未一致提高检测F1。这个后续只读测量区分“最终定位可读性提高”与“条件性跨帧匹配也改善”，不增加训练、不重新选checkpoint，也不改变全量DINO条件。

使用frozen_seed0及finetune_seed0/1/2最终保存的前缀。三个冻结臂均不更新前缀，故只算一次共同冻结表示。输入为原step8目标对应的真实连续RGB，所有模型都在CPU在线算float32特征；旧float16缓存仅作为参考身份的来源，不作为一臂的特征输入。

当前GT query位于native36×64格，预测历史位置。主要条件为双端VC1，主要匹配固定去空间均值的Δ1/R2、Δ2/R4及global精确格recall@1/5/10。其他原helper结果完整保存但不重新择优。VC3响应不作为正视觉对应；这也不是自动球发现或新motion算法的性能表。

## 最小实现与入口验证

scripts/probe_adapted_correspondence.py从run配置定位RGB，从best.pt严格加载prefix参数与归一化buffer，复用analyze_pair、summarize和summarize_groups。每个当前query只处理当前及需要的唯一历史图像，两个间隔复用当前特征；不执行定位头、不存新的大特征缓存、不读原图、不占用GPU。

新增分组只是在原摘要上补充双端VC1的同格/跨格，以及源visibility组合。来源引用或特征形状不符合固定输入时明确失败；现有时间和标签接线不重复审计。

实际smoke使用原参考JSON中game7/Clip1/8的两个帧对，历史分别为7和6。输入参考另存outputs/adaptation_correspondence/smoke_reference.json，注明源参考与smoke属性，没有修改原诊断。

```bash
python scripts/probe_adapted_correspondence.py --run outputs/adaptation_probe/frozen_seed0 --reference outputs/adaptation_correspondence/smoke_reference.json --output outputs/adaptation_correspondence/smoke.json
```

运行使用Conda zshihyc、CPU2线程；strict加载成功，特征为有限float32 192×36×64。保存1个query、2个pair，Δ1/Δ2身份分别为8→7、8→6，checkpoint epoch0；双端VC1分组与同格/跨格分组计数一致。索引、forward、匹配及汇总共3.24秒，其中归一化与prefix forward为3.07秒。计时不含初始metadata/checkpoint读取、模型构造、导入或最终JSON写入；首批成本不直接外推全部运行或GPU吞吐。

该smoke发生在新脚本提交前，记录的父提交4e0a516不包含新入口；正式诊断在实现提交后启动并另记实际revision。旧匹配算法的测试已覆盖向量对应、候选范围和并列规则，本次未重复跑其原数据或训练。

独立审阅确认来源、raw frame身份、prefix与归一化参数、native格、VC1/VC3分层、固定checkpoint选择及计时边界符合协议，无剩余问题。正式参考共有216个不同当前query，两个间隔共享当前特征；脚本语法与差异格式检查通过。

## 正式运行的固定内容

正式参考为outputs/correspondence_probe/stage1_cosine.json，含215/213个Δ1/Δ2合法坐标帧对，双端VC1为199/190。顺序测量frozen_seed0、finetune_seed0、finetune_seed1、finetune_seed2；结果保存outputs/adaptation_correspondence/<run>.json。命令模板如下，当前尚未开始四组正式运行：

```bash
python scripts/probe_adapted_correspondence.py --run outputs/adaptation_probe/frozen_seed0 --reference outputs/correspondence_probe/stage1_cosine.json --output outputs/adaptation_correspondence/frozen_seed0.json
```

每组保存原run配置、实际checkpoint epoch、参考身份、代码版本、CPU条件、逐帧结果、分组和耗时。完成后复用这些JSON作配对比较，不重新forward或按新指标另选checkpoint。此前自动定位/存在指标直接引用适配实验的保存预测，不混用不同目标范围的分数。

## 判断边界

若严格自动位置改善而固定匹配指标不改善，进一步收窄为定位适配收益；若匹配改善但检测不一致，说明条件性对应信号没有自动保证最终收益。无论何种结果，当前GT query都绕过自动发现，同一game7的开发测量也不能替代独立比赛。正式结果尚未得到，当前不填写性能判断。
