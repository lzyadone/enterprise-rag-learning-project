# Routing Holdout V2 Release Decision

## 为什么要再做一次留出评测

前一套 16 题已经用于定位 legacy planned 的证据稀释问题，也参与了 anchored v2 的开发判断，因此它只能作为开发集。要判断修改能否泛化，必须先冻结一套从未运行过规划和检索的问题，再生成候选和标签。

holdout v2 在提交 `079e51f` 中先冻结了以下内容：

- 20 道新问题，其中 focused 10 道、compound 10 道；
- direct 系统为 `direct_hybrid`；
- planned 系统为 anchored `planned_v2_hybrid`；
- 每个系统取 Top 10；
- 相关阈值为 2；
- 6 项通过标准。

冻结提交之前只检查了语料的标题、分类、章节和题目可答性，没有运行 query plan、检索或指标计算。

## 匿名候选池与标签

两个系统合计给出 400 个排名位置，按 query 和 chunk 去重后形成 217 个待标注 pair：

- direct 独有候选：17；
- planned v2 独有候选：17；
- 两个系统共享：183；
- 每题 union pool：10 到 12 条，平均 10.85 条。

候选顺序按问题确定性打乱，标注输入隐藏系统名、排名、距离、融合通道和 query plan。DeepSeek 对 217/217 个 pair 完成 0 到 3 级盲标：

| relevance | 数量 |
|---:|---:|
| 0 | 53 |
| 1 | 66 |
| 2 | 66 |
| 3 | 32 |

第一次以 10 条为一批时，模型连续三次遗漏一个 chunk ID，严格校验使运行中止。已有标签被原子保存，剩余任务改为每批 5 条后完成。遗漏项没有被默认记为 0 分。

## 总体结果

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median |
|---|---:|---:|---:|---:|---:|
| direct hybrid | 0.973 | 0.480 | 0.800 | **0.825** | **1.48s** |
| planned v2 hybrid | 0.955 | 0.475 | 0.702 | 0.752 | 4.95s |
| auto | **0.980** | 0.480 | **0.825** | 0.804 | 4.05s |

auto 选择 direct 7 次、planned 13 次，与逐题 nDCG oracle 一致 11/20。

## 预注册门槛

| check | 结果 | actual | required |
|---|---|---:|---:|
| auto nDCG 非劣 | FAIL | 0.8043 | >= 0.8050 |
| auto Recall 非劣 | PASS | 0.9800 | >= 0.9533 |
| auto MRR 非劣 | PASS | 0.8250 | >= 0.7500 |
| oracle agreement | FAIL | 0.5500 | >= 0.7500 |
| planned route 单题最大损失 | PASS | 0 cases | 0 cases |
| median latency | PASS | 4.05s | <= 12s |

总体结论为 **FAIL**。nDCG 只差 0.0007 仍然算失败，因为标准是在看结果前写下的，不能在看到分数后放宽。

## 分层诊断

| stratum | direct nDCG | planned v2 nDCG | auto nDCG | auto 选择 planned |
|---|---:|---:|---:|---:|
| focused | **0.840** | 0.719 | 0.791 | 7/10 |
| compound | 0.810 | 0.784 | **0.818** | 6/10 |

auto 在 compound 题上出现小幅收益，说明“复杂题可能值得规划”的方向仍有价值。主要问题是 focused 题过度路由：包含多个分类词或触发规则，不等于真正需要多路问题拆解。

最明显的例子是 `holdout2_contextual_recall`：direct nDCG 为 0.8542，planned v2 为 0.6065，auto 仍把它送入 planned。这个样例只能用于解释本次发布决定，不能继续用来修改阈值并再次声称通过 holdout。

## 工程决定

- Web 默认保持 `direct`。
- 手动 `planned` 继续使用 anchored v2，作为可观察的实验路径。
- `auto` 不升级为默认，也不标记为发布候选。
- holdout v2 从本次评测后封存，不参与下一轮路由或融合调参。
- 下一轮若继续研究路由，先从真实用户日志或新的开发集学习“术语多”与“需要拆解”的区别，再冻结第三套独立留出集。
- LLM 盲标结论仍需少量独立人工抽查；在此之前不把它表述为生产级证明。

## 可复现产物

- 协议：`eval/benchmarks/rag_routing_holdout_v2/README.md`
- 问题集：`eval/datasets/rag_routing_holdout_v2.jsonl`
- 系统排名：`eval/routing_holdout_v2/pool_manifest.jsonl`
- 完整标签：`eval/benchmarks/rag_routing_holdout_v2/qrels_llm.jsonl`
- 汇总：`eval/auto_retrieval_routing_holdout_v2/summary.md`

仓库的统一 `.gitignore` 会排除候选正文、逐题结果和 system run 缓存，避免重复提交可再生的大文件。clone 后先重建匿名候选：

```powershell
python experiments\26_retrieval_pooling\build_union_pool.py `
  --dataset eval\datasets\rag_routing_holdout_v2.jsonl `
  --systems direct_hybrid planned_v2_hybrid `
  --all-dataset-cases --no-inherit-qrels --pool-depth 10 `
  --output-dir eval\routing_holdout_v2 --force
```

然后使用仓库中已经冻结的完整 qrels 重算路由结果，不需要再次调用 DeepSeek：

```powershell
python experiments\28_auto_retrieval_routing\evaluate_router.py `
  --manifest eval\routing_holdout_v2\pool_manifest.jsonl `
  --candidate-pools eval\routing_holdout_v2\candidate_pools.jsonl `
  --qrels eval\benchmarks\rag_routing_holdout_v2\qrels_llm.jsonl `
  --latency-results eval\routing_holdout_v2\no_latency_cache.jsonl `
  --direct-system direct_hybrid --planned-system planned_v2_hybrid `
  --evaluation-role holdout --output-dir eval\auto_retrieval_routing_holdout_v2
```
