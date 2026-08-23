# Conservative Query Planner V3

## 开发目标

自然问题开发集显示，legacy planner 会因为“技术、组件、metadata、重排、阶段、是什么”等宽触发词，把具体问题扩展成通用 `RAG key components` 查询。anchored planned v2 虽然限制了扩展总权重，仍可能用十几到四十多个 retrieval runs 稀释原问题证据。

planner v3 的目标不是让所有问题都执行复杂规划，而是建立一个安全退化机制：只有存在至少两个明确、可独立检索的信息需求时才扩展，否则检索行为应等价于 direct。

## 设计

`plan_query_v3` 与 legacy planner 并存，不修改当前 Web 默认。主要约束包括：

1. `metadata`、`重排`、`技术`等单个术语不再自动生成通用 techniques aspect；
2. `是什么`不再把任意 RAG 问题扩展成分类、技术和瓶颈三件套；
3. citation aspect 只由明确引用需求触发，不再由普通“指标”触发；
4. 增量摄取问题明确拆成 `ingestion_identity` 和 `ingestion_cache`；
5. 少于两个独立 aspect 时只保留 original-query anchor；
6. 多方面问题最多使用 4 个 aspect，每个 aspect 最多一个全局扩展和一个分类扩展；
7. 所有 aspect query 都保留完整原问题，再追加检索重点；
8. 原查询融合权重为 3.0，全部全局扩展总权重为 0.75，全部分类扩展总权重为 0.25。

## 静态规划变化

对同一套 32 条自然问题只运行 planner，不执行检索：

| planner | 总 retrieval runs | 平均每题 | 真正扩展的问题 |
|---|---:|---:|---:|
| legacy + anchored v2 | 583 | 18.22 | 26/32 |
| conservative v3 | 58 | 1.81 | 4/32 |

v3 扩展的问题是：RAG badcase 评测流程、IngestionPipeline 幂等与缓存、citation 指标组合，以及摄取流水线一致性。具体 API、Chroma、ColBERT 和单指标解释问题都安全退化为原查询。

## 公平候选池

新的 union pool 同时包含 `direct_hybrid`、`planned_v2_hybrid` 和 `planned_v3_hybrid`：

- 32 个问题；
- 351 个 query/chunk pair；
- 从上一轮相同 DeepSeek judge 的完整 qrels 复用 349 条；
- v3 仅引入 2 条新候选，并以隐藏系统和排名的方式补充盲标；
- relevance 分布：0 分 99、1 分 105、2 分 87、3 分 60；
- 30 题存在可用证据，2 个 PDFLoader 页码问题仍为 coverage gap。

## 质量结果

| system | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| direct hybrid | 0.633 | 0.977 | 0.865 | 0.837 |
| planned v2 hybrid | 0.568 | 0.925 | 0.691 | 0.737 |
| planned v3 hybrid | **0.648** | **0.992** | **0.882** | **0.853** |

v3 相比 direct 没有出现 nDCG 下降超过 0.05 的问题。28 题结果相近，2 题 v3 明显更好，2 题为 coverage gap。

关键样例：

- `natural_dev_030`：direct 0.469，v2 0.169，v3 0.916；v3 找回了摄取一致性相关证据。
- `natural_dev_021`：direct 0.744，v2 0.808，v3 0.797；v3 保留了多指标 badcase 分析收益，但使用更少 runs。
- `natural_dev_024`：direct 与 v3 均为 0.858，v2 为 0.980；v3 在该题放弃了一部分潜在收益，换取更稳定的全局行为。
- `natural_dev_027`：direct 0.941，v3 0.920，差异低于 0.05，未形成有意义退化。

## 自动路由开发结果

使用 v3 plan shape 路由时，28 题选择 direct，4 题选择 planned v3：

| system | Recall@10 | MRR@10 | nDCG@10 | median |
|---|---:|---:|---:|---:|
| direct | 0.977 | 0.865 | 0.837 | 0.52s |
| planned v3 | 0.992 | 0.882 | 0.853 | 0.48s |
| conservative auto | **0.992** | **0.882** | **0.853** | 0.52s |

conservative auto 与逐题 oracle 一致 28/30，六项开发诊断全部通过。两次不一致分别是完全持平和仅下降 0.021 的扩展题，未触发 planned worst-case 门槛。

## 延迟校正

三系统候选生成按 direct、v2、v3 顺序运行，v3 的 direct-like 查询会命中前序 embedding 缓存，原始 `0.03s` 不是独立在线延迟。为避免误报，另起全新进程只运行 v3：

- 全部问题中位：0.479s；
- 不扩展问题中位：0.465s；
- 4 个扩展问题中位：2.026s；
- 扩展问题最大：2.937s。

正式摘要通过独立 latency manifest 覆盖缓存污染的 v3 耗时。

## 解释边界与决定

这是已经用于 planner v3 设计的 development set。当前结果证明改动值得进入独立验证，不证明它已经泛化，也不允许直接把 Web 默认切换为 auto 或 conservative planned。

工程决定：

- 保持 Web 默认 direct；
- 保留 anchored v2 作为已有实验基线；
- planner v3 作为 holdout v3 候选冻结；
- 下一阶段先冻结新问题、候选系统和通过门槛，再运行检索和盲标；
- 新 holdout 通过并完成人工抽查后，才考虑把 Web 手动 planned 升级为 v3。

## 可复现命令

```powershell
python experiments\26_retrieval_pooling\build_union_pool.py `
  --dataset eval\datasets\rag_natural_query_dev_v1.jsonl `
  --systems direct_hybrid planned_v2_hybrid planned_v3_hybrid `
  --all-dataset-cases `
  --source-qrels eval\benchmarks\rag_natural_query_dev_v1\qrels_llm.jsonl `
  --target-qrels eval\benchmarks\rag_natural_query_planner_v3_dev\qrels_llm.jsonl `
  --pool-depth 10 --output-dir eval\natural_query_planner_v3_dev --force

python experiments\26_retrieval_pooling\label_union_with_llm.py `
  --candidate-pools eval\natural_query_planner_v3_dev\candidate_pools.jsonl `
  --output eval\benchmarks\rag_natural_query_planner_v3_dev\qrels_llm.jsonl `
  --summary eval\benchmarks\rag_natural_query_planner_v3_dev\qrels_llm_summary.json `
  --rejudge-all --skip-human-audit --batch-size 5

python experiments\26_retrieval_pooling\build_union_pool.py `
  --dataset eval\datasets\rag_natural_query_dev_v1.jsonl `
  --systems planned_v3_hybrid --all-dataset-cases --no-inherit-qrels `
  --pool-depth 10 --output-dir data\runtime\planner_v3_latency --force

python experiments\26_retrieval_pooling\evaluate_candidate_generators.py `
  --candidate-pools eval\natural_query_planner_v3_dev\candidate_pools.jsonl `
  --manifest eval\natural_query_planner_v3_dev\pool_manifest.jsonl `
  --latency-manifest data\runtime\planner_v3_latency\pool_manifest.jsonl `
  --qrels eval\benchmarks\rag_natural_query_planner_v3_dev\qrels_llm.jsonl `
  --output-dir eval\natural_query_planner_v3_dev\system_evaluation
```
