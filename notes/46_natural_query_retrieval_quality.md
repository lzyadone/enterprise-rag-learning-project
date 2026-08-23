# Natural Query Retrieval Quality

## 为什么不能只相信问题标签

`rag_natural_query_dev_v1` 中的 `intended_route` 表示 DeepSeek 生成问题时希望它包含一个还是多个信息需求。它不是检索真值。一个看起来复杂的问题可能被原始查询完整表达，直接检索已经足够；规划器也可能生成宽泛子查询，反而稀释原问题。

本阶段对 32 条自然问题分别运行 `direct_hybrid` 和 anchored `planned_v2_hybrid`，取每路 Top 10 后按问题合并、去重并确定性打乱。标注输入隐藏系统、排名、分数、检索通道和 query plan。

## 候选池与盲标

- 问题数：32；
- query/chunk pair：349；
- 每题候选：10 到 12，平均 10.91；
- direct 独有候选：29；
- planned v2 独有候选：29；
- DeepSeek 完整盲标：349/349。

| relevance | 数量 |
|---:|---:|
| 0，无关 | 99 |
| 1，弱相关 | 105 |
| 2，可用证据 | 85 |
| 3，直接证据 | 60 |

两条 PDFLoader 页码问题的 union pool 中最高只有 1 分，因此标记为 `coverage_gap`。这类问题不能虚构 direct/planned oracle，也不能混入检索质量均值。评测器现在单独报告候选池可评率，并仅对存在 2/3 级证据的问题计算 Recall、MRR、nDCG 和 oracle agreement。

## 总体结果

30 条问题可评，候选池可评率为 93.8%。

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median |
|---|---:|---:|---:|---:|---:|
| direct hybrid | **0.991** | **0.477** | **0.865** | **0.840** | **0.55s** |
| planned v2 hybrid | 0.935 | 0.453 | 0.691 | 0.739 | 2.25s |
| current auto | 0.960 | 0.463 | 0.724 | 0.769 | 1.41s |

current auto 选择 direct 15 次、planned 17 次，只命中 10/30 个逐题 nDCG oracle，agreement 为 33.3%。开发门槛失败，Web 默认继续保持 direct。

按 `|planned nDCG - direct nDCG| > 0.05` 判断有意义差异：

- direct 明显更好：19；
- planned 明显更好：2；
- 两者相近：9；
- coverage gap：2。

## 生成意图与真实收益

生成时标记为 direct 的 18 题中有 16 题可评，planned 平均 nDCG 比 direct 低 0.099。生成时标记为 planned 的 14 题全部可评，但 planned 平均仍低 0.105；其中只有 2 题出现大于 0.05 的收益。

| intended / observed oracle | direct | planned |
|---|---:|---:|
| direct | 13 | 3 |
| planned | 11 | 3 |

intended route 与 oracle 只一致 16/30。结论是：信息需求数量可以描述问题，但不能直接当成检索路线标签。下一版路由器应该预测“规划检索是否会带来证据排序收益”，而不是机械识别长句、多术语或多个问号。

## 主要 badcase

`natural_dev_030` 询问增量摄取中缓存、文档 ID、去重和向量写入一致性。direct nDCG 为 0.531，planned v2 只有 0.192。规划器把具体问题改写成通用的“RAG 关键组件”，扩展到 chunking、embedding、vector DB、evaluation 等宽泛类别；direct 排第 6 的 `llamaindex_ingestion_pipeline::chunk_0003` 被挤出 planned Top 10。

相同模式也出现在 Chroma API、citation 指标、ColBERT 两阶段检索等问题：原问题已有高辨识度实体和技术关系，额外的通用扩展引入噪声。planned 真正明显受益的两题分别是完整 RAG badcase 评测流程，以及 IngestionPipeline 的幂等、缓存和旧向量清理组合问题。

这提示下一轮应先修 planner，而不是仅调 router 阈值：

1. 禁止在没有问题证据时生成通用 `RAG key components` aspect；
2. 子查询必须保留原问题中的产品名、组件名和关键约束；
3. 只有可独立回答的明确方面才获得 coverage slot；
4. 对扩展查询加入原查询一致性检查，低一致性时退回 direct；
5. 在本开发集上做受控 A/B，修复后另建新 holdout 验证。

## 解释边界

这是开发集，不是发布用 holdout。DeepSeek 同时参与自然问题生成与相关性盲标，可能存在共享偏差；本轮没有人工 overlap audit。结果足以定位工程问题和指导开发，但修复后的发布结论必须来自新冻结的问题集，并至少抽查部分 qrels。

## 可复现命令

```powershell
python experiments\26_retrieval_pooling\build_union_pool.py `
  --dataset eval\datasets\rag_natural_query_dev_v1.jsonl `
  --systems direct_hybrid planned_v2_hybrid --all-dataset-cases `
  --no-inherit-qrels --pool-depth 10 `
  --output-dir eval\natural_query_retrieval_dev_v1 --force

python experiments\28_auto_retrieval_routing\evaluate_router.py `
  --manifest eval\natural_query_retrieval_dev_v1\pool_manifest.jsonl `
  --candidate-pools eval\natural_query_retrieval_dev_v1\candidate_pools.jsonl `
  --qrels eval\benchmarks\rag_natural_query_dev_v1\qrels_llm.jsonl `
  --latency-results eval\natural_query_retrieval_dev_v1\no_latency_cache.jsonl `
  --direct-system direct_hybrid --planned-system planned_v2_hybrid `
  --evaluation-role development `
  --output-dir eval\natural_query_retrieval_dev_v1\routing_evaluation

python experiments\30_natural_query_development\compare_intent_to_retrieval.py
```

候选正文和逐题 system runs 可由命令重建，因此不提交；完整 qrels、manifest、摘要和分析结果保存在 `eval/`。
