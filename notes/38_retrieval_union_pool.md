# 38. 构建多检索系统候选并集

## 这一阶段解决什么问题

上一阶段的人工 qrels 只覆盖 `planned + hybrid` 产生的固定候选，因此可以比较候选池内的排序方法，但不能公平比较不同候选生成方法。

本阶段采用信息检索评测中常见的 depth pooling：每个系统提交前 10 条候选，按 query/chunk 去重后统一人工标注。

## 对比系统

`direct/planned` 和 `dense/BM25/hybrid` 是两个不同维度：

- `direct_dense`：原问题直接做向量检索。
- `direct_bm25`：原问题直接做关键词检索。
- `direct_hybrid`：原问题的 dense 与 BM25 结果做 RRF 融合。
- `planned_dense`：先拆解、扩展问题，再做多路向量检索。
- `planned_hybrid`：先规划问题，每一路同时做 dense 与 BM25，再融合。

## Pooling 结果

- 问题数：8
- 每系统 pooling depth：10
- 去重后 query/chunk 对：224
- 每题候选数：23–35，平均 28
- 已继承人工标签：128
- 新增待标注：96
- 当前完成度：57.14%

只被单个系统找到的候选：

| system | 独有候选数 |
|---|---:|
| direct_dense | 22 |
| direct_bm25 | 26 |
| direct_hybrid | 0 |
| planned_dense | 16 |
| planned_hybrid | 13 |

`direct_hybrid` 没有独有候选并不等于它没有价值。它的候选来源本来就是 dense 和 BM25，价值主要体现在把两路共有结果重新排序到更靠前的位置。

## 为什么要盲标

如果标注者看见系统名称、检索分数或原始 rank，容易下意识把排名高的内容标得更相关。标注文件因此使用固定哈希打乱顺序，并清空以下字段：

- retrieval score
- retrieval rank
- retrieval channel
- expanded source query
- query plan aspect

真实的系统来源和排名只保存在 `eval/retrieval_union_v1/pool_manifest.jsonl`，标注完成后才用于计算指标。

## 可断点续跑

生成器每完成一个“问题 × 系统”就写入本地 `system_runs.jsonl`。再次执行时会复用已经完成的 40 个检索任务，避免中断后重新运行耗时的 planned retrieval。

```powershell
python experiments\26_retrieval_pooling\build_union_pool.py
```

## 下一步

在 `http://127.0.0.1:8770` 补完 96 条新候选。完成后基于同一份 union qrels 计算五个系统的 Recall@5/10、MRR、nDCG@10 和平均延迟，再判断 query planning 与 hybrid retrieval 的真实收益。
