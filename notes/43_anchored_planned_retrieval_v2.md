# Anchored Planned Retrieval V2

## 问题

独立留出评测表明，legacy planned 在复杂问题上会把原问题证据稀释。进一步检查发现两个原因：

1. 原问题、通用 aspect 查询、分类扩展和 filtered 查询近似等权进入 RRF；重复的通用查询能够集体压过原问题。
2. `plan_boost` 最高直接加 0.24，而 RRF 分数通常只有 0.02-0.04，启发式信号比真实检索排序强一个数量级。

## 参考设计

- [RAG-Fusion](https://arxiv.org/abs/2402.03367) 使用多查询和 RRF 扩大检索视角，也明确记录了生成查询偏离原意时会跑题。
- [Elasticsearch RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) 给出 `1 / (k + rank)` 融合方式、rank constant 和窗口大小的生产参数。
- [Elasticsearch Weighted RRF](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf) 使用 `weight / (k + rank)` 表达不同检索信号的优先级。
- [Scaling RAG Fusion](https://arxiv.org/abs/2603.02153) 指出多查询的召回收益可能被固定 Top-K、重排预算和上下文截断抵消，必须同时评估质量与延迟。

## V2 设计

- 原问题全局检索是 anchor，权重固定为 2.0，并保持 direct 的候选窗口参数。
- 所有全局扩展查询合计权重不超过 1.0，不会因为 fanout 增加而无限放大。
- 所有 metadata/category filtered 扩展合计权重不超过 0.5。
- `(query, category)` 完全相同的 retrieval run 只执行一次。
- 最终 Top-K 最多允许两个 plan coverage 槽位，其余位置遵循融合相关性排序。
- v2 的 aspect boost 上限为 0.006，只能影响接近的候选，不能推翻 RRF 主排序。
- legacy 与 anchored 通过 `fusion_mode` 并存，支持严格 A/B。

## 完整标签原则

anchored v2 会带来旧 union pool 外的新片段。评测脚本发现未标注候选时会停止评分并输出匿名增量池。本阶段先后补充 8、44、3、1 个候选标签；旧标签保持不可变，新增 qrels 分层叠加。未标注片段从未被当成 0 分。

## 开发集结果

### 原 8 题校准集

| system | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| direct hybrid | 0.400 | **1.000** | 0.576 |
| legacy planned | **0.451** | 0.854 | **0.614** |
| anchored v2 | 0.403 | 0.917 | 0.552 |

### 暴露问题后的 16 题开发集

| system | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| direct hybrid | 0.797 | 0.812 | **0.730** |
| legacy planned | 0.413 | 0.498 | 0.453 |
| anchored v2 | **0.803** | **0.823** | 0.726 |

v2 明显消除了 legacy 的灾难性退化，但没有在所有开发问题上稳定超过 direct。它适合作为下一轮 planned 候选，不足以证明 auto 应重新成为默认。

## 工程决策

- Web 默认仍是 `direct`。
- Web 的 planned 分支默认使用 `anchored`，可通过 `RAG_PLANNED_FUSION_MODE=legacy` 回放旧行为。
- 当前 16 题已经用于开发诊断，不再称为独立 holdout。
- 下一阶段先冻结新的问题集，再生成候选与盲标，不能继续在现有 24 题上调参后宣称泛化。

## 产物

- 实现：`src/retrieval.py`
- A/B 脚本：`experiments/29_planned_retrieval_v2/compare_fusion.py`
- 原校准集结果：`eval/planned_retrieval_v2_dev/summary.md`
- 16 题开发结果：`eval/planned_retrieval_v2_holdout_v1_dev/summary.md`
