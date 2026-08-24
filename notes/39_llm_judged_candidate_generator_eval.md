# 39. 用完整盲审标签比较候选生成器

## 为什么使用单独的 LLM qrels

用户不继续补标新增的 96 条候选。项目没有把模型标签写成“人工标签”，而是保留两套数据：

- `qrels.jsonl`：128 条真实人工标签，仍是未完成状态。
- `qrels_llm.jsonl`：224 条 DeepSeek 盲审标签，用于本阶段横向实验。

全部 224 条都在固定哈希打乱后的候选顺序中重新判断。模型看不到检索系统、rank、score、检索通道、扩展 query、query plan 和人工标签。

## 模型标签质量检查

在与 128 条人工标签重叠的部分：

- 精确一致：63/128，`49.22%`
- 相差不超过一级：123/128，`96.09%`
- 严重分歧：5/128，`3.91%`

模型整体比人工更严格，尤其容易把“有实现细节但未完整解释问题”的 chunk 从 `2/3` 降为 `0/1`。因此这套 qrels 适合筛选方向和发现 badcase，不应在作品集中表述为黄金人工基准。

## 五种检索系统结果

相关阈值为 `relevance >= 2`，指标基于 depth-10 union pool：

| system | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | 中位耗时 |
|---|---:|---:|---:|---:|---:|
| direct_dense | 0.213 | 0.434 | 0.698 | 0.519 | 1.03s |
| direct_bm25 | 0.239 | 0.409 | 0.938 | 0.585 | **0.06s** |
| direct_hybrid | **0.257** | 0.408 | **1.000** | 0.580 | 2.41s |
| planned_dense | 0.219 | **0.463** | 0.688 | 0.547 | 22.18s |
| planned_hybrid | 0.251 | 0.461 | 0.854 | **0.620** | 21.86s |

## 结论

1. `direct_hybrid` 适合作为默认在线候选生成器。它的 Recall@5 和 MRR 最好，能更早给上下文组装层提供相关证据。
2. `direct_bm25` 不能被当作落后基线。技术文档包含大量准确英文术语，它以极低延迟得到很强的 nDCG 和 MRR。
3. query planning 确实提高深层候选覆盖。planned dense/hybrid 的 Recall@10 比对应 direct 方案高约 `0.029/0.053`。
4. 当前 planning 成本过高。约 22 秒的中位耗时不能为小幅平均收益提供足够工程价值，不应对所有问题全局启用。
5. planned 策略在 embedding、chunking 和 query expansion 等问题上有收益，但并非每题都改善。不能按测试题 ID 写死路由。

## 当前决策

- 默认路径：`direct_hybrid + lexical rerank`
- 快速降级路径：`direct_bm25`
- 复杂问题实验路径：保留 `planned_hybrid`，暂不作为全局默认
- 下一轮优化：让 planned 子检索并行执行，增加 query embedding 缓存，并基于 plan 的 aspect/sub-query 数量和延迟预算决定是否启用

## 边界

- 只有 8 个问题，结论仍需更多 query 验证。
- depth pooling 没有穷举知识库全部 938 个 chunk。
- 224 条完整标签来自单一 LLM judge，虽然经过人工重叠校准，仍可能存在系统性偏差。
