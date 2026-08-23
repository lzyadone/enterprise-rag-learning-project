# 40. Planned retrieval 查询向量复用

## 问题

Planned retrieval 会把一个复杂问题拆成多个 aspect 和 sub-query，并对同一个查询执行全库检索及多个 category 过滤检索。优化前，每次检索都会重新调用 Ollama `bge-m3`，同一句查询被重复向量化多次。

这不会增加召回范围，只会增加延迟。例如最复杂的测试问题有 9 个唯一规划查询，但实际触发了 48 次 embedding 请求。

## 实现

- 在 `src/ollama_http.py` 增加线程安全、上限为 256 项的进程内 LRU 查询向量缓存。
- 缓存键包含 embedding 模型、Ollama host 和规范化后的查询文本，避免不同模型或服务之间误用向量。
- Planned retrieval 在执行 Chroma/BM25 检索前，先为每个唯一规划查询生成一次向量。
- 同一个向量随后复用于全库检索和不同 category filter，不改变 Chroma 查询、BM25、RRF、重排或覆盖选择逻辑。
- 保留 `reuse_query_embeddings=False` 诊断开关，用于严格 A/B 和故障排查；生产默认启用复用。

## 为什么没有采用批量 embedding

最初尝试把所有唯一查询一次性发送给 Ollama，虽然中位耗时降至约 2.5 至 2.9 秒，但与历史 manifest 对比时候选集合出现变化。历史 manifest 还混入了代码和运行状态变化，不能作为严格基线，因此不能据此证明批量模式降低质量，但也不能证明等价。

最终采用更保守的方式：每个唯一查询仍单独调用一次 embedding，只跨过滤检索复用结果。这样更容易解释，也更适合作为当前生产默认。

## 严格 A/B

实验脚本：

```powershell
python experiments\27_planned_retrieval_latency\benchmark_latency.py
```

同一个进程内先预热 embedding 模型。每个问题使用相同的当前代码和确定性 query plan，分别运行：

1. baseline：每次 category/filter 检索都重新生成向量；
2. optimized：每个唯一规划查询只生成一次向量并复用。

两次运行之间清空进程内缓存，因此唯一变化是 embedding 是否复用。

| system | baseline 中位耗时 | optimized 中位耗时 | 中位提速 | Top 10 顺序一致 | 平均 embedding 调用 |
|---|---:|---:|---:|---:|---:|
| planned_dense | 23.13s | 6.03s | 3.95x | 8/8 | 22.50 -> 5.50 |
| planned_hybrid | 22.78s | 7.50s | 3.13x | 8/8 | 22.50 -> 5.50 |

完整逐题结果见 `eval/planned_retrieval_cache_benchmark/summary.md`。

## 结论与边界

这项优化可以进入默认路径：16 组 paired runs 的 Top 10 候选顺序全部一致，说明没有通过缩小检索范围换取速度。收益来自删除重复 embedding 计算。

测试集只有 8 个问题，耗时也会受本机 Ollama 和 Chroma 状态影响，因此 3.13 至 3.95 倍是本机实验结果，不应表述为所有部署环境的固定收益。

下一阶段不应继续堆缓存。更有意义的是根据问题复杂度自动选择 direct hybrid 或 planned hybrid，并在 planned 路径中并发执行相互独立的 Chroma/BM25 检索，同时设置延迟预算和降级策略。
