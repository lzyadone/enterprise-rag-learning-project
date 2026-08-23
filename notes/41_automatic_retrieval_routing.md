# 41. Direct / planned 自动检索路由

## 为什么需要路由

候选生成评测表明，planned retrieval 不是对每个问题都更好：

- 它能通过问题拆解和多路检索改善部分复合问题的召回与覆盖；
- 对已经足够明确的单点问题，扩展查询可能引入噪声，同时增加延迟；
- 缓存优化后 planned hybrid 的本机中位耗时仍约为 7.50 秒，direct hybrid 约为 2.41 秒。

因此成熟系统不应全局固定一种路径，而应保留 `auto / direct / planned` 三种模式。

## 路由方法

自动路由先调用现有确定性 query planner，但不会额外调用 LLM。路由器读取通用 plan 特征：

- 必须回答的 aspect 数量；
- sub-query 数量；
- 涉及的知识类别数量；
- 唯一检索查询数量；
- category 匹配置信度；
- 问题意图。

当前复杂度规则为：

1. 多个 answer aspects：倾向 planned；
2. 没有显式 aspect，但需要跨多个类别扩展：倾向 planned；
3. 已识别单个 aspect，但 category 置信度很低：倾向 planned；
4. 聚焦且类别明确的问题：倾向 direct；
5. 用户显式选择 `direct` 或 `planned` 时，不覆盖用户设置。

路由不会写死具体问题或实体名称。返回结果包含选择、分数、特征和原因码，便于 Web 展示与离线审计。

## 延迟预算

本机 planned 延迟估算使用：

```text
estimated_ms = 1500 + 1000 * unique_retrieval_query_count
```

该估算来自当前 Ollama、Chroma 和 embedding 缓存实验，只是路由用的经验模型。如果复杂度达到 planned 阈值，但预计耗时超过用户预算，自动模式会降级为 direct。

默认预算是 12000 ms。它不是硬实时 SLA；部署到其他机器后应重新校准系数，并增加请求超时和运行时降级。

## 校准评测

运行：

```powershell
python experiments\28_auto_retrieval_routing\evaluate_router.py
```

使用 8 个问题、224 条完整 DeepSeek 盲审 qrels，以及缓存优化后的 planned 逐题耗时：

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | 中位耗时 |
|---|---:|---:|---:|---:|---:|
| always direct hybrid | 0.408 | 0.550 | 1.000 | 0.580 | 2.41s |
| always planned hybrid | 0.461 | 0.600 | 0.854 | 0.620 | 7.50s |
| auto | **0.470** | **0.613** | 0.938 | **0.666** | 3.43s |

自动模式选择 5 次 planned、3 次 direct，与逐题 nDCG 较优路径一致 8/8。

## 解释边界

这 8 个问题参与了初始阈值设计，因此 8/8 是校准结果，不是独立泛化证明。下一步必须增加未参与规则设计的 holdout 问题，覆盖简短事实题、跨文档比较题、模糊追问、知识库外问题和不同延迟预算，再决定是否把阈值作为稳定默认值。

完整结果见 `eval/auto_retrieval_routing/summary.md`。
