# 35. Hybrid Retrieval 与 RRF

## 这一阶段解决什么问题

原系统的 Chroma + bge-m3 属于稠密向量检索。它擅长寻找语义相近的内容，但型号、API 名称、英文缩写、配置项等精确词不一定排在前面。企业知识库通常同时需要：

- 语义召回：用户说法和资料原文不一样时仍能找到相关内容。
- 精确召回：`BGE-M3`、`metadata filter`、`where`、错误码等词不能被语义近似结果挤掉。

因此本阶段增加 BM25 稀疏检索，并和已有 dense retrieval 组成 hybrid retrieval。

## 数据怎样流动

1. Chroma 使用 bge-m3 查询向量，返回 dense 候选及距离。
2. BM25 读取同一份 `chunks.jsonl`，对英文词和中文二/三元词组建立倒排统计。
3. 两路各自返回候选排名，不直接比较两种不可比的原始分数。
4. RRF 使用 `1 / (k + rank)` 合并排名；同一 chunk 被两路命中时会累加分数。
5. 融合后的候选再进入可选 rerank、计划覆盖选择和上下文组装。

RRF 的好处是不用假设 Chroma distance 和 BM25 score 在同一数值尺度上，工程上更稳定，也便于以后增加更多检索通道。

## 为什么给标题和结构字段加权

BM25 索引没有改写 chunk 正文，而是在建立稀疏索引时重复 `title`、`category` 和 `heading_path`。这样技术名称出现在标题时更容易排到前面，同时 Chroma 中保存的原文仍保持不变，引用和审计看到的还是原始资料。

## 对比实验

实验脚本：`experiments/23_hybrid_retrieval_eval/evaluate_hybrid_retrieval.py`

数据：端到端评测集中的 8 个知识检索问题；排除长期记忆题和没有检索金标准的天气边界题。

固定设置：`top_k=7`、`candidate_k=16`、相同 chunks、相同 Chroma collection。

指标：

- category pass：top-k 是否覆盖要求的知识类别。
- evidence-term pass：top-k 正文是否包含足够的预期证据词。
- both pass：上述两项同时通过。
- category MRR：第一个正确类别越靠前，分数越高。
- source term recall：预期证据词在 top-k 中的覆盖比例。

结果：

| 策略 | both pass | category MRR | term recall | 平均耗时 |
|---|---:|---:|---:|---:|
| dense | 62.5% | 0.573 | 0.738 | 0.35s |
| dense + lexical rerank | 62.5% | 0.594 | 0.738 | 0.35s |
| hybrid | 75.0% | 0.708 | 0.838 | 0.37s |
| hybrid + lexical rerank | 75.0% | 0.688 | 0.838 | 0.36s |
| planned dense + lexical | 100% | 0.875 | 0.950 | 8.31s |
| planned hybrid + lexical | 100% | 0.938 | 0.950 | 8.66s |

完整结果在 `eval/hybrid_retrieval_experiment/summary.md`。

## 如何理解结果

这次升级证明了三件事：

1. Hybrid 对直接检索有稳定收益，尤其改善精确术语和 chunking 类问题。
2. 复合问题仍需要 query planning 才能覆盖多个知识方面，hybrid 不能代替问题拆解。
3. 当前 lexical rerank 没有改善 hybrid 的通过率，MRR 还略有下降。因此不能把启发式重排写成已经验证有效的亮点。

端到端使用 `planned + hybrid + lexical` 再跑 10 题，结果 10/10、quality pass 10/10，平均 21.22 秒，说明新通道没有破坏生成、引用、审计和记忆流程。

## 下一阶段

下一步应引入真正的 cross-encoder reranker，固定 hybrid 候选集后只替换重排器，对比：

- none
- lexical heuristic
- BGE reranker

重点看 hit/MRR、答案引用质量、延迟和内存占用，再决定是否作为默认重排器。
