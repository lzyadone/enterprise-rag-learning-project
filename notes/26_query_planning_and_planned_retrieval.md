# Query Planning 与 Planned Retrieval 优化记录

日期：2026-08-19

## 为什么要做这一步

上一阶段已经完成：

```text
documents -> chunks -> embeddings -> Chroma -> direct retrieval -> answer
```

但 direct retrieval 的问题是：它基本拿用户原话去做向量检索。

这种方式在简单问题上能用，但遇到下面情况会不稳定：

- 用户用中文问，资料主要是英文。
- 用户问题里有多个概念，例如 RAG + metadata filter。
- 用户问的是某个工程子领域，例如 chunking、evaluation、rerank。
- 向量检索被大而泛的 RAG overview 或论文内容抢走 top1。

所以这一阶段补上企业级 RAG 常见的 pre-retrieval 处理：

```text
用户问题 -> QueryPlan -> 多路检索 -> RRF 融合 -> 去重排序
```

## 新增文件

Ollama HTTP 工具：

```text
src/ollama_http.py
```

查询规划：

```text
src/query_planning.py
```

检索融合：

```text
src/retrieval.py
```

问答脚本升级：

```text
experiments/19_llm_rag_qa/ask.py
```

检索评估：

```text
experiments/20_planned_retrieval_eval/evaluate_retrieval.py
```

评估输出：

```text
eval/planned_retrieval_smoke/results.jsonl
eval/planned_retrieval_smoke/summary.md
```

## QueryPlan 做了什么

当前是 schema-driven fallback planner，不依赖外部 API。

它会输出：

```text
intent
category_filters
sub_queries
confidence
warnings
```

例如问题：

```text
metadata filter 在 RAG 检索中有什么作用？
```

规划结果：

```text
intent: answer
category_filters: vector db, retrieval
sub_queries:
1. 原始问题
2. 原始问题 + Chroma/vector database/metadata filter 扩展词
3. 原始问题 + retriever/self-query/metadata filtering 扩展词
```

这样做的意义是：系统不是只拿原话搜索，而是先把问题映射到知识库 schema，再生成更适合检索的查询。

## Planned Retrieval 怎么做

Planned retrieval 会执行多路检索：

```text
1. 原始 query，全库检索
2. 扩展 query，全库检索
3. 原始/扩展 query + category filter 检索
4. 对所有结果做 reciprocal rank fusion
5. 按融合分数去重排序
```

这样可以兼顾：

- 语义召回
- metadata 过滤
- 多查询扩展
- 去重融合

## direct vs planned 对比

评估命令：

```powershell
python experiments\20_planned_retrieval_eval\evaluate_retrieval.py --top-k 3 --candidate-k 6
```

评估集包含 8 个问题，覆盖：

- RAG overview
- chunking
- metadata filter
- evaluation
- embedding
- reranking
- local model
- ingestion

结果：

```text
total: 8
direct hit@1: 4/8 = 50%
planned hit@1: 8/8 = 100%
direct hit@3: 6/8 = 75%
planned hit@3: 8/8 = 100%
```

说明 planned retrieval 对这个知识库确实有效，尤其是下面几类问题：

- 文档切分为什么不能只用固定窗口？
- metadata filter 在 RAG 检索中有什么作用？
- 如何评估 RAG 回答是否忠实于检索上下文？
- rerank 和普通向量检索有什么关系？

这些问题 direct retrieval 容易被泛 RAG 内容或论文结果带偏，planned retrieval 能把结果拉回对应 category。

## Prompt 优化

问答 prompt 增加了领域规则：

```text
RAG 固定指 Retrieval-Augmented Generation，即检索增强生成。
不要把 RAG 解释成其他缩写。
如果检索资料没有支持某个判断，不要写这个判断。
```

原因：本地 `qwen2.5:1.5b` 在一次回答中把 RAG 错误解释成其他缩写。

这说明企业级 RAG 里需要区分两类问题：

```text
检索质量问题
生成忠实性问题
```

这次 planned retrieval 解决的是检索质量；prompt 规则是对生成忠实性的第一层约束。

## 当前结论

当前系统已经从 baseline RAG 升级到：

```text
schema-driven planned retrieval RAG
```

它已经具备：

- query planning
- category routing
- multi-query retrieval
- metadata filter
- RRF fusion
- evidence-based answer prompt
- retrieval smoke evaluation

## 下一步

下一步建议做：

```text
answer quality / faithfulness audit
```

也就是自动检查回答有没有：

- 引用来源
- 使用检索证据
- 编造资料外内容
- 答非所问
- 把术语解释错

这会把系统从“检索增强”继续推进到“答案可信度控制”。
