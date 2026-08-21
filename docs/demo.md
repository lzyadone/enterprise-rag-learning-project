# Demo Cases

本页整理最新端到端评测中的代表性问答，用来展示项目效果。对应评测基线见：

- `eval/rag_system_full_after_chroma_query_docs/summary.md`
- `notes/34_component_scoped_planning_and_chroma_docs.md`

## Evaluation Snapshot

```text
total: 10
passed: 10
failed: 0
quality_pass_rate: 100%
retrieval_mode: planned
rerank_mode: lexical
top_k/candidate_k: 7/16
max_context_chars: 15000
```

## Case 1: 综合问题拆解

**Question**

```text
RAG系统分为哪些类别，有哪些关键技术，主要瓶颈有哪些？
```

**Planning Result**

- intent: `answer`
- categories: `RAG overview`, `RAG challenges`
- aspects: `classification`, `techniques`, `bottlenecks`
- sources: 10

**What It Shows**

这个问题不是单点问答，而是复合问题。系统会先拆出“分类、关键技术、瓶颈”三个必答方面，再围绕不同方面召回资料，避免只回答其中一小块。

**Answer Summary**

答案覆盖了 2-Step RAG、Agentic RAG、Hybrid RAG、Modular RAG，也解释了语义分块、混合检索、BGE-M3、重排、上下文组装和 RAG failure points。

**Representative Sources**

- LangChain Retrieval
- Retrieval-Augmented Generation for Large Language Models: A Survey
- Seven Failure Points When Engineering a Retrieval Augmented Generation System
- BGE-M3 Model Card
- Haystack Rankers

## Case 2: Chunking 不是固定窗口

**Question**

```text
文档切分为什么不能只用固定窗口？企业级RAG通常会考虑哪些边界？
```

**Planning Result**

- intent: `explain`
- categories: `chunking`, `RAG overview`
- sources: 7

**What It Shows**

系统能解释为什么真实 RAG 不应该只按固定长度切分，而要考虑 Markdown 标题、HTML 标签、JSON 对象、段落、句子和语义连续性。

**Answer Summary**

答案强调结构化边界和语义边界比固定窗口更适合企业知识库，并指出 Hybrid RAG 中还会加入查询增强、检索验证和答案验证。

**Representative Sources**

- LangChain Retrieval
- LangChain Text Splitter Integrations
- Haystack DocumentSplitter

## Case 3: Query Rewrite / Query Expansion

**Question**

```text
复杂问题在RAG里为什么要做query rewrite或query expansion？
```

**Planning Result**

- intent: `explain`
- categories: `querying`, `retrieval`
- aspects: `query_optimization`
- sources: 7

**What It Shows**

系统不是直接拿原问题做一次向量搜索，而是识别出这是 query optimization 问题，补充 query rewrite、query expansion、query routing、sub-query 等检索增强方向。

**Answer Summary**

答案说明 query rewrite 可以把原始问题改写成更适合检索的表达，query expansion 可以生成多个并行查询，query routing 可以把问题路由到更合适的 pipeline 或 metadata filter。

**Representative Sources**

- LlamaIndex Advanced Retrieval
- LlamaIndex Querying
- Retrieval-Augmented Generation for Large Language Models: A Survey

## Case 4: Chroma 和 Metadata Filter

**Question**

```text
Chroma在这个RAG项目里承担什么角色？metadata filter有什么用？
```

**Planning Result**

- intent: `answer`
- categories: `vector db`, `retrieval`
- aspects: `techniques`
- sources: 7

**What It Shows**

这个问题验证了向量数据库层是否讲得清楚。系统会检索 Chroma 官方文档，而不是只泛泛说“Chroma 是向量库”。

**Answer Summary**

答案说明 Chroma 在项目中承担向量数据库和检索基础设施角色，用于存储 documents、embeddings 和 metadata。`metadata filter` 通过 `where` 条件在查询时过滤记录，缩小检索范围，也可以和 `where_document` 配合使用。

**Representative Sources**

- Chroma Docs
- Chroma Query and Get
- Chroma Metadata Filtering

## Case 5: 企业级检索策略

**Question**

```text
企业级RAG为什么需要混合检索、metadata过滤和重排，而不是只做一次向量相似度搜索？
```

**Planning Result**

- intent: `explain`
- categories: `retrieval`, `vector db`
- aspects: `techniques`
- sources: 7

**What It Shows**

这是本阶段重点优化的 badcase。系统现在会把它识别为“企业级检索策略”问题，而不是泛泛的 RAG 技术列表问题。

**Answer Summary**

答案把三类能力分别对应到问题：

- 混合检索：提升召回覆盖和泛化能力。
- metadata filter：用结构化条件缩小候选范围。
- 重排：对初步召回的候选做更精细排序。
- 单路向量检索风险：容易受 embedding 质量、查询歧义、领域词汇和精确过滤需求影响。

**Representative Sources**

- Chroma Query and Get
- Chroma Metadata Filtering
- BGE-M3 Model Card
- Haystack Rankers
- Seven Failure Points When Engineering a Retrieval Augmented Generation System

## Case 6: 知识边界

**Question**

```text
根据当前知识库，RAG能不能直接预测明天杭州天气？为什么？
```

**Planning Result**

- intent: `explain`
- categories: `RAG overview`
- sources: 7

**What It Shows**

这个问题验证系统是否知道知识边界。RAG 不应该把静态知识库当成实时天气服务，也不应该编造未来天气。

**Answer Summary**

答案指出当前知识库不能直接预测明天杭州天气，因为天气预报需要实时气象数据。若要回答这类问题，应接入外部天气 API 或工具调用；RAG 只是检索和组织已有信息，不等于自身具备预测能力。

**Representative Sources**

- LangChain Retrieval
- Retrieval-Augmented Generation for Large Language Models: A Survey

## Why These Demos Matter

这些 demo 覆盖了一个成熟 RAG 项目最容易被追问的几个点：

- 复杂问题能不能拆解？
- chunking 是否有边界意识？
- 检索前是否理解用户问题？
- 向量库和 metadata filter 是否讲得清楚？
- 企业级检索为什么不只做一次向量搜索？
- 超出知识库范围时是否会拒绝编造？

当前项目还不是最终形态，但这些案例已经能说明：它不是一个只会“向量搜索 + 拼 prompt”的简单 Demo，而是具备数据、检索、上下文、生成、审计和记忆链路的 RAG 学习项目。
