# Chunk、Top-k 与 Rerank 设计依据

日期：2026-08-19

## 目标

这份文档记录本项目为什么这样设置：

- chunk 怎么切
- chunk 边界怎么处理
- candidate_k 和 top_k 为什么分开
- 为什么要加 rerank
- 当前实现和成熟 RAG 系统还有什么差距

## 参考资料

本轮主要参考主流框架和服务的官方文档：

- LangChain Text Splitters: https://docs.langchain.com/oss/python/integrations/splitters
- LangChain Recursive Text Splitter: https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
- LlamaIndex Node Parsers: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/
- Haystack DocumentSplitter: https://docs.haystack.deepset.ai/docs/documentsplitter
- Chroma query and metadata filtering: https://docs.trychroma.com/docs/querying-collections/query-and-get
- Chroma metadata filtering: https://docs.trychroma.com/docs/querying-collections/metadata-filtering
- Cohere Rerank: https://docs.cohere.com/docs/reranking-with-cohere
- LlamaIndex Node Postprocessors: https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/
- Haystack Rankers: https://docs.haystack.deepset.ai/docs/rankers

## Chunk 不是只看窗口大小

成熟 RAG 系统通常不会只按固定字符数硬切。更合理的顺序是：

1. 优先利用文档结构，例如 Markdown 标题、HTML 标题、JSON 字段、PDF 页码。
2. 再利用自然语言边界，例如段落、句子、标点。
3. 最后才用字符数或 token 数做安全上限，避免单个 chunk 太长。

原因：

- 固定窗口可能把一个概念切断，检索时只召回半句话。
- 太大的 chunk 会带入噪音，影响向量匹配和最终回答。
- 太小的 chunk 会丢上下文，模型需要的定义、条件、例子可能分散在多个片段里。
- 中文、日文、泰文等语言没有稳定空格分词时，要特别注意标点和句子边界。

## 当前项目的 chunk 策略

当前实现文件：

```text
src/chunking.py
```

策略：

```text
Markdown 标题 -> section -> 段落 -> 句子 -> hard cap
```

默认参数：

```text
soft_max_chars = 1800
hard_max_chars = 3500
min_chars = 280
```

这里的窗口不是主逻辑，而是安全边界：

- `soft_max_chars`：段落累计到这个范围附近就倾向于切开。
- `hard_max_chars`：超过这个值必须继续切，避免上下文爆掉。
- `min_chars`：太短的 chunk 会和相邻 chunk 合并，避免碎片化。

最新结果：

```text
documents: 32
chunks: 487
char_count min/p50/p90/max: 288 / 1213 / 2697 / 3647
too_long_chunks: 0
tiny_chunks: 0
```

## Top-k 要拆成 candidate_k 和 top_k

真实 RAG 里通常要区分两个 k：

```text
candidate_k: 第一阶段召回多少候选
top_k: 最终放进 prompt 的证据数量
```

原因：

- 第一阶段向量召回不一定排序最准，所以要多拿一些候选。
- LLM 上下文不是越多越好，太多会增加噪音和成本。
- rerank 需要候选集，如果 candidate_k 太小，rerank 没有发挥空间。

当前默认：

```text
candidate_k = 12
top_k = 5
```

评估脚本默认用：

```text
candidate_k = 12
top_k = 3
```

这样可以先看检索质量，再控制最终上下文规模。

## Rerank 放在向量召回之后

Rerank 的位置：

```text
query -> embedding retrieval -> candidates -> rerank -> final context
```

参考 Cohere、LlamaIndex、Haystack 的设计，rerank 通常不是替代向量检索，而是放在检索之后重排候选文档。

当前项目先实现轻量本地版：

```text
src/reranking.py
```

它综合：

- 第一阶段检索分
- query 和 chunk 正文的术语命中
- query 和标题/分类的命中
- 常见中英文术语别名，例如“切分/chunk/splitter”“重排/rerank/ranker”

这不是最终企业级 reranker，但它先把系统结构搭起来。以后可以替换成：

- BGE reranker
- Cohere Rerank
- Jina Reranker
- cross-encoder reranker
- LLM reranker

## 多意图问题需要分类覆盖

这类问题很常见：

```text
文档切分为什么不能只用固定窗口？top-k 和 rerank 应该怎么设置？
```

它同时涉及：

- chunking
- retrieval/top-k
- reranking

如果只按综合分排序，最终上下文可能被某一类资料占满。当前 planned retrieval 增加了分类覆盖：

```text
保留综合排序第一条 -> 尽量补齐 QueryPlan 规划出的类别 -> 再按分数填满 top_k
```

这样回答会更完整，也更接近企业知识库里处理复合问题的方式。

## 当前推荐流程

```text
用户问题
-> QueryPlan
-> 多路检索：原始 query + 扩展 query + category filter
-> RRF 融合
-> lexical rerank
-> category coverage
-> top_k context
-> 带引用回答
```

当前命令：

```powershell
python experiments\19_llm_rag_qa\ask.py --query "你的问题" --retrieval-mode planned --rerank-mode lexical --top-k 5 --candidate-k 12
```

评估命令：

```powershell
python experiments\20_planned_retrieval_eval\evaluate_retrieval.py --top-k 3 --candidate-k 12 --rerank-mode lexical --output-dir eval\planned_retrieval_smoke_rerank
```

## 当前边界

现在还没有做到真正企业级 rerank，因为：

- lexical rerank 只是轻量启发式，不是训练好的相关性模型。
- 评估集还是 smoke eval，问题数量少。
- 还没有加入 answer faithfulness 自动评估。
- 还没有增量更新、版本化索引、权限过滤和外部搜索。

但当前结构已经比普通 demo 更接近真实系统，因为它具备：

- 高质量资料准入
- 结构化 chunk
- metadata
- query planning
- candidate_k / top_k 分离
- rerank
- category coverage
- 检索评估
