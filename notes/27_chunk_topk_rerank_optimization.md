# Chunk / Top-k / Rerank 优化记录

日期：2026-08-19

## 这轮做了什么

目标：

```text
让 RAG 项目的切分、top-k、重排更接近真实系统，而不是只跑通 demo。
```

完成内容：

1. 查阅主流框架/服务官方资料。
2. 补充 chunking 和 reranking 相关资料源。
3. 重新生成 documents、chunks、Chroma 向量索引。
4. 新增轻量本地 rerank。
5. planned retrieval 增加 category coverage。
6. 重新跑检索评估。

## 新增资料

在资料清单中新增 5 个 P0 来源：

```text
llamaindex_node_parsers
haystack_document_splitter
llamaindex_node_postprocessors
cohere_rerank_docs
haystack_rankers
```

原因：

- 原知识库里 chunking 资料偏少。
- reranking 资料主要是 ColBERT 项目说明，工程解释不够直接。
- Cohere、Haystack、LlamaIndex 更适合解释“检索后重排”的实际系统位置。

## 数据规模变化

重新抓取后：

```text
documents: 32
```

重新切分后：

```text
chunks: 487
char_count:
  min: 288
  p50: 1213
  p90: 2697
  max: 3647
  avg: 1361.64
too_long_chunks: 0
tiny_chunks: 0
```

重新建索引后：

```text
indexed_count: 487
elapsed_seconds: 234.36
```

## 代码变化

新增：

```text
src/reranking.py
docs/11_chunk_topk_rerank_research.md
notes/27_chunk_topk_rerank_optimization.md
```

修改：

```text
data/source_manifests/llm_rag_sources.csv
src/retrieval.py
experiments/19_llm_rag_qa/ask.py
experiments/20_planned_retrieval_eval/evaluate_retrieval.py
```

## 检索链路变化

之前：

```text
query -> embedding search -> top_k -> prompt
```

现在：

```text
query
-> QueryPlan
-> original / expanded / category-filtered retrieval
-> RRF fusion
-> lexical rerank
-> category coverage
-> top_k context
-> prompt
```

## 参数变化

问答默认：

```text
candidate_k: 12
top_k: 5
rerank_mode: lexical
```

评估默认：

```text
candidate_k: 12
top_k: 3
rerank_mode: lexical
```

为什么这样设：

- `candidate_k` 大一点，给 rerank 留候选空间。
- `top_k` 小一点，控制 prompt 噪音。
- 先用 lexical rerank 保证本地可跑，后面可替换成模型 rerank。

## Badcase 与修复

人工预览问题：

```text
Why should RAG chunking not only use fixed windows, and how should top-k and reranking be set?
```

发现：

- 没有 category coverage 时，结果容易被 reranking 资料占满。
- 加入 category coverage 后，结果同时包含 Cohere Rerank 和 Haystack DocumentSplitter。

这说明多意图问题需要覆盖多个知识类别，而不是只看单一综合排序。

另一个修复：

- 某些网页资料带零宽空格，Windows 控制台 GBK 输出会报错。
- 已在问答和评估脚本中设置 UTF-8 输出，避免 demo 中断。

## 评估结果

命令：

```powershell
python experiments\20_planned_retrieval_eval\evaluate_retrieval.py --top-k 3 --candidate-k 12 --rerank-mode lexical --output-dir eval\planned_retrieval_smoke_rerank
```

结果：

```text
total: 8
direct hit@1: 5/8 = 62.5%
planned hit@1: 8/8 = 100%
direct hit@3: 7/8 = 87.5%
planned hit@3: 8/8 = 100%
```

结论：

- direct retrieval 仍然会被泛 RAG 或论文内容带偏。
- planned retrieval + rerank + category coverage 在 smoke eval 上保持稳定。
- 当前评估还不能证明企业级效果，只能说明这一版检索策略比 direct baseline 更稳。

## 实际问答检查

命令：

```powershell
python experiments\19_llm_rag_qa\ask.py --query "How should top-k and reranking be set in a RAG system?" --retrieval-mode planned --rerank-mode lexical --top-k 5 --candidate-k 12
```

观察：

- 检索阶段能找到 Cohere Rerank 和 LangChain RAG architecture 资料。
- 生成阶段能给出中文回答。
- 但本地小模型对 top-k 的解释仍偏泛，来源列表也没有完全按 prompt 要求输出。

结论：

```text
检索链路已经可继续推进，下一步重点应从 retrieval quality 转到 answer faithfulness。
```

## 下一步建议

下一步不要继续盲目堆资料，建议做：

```text
answer quality / faithfulness audit
```

也就是检查答案是否：

- 使用了检索上下文
- 给了来源编号
- 没有编造资料外事实
- 没有答非所问
- 能在资料不足时拒答

这是从“检索更准”走向“答案可信”的下一步。
