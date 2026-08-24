# Natural Query Routing Development Analysis

- cases: 32
- intended routes: {'direct': 18, 'planned': 14}
- selected routes: {'direct': 15, 'planned': 17}
- route intent agreement: 15/32 (46.9%)
- direct retention: 44.4%
- planned detection: 50.0%

## Confusion Matrix

| intended / selected | direct | planned |
|---|---:|---:|
| direct | 8 | 10 |
| planned | 7 | 7 |

## Mismatches

| case | intended | selected | score | reasons | question |
|---|---|---|---:|---|---|
| natural_dev_002 | direct | planned | 3 | complexity_threshold_reached, multi_category_expansion_without_aspects | 我在用Chroma，想给集合设置一个embedding函数，但不确定这个函数在写入和查询时分别会怎么影响我的向量，能给我讲讲吗？ |
| natural_dev_003 | direct | planned | 3 | complexity_threshold_reached, aspect_detected_with_low_category_confidence | 重排在RAG里到底是干什么的？我现在的检索结果已经出来了，为什么还要再排一次序？ |
| natural_dev_004 | direct | planned | 4 | complexity_threshold_reached, multiple_answer_aspects | 我在用Ollama跑本地模型做RAG，想调用它的embedding接口来生成向量，但不知道这个接口具体输入什么格式，输出的是什么样的向量？ |
| natural_dev_005 | direct | planned | 3 | complexity_threshold_reached, multi_category_expansion_without_aspects | 我刚开始学RAG，想知道文档切分这一步具体是做什么的，为什么不能直接把整个文档扔给模型？ |
| natural_dev_006 | direct | planned | 3 | complexity_threshold_reached, multi_category_expansion_without_aspects | 我的 Chroma 集合不小心删了，有没有办法恢复？还是只能重新入库？ |
| natural_dev_007 | direct | planned | 3 | complexity_threshold_reached, multi_category_expansion_without_aspects | 我现在要把一个 PDF 文档加载进 RAG，用的是 LangChain，想知道它的 PDFLoader 会保留页码信息吗？还是需要自己额外处理？ |
| natural_dev_014 | direct | planned | 3 | complexity_threshold_reached, multi_category_expansion_without_aspects | 我往Chroma里写入了一批文档，但发现有些文档的向量没写进去，索引写入是不是有部分失败的？这种情况怎么排查？ |
| natural_dev_016 | direct | planned | 4 | complexity_threshold_reached, multiple_answer_aspects | 我现在要评估RAG系统的效果，想知道faithfulness这个指标具体是检查答案的什么性质？我该怎么理解它？ |
| natural_dev_018 | direct | planned | 4 | complexity_threshold_reached, multiple_answer_aspects | 我现在在调RAG的评测指标，想知道contextual precision具体是衡量什么的？它和普通的precision有什么不一样？ |
| natural_dev_019 | direct | planned | 3 | complexity_threshold_reached, multi_category_expansion_without_aspects | 我在用Chroma做向量检索，但发现有时候查出来的结果和我想的不太一样，我想了解一下vector database里的collection到底是个什么概念？ |
| natural_dev_021 | planned | direct | 4 | estimated_planned_latency_exceeds_budget, multiple_answer_aspects | 我想评估一下我搭的RAG系统，现在有答案正确性、忠实度、引用质量这些指标，但不知道该怎么组合起来看。比如有些badcase是回答内容完全正确但引用错了，有些是引用对了但答案不全。我该怎么设计一套评测流程来系统地分析这些badcase？ |
| natural_dev_022 | planned | direct | 0 | simple_or_specific_query | 我们的RAG系统现在用混合检索，dense和sparse结果合并后直接进入LLM，但发现有时候两个检索器返回的结果有重复，而且排序靠后的文档其实很相关，但被排到后面了。我想加入重排，但不知道应该选哪种重排模型，是ColBERT还是cross-encoder？另外重排后还需要再做上下文选择吗？ |
| natural_dev_023 | planned | direct | 0 | simple_or_specific_query | 我现在要做一个多知识库的RAG系统，不同团队有不同的文档库，我希望用户提问时能自动路由到正确的知识库。我打算用QueryPipeline里的router，但不知道router基于什么条件选择？是根据关键词还是语义？另外，如果一个问题涉及多个知识库，该怎么处理？ |
| natural_dev_024 | planned | direct | 0 | simple_or_specific_query | 我们知识库每天会有新增和更新的文档，我现在用IngestionPipeline跑增量，但发现有时候重复导入会生成重复的向量，而更新文档时旧向量还在。缓存和文档ID应该怎么设计才能保证幂等？另外，清理缓存的时机是什么？ |
| natural_dev_025 | planned | direct | 0 | simple_or_specific_query | 我在用LlamaIndex搭建RAG，一个问题进来后经过retriever、node postprocessor和response synthesizer，但我不是很清楚它们各自具体负责什么，尤其是postprocessor在什么位置，它到底能对检索出来的节点做哪些处理？如果我想在重排后过滤掉不相关的节点，应该在哪一步做？ |
| natural_dev_030 | planned | direct | 0 | simple_or_specific_query | 我们的知识库更新频繁，我担心增量更新时向量索引和metadata不一致，导致检索结果过期。我应该怎么设计摄取流水线，使得缓存、文档ID、去重和向量写入保持一致性？ |
| natural_dev_031 | planned | direct | 0 | simple_or_specific_query | 我在用Chroma存向量，现在需要按文档的metadata字段过滤，比如‘部门’等于‘销售部’。但我不确定metadata过滤的语法，而且如果过滤条件很复杂，比如多个条件组合，应该怎么写？另外，过滤是在向量检索之前还是之后进行？ |

## Boundary

Agreement measures whether the rule router recognizes the generator's intended complexity. It does not show which retrieval path has better evidence quality; that requires pooled qrels.
