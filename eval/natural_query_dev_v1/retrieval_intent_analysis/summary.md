# Natural Query Intent vs Retrieval Quality

- cases: 32
- evaluable cases: 30
- candidate-pool coverage gaps: 2
- meaningful nDCG margin: 0.05
- benefit classes: {'comparable': 9, 'direct_better': 19, 'coverage_gap': 2, 'planned_better': 2}
- intended route / oracle agreement: 16/30 (53.3%)
- current auto / oracle agreement: 10/30 (33.3%)

## Intended vs Oracle

| intended / oracle | direct | planned |
|---|---:|---:|
| direct | 13 | 3 |
| planned | 11 | 3 |

## Quality by Intended Route

| intended | cases | evaluable | gaps | direct nDCG | planned nDCG | planned delta | benefit classes |
|---|---:|---:|---:|---:|---:|---:|---|
| direct | 18 | 16 | 2 | 0.811 | 0.712 | -0.099 | {'comparable': 6, 'direct_better': 10} |
| planned | 14 | 14 | 0 | 0.874 | 0.769 | -0.105 | {'planned_better': 2, 'direct_better': 9, 'comparable': 3} |

## Largest Planned Deltas

| case | intended | direct | planned | delta | class | question |
|---|---|---:|---:|---:|---|---|
| natural_dev_024 | planned | 0.858 | 0.980 | +0.122 | planned_better | 我们知识库每天会有新增和更新的文档，我现在用IngestionPipeline跑增量，但发现有时候重复导入会生成重复的向量，而更新文档时旧向量还在。缓存和文档ID应该怎么设计才能保证幂等？另外，清理缓存的时机是什么？ |
| natural_dev_021 | planned | 0.771 | 0.838 | +0.067 | planned_better | 我想评估一下我搭的RAG系统，现在有答案正确性、忠实度、引用质量这些指标，但不知道该怎么组合起来看。比如有些badcase是回答内容完全正确但引用错了，有些是引用对了但答案不全。我该怎么设计一套评测流程来系统地分析这些badcase？ |
| natural_dev_001 | direct | 0.825 | 0.858 | +0.033 | comparable | 我现在在做一个RAG项目，刚开始接触，不太明白向量数据库具体是干什么的，它和普通的数据库有什么区别？ |
| natural_dev_023 | planned | 0.905 | 0.929 | +0.024 | comparable | 我现在要做一个多知识库的RAG系统，不同团队有不同的文档库，我希望用户提问时能自动路由到正确的知识库。我打算用QueryPipeline里的router，但不知道router基于什么条件选择？是根据关键词还是语义？另外，如果一个问题涉及多个知识库，该怎么处理？ |
| natural_dev_014 | direct | 0.562 | 0.583 | +0.021 | comparable | 我往Chroma里写入了一批文档，但发现有些文档的向量没写进去，索引写入是不是有部分失败的？这种情况怎么排查？ |
| natural_dev_020 | direct | 0.946 | 0.955 | +0.010 | comparable | 我在用LangChain做RAG，想知道metadata filter具体是干什么用的？它是在检索前过滤还是检索后过滤？ |
| natural_dev_013 | direct | 1.000 | 1.000 | +0.000 | comparable | 我现在在管理Chroma里的多个collection，想知道怎么删除一个不再使用的collection，是直接调用delete_collection吗？ |
| natural_dev_037 | planned | 0.801 | 0.777 | -0.024 | comparable | 我们上线的RAG系统最近频繁出现答案不完整，明明知识库里有相关内容。我想排查是检索没召回到，还是召回了但排序靠后被截断了，或者是上下文选择时没用上。我该从哪些日志或指标入手一步步定位？另外，如果发现是重排问题，应该怎么调整？ |
| natural_dev_006 | direct | 0.631 | 0.431 | -0.200 | direct_better | 我的 Chroma 集合不小心删了，有没有办法恢复？还是只能重新入库？ |
| natural_dev_004 | direct | 0.708 | 0.504 | -0.203 | direct_better | 我在用Ollama跑本地模型做RAG，想调用它的embedding接口来生成向量，但不知道这个接口具体输入什么格式，输出的是什么样的向量？ |
| natural_dev_002 | direct | 0.817 | 0.609 | -0.208 | direct_better | 我在用Chroma，想给集合设置一个embedding函数，但不确定这个函数在写入和查询时分别会怎么影响我的向量，能给我讲讲吗？ |
| natural_dev_010 | direct | 0.904 | 0.694 | -0.210 | direct_better | 我在用Chroma做向量数据库，想按文档的部门字段过滤搜索结果，但不确定metadata过滤的语法怎么写，能告诉我具体怎么用吗？ |
| natural_dev_027 | planned | 0.941 | 0.725 | -0.216 | direct_better | 我搭了一个RAG系统，现在要评估答案的引用质量，发现有些答案中的每个引用都支持相邻陈述，但部分事实没有引用。这种情况下citation precision和citation recall会分别表现怎样？我该如何改进？ |
| natural_dev_022 | planned | 0.965 | 0.734 | -0.231 | direct_better | 我们的RAG系统现在用混合检索，dense和sparse结果合并后直接进入LLM，但发现有时候两个检索器返回的结果有重复，而且排序靠后的文档其实很相关，但被排到后面了。我想加入重排，但不知道应该选哪种重排模型，是ColBERT还是cross-encoder？另外重排后还需要再做上下文选择吗？ |
| natural_dev_029 | planned | 1.000 | 0.749 | -0.251 | direct_better | 我现在用Sentence-BERT做向量召回，但发现召回结果不够精确。我考虑用ColBERT做重排，但不太清楚为什么可以先用双塔向量召回再用late interaction重排？这种两阶段设计到底交换了什么？ |
| natural_dev_030 | planned | 0.531 | 0.192 | -0.340 | direct_better | 我们的知识库更新频繁，我担心增量更新时向量索引和metadata不一致，导致检索结果过期。我应该怎么设计摄取流水线，使得缓存、文档ID、去重和向量写入保持一致性？ |

## Boundary

This is development analysis. LLM-generated intent labels and LLM relevance qrels may share model biases. Observed retrieval deltas can guide development but require a new independent holdout before any release claim.
