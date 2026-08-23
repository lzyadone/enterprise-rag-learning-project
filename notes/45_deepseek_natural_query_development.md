# DeepSeek Natural Query Development

## 目标

前两轮评测问题由人工按知识点编写，表达更接近课程题目。真实用户往往会带着当前场景、已有工具、异常现象和不完整猜测提问。新开发集让 DeepSeek 模拟不同角色提问，用来暴露当前规则 planner/router 对自然语言复杂度的误判。

这不是新的 holdout。问题生成、清洗和后续路由开发都发生在同一阶段，因此它只能作为开发数据。

## 生成协议

生成范围限定在现有 938-chunk 大模型/RAG 知识库能够覆盖的主题，但不给模型资料标题或标准答案。角色包括：

- 刚入门的学习者；
- 正在接入知识库的开发者；
- 排查线上问题的工程师；
- 负责方案选型的技术负责人；
- 维护内部资料的数据工程师；
- 负责效果验收的测试或产品人员。

问题先按 intended complexity 生成：direct 只有一个主要信息需求，planned 至少有两个需要独立检索再组合的信息需求。该标签描述生成意图，不是检索质量真值。

每条问题经过三层检查：

1. 规则检查长度、字段、考试腔、目标标签泄漏和字符近似重复。
2. bge-m3 检查与旧数据集及本轮问题的语义相似度。
3. DeepSeek 在看不到 intended route 的情况下复审自然度、独立可理解性和实际信息需求数量。

生成过程还增加了批次 checkpoint。某个严格批次失败后，已经完成的批次和批次内部分进度可以继续使用，不会重复消耗 API。

## 为什么没有硬凑 40 条

固定要求“10 条单组件排错题”时，大部分候选被盲审判为 planned，因为真实排错自然会同时怀疑多个环节。固定要求第 20 条 planned 时，新增候选连续落入已有语义模板。

最终保留 38 条通过生成审查的原始问题，而不是降低标准凑整数。pairwise bge-m3 检查又删除了 6 条同路由重复意图：

| removed | kept | route | cosine |
|---|---|---|---:|
| natural_dev_015 | natural_dev_012 | direct | 0.8672 |
| natural_dev_017 | natural_dev_016 | direct | 0.8666 |
| natural_dev_034 | natural_dev_026 | planned | 0.8790 |
| natural_dev_035 | natural_dev_024 | planned | 0.9241 |
| natural_dev_036 | natural_dev_029 | planned | 0.8789 |
| natural_dev_038 | natural_dev_037 | planned | 0.8875 |

最终开发集为 32 条：direct 18 条、planned 14 条。相似但 intended route 不同的问题被保留，因为它们可以检验路由器能否区分“相同技术主题、不同任务复杂度”。

## 当前规则路由结果

| intended / selected | direct | planned |
|---|---:|---:|
| direct | 8 | 10 |
| planned | 7 | 7 |

- intent agreement：15/32，46.9%；
- direct retention：44.4%；
- planned detection：50.0%；
- 当前规则选择 direct 15 次、planned 17 次。

过度规划主要来自 `multi_category_expansion_without_aspects` 和关键词触发的多个 aspect。漏规划问题则经常带有真实业务约束，但没有命中固定触发词；另有一题因 estimated planned latency 超过 12 秒而被强制降级 direct。

## 解释边界

这个结果只说明当前规则能否识别生成器设定的信息需求复杂度。它没有运行 direct/planned 检索，也没有 qrels，因此不能证明 planned 在这些题上质量更高，更不能据此修改线上默认。

下一步应对 32 条问题生成 direct 与 anchored planned v2 的匿名 union pool，完整标注候选相关性，再比较逐题 nDCG oracle 与 intended route。只有两者结合，才能决定路由层应该学习“问题复杂度”还是直接预测“哪条检索路径更可能受益”。

## 产物

- 生成器：`experiments/30_natural_query_development/generate_queries.py`
- 清洗器：`experiments/30_natural_query_development/curate_queries.py`
- 路由分析：`experiments/30_natural_query_development/analyze_routing.py`
- 原始问题：`eval/datasets/rag_natural_query_dev_v1_raw.jsonl`
- 清洗问题：`eval/datasets/rag_natural_query_dev_v1.jsonl`
- 生成摘要：`eval/natural_query_dev_v1/generation_summary.md`
- 清洗摘要：`eval/natural_query_dev_v1/curation_summary.md`
- 路由摘要：`eval/natural_query_dev_v1/routing_analysis/summary.md`
