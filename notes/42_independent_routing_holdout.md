# 独立路由留出集

## 为什么要做

自动路由在最初 8 题上与逐题最优路线一致 8/8，但这些问题参与过规则阈值设计，只能算校准结果。为了验证泛化能力，本阶段先冻结 16 道新问题，再生成候选和标签。

## 实验协议

- 8 道 focused 问题，覆盖 RAG 概览、切分、embedding、Chroma、重排、评测和 Ollama。
- 8 道 compound 问题，覆盖入库、混合检索、评测设计、失败诊断、上下文和架构选择。
- 每题分别运行 `direct_hybrid` 和 `planned_hybrid`，各取 Top-10。
- 合并去重后得到 279 个 query/chunk 对。
- 候选顺序确定性打乱；DeepSeek 看不到系统、排名、分数、通道或 query plan。
- 相关性采用 0-3 分，`>=2` 视为相关。

## 结果

| system | Recall@10 | MRR@10 | nDCG@10 | median latency |
|---|---:|---:|---:|---:|
| direct hybrid | **0.862** | **0.812** | **0.750** | **0.83s** |
| planned hybrid | 0.448 | 0.498 | 0.465 | 4.62s |
| auto | 0.673 | 0.652 | 0.596 | 1.32s |

自动路由选择 9 次 direct、7 次 planned，与逐题 nDCG oracle 一致 10/16（62.5%）。

分层结果更加明确：focused 题的 direct/planned/auto nDCG 分别为 0.725/0.523/0.701；compound 题分别为 0.775/0.406/0.491。当前 planned 在它本应帮助的复合问题上反而退化更严重。

## 失败分型

1. **原问题信号被稀释**：planned 把原问题、通用 aspect 查询和分类扩展都送入 RRF；多个通用扩展会共同抬高“泛相关”片段。
2. **覆盖不等于相关**：`select_with_plan_coverage` 优先给规划分类分配位置，但错误或过宽的分类会把直接证据挤出 Top-10。
3. **复杂度不等于收益**：当前 router 预测的是 query plan 结构复杂度，没有估计 planned 相对 direct 的实际检索增益。
4. **典型 badcase**：架构选择题 direct nDCG 为 0.974，planned 为 0.000；planned Top-10 被 loading、evaluation、splitter、embedding 等通用组件片段占据，未保留 direct 找到的 RAG 架构证据。

## 工程决策

- 不隐藏失败结果，也不在这 16 题上调参后继续称它为 holdout。
- Web 默认路线改为 `direct`；`auto` 和 `planned` 暂时是实验功能。
- 下一阶段把这批失败样本作为 planned v2 的开发诊断材料。
- planned v2 应研究原问题加权锚点、扩展查询降权、覆盖槽位上限和基于检索置信度的二阶段路由。
- 完成开发后必须再冻结一套未见 holdout v2；当前 LLM 标签还需要抽样人工审计。

## 产物

- 问题集：`eval/datasets/rag_routing_holdout_v1.jsonl`
- 匿名候选清单：`eval/routing_holdout_v1/pool_manifest.jsonl`
- LLM qrels：`eval/benchmarks/rag_routing_holdout_v1/qrels_llm.jsonl`
- 汇总：`eval/auto_retrieval_routing_holdout_v1/summary.md`
