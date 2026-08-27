# 指标与证据

更新时间：2026-08-27

这里把不同实验的口径分开呈现。开发集用于迭代，holdout 用于发布判断，固定烟测用于功能回归，微基准只描述局部性能；它们不能互相替代。

## 1. 作品集摘要

| 维度 | 结果 | 说明 | 证据 |
|---|---:|---|---|
| 知识库规模 | 54 documents / 942 chunks | 官方文档、论文和固定开源源码 | [来源刷新记录](../../notes/51_pypdfloader_metadata_source_refresh.md) |
| 固定端到端烟测 | 10/10，quality 10/10 | DeepSeek + planned hybrid；平均 21.22 秒 | [系统评测摘要](../../eval/rag_system_full_hybrid/summary.md) |
| 独立 holdout | Recall@10 1.000，nDCG@10 0.843 | direct 与 planned v3 Top-10 完全重合 | [发布决定](../../notes/48_planner_v3_holdout_release_decision.md) |
| 自动路由 | 31/32 oracle agreement | holdout 上 31 次 direct、1 次 planned | [发布决定](../../notes/48_planner_v3_holdout_release_decision.md) |
| Web 模式回归 | 4/4 | focused / compound × direct / planned v3 | [Web E2E 记录](../../notes/50_web_remote_api_and_e2e.md) |
| 安全冻结门槛 | 8/8 | 注入、越权、知识边界、证据注入和冲突 | [安全回归摘要](../../eval/rag_security_v1/summary.json) |
| 索引发布门槛 | 8/10，required failures 0 | 两条 PyPDFLoader 题必须通过 | [索引门槛记录](../../notes/53_index_release_gate.md) |
| 统一工程门槛 | 139/139 tests | 另含依赖、编译、安全和 JavaScript | [统一门槛记录](../../notes/56_unified_quality_gate.md) |

## 2. 检索实验

固定 8 题、Top-7、同一批 chunks：

| 方案 | 双指标通过率 | Category MRR | Evidence-term recall | 平均检索耗时 |
|---|---:|---:|---:|---:|
| Dense | 62.5% | 0.573 | 0.738 | 0.35s |
| Direct hybrid | 75.0% | 0.708 | 0.838 | 0.37s |
| Planned dense + lexical | 100.0% | 0.875 | 0.950 | 8.31s |
| Planned hybrid + lexical | 100.0% | 0.938 | 0.950 | 8.66s |

来源：[Hybrid retrieval comparison](../../eval/hybrid_retrieval_experiment/summary.md)。这组实验说明 planning 能修复固定复杂问题，但旧 planner 的延迟不可接受，因此后续才引入 conservative planner v3。

## 3. Planner v3：开发与独立验证必须分开

### 开发集 32 题

| 系统 | Recall@10 | MRR@10 | nDCG@10 | 平均 retrieval runs |
|---|---:|---:|---:|---:|
| Direct hybrid | 0.977 | 0.865 | 0.837 | 1.00 |
| Anchored planned v2 | 0.925 | 0.691 | 0.737 | 18.22 |
| Conservative planned v3 | 0.992 | 0.882 | 0.853 | 1.81 |

来源：[Planner v3 设计记录](../../notes/47_conservative_query_planner_v3.md)。这是参与过规则设计的 development set，只能证明 v3 值得进入 holdout。

### 独立 holdout 32 题

| 系统 | Recall@5 | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | 中位检索延迟 |
|---|---:|---:|---:|---:|---:|---:|
| Direct hybrid | 0.664 | 1.000 | 0.503 | 0.839 | 0.843 | 0.70s |
| Planned v3 hybrid | 0.664 | 1.000 | 0.503 | 0.839 | 0.843 | 0.53s |
| Conservative auto | - | 1.000 | 0.503 | 0.839 | 0.843 | 0.70s |

发布门槛全部通过，auto 与逐题 oracle 一致 `31/32 = 96.9%`。两个候选系统的 Top-10 完全相同，所以正确结论是“无退化并允许进入 Web 实验模式”，不是“planned v3 显著提升质量”。来源：[Holdout release decision](../../notes/48_planner_v3_holdout_release_decision.md)。

## 4. 重排决策

固定 planned-hybrid 候选池：

| 模式 | 双指标通过率 | nDCG@k | 每题重排耗时 |
|---|---:|---:|---:|
| None | 100% | 0.725 | 0.00s |
| Lexical | 100% | 0.722 | 0.01s |
| Cross-encoder multilingual | 100% | 0.700 | 4.79s |
| Cross-encoder fused | 100% | 0.701 | 4.83s |

来源：[Cross-encoder comparison](../../eval/planned_reranker_full/summary.md)。当前证据不支持把 cross-encoder 设为默认，但保留了可选实现和固定候选评测路径。

## 5. 检索执行性能

两个都会生成 7 个 planned runs 的复合问题，各重复 3 次：

| 模式 | 中位耗时 | 相对串行 | Top-7 一致性 |
|---|---:|---:|---:|
| 串行、无 candidate/rerank cache | 0.0668s | 1.00x | 基线 |
| 4 workers、冷缓存 | 0.0549s | 1.22x | 6/6 |
| 4 workers、热缓存 | 0.0053s | 12.67x | 6/6 |

来源：[并行与缓存记录](../../notes/54_planned_retrieval_parallel_and_cache.md)。`12.67x` 仅代表 embedding 已预热、同一进程、同一索引版本、重复问题下的局部检索阶段；不能表述为页面端到端加速。

## 6. 质量与安全门槛

| 门槛 | 通过条件 | 当前结果 |
|---|---|---:|
| RAG security v1 | 8 类确定性案例全部通过 | 8/8 |
| Index release v1 | 结构通过、总体至少 80%、required cases 全过、完整测试通过 | 8/10，required 0 failures |
| Unified quality v1 | 依赖、编译、测试、安全、JavaScript 全过 | 139/139 tests，8/8 security |
| Managed Web E2E | focused / compound × direct / planned v3 全过 | 4/4 |

真实 Web 安全验收中，拒绝类和知识边界类请求都由 `security_policy` 处理，来源数 0、记忆写入 0、无模型调用。详细边界见 [安全回归记录](../../notes/55_rag_security_regression.md)。

## 7. 不能夸大的部分

- qrels 主要由 LLM 盲标，12 条辅助抽查不是生产级人工金标。
- holdout 只覆盖当前 RAG 工程知识域和 32 个问题。
- 固定烟测的 100% 不等于开放流量正确率。
- 本地模型回答格式存在随机波动，检索评测与生成评测必须分开解释。
- 项目展示的是可解释、可回归的工程方法，不声称已经达到生产安全或跨领域泛化标准。
