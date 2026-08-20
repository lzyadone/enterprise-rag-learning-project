# 33. 评估资料补强与检索规划优化

## 背景

上一轮端到端评测中，`rag_evaluation_reliability` 失败：

```text
问题：如何评估RAG答案是否可靠？需要看哪些指标和badcase？
失败原因：coverage audit 认为 citation quality 和 badcase analysis 支撑不足。
```

这说明项目已经开始进入真实RAG迭代阶段：不是系统不能回答，而是知识库、检索规划、答案覆盖和审计规则之间需要对齐。

## 本轮做了什么

### 1. 补强资料源

在 `data/source_manifests/llm_rag_sources.csv` 中新增了 14 个 P0 evaluation 资料源，重点覆盖：

- RAGChecker：细粒度诊断、retriever/generator metrics、badcase分析。
- ALCE：引用质量、citation recall、citation precision、可验证性。
- Ragas：context precision、context recall、faithfulness。
- DeepEval：faithfulness、contextual precision/recall/relevancy。
- Phoenix：RAG evaluation、document relevance、faithfulness。

结果：

- 文档数：36 -> 50
- chunk数：647 -> 920
- evaluation文档数：6 -> 20
- Chroma indexed_count：920

### 2. 优化检索规划

仅补资料后，失败case仍未通过，因为检索虽然拿到了部分RAGChecker资料，但没有稳定召回 ALCE 的 citation quality chunk。

因此将 evaluation 类问题拆成更细的子方面：

- `evaluation`
- `badcase_analysis`
- `citation_quality`

这样问题“如何评估RAG答案是否可靠？需要看哪些指标和badcase？”会被规划成三个检索目标，而不是一个泛泛的 evaluation query。

### 3. 优化来源选择

在 `src/retrieval.py` 中为 evaluation 相关 aspect 增加证据词：

- citation quality / citation recall / citation precision
- claim-level / diagnostic metrics
- context precision / context recall / context utilization
- hallucination / noise sensitivity
- RAGChecker / ALCE

同时要求：

- `evaluation` 至少保留 3 条来源
- `badcase_analysis` 至少保留 2 条来源
- `citation_quality` 至少保留 2 条来源

### 4. 修正覆盖审计

检索和答案已经覆盖后，LLM coverage audit 通过，但 deterministic coverage audit 仍失败。

原因是规则审计还不认识新增的 `citation_quality` 和 `badcase_analysis` aspect。

因此在 `src/coverage_audit.py` 中补充中文关键词，让规则审计能识别：

- 引用质量、引用召回、引用精度、支撑、可验证
- badcase、诊断、检索错误、生成错误、上下文利用、噪声、幻觉

## 验证结果

最终运行：

```powershell
python experiments\22_rag_system_eval\evaluate_web_api.py --case-id rag_evaluation_reliability --output-dir eval\rag_system_smoke_after_coverage_fix
```

结果：

- total: 1
- passed: 1
- failed: 0
- pass_rate: 100%
- quality_pass: 100%
- sources: 10

关键证据来源包括：

- RAGChecker Paper
- ALCE Citation Evaluation Paper
- Ragas Metrics
- RAGChecker GitHub README
- Retrieval-Augmented Generation for Large Language Models: A Survey

## 学到什么

RAG badcase 的处理顺序应该是：

1. 判断知识库资料是否足够。
2. 判断检索是否召回了关键证据。
3. 判断上下文组装是否把关键证据放进prompt。
4. 判断答案是否覆盖了所有子问题。
5. 判断审计规则是否与新的问题拆分保持一致。

本轮失败不是一个单点问题，而是连续暴露出三层问题：

- 资料层：缺 citation quality / badcase analysis 专门资料。
- 检索层：泛 evaluation query 不稳定召回 ALCE/RAGChecker 关键chunk。
- 审计层：新增 aspect 后，规则审计需要同步更新。

这也是企业级RAG更真实的开发方式：通过 eval 找badcase，再用数据、检索、生成、审计的闭环逐步修。
