# 32. RAG端到端评测体系

## 为什么要做

前面我们已经把本地RAG项目做到了“能检索、能生成、能审计、能修复、能做长记忆”。但如果只靠网页上随手问几个问题，很难判断优化是真的变好，还是只是这一次刚好回答得顺眼。

所以这一步补一个端到端评测脚本：直接调用网页使用的 `/api/ask` 接口，用固定问题集反复测试系统。

## 评测覆盖什么

第一版数据集在 `eval/datasets/rag_system_smoke_eval.jsonl`，覆盖五类能力：

1. 复合问题：同时问RAG类别、关键技术、瓶颈，验证query planning和多aspect检索。
2. 技术细节：chunking、embedding、rerank、query expansion、metadata filter。
3. 可靠性：评估faithfulness、relevance、badcase等指标。
4. 长记忆：先写入学习偏好，再换新session提问，验证长期记忆是否可用。
5. 边界问题：问知识库没有的天气预测，验证系统是否会承认资料不足。

## 检查项

脚本会检查：

- response_ok：接口是否正常返回。
- quality：后端答案审计是否通过。
- aspects：规划出来的子问题是否覆盖预期方面。
- categories：检索来源是否命中预期知识类别。
- source_terms：检索证据里是否包含关键术语。
- answer_terms：最终答案是否覆盖关键表达。
- long_memory：记忆问答是否真的检索到长期记忆。
- insufficient_boundary：超出知识库范围的问题是否明确说资料不足。

## 运行方式

先保证Web服务已启动：

```powershell
conda activate rag-book
cd "C:\Users\Lenovo\Desktop\大模型官方课程-视频资料\学习产出\enterprise-rag-learning-project"
python webapp\server.py --host 127.0.0.1 --port 8765
```

再运行小规模评测：

```powershell
python experiments\22_rag_system_eval\evaluate_web_api.py --limit 4
```

完整评测：

```powershell
python experiments\22_rag_system_eval\evaluate_web_api.py
```

评测结果会输出到：

- `eval/rag_system_smoke/results.jsonl`
- `eval/rag_system_smoke/summary.json`
- `eval/rag_system_smoke/summary.md`

## 这一步的意义

这不是为了追求一次性全通过，而是建立项目的“仪表盘”。后面每次改chunk策略、top_k、重排、prompt、上下文组装、知识库内容，都可以重新跑这套评测，看哪些case变好，哪些case变差。

## 本次小样本运行结果

运行命令：

```powershell
python experiments\22_rag_system_eval\evaluate_web_api.py --limit 4
```

结果：

- total: 4
- passed: 3
- failed: 1
- pass_rate: 75%
- avg_seconds: 39.81
- 通过项：复合RAG概览、chunk切分、长期记忆偏好问答
- 失败项：RAG答案可靠性评估

失败原因：

`rag_evaluation_reliability` 的检索命中了 evaluation 资料，答案的忠实性、引用和相关性都拿到了高分，但 coverage audit 认为资料对“引用质量”和“badcase分析”的支撑不足。因此这不是单纯的生成问题，而是知识库内容覆盖不足。

下一步优化：

1. 补充更专业的RAG评估资料，重点覆盖 citation quality、badcase/error analysis、dataset design。
2. 更新知识库后重新跑 `rag_evaluation_reliability`，确认失败项是否变成通过。
3. 再扩大评测集，从4条小样本扩展到完整10条。

## 后续修复结果

在 `33_eval_source_and_retrieval_tuning.md` 中已完成修复：

- 补充 evaluation 资料源：RAGChecker、ALCE、Ragas、DeepEval、Phoenix。
- 重建知识库：50 documents / 920 chunks。
- 将 evaluation 问题拆成 `evaluation`、`badcase_analysis`、`citation_quality`。
- 修正 deterministic coverage audit 对新增 aspect 的识别。

最终 `rag_evaluation_reliability` 单case回归：

- passed: 1/1
- quality_pass: true
- failed: 0
