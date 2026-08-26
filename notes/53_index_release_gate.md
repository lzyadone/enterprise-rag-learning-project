# 知识库索引离线发布门禁

## 目的

版本化增量索引解决了如何安全构建、切换和回滚，但上一阶段仍需要人工依次运行结构验证、固定问题检索和完整测试。本阶段将这些检查收敛为一个离线发布门禁，避免漏跑步骤，也让每次索引更新都有可追溯报告。

## 门禁组成

入口：`experiments/34_index_release_gate/run_gate.py`

一次运行包含三个必须通过的阶段：

1. `structure`：验证 manifest 状态和哈希、来源状态、chunks 内容、Chroma 文本、关键 metadata、ID 与数量；
2. `retrieval`：按冻结规格运行 direct + hybrid + lexical 检索；
3. `tests`：使用项目既有 `unittest discover` 入口运行完整 Python 回归。

任一阶段失败时，整体状态为 failed。指定 `--activate` 也不会绕过门禁，激活状态会记录为 blocked。

## 冻结检索规格

规格位于 `eval/benchmarks/rag_index_release_gate_v1/gate.json`，并在报告中记录 SHA-256。它包含：

- 8 条既有跨组件烟测题，覆盖 RAG overview、chunking、evaluation、embedding、reranking、query planning、Chroma 和 enterprise retrieval；
- 2 条 PyPDFLoader metadata 覆盖修复题。

历史 direct hybrid 基线在前 8 题上是 75%，因此本门禁没有把已知失败伪装成逐题全过。加入两条必须通过的 PyPDF 题后，总门槛冻结为 `8/10 = 80%`，并额外要求：

- 页码 metadata 题的官方 PyPDF chunk 必须排第 1；
- chunk metadata 继承题的 4 条直接证据必须全部进入前 5；
- 任一 required case 失败都会阻止发布，即使总体通过率仍达标。

## 使用方式

只运行门禁并生成报告，不切换索引：

```powershell
python experiments\34_index_release_gate\run_gate.py `
  --manifest data\indexes\llm_rag_versions\index-YYYYMMDD\manifest.json
```

全部通过后，在同一次受控流程中启用：

```powershell
python experiments\34_index_release_gate\run_gate.py `
  --manifest data\indexes\llm_rag_versions\index-YYYYMMDD\manifest.json `
  --activate
```

报告默认写入 `data/runtime/index_release_gate/<version>-<UTC timestamp>/`：

- `report.json`：供自动化读取；
- `report.md`：供人工审阅。

报告只记录版本、哈希、计数、检查结果和返回 chunk ID，不保存文档正文、模型回答或凭据。激活前先原子写入 approved 报告，随后才切换指针；激活完成后再把状态更新为 activated。

## 真实验收

对不可变候选 `validation-copy-20260825` 运行门禁：

- structure：passed；
- retrieval：8/10，pass rate `0.8`，required failures `0`；
- tests：116 passed；
- dry run：passed，未激活；
- `--activate`：passed 并成功切换；
- 随后 rollback 成功恢复 `baseline-20260825-942`；
- 最终本地索引保持 54 documents / 942 chunks / 942 Chroma rows。

两个未通过的非强制题仍是既有 `rag_compound_overview` 和 `enterprise_rag_retrieval` 类别覆盖限制，与历史 direct hybrid 75% 结论一致，不是本阶段回归。

## 边界与下一步

- 门禁验证检索和代码回归，不调用随机性较高的答案生成模型；生成质量仍由独立 Web E2E 流程评估。
- v1 规格针对当前 RAG 学习知识库。新增业务域时应创建新版本规格，不覆盖 v1 历史门槛。
- 下一项工程优化转为并行执行 planned retrieval 的独立检索请求，并评估候选/重排缓存。
