# PyPDFLoader Metadata 官方来源与安全刷新

日期：2026-08-25

## 目标

旧自然开发集中 `natural_dev_007` 和 `natural_dev_009` 都在询问 PyPDFLoader 是否保留页码，以及切分后 chunk 是否继续携带页码。此前候选池没有足够直接的官方证据。这是知识覆盖缺口，不应通过改 prompt、改 qrels 或覆盖冻结 holdout 来修复。

## 来源设计

新增两个 P0、优先入库的官方 GitHub 来源：

1. `langchain_pypdf_metadata`：固定到 `langchain-community` 的具体提交，只提取 `PyPDFParser.__init__` 与 `PyPDFParser.lazy_parse`。
2. `langchain_text_splitter_metadata`：固定到 `langchain` 的具体提交，只提取 `TextSplitter.create_documents` 与 `TextSplitter.split_documents`。

第一份源码证明默认 `mode="page"`、逐页 Document 的 `page`、`page_label`、`total_pages` 等 metadata；第二份源码证明 `split_documents` 将原 Document metadata 传入 `create_documents`，并为每个 chunk 深拷贝 metadata。

没有索引整份 PDF parser 文件。这样可排除相邻的 `PDFPlumberParser` 等实现，避免小模型把其他解析器细节误当成 PyPDFParser 行为。清单标题使用中英双语检索词，正文仍来自固定提交的官方源码。

## 安全刷新

`fetch_sources.py` 的定向刷新行为已收紧：

- `--source-id` 只控制本次下载对象，不再把 `documents.jsonl` 缩成局部语料。
- 聚合阶段始终包含全部符合条件的 P0、`ingest_first=yes` 来源。
- 任一预期来源缺失时拒绝替换旧 `documents.jsonl`。
- 新产物先写临时文件，再原子替换正式文件。
- `extract_symbol` 支持顶层类、嵌套方法和逗号分隔的多个 Python symbol；找不到目标或源码无法解析时直接失败。

## 构建结果

| 项目 | 更新前 | 更新后 |
|---|---:|---:|
| documents | 52 | 54 |
| chunks | 938 | 942 |
| Chroma rows | 938 | 942 |

新增资料最终形成 4 个高密度片段：2 个 PyPDFParser 片段和 2 个 TextSplitter metadata 片段。切分检查为 `too_long_chunks=0`、`tiny_chunks=0`。

候选索引先在 `llm_rag_chroma_pypdf_metadata_v1` 隔离构建和验证，通过后才晋升为默认 `llm_rag_chroma`。旧 938 条索引完整保留在本地 `llm_rag_chroma_pre_pypdf_938`，索引目录均不提交 Git。

## 检索与答案验收

使用网页默认参数 `direct + hybrid + lexical + top_k=7 + candidate_k=16`：

- `natural_dev_007`：`langchain_pypdf_metadata::chunk_0000` 排名第 1。
- `natural_dev_009`：两个 PyPDFParser 片段排名第 1、2；两个 TextSplitter 片段排名第 3、5。

真实 Web API 使用本地 `qwen2.5:1.5b` 生成答案。最终上下文同时包含 2 个 PyPDFParser 和 2 个 TextSplitter 片段，共使用 8668/9000 字。答案正确说明：

- PyPDFParser 默认是 `mode="page"`；
- page 模式自动写入 `page` 和 `page_label`；
- single 模式只有文档级 metadata，不提供逐页定位；
- 使用 `split_documents` 时，原 Document metadata 会传给每个 chunk，无需手工重复添加页码。

规则审计与 LLM 审计均通过，faithfulness 和 citation 均为 5/5。完整 Python 回归为 106/106 通过。

## 兼容边界与决定

- 未修改 `natural_dev_007`、`natural_dev_009` 的问题文本或历史评测结果。
- 未修改任何旧 qrels、候选池、冻结 holdout、source anchors 或发布结论。
- 本次结果证明知识库已补齐这两个问题的直接官方证据，不代表旧 benchmark 分数可被追溯改写。
- 该来源与索引版本通过验收，可以作为默认知识库使用。

下一项工程工作是实现通用的增量索引更新、来源版本记录、旧向量删除和缓存失效，不再依赖人工目录晋升。
