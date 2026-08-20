# LLM/RAG 知识库资料抓取记录

日期：2026-08-19

## 今天做了什么

把项目主线从金融财报 RAG 调整为：

```text
大模型工程知识库 RAG
```

第一版聚焦：

```text
RAG 工程知识库
```

并完成了第一版资料源准备：

- 建立高质量资料 manifest。
- 新增资料抓取脚本。
- 用前 3 个 P0 资料源完成小样本验证。
- 加入网页噪音清洗。
- 优先抓取官方 Markdown 或 arXiv HTML，避免网页导航噪音和 PDF 抽取粘连。

## 为什么先做资料层

RAG 的效果不是只靠模型，也不是只靠 prompt。

如果知识库资料质量差，后面会出现：

- 检索结果全是网页导航、广告、页脚。
- chunk 被无意义内容污染。
- LLM 引用错误来源。
- 评估结果看起来能跑，但没有真实价值。

所以第一步必须先把资料源、清洗、metadata 做好。

## 已完成文件

资料规划：

```text
docs/10_llm_rag_knowledge_base_source_plan.md
```

资料清单：

```text
data/source_manifests/llm_rag_sources.csv
```

抓取脚本：

```text
experiments/16_llm_rag_sources/fetch_sources.py
```

小样本输出：

```text
data/raw/llm_rag_docs/langchain_retrieval/document.md
data/raw/llm_rag_docs/llamaindex_rag_concepts/document.md
data/raw/llm_rag_docs/rag_paper/document.md
data/processed/llm_rag_docs/documents.jsonl
```

## 小样本验证结果

小样本包含 3 个源：

1. LangChain Retrieval
2. LlamaIndex RAG Concepts
3. Retrieval-Augmented Generation 原始论文

验证结果：

```text
records: 3
langchain_retrieval: 14055 chars
llamaindex_rag_concepts: 5284 chars
rag_paper: 79395 chars
```

网页噪音检查：

```text
Skip to / Search / Copy page / On this page / Was this page helpful: 0 hits
```

说明：这 3 个源已经可以作为后续 chunk 和向量化的小样本。

## 全量下载结果

用户确认联网后，已经完成 27 个 P0 资料源的下载、转换和汇总。

最终输出：

```text
data/processed/llm_rag_docs/documents.jsonl
```

质量检查结果：

```text
records: 27
total_chars: 591979
too_short: []
fetch errors: []
navigation noise hits: 0
```

按知识点分类：

```text
RAG overview: 2
RAG paper: 1
document loading: 1
chunking: 2
ingestion: 2
indexing: 1
querying: 1
vector db: 3
local model: 2
embedding: 3
retrieval: 2
reranking: 1
evaluation: 6
```

## 下载过程中修过的问题

### 1. 批量下载一开始没有实时输出

原因：Python 输出被缓冲，而且某些网页/PDF 转换可能耗时较长。

修复：

- 每条资料开始下载、开始转换、完成状态都实时打印。
- 给 MarkItDown 转换加 `--convert-timeout`，避免单个源卡死整个流程。

### 2. 旧 LangChain 链接内容不匹配

几个旧的 `python.langchain.com/docs/concepts/...` 链接被重定向到了 Agent 概览，和资料标题不匹配。

修复：

- 改用 LangChain 当前官方 `docs.langchain.com/oss/python/...` Markdown 地址。
- 重抓 document loaders、text splitters、recursive splitter、retrievers 等资料。

### 3. LlamaIndex 链接迁移

部分 LlamaIndex 页面从旧 `docs.llamaindex.ai/en/stable/...` 路径迁移。

修复：

- 改用 `developers.llamaindex.ai/.../index.md` 的官方 Markdown 路径。

### 4. HTML 页面残留导航噪音

部分 HTML 页面有 `Copy page`、`On this page` 等导航残留。

修复：

- 增加通用清洗规则。
- 重新转换残留源。

## 以后重建资料层的命令

```powershell
conda activate rag-book
python experiments\16_llm_rag_sources\fetch_sources.py --sleep 0.5 --convert-timeout 60
```

这个命令会：

1. 读取 `llm_rag_sources.csv`。
2. 只抓 `priority=P0` 且 `ingest_first=yes` 的 27 个资料源。
3. 下载到 `data/raw/llm_rag_docs/`。
4. 转成统一 Markdown。
5. 生成 `data/processed/llm_rag_docs/documents.jsonl`。

## 下一步怎么继续

资料准备好后，下一阶段是：

```text
documents.jsonl -> structure-aware chunk -> embeddings -> Chroma index -> retrieval QA
```

这一步会把资料从“可读文档”变成“可检索知识库”。
