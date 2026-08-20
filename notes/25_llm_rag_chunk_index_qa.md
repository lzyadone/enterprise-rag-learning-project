# LLM/RAG 知识库：chunk、索引和问答记录

日期：2026-08-19

## 本阶段目标

把已经整理好的 27 篇高质量资料从“可读文档”变成“可检索知识库”：

```text
documents.jsonl -> chunks.jsonl -> embeddings -> Chroma index -> retrieval QA
```

## 为什么要先 chunk

RAG 检索的基本单位不是整篇文档，而是 chunk。

如果 chunk 太大：

- 检索命中后上下文太长。
- 模型容易抓不住重点。
- 不同知识点混在一起，回答不够精确。

如果 chunk 太小：

- 单个片段信息不足。
- 模型拿不到完整解释。
- 容易出现断章取义。

所以本项目使用结构化切分：

```text
优先按 Markdown 标题/章节切分
只有章节过长时才按段落做安全拆分
不把固定窗口作为主切分策略
```

## 已新增文件

结构化切分模块：

```text
src/chunking.py
```

chunk 构建脚本：

```text
experiments/17_llm_rag_chunking/build_chunks.py
```

Chroma 建库脚本：

```text
experiments/18_llm_rag_index/build_index.py
```

检索脚本：

```text
experiments/18_llm_rag_index/search_index.py
```

问答脚本：

```text
experiments/19_llm_rag_qa/ask.py
```

## chunk 结果

运行命令：

```powershell
python experiments\17_llm_rag_chunking\build_chunks.py
```

结果：

```text
documents: 27
chunks: 457
```

后续优化后去掉孤立短片段，迭代过程：

```text
478 chunks: 初版，有 2 个 header/source 孤立短片段
476 chunks: 去掉 Source 元数据行
469 chunks: 合并普通短片段
457 chunks: 允许短片段合并到后续正文，去掉最后的孤立短片段
最终确认：
chunks: 457
tiny chunks: 0
min char_count: 288
max char_count: 3647
```

最终输出：

```text
data/processed/llm_rag_docs/chunks.jsonl
data/processed/llm_rag_docs/chunk_summary.json
```

## embedding 和 Chroma 建库

本地 embedding 模型：

```text
bge-m3
```

向量维度：

```text
1024
```

一开始 Python `ollama.Client.embed()` 返回 502，但 Ollama 官方 HTTP API 可用。

因此建库脚本直接调用：

```text
POST http://127.0.0.1:11434/api/embed
```

这样更稳定，也更接近工程系统里的显式 API 调用。

运行命令：

```powershell
python experiments\18_llm_rag_index\build_index.py --rebuild --batch-size 8
```

结果：

```text
chunks: 457
batches: 58
indexed_count: 457
elapsed_seconds: 222.81
```

索引目录：

```text
data/indexes/llm_rag_chroma
```

## 检索验证

检索问题：

```text
RAG 的完整流程包括哪些阶段？
```

命中：

- LlamaIndex RAG Concepts > Stages within RAG
- LangChain Retrieval > RAG architectures
- LangSmith Evaluate RAG Tutorial > Overview

检索问题：

```text
文档切分为什么不能只用固定窗口？
```

命中：

- Hugging Face RAG Evaluation Cookbook > Preprocessing documents
- LangChain Recursive Text Splitter
- LangChain Text Splitter Integrations > Document structure-based

检索问题：

```text
metadata filter 在检索中有什么作用？
```

命中：

- Chroma Docs
- Chroma Collections > Collection Metadata
- MongoDB Self-Query Retrieval with LangChain

说明：知识库不是只能搜索英文原文，中文问题也能通过 `bge-m3` 检索到英文资料。

## 问答验证

运行命令：

```powershell
python experiments\19_llm_rag_qa\ask.py --query "如何评估 RAG 回答是否忠实于检索上下文？" --category evaluation --top-k 4
```

检索来源：

- LangSmith Evaluate RAG Tutorial > Define evaluators
- LangSmith Evaluate RAG Tutorial > Reference code

回答能够基于资料说明：

- correctness
- relevance
- groundedness
- retrieval relevance

当前小问题：

- 本地 `qwen2.5:1.5b` 能回答，但会复制一些文档里的 Markdown 锚点。
- 后续需要继续优化 prompt 和 context 清洗，让来源引用格式更稳定。

## 当前状态

第一版知识库已经跑通：

```text
资料源选择 -> 下载清洗 -> 结构化 chunk -> embedding -> Chroma -> 检索 -> 基于资料回答
```

下一步建议：

```text
做 query planning 和 answer prompt 优化
```

也就是让系统先理解用户问题属于哪个知识点，再决定是否加 category filter、是否需要多路检索、以及答案应该怎么引用来源。
