# 大模型工程知识库 RAG：资料源筛选方案

更新时间：2026-08-19

## 新方向

项目主线调整为：

```text
大模型工程知识库 RAG
```

第一版聚焦：

```text
RAG 工程知识库
```

后续再扩展到 Prompt、Agent、微调、推理部署、评估和 LLMOps。

## 为什么这个方向更适合

这个方向和你的求职目标更贴近。你不是只做一个“能问答的 demo”，而是做一个能回答大模型工程问题的学习助手。它可以支撑你学习，也可以作为作品集展示：

- 你知道 RAG 的完整链路。
- 你知道资料怎么筛选和入库。
- 你知道 chunk、embedding、metadata、rerank、prompt、eval 分别解决什么问题。
- 你知道知识库答案为什么可信，以及什么时候应该拒答。

## 资料准入标准

资料不是越多越好。我们按下面规则筛：

### P0：必须进入第一版知识库

只收高质量、权威、能直接回答工程问题的资料：

- 官方文档
- 官方教程
- 经典论文
- 成熟开源项目官方文档

### P1：第二阶段进入知识库

用于拓展能力，但第一版不急：

- Agent
- GraphRAG
- 微调/LoRA
- vLLM 推理部署
- RAG 评估平台

### P2：只作为学习参考

不直接入库，避免噪音：

- 个人博客
- 二手教程
- 未维护的小项目
- 内容重复的文章

## 第一版知识库主题

第一版只做这些主题：

1. RAG 基本概念
2. 文档加载与解析
3. 文档切分与 chunk 设计
4. Embedding 和向量库
5. 检索、metadata filter、rerank
6. Prompt 与上下文组装
7. RAG 评估和 badcase 分析
8. 本地模型调用，Ollama + bge-m3 + Chroma

暂时不做：

- 大规模微调
- 复杂 Agent
- 多模态 RAG
- GraphRAG 全量实现
- 企业私有数据权限系统

原因：第一版要先把 RAG 工程主链路做扎实。

## 推荐资料源

完整清单见：

```text
data/source_manifests/llm_rag_sources.csv
```

清单里的字段：

- `priority`：P0/P1/P2
- `category`：知识点分类
- `title`：资料名称
- `source_type`：official_doc / paper / github / cookbook
- `url`：来源链接
- `ingest_first`：是否第一批入库
- `why`：为什么值得学
- `notes`：处理注意事项

## 第一批真正入库的资料

第一批建议入库 P0 资料，大约 20-25 个页面/文档，覆盖 RAG 主流程：

### 1. RAG 总览

- LangChain Retrieval
- LlamaIndex RAG concepts
- 原始 RAG 论文

### 2. 文档加载和切分

- LangChain Document Loaders
- LangChain Text Splitters
- LangChain RecursiveCharacterTextSplitter
- LlamaIndex Loading Data
- LlamaIndex Ingestion Pipeline

### 3. 向量库和 embedding

- Chroma Look at Your Data
- Chroma Embedding Functions
- Ollama Embed API
- BGE-M3 模型卡
- BGE-M3 论文
- Sentence-BERT 论文

### 4. 检索和问答

- LangChain Retrievers
- LlamaIndex Querying
- ColBERT 项目说明

### 5. 评估

- LlamaIndex Evaluating
- LangSmith Evaluation Concepts
- LangSmith Evaluate RAG
- Ragas Metrics
- TruLens RAG Triad
- Hugging Face RAG Evaluation Cookbook

### 6. 本地运行

- Ollama API Introduction
- Ollama Chat API
- Chroma docs

## 不直接上传原文到 GitHub

这点很重要。

我们可以：

- 上传资料清单
- 上传下载/解析脚本
- 上传你自己的学习笔记
- 上传索引构建代码
- 上传评估集和评估结果

不建议：

- 把官方文档全文复制进仓库
- 把书籍 PDF 原文放进仓库
- 把课程资料原文放进仓库

本地可以缓存资料用于学习和建库，但 GitHub 作品集应该展示“如何构建知识库”，而不是直接搬运资料。

## 知识库目录设计

建议新增：

```text
data/
  source_manifests/
    llm_rag_sources.csv
  raw/
    llm_rag_docs/
      README.md
  processed/
    llm_rag_docs/
      documents.jsonl
      chunks.jsonl
      metadata_schema.json
  indexes/
    llm_rag_chroma/
```

说明：

- `source_manifests` 保存资料来源。
- `raw` 保存本地下载/提取的原始文本，不一定上传 GitHub。
- `processed` 保存清洗后的文档和 chunk。
- `indexes` 保存向量库，可以本地保留，GitHub 一般不上传。

## 后续推进顺序

下一步不急着写问答代码，而是先做数据准备：

1. 建资料清单。
2. 写下载/解析脚本。
3. 只抓第一批 P0 资料。
4. 用 MarkItDown 或网页转 Markdown 工具统一成 Markdown。
5. 给每篇资料打 metadata。
6. 再做 chunk 和入库。

这样做的好处是：知识库的质量从源头开始控制，不会后面检索时发现全是噪音。

## 面试时怎么讲

这个项目可以这样介绍：

```text
我做了一个面向大模型工程学习和工程实践的 RAG 知识库系统。
它的数据不是随便抓网页，而是按官方文档、经典论文、成熟开源项目文档分级筛选。
系统支持文档解析、结构化 chunk、metadata 检索、rerank、引用回答和评估。
我还用 RAG evaluation 指标和 badcase 分析来迭代检索质量。
```

这个说法比“我做了一个本地 RAG demo”强很多。
