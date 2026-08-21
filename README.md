# Enterprise RAG Learning Project

一个面向学习和作品集展示的企业级 RAG Demo。项目从官方文档、论文和开源项目资料构建知识库，支持结构化切分、向量索引、query planning、planned retrieval、上下文组装、答案生成、答案审计、长对话记忆和 Web 可视化。

## 当前结果

- 知识库文档：52 篇
- 结构化 chunks：938 个
- 向量库：Chroma
- Embedding：Ollama `bge-m3`
- 生成模型：DeepSeek API 或 Ollama 本地模型
- 评测：10/10 通过，quality pass rate 100%

最新评测摘要：

- `eval/rag_system_full_after_chroma_query_docs/summary.md`
- `notes/34_component_scoped_planning_and_chroma_docs.md`

## Demo 展示

代表性问答案例见：

- `docs/demo.md`

Demo 覆盖复合问题拆解、非固定窗口 chunking、query rewrite/query expansion、Chroma metadata filter、企业级混合检索/重排，以及知识边界拒答。

## 核心能力

- 资料来源可追踪：通过 `data/source_manifests/llm_rag_sources.csv` 维护高质量来源。
- 正式入库流程：fetch -> markdown -> structure-aware chunking -> embedding -> Chroma index。
- 非固定窗口切分：按 Markdown 结构、标题和段落边界切分，再用软/硬长度约束兜底。
- Query planning：先理解问题意图，再生成子查询、分类过滤和必答方面。
- Planned retrieval：多子查询、多分类召回，再做融合、重排和覆盖选择。
- 上下文组装：区分检索证据和对话记忆，避免把记忆当事实来源引用。
- 答案审计：检查引用、忠实性、相关性、覆盖度和 badcase。
- 长记忆：记录用户偏好和目标，用于连续学习场景。
- Web 工作台：可查看规划、来源、上下文、审计和记忆。

## 架构流程

```mermaid
flowchart LR
    A["Curated sources"] --> B["Fetch and convert to Markdown"]
    B --> C["Structure-aware chunking"]
    C --> D["bge-m3 embeddings"]
    D --> E["Chroma vector DB"]
    Q["User question"] --> P["Query planning"]
    P --> R["Planned retrieval"]
    E --> R
    R --> K["Context assembly"]
    M["Short/long memory"] --> K
    K --> G["Answer generation"]
    G --> AU["Faithfulness and coverage audit"]
    AU --> W["Web workbench"]
```

## 项目结构

```text
src/                         核心 RAG 逻辑
webapp/                      本地 Web 工作台
experiments/16_llm_rag_sources/  资料抓取与 Markdown 转换
experiments/17_llm_rag_chunking/  文档切分
experiments/18_llm_rag_index/     Chroma 索引构建与搜索
experiments/22_rag_system_eval/   Web API 端到端评测
data/source_manifests/       高质量资料来源清单
docs/                        调研记录
notes/                       学习过程与阶段复盘
eval/                        评测摘要和可再生成结果
```

## 环境准备

已经验证过的本地环境：

- Python/conda 环境：`rag-book`
- Ollama：已安装
- 本地 embedding 模型：`bge-m3`
- 可选本地生成模型：`qwen2.5:1.5b`
- 可选远程生成模型：DeepSeek API

创建或进入环境后安装依赖：

```powershell
conda activate rag-book
python -m pip install -r requirements.txt
python -c "import ollama, chromadb; print('ok')"
ollama list
```

配置 API key 时，不要提交真实 key。复制 `.env.example` 或在当前终端设置：

```powershell
$env:DEEPSEEK_API_KEY="你的key"
```

## 重建知识库

这些命令会重新生成文档、chunks 和 Chroma 向量库：

```powershell
python experiments\16_llm_rag_sources\fetch_sources.py --priority P0 --sleep 0.1
python experiments\17_llm_rag_chunking\build_chunks.py
python experiments\18_llm_rag_index\build_index.py --rebuild --batch-size 8
```

当前构建结果：

```text
documents: 52
chunks: 938
indexed_count: 938
too_long_chunks: 0
tiny_chunks: 0
```

## 启动 Web 工作台

```powershell
python webapp\server.py --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765
```

## 运行评测

完整端到端评测：

```powershell
python experiments\22_rag_system_eval\evaluate_web_api.py --output-dir eval\rag_system_full_after_chroma_query_docs
```

最近一次结果：

```text
total: 10
passed: 10
failed: 0
pass_rate: 100%
quality_pass_rate: 100%
```

## 学习记录

建议按顺序阅读这些阶段记录：

- `notes/24_llm_rag_source_fetching.md`
- `notes/25_llm_rag_chunk_index_qa.md`
- `notes/26_query_planning_and_planned_retrieval.md`
- `notes/27_chunk_topk_rerank_optimization.md`
- `notes/29_web_workbench_and_context_assembly.md`
- `notes/31_long_term_memory_upgrade.md`
- `notes/32_rag_eval_harness.md`
- `notes/34_component_scoped_planning_and_chroma_docs.md`

## 适合作品集展示的点

- 不是只做一个简单 QA Demo，而是覆盖了 RAG 的数据、检索、生成、审计、记忆和评测链路。
- badcase 修复不是简单改 prompt，而是先定位问题属于数据、规划、检索、上下文还是审计，再做针对性改进。
- 资料来源偏官方文档、论文和成熟开源项目，适合解释“为什么这样设计”。
- 可以展示从 8/10 到 10/10 的质量迭代过程。

## 下一步方向

- 补充 `environment.yml`，让 conda 环境也能一键创建。
- 给 Web 页面增加更清晰的“答案/来源/审计”展示。
- 增加更多真实业务数据集，验证跨领域泛化。
- 引入真正的 reranker 模型，对比 lexical rerank 与 cross-encoder rerank。
- 增加 GitHub Actions 或本地一键评测脚本。
