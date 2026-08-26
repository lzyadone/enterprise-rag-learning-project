# Enterprise RAG Learning Project Handoff

更新日期：2026-08-26

项目目录：

`C:\Users\Lenovo\Desktop\大模型官方课程-视频资料\学习产出\enterprise-rag-learning-project`

当前开发分支：`feature/planned-retrieval-parallel-cache`

本次功能提交：`14b2b2e Parallelize and cache planned retrieval`

索引发布门禁已通过 `e8b961b Merge index release quality gate` 合并并推送到 `main`。

上一阶段合并包含：

- `7f841d9 Protect targeted source refreshes`
- `379fe71 Add pinned PyPDF metadata sources`
- `844277a Update handoff after source refresh`

Planner v3 阶段的主要提交已全部合并并推送到 `main`：

- `6fc4d22 Freeze planner v3 holdout dataset`
- `127dfbc Evaluate planner v3 holdout`
- `cc4f242 Add assisted qrels spot audit`
- `41e7689 Expose planner v3 planned mode in web`
- `5a16a2a Update project handoff after planner v3 validation`
- `9859ad7 Merge planner v3 evaluation benchmark`
- `9b6c8bb Update handoff after main merge`
- `bd8c327 Add temporary remote API support`
- `b6b8624 Merge temporary remote API support`
- `c44b57e Update handoff after remote API merge`

本文档不包含任何密钥、token、环境变量值、`.env` 内容或其他凭据。

## 1. 项目目标

这是一个用于学习和作品集展示的企业级 RAG 项目。目标不是只做一个能回答问题的 Demo，而是建立一套可解释、可评测、可更新、可回退的完整系统，包括：

1. 高质量资料采集与来源管理；
2. 文档解析和结构化切分；
3. embedding、向量数据库和关键词索引；
4. direct、planned 和 hybrid 检索；
5. 重排与上下文组装；
6. 答案生成、引用和质量审计；
7. 短期对话记忆与长期记忆；
8. Web 可视化工作台；
9. 离线评测、盲标、自动路由和发布门槛。

当前默认策略仍保持稳健配置：本地 Ollama 生成、direct retrieval。Planner v3 已通过独立 holdout 门槛并进入 Web 手动实验模式，但没有替换默认 direct。

## 2. 项目结构

| 路径 | 用途 |
|---|---|
| `src/` | 核心 RAG 模块：检索、规划、重排、上下文、记忆、审计和模型客户端 |
| `webapp/` | 本地 Web 服务和前端工作台 |
| `experiments/` | 各阶段可复现实验脚本 |
| `eval/datasets/` | 评测问题集 |
| `eval/benchmarks/` | qrels、评测协议和来源锚点 |
| `eval/` 其他目录 | 检索、路由和系统评测摘要 |
| `data/source_manifests/` | 知识来源清单 |
| `data/processed/` | 本地处理后的文档和 chunk，不提交 Git |
| `data/indexes/` | 本地 Chroma 索引，不提交 Git |
| `data/runtime/` | checkpoint、缓存和长期记忆数据，不提交 Git |
| `tests/` | 单元测试和行为回归测试 |
| `notes/` | 每个工程阶段的学习记录和实验结论 |

当前 RAG 知识库包含 54 份已处理文档、942 个结构化 chunk，主要来源是官方文档、论文和高质量开源项目资料。

## 3. 已完成能力

### 3.1 数据与索引

- 建立来源 manifest，记录标题、类别、优先级、来源类型和 URL。
- manifest 支持可选 `extract_symbol`，可从固定提交的 Python 源码中只提取指定类或方法，避免同文件相邻实现污染检索证据。
- 定向来源刷新只控制下载对象，聚合仍要求全部符合条件的 P0 来源完整存在；缺少来源时保留旧 `documents.jsonl`，完整时通过临时文件原子替换。
- 跑通 fetch -> Markdown/文本 -> 结构化 chunk -> embedding -> Chroma。
- chunking 以标题、段落和文档结构为主，长度只是软限制和异常兜底，不是固定窗口硬切。
- 使用 `bge-m3` 生成本地向量。
- 使用 Chroma 保存向量和来源 metadata。
- 为同一批 chunk 建立 BM25 关键词索引。

### 3.2 检索与重排

- dense retrieval：向量语义召回。
- BM25 retrieval：关键词召回，支持中英文 token 和中文二元/三元切分。
- hybrid retrieval：dense 与 BM25 并行召回，通过 RRF 融合。
- metadata/category filter：按知识类别约束检索。
- lexical rerank：轻量本地重排。
- cross-encoder rerank：可选精排，并完成过固定候选对比实验。
- direct retrieval：保留用户原始问题直接检索。
- planned retrieval：问题规划、多路召回、融合和回答面覆盖。

### 3.3 Query Planner

- legacy planner：基于知识类别、子查询和回答面的初始规划器。
- anchored planned v2：提高原始问题权重，限制扩展查询的影响。
- conservative planner v3：
  - 普通具体问题只使用原始问题；
  - 至少识别出两个明确且可独立检索的回答面时才扩展；
  - 扩展查询保留完整用户原话；
  - 限制 category、aspect 和 retrieval run 数量；
  - 原始查询在融合中保持最高权重。

开发集上，Planner v3 将平均 retrieval runs 从 `18.22` 降到 `1.81`。

三系统在同一个 351-pair 盲标 union pool 上的开发集结果：

| system | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| direct hybrid | 0.977 | 0.865 | 0.837 |
| planned v2 hybrid | 0.925 | 0.691 | 0.737 |
| planned v3 hybrid | 0.992 | 0.882 | 0.853 |

这些开发集结果已经由独立 holdout 补充验证。32 道 holdout 上，direct 与 planned v3 的 depth-10 候选完全重合，二者均为 Recall@10 `1.000`、MRR@10 `0.839`、nDCG@10 `0.843`。这证明 v3 没有造成退化，但本次 holdout 没有证明它带来检索质量提升。

自动路由在 32 题中选择 direct 31 次、planned 1 次，与逐题 oracle 的一致率为 `31/32 = 96.9%`，全部预注册门槛通过。

### 3.4 上下文、生成与审计

- 将检索证据、来源编号、回答要求和对话记忆组装成受长度限制的上下文。
- 生成提示要求答案只使用检索证据，并输出来源编号。
- 支持 Ollama 本地生成和显式选择远程生成模型。
- deterministic audit 检查引用编号、来源范围和基本格式。
- LLM audit 检查忠实性、引用质量、相关性和回答面覆盖。
- 审计不通过时可以尝试修复答案，并比较修复前后的质量分数。

### 3.5 Web 工作台与记忆

- Web 页面展示答案、来源、规划、上下文、审计和记忆。
- 支持短期对话记忆。
- 支持本地长期记忆存储、召回和清理。
- 页面展示 requested retrieval mode、actual route 和实际生成模型路径。
- 问题处理支持 `direct`、`auto（实验）` 和 `planned v3（实验）`。
- `planned v3` 使用 conservative planner；页面默认值仍是 `direct`。

### 3.6 最近完成的稳定性修复

相关提交：`620f4a3`

- Web 默认生成模型改为 Ollama。
- 只有显式配置远程 provider 时才将其作为默认值。
- 远程模型拒绝 prompt 或返回空内容时自动回退 Ollama。
- API 返回实际生成路径，例如远程模型、本地回退和后续修复模型。
- 浏览器不接收云端原始错误正文。
- `rank-bm25` 作为可选依赖；缺失时使用内置 `SimpleBM25Okapi`。
- 为默认 provider、生成回退、最终 provider 路径和 BM25 fallback 增加测试。
- 该阶段完成时测试结果为 `81 passed`；当前完整 Python 回归已扩展到 106 项并全部通过。
- Web 后端真实烟测成功：返回非空答案和检索来源。

### 3.7 独立 holdout 与发布决定

- 已冻结 `eval/datasets/rag_natural_query_holdout_v3.jsonl`，共 32 题：16 focused、16 compound。
- 已冻结 48 个 source anchor，来自 48 个互不重复的 chunk。
- 已记录固定随机种子、数据集 SHA-256、source-anchor SHA-256 和评测协议。
- 已建立 320 个 query/chunk pair 的 depth-10 union pool，并完成 0-3 relevance 盲标。
- direct 与 planned v3 的 Recall@5 均为 `0.664`，Recall@10 均为 `1.000`，Precision@10 均为 `0.503`，MRR@10 均为 `0.839`，nDCG@10 均为 `0.843`。
- 独立检索中位延迟：direct 约 `0.70s`，planned v3 约 `0.53s`。
- 结论记录在 `notes/48_planner_v3_holdout_release_decision.md`：允许进入 Web 实验模式，不替换默认 direct。

### 3.8 辅助标签抽查

- 使用按 relevance 等级分层的确定性样本抽查 12 个 query/chunk pair，覆盖 10 道问题。
- 9 条与 LLM 标签完全一致，12 条均在一个等级误差内，严重分歧为 0。
- 发现 3 条轻微高估，并完成修正敏感性评测；发布结论不变。
- 该结果是 Codex 辅助审计，不冒充真实人工金标。审计记录位于 `eval/benchmarks/rag_natural_query_holdout_v3/codex_spot_audit.jsonl`。

### 3.9 Web 端到端验收

在 `http://127.0.0.1:8766` 使用真实页面完成 4 次问答：简单问题和复合问题分别运行 direct 与 planned v3。

| 问题 | 模式 | 页面耗时 | 来源数 | 结果 |
|---|---|---:|---:|---|
| 简单问题 | direct | 30.3s | 7 | 通过 |
| 简单问题 | planned v3 | 8.6s | 7 | 通过 |
| 复合问题 | direct | 12.6s | 7 | 通过 |
| 复合问题 | planned v3 | 17.5s | 7 | 通过 |

- 四次回答均成功并包含相关来源，页面无控制台错误或警告。
- planned v3 页面元数据显示 `planned + conservative`。
- 页面刷新后默认模式仍为 direct。
- 具体产品/API 类复合问题会按 v3 的保守规则退化为原查询，不强制拆解；这是当前设计边界，不是运行故障。

### 3.10 临时远程 API 与自动 Web 回归

- Web 生成模型新增 `远程 API（临时）`，支持 Bearer Authorization 的 OpenAI Chat Completions 兼容接口。
- 用户可临时填写 API Base URL 或完整 `/chat/completions` 地址、模型名和密钥，并单独测试连接。
- 密钥不写入文件、浏览器存储、长期记忆、日志或响应；远程错误不返回 provider 原始正文。
- 默认生成模型仍是 Ollama，DeepSeek 环境配置模式保持兼容。
- 新增 managed Web E2E regression：自动启动临时服务，运行 focused/compound × direct/planned v3 共 4 项；功能分支首次完整验收 4/4 通过。
- 完整 Python 回归共 101 项通过，远程配置页面完成桌面与手机视口验收，浏览器控制台无错误。
- 合并后完整复验为 3/4；唯一失败是本地 Ollama 偶发未输出来源编号，失败项的 direct/planned 定向复验随后 2/2 通过。检索来源、类别、模式、规划器形态和耗时检查均正常，因此记录为生成格式波动，不判定为合并回归。
- 详细记录位于 `notes/50_web_remote_api_and_e2e.md`。

### 3.11 PyPDFLoader Metadata 官方来源与安全刷新

- 新增固定到具体 Git 提交的 `PyPDFParser` 和 `TextSplitter` 官方源码，分别提取构造/解析方法与 metadata 传递方法。
- 知识库由 52 documents / 938 chunks 升级为 54 documents / 942 chunks，切分检查无超长或过短片段。
- 默认 Web 检索参数下，`natural_dev_007` 的 PyPDF 官方证据排名第 1；`natural_dev_009` 的两条 PyPDF 证据排名第 1、2，两条 TextSplitter 证据排名第 3、5。
- 真实本地 Web API 答案正确区分默认 page 模式、single 模式和 chunk metadata 继承，规则与 LLM 审计均通过，faithfulness/citation 均为 5/5。
- 候选索引验证通过后已晋升为默认 `llm_rag_chroma`；旧 938 条索引本地保留为 `llm_rag_chroma_pre_pypdf_938`，两者都不提交 Git。
- 未修改旧自然开发问题、qrels、候选池、冻结 holdout、source anchors 或历史评测结果。详细记录位于 `notes/51_pypdfloader_metadata_source_refresh.md`。

### 3.12 版本化增量索引

- 新增来源级 SHA-256 差异计划，识别 added、changed、deleted 和 unchanged 文档。
- unchanged 文档直接复用上一版本 chunks；同 embedding 模型下按 chunk ID/text hash 或 source/text hash 复用旧向量，只为缺失向量调用 embedding。
- 每个版本保存不可变 manifest、chunks 和 Chroma；候选只包含当前期望 chunks，删除来源不会残留旧向量。
- 构建与启用分离；manifest、内容哈希、来源状态、Chroma 文本、metadata、ID 和数量必须一致才能启用。
- 激活指针通过原子文件替换切换版本并记录上一版本，可一步回滚。
- Web 在请求间检测激活指针，无需重启即可热加载新版本；切换时清空 BM25 语料缓存，单次请求固定使用同一版本快照。
- 真实 54/942/942 索引已封装为 `baseline-20260825-942`，942 个 embedding 全部复用；候选启用和回滚均在同一 Web 进程中通过。
- 两个 PyPDFLoader 固定问题的候选检索排名与上一阶段完全一致。详细记录位于 `notes/52_versioned_incremental_index.md`。

### 3.13 索引离线发布门禁

- 新增单一命令，依次执行候选结构校验、冻结检索回归和完整 Python 测试，并生成 JSON/Markdown 报告。
- 冻结 `rag_index_release_gate_v1`：8 条既有跨组件烟测题加 2 条 PyPDFLoader 强制题，总门槛保持历史 direct hybrid 基线 `8/10 = 80%`。
- 两条 PyPDFLoader 题是 required cases；任一直接证据排名退化都会阻止发布，即使总体通过率仍达标。
- `--activate` 只有在三个阶段全部通过时才生效；失败或测试进程异常都会留下报告并标记 blocked。
- 激活前先原子持久化 approved 报告，再切换 active pointer，确保发布决定可追溯。
- 对 `validation-copy-20260825` 的真实 dry run 和激活演练均通过：structure passed、retrieval 8/10、required failures 0、tests 116/116。
- 激活演练后已成功 rollback，当前恢复 `baseline-20260825-942`。详细记录位于 `notes/53_index_release_gate.md`。

### 3.14 Planned Retrieval 并行与缓存

- embedding 预计算后，planned retrieval 使用最多 4 workers 并行执行独立 Chroma/BM25 runs；单 run 自动保持串行。
- 并行结果按原请求顺序收集，加权 RRF、plan boost 和覆盖选择逻辑不变；cross-encoder 仍只集中执行一次。
- 新增线程安全有界 LRU：candidate 512 项、rerank 128 项，读写均深拷贝，避免可变候选污染缓存。
- candidate key 包含索引命名空间、模型/host、策略、查询、category 和召回窗口；rerank key 还包含候选及重排器配置指纹。
- Web active index 热切换会同时清空 BM25、candidate 和 rerank 缓存；自定义 legacy DB 使用实际路径隔离。
- `reuse_query_embeddings=False` 会同时绕过新缓存，旧严格 A/B 语义保持不变。
- 两个 7-run 复合问题各重复 3 次：串行中位 `0.0668s`，并行冷缓存 `0.0549s`（`1.22x`），并行热缓存 `0.0053s`（`12.67x`），Top-7 顺序 `6/6` 完全一致。
- 最终索引发布门禁通过，完整测试 `122/122`；真实 Web direct/planned v3 回归 `4/4` 通过。详细记录位于 `notes/54_planned_retrieval_parallel_and_cache.md`。

## 4. 当前未完成工作

P0 独立验证集、P1 独立检索评测、P2 Web 实验发布决定、PyPDFLoader metadata 官方来源、版本化增量索引、索引离线发布门禁和 planned retrieval 并行/缓存均已完成。当前没有功能阻塞。下一项是增加越权、提示注入、来源冲突和知识库外问题的安全回归，并明确拒答与冲突处理门槛。

### 4.1 Holdout 状态

- 生成阶段的 `information_needs` 误判已修复，对应 7 个单元测试通过。
- checkpoint 已从 21/32 补齐到 32/32，并成功生成冻结数据集、source anchors 和 summary。
- 生成完成后 checkpoint 已由生成器正常清理，不需要恢复或重新生成。
- 题目数量、ID 对齐、stratum、审稿结论、自然度、证据覆盖、重复度和 anchor 数量检查均为 0 failure。

### 4.2 当前边界

- holdout 的 direct 与 planned v3 top-10 候选完全重合，因此当前证据支持“无退化”，不支持“质量显著提升”。
- qrels 主要由 LLM 盲标，并经过 Codex 辅助抽查；它不是生产级人工金标。
- conservative planner 只扩展已定义、证据明确的 RAG 问题模式。具体 API、产品事实或不确定复合问题会安全退化为原查询。
- Web 的覆盖审计依赖 planner 识别出明确 aspects；安全退化的问题会显示“未运行”。
- 本地 Ollama 的答案生成具有随机性，偶尔会漏写来源编号；自动 Web 回归会将其标记为 `citation_present` 失败，需结合定向复验区分格式波动与功能回归。
- 当前使用项目既有 unittest 入口完成 122 项完整 Python 回归；项目尚未建立独立 CI 门禁。
- candidate/rerank cache 是进程内缓存，不跨进程持久化；热缓存 `12.67x` 只代表预热 embedding 后的重复检索阶段，不代表页面端到端提速。

### 4.3 当前工作区状态

- 分支：`feature/planned-retrieval-parallel-cache`
- 本阶段功能提交：`14b2b2e Parallelize and cache planned retrieval`。
- `feature/safe-source-refresh` 已通过 `e8a0583` 合并到 `main`；功能分支仍保留在本地和远程，没有删除。
- `feature/web-remote-api` 已通过 `b6b8624` 合并到 `main` 并推送；功能分支仍保留在本地和远程，没有删除。
- 功能分支 `feature/rag-evaluation-benchmark` 已完整合并，目前仍保留在本地和远程，没有删除。
- 默认激活版本为 `baseline-20260825-942`，包含 54 documents / 942 chunks / 942 Chroma rows；旧 938 条索引和 legacy 942 条索引均保留，生成语料、版本目录和激活指针由 `.gitignore` 排除。
- 完整运行 122 个 Python 单元测试，全部通过；索引发布门禁和前端 JavaScript 语法检查通过。
- managed Web E2E regression 为 4/4：focused/compound × direct/planned v3 全部通过。
- `eval/` 仅新增冻结门禁规格 `eval/benchmarks/rag_index_release_gate_v1/gate.json`，未修改旧数据集、qrels 或历史结果。
- 索引发布门禁已通过 `e8b961b` 合并到 `main`；当前性能功能分支从该提交创建，尚未合并；本文件继续按惯例单独提交。

## 5. 已知问题与边界

1. Planner v3 已通过独立 holdout 门槛，但目前只允许作为 Web 实验模式，不能替换默认 direct。
2. 旧评测中 2 个 PDFLoader 页码 metadata 问题的知识覆盖缺口已补齐；历史候选池和分数保持原样，若重新评测必须建立新版本。
3. holdout 已冻结并完成评测。后续若修改题目、资料、planner 或检索参数，必须建立新版本，不能覆盖 v3 结果。
4. 远程模型和本地小模型的答案质量不同，检索评测与答案生成评测必须分开。
5. LLM qrels 不是人工真值；当前 12 条辅助抽查足以支持学习项目实验结论，不足以替代生产发布前的真人标注。
6. 独立检索延迟已达标，但页面端到端生成耗时约为 8.6-30.3 秒，受本地模型冷启动和答案生成影响较大。
7. 当前长 Codex 对话曾在较长工作流中被平台中断。新对话应使用本文件恢复上下文，并采用短步骤、持续 checkpoint 和分阶段提交。
8. 临时远程 API 密钥只存在于当前页面内存和单次请求中；刷新页面后需重新填写。远程 API 必须使用 HTTPS，本机回环地址除外。

## 6. 验证命令

进入项目并激活环境：

```powershell
cd "C:\Users\Lenovo\Desktop\大模型官方课程-视频资料\学习产出\enterprise-rag-learning-project"
conda activate rag-book
```

完整 Python 测试：

```powershell
python -m pytest -q
```

Web 与 BM25 针对性测试：

```powershell
python -m unittest tests.test_bm25_retrieval tests.test_web_defaults
```

独立验证集生成器测试：

```powershell
python -m unittest tests.test_grounded_holdout_generation
```

Planner v3 与 Web 针对性回归：

```powershell
python -m unittest tests.test_web_defaults tests.test_query_planning_v3 tests.test_retrieval_fusion tests.test_retrieval_routing tests.test_auto_routing_evaluation
```

增量索引生命周期测试：

```powershell
python -m unittest tests.test_index_versioning
python experiments\33_incremental_index\manage_index.py status
python experiments\33_incremental_index\manage_index.py plan
```

索引离线发布门禁：

```powershell
python -m unittest tests.test_index_release_gate
python experiments\34_index_release_gate\run_gate.py --manifest data\indexes\llm_rag_versions\validation-copy-20260825\manifest.json
```

Planned retrieval 并行与缓存：

```powershell
python -m unittest tests.test_planned_retrieval_execution
python experiments\35_planned_retrieval_parallel_cache\benchmark.py
```

JavaScript 语法检查：

```powershell
node --check webapp\static\app.js
```

如果系统 PATH 没有 Node，可以使用 Codex 工作区依赖中提供的 Node 可执行文件。

检查本地模型：

```powershell
ollama list
```

启动 Web：

```powershell
python webapp\server.py --host 127.0.0.1 --port 8766
```

浏览器地址：`http://127.0.0.1:8766`

独立 holdout 已完成并冻结。除非明确创建新版本，否则不要重新生成或覆盖 v3 数据集。

## 7. 下一步任务清单

### P0：冻结独立验证集（已完成）

32 道题、48 个 source anchor、数据集哈希、固定随机种子、评测协议和预注册门槛均已冻结并提交。

### P1：运行独立检索评测（已完成）

候选池、确定性盲标、qrels、独立延迟、系统指标、自动路由、oracle 一致率和辅助抽查均已完成。

预注册建议门槛：

- 可评测问题覆盖率至少 90%；
- auto nDCG 不低于 direct 超过 0.02；
- auto Recall 不低于 direct 超过 0.02；
- auto MRR 不低于 direct 超过 0.05；
- oracle agreement 至少 75%；
- planned 路由不得出现 nDCG 下降超过 0.25 的严重样例；
- 中位端到端检索延迟不超过 12 秒。

门槛已在候选生成和 qrels 标注之前冻结，全部通过。

### P2：根据独立结果做发布决定（已完成）

决定：将 Planner v3 加入 Web 手动实验模式，默认 direct 不变。辅助抽查没有严重分歧，Web 端到端验收通过。

### P3：独立验证完成后的工程优化

Planner v3、direct/planned v3 自动端到端回归、临时远程 API、安全来源刷新、PyPDFLoader metadata 官方证据、版本化增量索引和索引离线发布门禁均已实现、验收并合并到 `main`。Planned retrieval 并行与缓存已在功能分支完成并通过真实数据验收，等待合并。

本轮已完成：有上限的并行 planned runs、版本感知 candidate/rerank LRU、索引切换失效、诊断绕过和串并行一致性 A/B。

后续工程优先级：

1. 增加越权、提示注入、来源冲突和知识库外问题测试，并冻结安全门槛。
2. 增加 GitHub Actions 或本地一键评测入口。
3. 准备作品集架构图、演示问题、指标表和技术决策说明。

## 8. 新任务启动方式

必须从 Codex 的“新任务”入口创建空白任务，不要在以前做 MarkItDown 批量转换的任务中继续。旧任务包含数百条转换进度消息，会把无关历史继续带入模型上下文。

第一条消息只做对齐，不修改文件：

> 进入 `学习产出/enterprise-rag-learning-project`，阅读根目录的 `PROJECT_HANDOFF.md`，查看 git status 和最近提交。只总结已完成阶段、当前边界和下一个短步骤，不执行修改，不读取任何密钥或 `.env` 内容。

确认总结正确后，再发送第二条消息：

> 按交接文档执行下一个短步骤。不要回滚现有改动，每次只完成一个可验证的小步骤，并说明原因和验证结果。
