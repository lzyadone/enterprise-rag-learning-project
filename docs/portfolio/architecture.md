# 系统架构

这套架构把 RAG 拆成三个可以独立解释和验收的平面：数据与索引、在线问答、质量与发布。默认在线路径保持本地 Ollama + direct retrieval；planned v3、cross-encoder 和远程 API 都是显式选择的能力。

## 1. 全局架构

```mermaid
flowchart TB
    subgraph DataPlane["数据与索引平面"]
        Sources["官方文档 / 论文 / 固定源码"] --> Manifest["来源 manifest 与版本固定"]
        Manifest --> Fetch["安全定向抓取与完整性检查"]
        Fetch --> Chunk["结构感知切分与 metadata 继承"]
        Chunk --> Delta["文档 / chunk 指纹与增量计划"]
        Delta --> Embed["复用或生成 bge-m3 embedding"]
        Embed --> Candidate["不可变候选索引版本"]
        Candidate --> ReleaseGate["结构 + 冻结检索 + 测试门槛"]
        ReleaseGate --> Active["原子 active pointer / rollback"]
        Active --> Chroma["Chroma dense index"]
        Active --> BM25["BM25 sparse index"]
    end

    subgraph RequestPlane["在线问答平面"]
        User["Web 用户问题"] --> QuerySecurity["查询安全与知识边界"]
        QuerySecurity --> Route["direct / auto / planned v3"]
        Route --> Dense["Dense retrieval"]
        Route --> Sparse["BM25 retrieval"]
        Dense --> RRF["RRF 融合"]
        Sparse --> RRF
        RRF --> Rerank["Lexical / 可选 cross-encoder"]
        Rerank --> Context["证据隔离与上下文组装"]
        ShortMemory["短期记忆"] --> Context
        LongMemory["长期记忆"] --> Context
        Context --> Generate["Ollama / DeepSeek / 临时远程 API"]
        Generate --> Audit["引用 + 忠实度 + 覆盖 + 安全审计"]
        Audit --> Response["答案 / 来源 / 路由 / 审计"]
    end

    subgraph QualityPlane["质量与发布平面"]
        FrozenData["冻结数据集与 source anchors"] --> Pool["隐藏系统与排名的 union pool"]
        Pool --> Qrels["0-3 relevance 盲标与辅助抽查"]
        Qrels --> Metrics["Recall / MRR / nDCG / latency"]
        SecurityCases["8 项安全冻结样例"] --> UnifiedGate["统一质量门槛"]
        Tests["139 项测试 + 编译 + JavaScript"] --> UnifiedGate
        Metrics --> Decision["默认策略与实验发布决定"]
        UnifiedGate --> Decision
    end

    Chroma --> Dense
    BM25 --> Sparse
    Decision -. "控制默认值与发布" .-> Route
    Decision -. "阻止不合格索引" .-> ReleaseGate
```

## 2. 单次请求时序

```mermaid
sequenceDiagram
    actor U as 用户
    participant W as Web API
    participant S as 安全策略
    participant M as 记忆
    participant P as 路由 / Planner v3
    participant R as Chroma + BM25
    participant G as 生成模型
    participant A as 审计

    U->>W: POST /api/ask
    W->>W: 立即移除临时远程凭据字段
    W->>S: 判断越权、注入和知识边界
    alt 拒绝或知识库外实时请求
        S-->>W: 固定策略响应
        W-->>U: 0 来源、0 记忆写入、无模型调用
    else 正常问题
        S-->>W: allow
        W->>M: 读取短期 / 长期上下文
        W->>P: 选择 direct、auto 或 planned v3
        P->>R: dense / hybrid 检索与可选过滤
        R-->>P: 已融合和重排的证据
        P-->>W: 来源、计划与诊断
        W->>G: 不可信证据边界 + 回答格式
        G-->>W: 带引用的回答
        W->>A: 忠实度、引用、覆盖和安全检查
        A-->>W: quality_pass 与问题列表
        W->>M: 仅正常流程写入本轮记忆
        W-->>U: 答案、来源、路由、审计和耗时
    end
```

## 3. 索引发布生命周期

```mermaid
stateDiagram-v2
    [*] --> Planned: 比较来源与 chunk 指纹
    Planned --> Built: 构建不可变候选并复用 embedding
    Built --> Validated: manifest / hash / Chroma 一致
    Validated --> Approved: 冻结检索与完整测试通过
    Validated --> Blocked: 任一 required case 或结构检查失败
    Approved --> Active: 原子切换 active pointer
    Active --> Previous: 新版本出现回归
    Previous --> Active: rollback 到上一版本
    Blocked --> [*]
```

构建与启用分离。`--activate` 不能绕过发布门槛；Web 在请求之间检测 active pointer，单次请求始终固定使用同一个索引快照。

## 4. 主要模块映射

| 责任 | 主要实现 |
|---|---|
| 来源、切分与索引构建 | `experiments/16_llm_rag_sources/`、`17_llm_rag_chunking/`、`33_incremental_index/` |
| Dense、BM25、融合与 planned retrieval | `src/retrieval.py`、`src/bm25_retrieval.py`、`src/query_planning.py` |
| 并行召回与缓存 | `src/retrieval_cache.py`、`experiments/35_planned_retrieval_parallel_cache/` |
| 上下文与证据边界 | `src/context_assembly.py` |
| 安全前置与安全审计 | `src/rag_security.py`、`experiments/36_rag_security_regression/` |
| 生成、记忆与 Web API | `webapp/server.py`、`src/long_memory.py` |
| 答案与覆盖审计 | `src/answer_audit.py`、`src/coverage_audit.py` |
| 索引和代码发布门槛 | `src/index_release_gate.py`、`experiments/37_unified_quality_gate/` |

## 5. 设计边界

- `direct` 仍是默认检索模式；holdout 证明 planned v3 无退化，但没有证明显著提升。
- 对话记忆只帮助解释指代和偏好，不是外部事实来源。
- cross-encoder 保留为可选实验能力，固定候选实验没有支持把它设为默认。
- CI 不运行真实索引门槛，因为本地 Chroma、chunks 和 embedding 模型不提交 Git；本地传入 manifest 后补齐这层检查。
- 安全分类器采用高精度确定性规则，目标是建立可重复的工程底线，不冒充通用内容安全系统。

指标证据见 [metrics.md](metrics.md)，决策依据见 [technical_decisions.md](technical_decisions.md)。
