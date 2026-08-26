# Planned Retrieval 并行召回与结果缓存

## 问题

查询 embedding 复用已经消除了同一句规划查询的重复向量化，但多 aspect 的 planned retrieval 仍按 run 串行执行 Chroma/BM25。相同问题在短时间内重复请求时，也会重复召回候选和执行重排。

本阶段只优化检索执行层，不修改 query plan、候选窗口、RRF 权重、覆盖选择或答案生成。

## 并行边界

planned retrieval 先按原有方式为唯一查询准备 embedding，然后把彼此独立的召回 run 交给有上限的线程池：

- 默认最多 4 workers；
- 单 run 自动走串行路径，不创建线程池；
- Chroma dense 与 BM25 hybrid run 可以并行；
- `executor.map` 按原请求顺序收集结果，保持 weighted RRF 输入顺序；
- 所有召回完成后仍只执行一次融合、一次集中重排和一次覆盖选择；
- cross-encoder 推理不在线程池内重复执行，原有 GPU 推理锁继续有效。

## 候选与重排缓存

`src/retrieval_cache.py` 提供两个线程安全的进程内 LRU：

- candidate cache：最多 512 项；
- rerank cache：最多 128 项。

candidate key 包含索引命名空间、embedding 模型与 host、检索策略、规范化查询、category、top_k 和 candidate_k。版本化索引使用 `version_id`；自定义 legacy DB 使用实际数据库路径，避免多个本地库共用 `legacy` 键。

rerank key 额外包含候选顺序、chunk/text hash、检索分数、距离、aspect 和重排器配置。缓存写入与读取都使用深拷贝，因为 RRF、lexical rerank 和 plan boost 会原地修改候选对象。

Web 检测到 active index 切换时，同时清空 BM25、candidate 和 rerank 缓存。`reuse_query_embeddings=False` 会连同新缓存一起绕过，保证旧的严格 A/B 仍是无缓存控制组。

## 测试

新增测试验证：

- 多 run 的实际并发数大于 1；
- 并行结果按输入 run 顺序返回；
- 串行与并行的最终 chunk 顺序和分数一致；
- candidate/rerank 命中不共享可变对象；
- 不同索引命名空间不会串用候选；
- 非法 worker 数被拒绝；
- embedding 诊断绕过不会误命中新缓存；
- Web 热切换索引后缓存归零。

## 真实 A/B

脚本：

```powershell
python experiments\35_planned_retrieval_parallel_cache\benchmark.py
```

使用 `baseline-20260825-942`，选择两个都会生成 7 个 conservative v3 runs 的复合问题。每题重复 3 次；查询 embedding 先预热，以隔离召回并行和结果缓存的贡献。比较：

1. serial：串行召回，不使用 candidate/rerank cache；
2. parallel cold：4 workers，并启用空缓存；
3. parallel warm：立即重复同一请求，复用候选和重排结果。

| case | runs | serial median | parallel cold | cold speedup | warm cache | warm speedup | Top-7 一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rag_compound_overview | 7 | 0.0655s | 0.0560s | 1.17x | 0.0055s | 11.92x | 3/3 |
| rag_evaluation_reliability | 7 | 0.0677s | 0.0538s | 1.26x | 0.0047s | 14.30x | 3/3 |
| overall | 7 | 0.0668s | 0.0549s | 1.22x | 0.0053s | 12.67x | 6/6 |

冷缓存每题记录 7 次 candidate miss 和 1 次 rerank miss；紧接的重复请求记录 7 次 candidate hit 和 1 次 rerank hit。全部 6 组串行、并行冷缓存和并行热缓存 Top-7 顺序完全相同。

## 结论与边界

并行召回可以作为默认 planned 路径，真实多 run 问题获得约 17%-26% 的召回阶段缩短。热缓存对重复问题非常有效，但 `12.67x` 只代表已预热 embedding、相同进程和相同索引版本下的局部检索阶段，不能表述为页面端到端提速。

当前 Chroma/BM25 本地查询本身已经很快，因此冷并行收益温和。答案生成仍是页面延迟大头。单 run 问题不会获得并行收益，也不会承担线程池开销。

下一阶段不继续扩大缓存范围，优先增加越权、提示注入、来源冲突和知识库外问题的安全回归。
