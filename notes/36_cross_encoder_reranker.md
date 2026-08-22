# 36. Cross-encoder 重排与固定候选评测

## 本阶段目标

在现有 `planned retrieval + dense/BM25 + RRF` 后加入真正的语义重排器，并回答三个问题：

1. 这台 RTX 2050 4GB 笔记本能否运行多语种 cross-encoder？
2. 模型重排是否比无重排和词法重排更好？
3. 它是否应该进入默认在线链路？

## 为什么选择 bge-reranker-v2-m3

- 官方模型卡标注为多语种 encoder-only reranker，许可证为 Apache-2.0。
- 与知识库当前的 `bge-m3` embedding 同属 BGE-M3 系列，适合中文问题与英文资料的跨语言场景。
- 官方 FlagEmbedding 文档支持 FP16、CUDA 设备、query/passage 最大长度和批量推理。

参考：

- [BAAI/bge-reranker-v2-m3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [FlagEmbedding reranker inference](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/inference/reranker/README.md)
- [Ollama embedding API](https://github.com/ollama/ollama/blob/main/docs/api.md)

## 实现

`src/cross_encoder_reranking.py` 提供统一接口：

- `fastembed`：`bge-reranker-base`，ONNX CPU，对照基线。
- `transformers`：`bge-reranker-v2-m3`，CUDA FP16，多语种实验后端。
- 懒加载和进程内模型缓存。
- GPU 推理锁，避免并发请求在 4GB 显存上重复占用。
- batch size、max length、device、model 和 cache dir 均可配置。
- CUDA OOM 给出明确修复建议，不静默退化。

本机短样本结果：

```text
GPU: NVIDIA GeForce RTX 2050 4GB
peak allocated GPU memory: 1093.8 MB
3 pairs inference: 4.526s
relevant English passage > BM25 passage > weather passage
```

## 小显存模型生命周期

第一次评测失败于 `Memory allocation failure`。模型能放入显存，但 Ollama `bge-m3`、Chroma/ONNX 和 Transformers reranker 同时驻留时，16GB 系统内存出现瞬时峰值。

修复方式：

1. 先生成并保存固定候选池。
2. 通过 Ollama `/api/embed` 的 `keep_alive: 0` 卸载 embedding 模型。
3. 释放 Chroma client 后再加载 GPU reranker。
4. 在线 cross-encoder 请求默认采用同样的互斥驻留策略。

`RAG_EXCLUSIVE_MODEL_RESIDENCY=0` 可在资源充足的服务器上关闭该策略。

## 为什么固定候选池

如果每个重排器都重新检索，候选差异会混入结果，无法确定改善来自召回还是排序。实验先把每个问题的 16 个候选保存为 JSONL，然后所有模式读取相同候选，只改变排序方法。

候选快照可能包含较长原文，因此被 `.gitignore` 排除；评测摘要保留在 Git 中。

## 全量结果

数据为 8 个知识库问题，`top_k=7`、`candidate_k=16`，候选来自 planned hybrid retrieval。

| 模式 | 双指标通过率 | nDCG@7 | 类别 MRR | 证据词召回 | 平均重排耗时 |
|---|---:|---:|---:|---:|---:|
| none | 100% | 0.725 | 0.938 | 0.950 | 0.00s |
| lexical | 100% | 0.722 | 0.938 | 0.950 | 0.01s |
| cross_encoder_multilingual | 100% | 0.700 | 0.938 | 0.950 | 4.79s |
| cross_encoder_fused | 100% | 0.701 | 0.875 | 0.950 | 4.83s |

纯 cross-encoder 在 embedding、query planning 和 vector DB 问题上改善 nDCG，但在 chunking、reranking 和企业复合检索问题上退化。规划层仍能保证类别与证据覆盖，因此所有模式的双指标通过率保持 100%。

## 秩融合实验

模型 logits 与 RRF/检索分数不在同一量纲，不能直接线性相加。实验使用未调参的 rank fusion：

```text
final = 0.65 / (60 + retrieval_rank) + 0.35 / (60 + model_rank)
```

融合减轻了部分坏例，但总体没有超过无重排基线。不能为了加入 cross-encoder 而选择更差、更慢的默认链路。

## 当前决策

- 默认：`planned + hybrid + lexical`，保持已验证端到端结果。
- 可选：Web 中提供多语种 GPU semantic rerank，用于观察单问题变化。
- 不宣称：当前 cross-encoder 提升了整体检索质量。
- 下一步：建立人工 chunk 相关性标签，按问题类型做独立验证，再研究自适应 reranker routing。

## 评测边界

当前 nDCG 标签由预期类别和证据词自动生成，只适合在相同候选池内比较，不能替代人工逐 chunk 判断。8 个问题也不足以证明跨领域泛化；本阶段结论仅适用于当前 52 篇、938 chunks 的大模型/RAG 知识库。
