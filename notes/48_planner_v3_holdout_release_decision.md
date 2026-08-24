# Planner V3 Holdout Release Decision

## 为什么这次可以算独立验证

Planner v3 的开发集已经参与过规则设计，因此不能作为泛化证据。本轮先冻结
`rag_natural_query_holdout_v3`，再运行候选检索、盲标和指标计算。

冻结提交为 `6fc4d22 Freeze planner v3 holdout dataset`，包含：

- 32 道自然问题：16 focused、16 compound；
- 48 个唯一 source anchor chunk；
- 数据集 SHA-256：
  `85d8ff3c55183ba83ed6d116438d9fcbac36e0e6e31af3ef4beb1f5e16b18b01`；
- source anchor SHA-256：
  `daf3cd40a626d4609de8b049feef8ef08cd0c0aa918a5f459e55bac81aeb9805`；
- 固定随机种子：`20260824`；
- 预注册协议：`eval/benchmarks/rag_natural_query_holdout_v3/README.md`。

source anchors 只证明题目构造时的可答性，不是 qrels，也没有提供给候选生成器或
LLM 标注器。

## 候选池与盲标

候选系统固定为：

- `direct_hybrid`
- `planned_v3_hybrid`

每题每系统取 Top 10，合并后形成 320 个 query/chunk pair。这个 holdout 上两个
系统的 Top 10 完全重合，因此每题 union pool 为 10 条。

DeepSeek 在隐藏系统名、排名、分数、检索通道和 query plan 的条件下完成 320/320
个 pair 的 0-3 级盲标：

| relevance | 数量 |
|---:|---:|
| 0 | 43 |
| 1 | 116 |
| 2 | 92 |
| 3 | 69 |

相关阈值仍为 grade >= 2。32/32 个问题都有至少一个可用相关候选，覆盖率为 100%。

## 系统结果

| system | Recall@5 | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median |
|---|---:|---:|---:|---:|---:|---:|
| direct hybrid | 0.664 | 1.000 | 0.503 | 0.839 | 0.843 | 0.70s |
| planned v3 hybrid | 0.664 | 1.000 | 0.503 | 0.839 | 0.843 | 0.53s |
| conservative auto | - | 1.000 | 0.503 | 0.839 | 0.843 | 0.70s |

`planned_v3_hybrid` 的正式延迟使用单独运行的
`data/runtime/planner_v3_holdout_latency/pool_manifest.jsonl` 覆盖 pooled-run
中的缓存耗时。首题有冷启动；中位数仍低于 12 秒预算。

## 自动路由结果

conservative auto 选择：

- direct：31 题；
- planned v3：1 题。

逐题 oracle agreement 为 31/32，即 96.9%。唯一不一致样例中 direct 与 planned v3
nDCG 完全相同，脚本按 tie-break 把 oracle 记为 direct，因此不是质量退化。

## 预注册门槛

| check | 结果 | actual | required |
|---|---|---:|---:|
| 可评测问题覆盖率 | PASS | 100.0% | >= 90.0% |
| auto nDCG 非劣 | PASS | 0.8434 | >= 0.8234 |
| auto Recall 非劣 | PASS | 1.0000 | >= 0.9800 |
| auto MRR 非劣 | PASS | 0.8385 | >= 0.7885 |
| oracle agreement | PASS | 0.9688 | >= 0.7500 |
| planned route 单题严重损失 | PASS | 0 cases | 0 cases |
| median latency | PASS | 0.7018s | <= 12.0s |

总体结果为 **PASS**。

## 辅助抽查

已准备 12 条固定抽查样本：
  `eval/benchmarks/rag_natural_query_holdout_v3/human_spot_audit_sample.jsonl`。

Codex 对这 12 条做了独立辅助复核，结果记录在：

- `eval/benchmarks/rag_natural_query_holdout_v3/codex_spot_audit.jsonl`
- `eval/benchmarks/rag_natural_query_holdout_v3/codex_spot_audit_summary.json`

结果：

- exact agreement：9/12，即 75.0%；
- within-one agreement：12/12，即 100.0%；
- severe disagreement：0/12；
- 3 条分歧均为轻微偏高，主要出现在候选 chunk 只回答复合问题的一部分，或命中相邻 API 而不是目标 API。

将这 12 条辅助复核分数临时套入 qrels 后，direct、planned v3 和 conservative auto
的相对结论不变，预注册门槛仍然通过。这个结果可以作为独立辅助抽查记录，但不能
宣传为人工金标或生产级人工审计。

## 工程决定

- Web 默认继续保持 `direct`。
- Planner v3 通过本轮 LLM-labeled holdout 的预注册门槛，可作为 release candidate。
- 辅助抽查未发现严重错标，足以支持在作品集/学习项目中进入 Web 实验模式。
- Web 手动 planned 实验路径可以升级到 Planner v3；仍不立即替换默认 direct。
- 如果后续要对外发布生产级结论，仍需要真人复核一小批 qrels。

## 可复现产物

- 冻结协议：`eval/benchmarks/rag_natural_query_holdout_v3/README.md`
- 问题集：`eval/datasets/rag_natural_query_holdout_v3.jsonl`
- source anchors：`eval/benchmarks/rag_natural_query_holdout_v3/source_anchors.jsonl`
- LLM qrels：`eval/benchmarks/rag_natural_query_holdout_v3/qrels_llm.jsonl`
- LLM qrels summary：`eval/benchmarks/rag_natural_query_holdout_v3/qrels_llm_summary.json`
- 候选池 manifest：`eval/natural_query_holdout_v3/pool_manifest.jsonl`
- 系统评估摘要：`eval/natural_query_holdout_v3/system_evaluation/summary.md`
- 自动路由摘要：`eval/natural_query_holdout_v3/conservative_routing_evaluation/summary.md`

仓库的 `.gitignore` 会排除候选正文、逐题结果、system runs 和本地 runtime
latency 文件。clone 后可按协议重建这些可再生产物。

## 复现命令

```powershell
python experiments\26_retrieval_pooling\build_union_pool.py `
  --dataset eval\datasets\rag_natural_query_holdout_v3.jsonl `
  --systems direct_hybrid planned_v3_hybrid `
  --all-dataset-cases --no-inherit-qrels --pool-depth 10 `
  --output-dir eval\natural_query_holdout_v3 --force

python experiments\26_retrieval_pooling\label_union_with_llm.py `
  --candidate-pools eval\natural_query_holdout_v3\candidate_pools.jsonl `
  --output eval\benchmarks\rag_natural_query_holdout_v3\qrels_llm.jsonl `
  --summary eval\benchmarks\rag_natural_query_holdout_v3\qrels_llm_summary.json `
  --rejudge-all --skip-human-audit --batch-size 5 --force

python experiments\26_retrieval_pooling\build_union_pool.py `
  --dataset eval\datasets\rag_natural_query_holdout_v3.jsonl `
  --systems planned_v3_hybrid --all-dataset-cases --no-inherit-qrels `
  --pool-depth 10 --output-dir data\runtime\planner_v3_holdout_latency --force

python experiments\26_retrieval_pooling\evaluate_candidate_generators.py `
  --candidate-pools eval\natural_query_holdout_v3\candidate_pools.jsonl `
  --manifest eval\natural_query_holdout_v3\pool_manifest.jsonl `
  --latency-manifest data\runtime\planner_v3_holdout_latency\pool_manifest.jsonl `
  --qrels eval\benchmarks\rag_natural_query_holdout_v3\qrels_llm.jsonl `
  --output-dir eval\natural_query_holdout_v3\system_evaluation

python experiments\28_auto_retrieval_routing\evaluate_router.py `
  --manifest eval\natural_query_holdout_v3\pool_manifest.jsonl `
  --candidate-pools eval\natural_query_holdout_v3\candidate_pools.jsonl `
  --qrels eval\benchmarks\rag_natural_query_holdout_v3\qrels_llm.jsonl `
  --latency-results data\runtime\planner_v3_holdout_latency\pool_manifest.jsonl `
  --direct-system direct_hybrid --planned-system planned_v3_hybrid `
  --planner-version conservative --evaluation-role holdout `
  --output-dir eval\natural_query_holdout_v3\conservative_routing_evaluation
```
