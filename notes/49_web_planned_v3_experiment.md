# Web Planned V3 Experiment

## 变更背景

Planner v3 在 `rag_natural_query_holdout_v3` 上通过预注册门槛，且 Codex 辅助抽查
未发现严重 qrels 分歧。按 release decision，下一步可以把 Web 的手动 `planned`
实验路径升级到 Planner v3，但仍不替换默认 `direct`。

## 工程变更

- `RAG_DEFAULT_RETRIEVAL_MODE` 默认仍为 `direct`。
- `RAG_PLANNED_FUSION_MODE` 默认从 `anchored` 改为 `conservative`。
- Web 后端允许 `planned_fusion_mode=conservative`。
- 当 planned fusion 为 `conservative` 时，Web 后端使用 `plan_query_v3()` 构造路由计划。
- 显式设置 `RAG_PLANNED_FUSION_MODE=legacy` 或 `anchored` 仍可回到旧实验路径。
- 前端的 planned 选项标记为 `planned v3 (实验)`。

## 验证

```powershell
python -m unittest tests.test_web_defaults tests.test_query_planning_v3 tests.test_retrieval_fusion tests.test_retrieval_routing tests.test_auto_routing_evaluation
```

结果：37 tests passed。

```powershell
C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe --check webapp\static\app.js
```

结果：JavaScript syntax check passed。

## 边界

这不是默认检索策略切换。用户不手动选择 `planned` 或 `auto` 时，Web 仍走 direct。
