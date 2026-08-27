# 本地与 CI 统一质量门槛

日期：2026-08-27

## 目的

项目原有 GitHub Actions 和各阶段脚本能够分别运行测试，但本地、CI、安全回归和索引发布使用不同入口，容易出现漏跑或标准漂移。本阶段将稳定检查收敛到一个共享命令，同时保留本地索引不上传 Git 的边界。

## 统一入口

入口：`experiments/37_unified_quality_gate/run_quality_gate.py`

核心阶段：

1. `dependency_check`：检查已安装 Python 依赖的一致性；
2. `compile_python`：编译检查 `src`、`experiments`、`webapp` 和 `tests`；
3. `unit_tests`：运行完整 `unittest discover`；
4. `security_gate`：运行冻结的 `rag_security_v1`；
5. `javascript_syntax`：使用 Node 检查 Web 前端脚本；
6. `index_release_gate`：只有显式传入候选 manifest 时才运行。

任一必需阶段失败，统一门槛即失败。报告只保存阶段状态、返回码、计数、耗时和失败测试名称，不保存断言正文、子进程输出、文档正文、模型回答或凭据。

## CI 与本地边界

- GitHub Actions 在 pull request、`main` push 和手动触发时运行核心门槛，并强制要求 Node。
- CI 不运行索引门槛，因为 Chroma、处理后 chunks 和本地 embedding 模型按项目约定不提交 Git。
- 本地默认自动查找 Node；缺少 Node 时标记为 skipped。`--require-node` 可将其变成强制检查，`--node-executable` 可显式指定运行时。
- 传入 `--manifest` 后，统一入口调用既有索引发布门禁，执行候选结构、冻结检索和完整测试检查，但不会自动激活索引。

## 验收结果

核心本地门槛：

- dependency check：passed；
- Python compile：passed；
- unit tests：135/135 passed；
- security gate：8/8 passed；
- JavaScript syntax：passed。

追加 `validation-copy-20260825` 候选 manifest 后，`index_release_gate` 同样 passed，统一门槛最终状态为 passed。生成报告位于本地 `data/runtime/unified_quality_gate/`，不提交 Git。

## 后续方向

工程质量门槛已形成持续执行入口。下一阶段转向作品集交付：架构图、演示问题、指标表和关键技术决策说明。
