# 版本化增量索引与无重启切换

## 目标

此前知识库更新依赖人工构建候选目录、验证后再替换默认 Chroma 目录。这个流程能保护旧索引，但缺少统一的来源差异判断、向量复用、版本清单、原子切换和回滚入口。本阶段把这些步骤收敛为可验证的版本化流程，同时保持现有检索和生成策略不变。

## 设计

`src/index_versioning.py` 负责索引生命周期：

1. 对规范化文档内容、来源 metadata 和切分配置计算 SHA-256 指纹；
2. 将来源分为 added、changed、deleted 和 unchanged；
3. unchanged 文档直接复用上一版本的不可变 chunks，只有新增或变化文档重新切分；
4. chunk 使用 `chunk_id + text_hash` 校验，同模型下优先复用旧 embedding；
5. 候选 Chroma 只写入当前期望的 chunks，因此删除来源不会残留旧向量；
6. 每个版本保存独立 `manifest.json`、`chunks.jsonl` 和 Chroma 目录；
7. `data/runtime/active_index.json` 通过临时文件替换完成原子切换，并记录上一版本用于回滚。

版本目录和激活指针属于本地运行产物，继续由 `.gitignore` 排除。代码不会自动清理旧版本，避免误删仍需审计或回滚的索引。

## 操作流程

管理入口为 `experiments/33_incremental_index/manage_index.py`：

```powershell
# 仅首次执行：把现有索引封装为不可变基线
python experiments\33_incremental_index\manage_index.py bootstrap --version-id baseline-YYYYMMDD

# 查看来源差异，不写索引
python experiments\33_incremental_index\manage_index.py plan

# 构建候选；不会自动启用
python experiments\33_incremental_index\manage_index.py build --version-id index-YYYYMMDD

# 校验 manifest、chunks 和 Chroma ID/数量完全一致
python experiments\33_incremental_index\manage_index.py validate --manifest data\indexes\llm_rag_versions\index-YYYYMMDD\manifest.json

# 离线检索/测试通过后显式启用
python experiments\33_incremental_index\manage_index.py activate --manifest data\indexes\llm_rag_versions\index-YYYYMMDD\manifest.json

# 一步回到上一版本
python experiments\33_incremental_index\manage_index.py rollback
```

`build` 与 `activate` 有意分离。候选必须先通过结构校验、固定问题检索和完整测试，才能进入运行路径。

## Web 热加载与缓存

Web 默认优先读取激活指针；没有指针时继续兼容旧的 `data/indexes/llm_rag_chroma`。每次请求前轻量检查指针文件，版本变化时先加载并验证新 Chroma，再一次性替换运行时引用，不需要重启服务。

版本切换会清空依赖 chunks 的 BM25 缓存。查询 embedding 缓存与知识库内容无关，因此保留，避免无意义的重新计算。当前请求持有自己的索引快照，切换不会把一次问答拆到两个版本上。

## 真实验收

现有 54 documents / 942 chunks / 942 Chroma rows 已封装为 `baseline-20260825-942`：

- 复用 embedding：942；
- 新生成 embedding：0；
- 删除向量：0；
- 网页进程未重启，从 legacy 自动切换到 baseline。

随后强制构建等价候选 `validation-copy-20260825`，54 份文档和 942 个 embedding 全部复用。候选结构验证通过后，固定问题检索结果与上一阶段一致：

- `natural_dev_007`：`langchain_pypdf_metadata::chunk_0000` 排名第 1；
- `natural_dev_009`：两个 PyPDFParser 片段排名第 1、2，两个 TextSplitter metadata 片段排名第 3、5。

候选启用后，同一 Web 进程识别到 `validation-copy-20260825`；执行 rollback 后恢复 `baseline-20260825-942`，最终计数保持 54/942/942。

## 边界

- 首次 bootstrap 没有更早的版本可回滚；从第二个已启用版本开始，指针始终记录上一版本。
- embedding 模型变化时不会复用旧向量，所有期望 chunks 都会重新生成 embedding。
- 当前离线检索回归仍由既有实验脚本和固定问题执行；下一阶段应把知识库更新验收收敛为一个可重复的发布门禁命令。
