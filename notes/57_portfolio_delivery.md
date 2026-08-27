# 作品集交付材料

日期：2026-08-27

## 目标

把已经完成的 RAG 工程整理成一套可以快速阅读、现场演示并追溯证据的作品集材料。重点不是增加新功能，而是让架构、指标、取舍和边界保持一致且可验证。

## 交付内容

入口：`docs/portfolio/README.md`

1. `architecture.md`：数据与索引、在线问答、质量与发布三个平面；另含请求时序和索引发布状态图。
2. `metrics.md`：知识库、检索、holdout、Web、安全、索引和 CI 指标总表，每项链接到仓库内证据。
3. `technical_decisions.md`：10 项关键决策，记录问题、选择、证据和代价。
4. `demo_script.md`：10-12 分钟演示流程，覆盖 direct、planned v3、PyPDFLoader、记忆、安全拒绝和知识边界。
5. `tests/test_portfolio_docs.py`：检查文档存在性、相对链接、结构化指标和演示/架构覆盖。

根 `README.md` 已增加作品集入口并更新简化架构；旧 `docs/demo.md` 保留为历史案例，同时指向新的演示脚本。

## 指标口径

- 开发集只说明 planner v3 值得进入独立验证。
- 32 题 holdout 的正确结论是 direct 与 planned v3 无退化，不宣传显著提升。
- 固定端到端 10/10 和 Web E2E 4/4 是回归结果，不冒充开放流量正确率。
- 热缓存 12.67x 只表示局部检索微基准，不表述为页面端到端加速。
- LLM qrels 和 12 条辅助抽查不冒充生产级人工金标。

## 验收

- 作品集文档专项测试：4/4 passed。
- 统一质量门槛：dependency、compile、139/139 tests、8/8 security、JavaScript 全部 passed。
- 文档相对链接、关键结构化指标和演示覆盖由测试持续检查。
- `git diff --check` 在提交前通过。

## 下一步

工程与书面作品集已经形成完整闭环。若继续发布包装，建议只做最终发行工作：从真实 Web 捕获不含凭据的桌面截图、录制短演示，并创建带版本说明的 GitHub release；若继续工程研究，则应引入新的真实业务域和小规模真人 qrels，而不是继续在现有开发集调参。
