# Web Remote API and E2E Regression

日期：2026-08-24

## 目标

在保留 Ollama 默认路径和 DeepSeek 环境配置路径的同时，允许用户在本地 Web 工作台临时使用 OpenAI 兼容远程 API，并把 direct/planned v3 手工验收固化为自动端到端回归。

## 实现

- 新增 `src/openai_compatible_client.py`：支持 API Base URL 或完整 `/chat/completions` 地址。
- 新增生成模式 `openai_compatible`，请求只记录远程模型名和 provider path。
- 新增 `POST /api/providers/test`，用最小请求检查地址、密钥和模型是否可用。
- Web 增加临时 API 地址、模型名、密码式密钥输入和连接状态。
- 新增 `experiments/32_web_e2e_regression/run_web_regression.py`，自动管理临时服务并运行 direct/planned v3 回归矩阵。
- 新增 `eval/datasets/rag_web_mode_smoke_v1.jsonl`，固定一个 focused 和一个 compound 问题。

## 安全边界

- 远程密钥不写入文件、长期记忆、浏览器存储、日志或 API 响应。
- 网页刷新后临时密钥消失。
- 远程主机必须使用 HTTPS；仅 localhost、127.0.0.1 和 ::1 允许 HTTP。
- URL 不允许嵌入用户名、密码、query 或 fragment。
- 远程 HTTP 错误不会把 provider 响应正文返回给浏览器。
- 通用远程模式不自动回退 Ollama，也不使用本机 DeepSeek 做答案修复或 LLM 审计，避免隐藏的第二次远程传输。
- 当前只支持 Bearer Authorization 的 OpenAI Chat Completions 兼容接口，不包含 Azure 特有的 `api-key` header 和 `api-version` query 形式。

## 自动回归结果

命令：

```powershell
python experiments\32_web_e2e_regression\run_web_regression.py
```

结果：4/4 通过。

| case | mode | sources | seconds |
|---|---|---:|---:|
| focused_chunking | direct | 7 | 4.99 |
| focused_chunking | planned | 7 | 4.67 |
| compound_badcase | direct | 7 | 14.37 |
| compound_badcase | planned | 10 | 14.97 |

检查范围包括服务就绪、回答非空、来源 URL、有效引用、预期类别、requested/selected route、conservative fusion、Planner v3 版本、本地生成路径和 60 秒端到端上限。

Python 完整测试：`101 tests OK`。JavaScript 语法检查和 Python 编译检查通过。

## 页面验收

- 桌面与 390px 手机视口均通过，远程配置字段和按钮没有横向溢出。
- 默认生成模型仍为 Ollama；选择远程模式后才显示配置面板。
- 密钥输入为 password 类型，刷新页面后字段为空。
- 未填写配置时由前端阻止请求；非 HTTPS 远程地址由后端拒绝。
- 安全错误只显示 `Remote API URL must use HTTPS`，没有回显测试密钥。
- 浏览器控制台没有 warning 或 error。

## 发布决定

远程 API 保持用户手动选择，不替换默认 Ollama。真实远程账号的可用性取决于用户填写的 provider、模型权限和余额；项目测试只验证协议、错误收敛和无凭据泄漏，不保存或提交真实凭据。
