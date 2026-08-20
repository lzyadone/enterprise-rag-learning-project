# Web 工作台与上下文组装记录

日期：2026-08-20

## 本轮目标

把命令行 RAG 系统做成可视化本地 Web 工作台，并开始加入：

- QueryPlan 可视化
- 检索来源可视化
- 上下文组装可视化
- 答案审计可视化
- 会话内对话记忆

## 新增文件

上下文组装：

```text
src/context_assembly.py
```

本地 Web 后端：

```text
webapp/server.py
```

前端页面：

```text
webapp/static/index.html
webapp/static/styles.css
webapp/static/app.js
```

## 当前 Web 链路

```text
用户问题
-> 会话记忆生成 effective query
-> planned/direct retrieval
-> rerank
-> category coverage
-> context assembly
-> DeepSeek/Ollama 生成
-> deterministic audit + DeepSeek audit
-> 页面展示 answer / plan / sources / context / audit / memory
```

## 上下文组装规则

上下文被分成两类：

```text
conversation_memory: 只用于理解指代和用户偏好
retrieved_evidence: 唯一允许作为事实来源引用的资料
```

Prompt 明确要求：

```text
对话记忆不得作为事实来源。
事实判断必须来自检索资料，并标注来源编号。
```

这样可以避免把聊天历史当成知识库证据。

## 会话记忆现状

当前是第一版轻量记忆：

- 只保存在服务进程内。
- 默认保留最近 8 轮。
- 检索时如果问题较短或包含“这个、上面、刚才、继续、下一步”等指代，会把最近问题并入 effective query。
- 页面 Memory 标签会展示 session、turns、effective query 和最近对话摘要。

后续可以升级为：

- 持久化 memory store
- DeepSeek 摘要长期记忆
- 用户偏好记忆
- 项目状态记忆
- memory 与 evidence 分权重组装

## 运行命令

```powershell
python webapp\server.py --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765
```

## 2026-08-20 UI 可读性优化

用户反馈：第一版页面能展示数据，但不容易看出 RAG 效果。

本次优化：

```text
答案上方新增“本轮效果”看板：检索覆盖、上下文组装、答案审计、对话记忆。
来源卡片增加相关性条。
上下文页面增加预算进度条，并明确 memory 与 evidence 的区别。
审计页面增加“答案可以交付 / 需要复查”的结论块。
页面标签和控件改成中文。
```

验证：

```text
/api/status 正常返回。
首页可访问，HTTP 200。
app.js 使用 bundled Node 完成语法检查。
```

说明：

```text
这次没有改变 RAG 后端逻辑，只优化可视化表达，让用户一眼能看到本轮问答是否检索到资料、上下文用了多少、审计是否通过、记忆是否参与。
```

## 2026-08-20 空答案修复

现象：

```text
页面显示检索和上下文已经完成，但答案区为空，审计显示引用格式失败。
```

原因：

```text
DeepSeek 生成模型偶发返回空 content，后端没有重试，前端也没有提示空答案。
```

修复：

```text
webapp/server.py: DeepSeek 生成返回空时自动 fallback 到 deepseek-chat。
experiments/19_llm_rag_qa/ask.py: CLI 同步增加 fallback。
webapp/static/app.js: 空答案时显示明确提示，不再留白。
```

验证：

```text
问题：RAG系统分为哪些类别，有哪些关键技术，瓶颈有哪些
answer_length: 1080
overall_pass: True
```
