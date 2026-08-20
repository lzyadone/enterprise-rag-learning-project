# DeepSeek 答案审计与修复优化记录

日期：2026-08-19
更新：2026-08-20

## 为什么做这一步

上一轮已经把检索链路优化到：

```text
QueryPlan -> 多路检索 -> RRF -> rerank -> category coverage -> top_k context
```

检索评估结果稳定：

```text
planned hit@1: 8/8
planned hit@3: 8/8
```

但实际问答暴露出新问题：

- 本地 `qwen2.5:1.5b` 能复述资料，但经常不按要求输出引用编号。
- 没有“来源”列表时，答案很难自动审计。
- 这属于 answer faithfulness / citation control 问题，不是检索问题。

所以这一轮引入 DeepSeek 作为可选的强模型生成、审计和修复层。

## 参考资料

DeepSeek 官方 API 文档：

```text
https://api-docs.deepseek.com/
```

当前实现按 OpenAI-compatible Chat Completions 方式调用：

```text
POST https://api.deepseek.com/chat/completions
```

默认模型：

```text
deepseek-v4-flash
```

审计模型：

```text
deepseek-chat
```

原因：

```text
deepseek-v4-flash 在审计场景中会输出 reasoning_content，若 max_tokens 不够，可能还没输出 JSON 就被截断。
deepseek-chat 更适合稳定返回结构化审计 JSON。
```

## 安全处理

API key 不写入代码，不写入日志，不写入文档。

代码只读取环境变量：

```text
DEEPSEEK_API_KEY
```

新增示例文件：

```text
.env.example
```

真实 `.env` 和 `.env.*` 已在 `.gitignore` 中忽略。

PowerShell 临时设置方式：

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
```

只对当前终端窗口有效，关闭终端后失效。

## 新增代码

DeepSeek 客户端：

```text
src/deepseek_client.py
```

答案审计：

```text
src/answer_audit.py
```

答案修复：

```text
src/answer_repair.py
```

批量答案审计实验：

```text
experiments/21_answer_faithfulness_audit/evaluate_answers.py
```

问答脚本增强：

```text
experiments/19_llm_rag_qa/ask.py
```

## 新链路

现在系统可以支持三种模式。

### 模式 1：本地生成，只做规则审计

```text
Ollama 生成 -> deterministic audit
```

命令：

```powershell
python experiments\21_answer_faithfulness_audit\evaluate_answers.py --skip-llm-audit --output-dir eval\answer_faithfulness_smoke_rule_only
```

当前结果：

```text
total: 4
rule_pass: 0/4
```

结论：

```text
本地小模型生成质量不足，尤其是不稳定输出引用和来源。
```

强化 prompt 后再次运行：

```powershell
python experiments\21_answer_faithfulness_audit\evaluate_answers.py --skip-llm-audit --output-dir eval\answer_faithfulness_smoke_rule_only_prompt_v3
```

结果仍为：

```text
rule_pass: 0/4
```

但单次问答中，简单问题已经可以被 prompt 约束到通过规则审计：

```powershell
python experiments\19_llm_rag_qa\ask.py --query "rerank 和普通向量检索有什么关系？" --audit-answer --top-k 3 --candidate-k 8
```

观察：

- answer 输出了引用和来源。
- deterministic audit 通过。
- DeepSeek audit 因未设置 `DEEPSEEK_API_KEY` 被记录为 `audit_error`，脚本没有中断。

结论：

```text
Prompt 能改善单次输出，但本地小模型批量稳定性不足；DeepSeek 生成/修复层是必要的。
```

### 模式 2：本地生成，DeepSeek 修复

```text
Ollama 生成 -> deterministic audit -> DeepSeek repair -> audit final answer
```

命令：

```powershell
python experiments\21_answer_faithfulness_audit\evaluate_answers.py --skip-llm-audit --repair-with-deepseek --output-dir eval\answer_faithfulness_smoke_repair
```

适用场景：

- 想保留本地模型演示。
- 但又希望最终答案格式、引用和来源更稳定。

### 模式 3：DeepSeek 生成 + DeepSeek 审计

```text
DeepSeek 生成 -> deterministic audit -> DeepSeek judge audit
```

命令：

```powershell
python experiments\21_answer_faithfulness_audit\evaluate_answers.py --llm-provider deepseek --output-dir eval\answer_faithfulness_smoke_deepseek
```

当前推荐：

```text
generation model: deepseek-v4-flash
audit model: deepseek-chat
```

适用场景：

- 追求答案质量。
- 展示更接近真实产品的“强模型生成 + 自动评估”链路。

## 单次问答命令

DeepSeek 直接生成：

```powershell
python experiments\19_llm_rag_qa\ask.py --query "文档切分为什么不能只用固定窗口？" --llm-provider deepseek --audit-answer
```

本地生成，DeepSeek 修复和审计：

```powershell
python experiments\19_llm_rag_qa\ask.py --query "文档切分为什么不能只用固定窗口？" --repair-answer --audit-answer
```

## 当前验证

已验证：

- Python 语法检查通过。
- 无 key 时脚本不会崩溃，会把错误记录为 `repair_error` 或 `audit_error`。
- 本地生成的答案规则审计为 `0/4`，说明审计层能抓到引用格式问题。
- 单次问答缺少 key 时仍可输出本地答案和 deterministic audit。

未验证：

- 这一条已经在 2026-08-20 完成验证。

实际运行命令：

```powershell
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
python experiments\21_answer_faithfulness_audit\evaluate_answers.py --llm-provider deepseek --output-dir eval\answer_faithfulness_smoke_deepseek_v3
```

结果：

```text
total: 4
llm_provider: deepseek
deepseek_model: deepseek-v4-flash
deepseek_audit_model: deepseek-chat
rule_pass: 4/4 = 100%
overall_pass: 4/4 = 100%
audit_errors: 0
repair_errors: 0
```

结果文件：

```text
eval/answer_faithfulness_smoke_deepseek_v3/results.jsonl
eval/answer_faithfulness_smoke_deepseek_v3/summary.md
```

结论：

```text
检索层已经稳定，DeepSeek 生成 + DeepSeek 审计可以产出带来源、可审计、忠实于上下文的回答。
```
