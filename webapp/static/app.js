const state = {
  sessionId: localStorage.getItem("rag_session_id") || "",
  lastResponse: null,
};

const el = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  bindTabs();
  el("askBtn").addEventListener("click", ask);
  el("newSessionBtn").addEventListener("click", newSession);
  el("clearMemoryBtn").addEventListener("click", clearLongMemory);
  loadStatus();
});

function bindTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      el(`${button.dataset.tab}Tab`).classList.add("active");
    });
  });
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const longCount = data.long_memory?.memory_count ?? 0;
    el("statusLine").textContent = `知识库 ${data.collection} · ${data.indexed_count} 个 chunk · 长期记忆 ${longCount} 条 · DeepSeek ${data.deepseek_key ? "已配置（未验证）" : "未配置"}`;
    el("llmProvider").value = data.default_llm_provider || (data.deepseek_key ? "deepseek" : "ollama");
  } catch (error) {
    el("statusLine").textContent = `status error: ${error.message}`;
  }
}

function newSession() {
  state.sessionId = crypto.randomUUID();
  localStorage.setItem("rag_session_id", state.sessionId);
  state.lastResponse = null;
  el("answerBox").textContent = "新会话已创建。";
  el("effectSummary").innerHTML = `<div class="empty-state">新会话已创建。短期记忆重新累计，长期记忆继续保留。</div>`;
  renderEmpty();
}

async function clearLongMemory() {
  if (!window.confirm("确定清空本地长期记忆吗？知识库资料不会被删除。")) return;
  try {
    const res = await fetch("/api/memory/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    state.lastResponse = null;
    el("answerBox").textContent = "长期记忆已清空。";
    el("effectSummary").innerHTML = `<div class="empty-state">长期记忆已清空，知识库索引未改变。</div>`;
    renderEmpty();
    loadStatus();
  } catch (error) {
    showError(error.message);
  }
}

async function ask() {
  const query = el("queryInput").value.trim();
  if (!query) return;

  setBusy(true);
  showError("");
  el("answerBox").textContent = "检索、组装上下文、生成答案中...";
  el("effectSummary").innerHTML = `<div class="empty-state">正在执行：问题理解 -> 检索 -> 上下文组装 -> 生成 -> 审计</div>`;

  const payload = {
    session_id: state.sessionId,
    query,
    llm_provider: el("llmProvider").value,
    retrieval_mode: el("retrievalMode").value,
    retrieval_strategy: el("retrievalStrategy").value,
    rerank_mode: el("rerankMode").value,
    top_k: Number(el("topK").value),
    candidate_k: Number(el("candidateK").value),
    max_context_chars: Number(el("maxContextChars").value),
    latency_budget_ms: Number(el("latencyBudgetMs").value),
    use_memory: el("useMemory").checked,
    use_long_memory: el("useLongMemory").checked,
    audit_answer: el("auditAnswer").checked,
  };

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    state.sessionId = data.session_id;
    localStorage.setItem("rag_session_id", state.sessionId);
    state.lastResponse = data;
    renderResponse(data);
  } catch (error) {
    el("answerBox").textContent = "请求失败。";
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

function setBusy(isBusy) {
  el("askBtn").disabled = isBusy;
  el("askBtn").textContent = isBusy ? "运行中..." : "运行 RAG";
}

function showError(message) {
  const box = el("errorBox");
  box.hidden = !message;
  box.textContent = message;
}

function renderResponse(data) {
  el("answerBox").textContent = data.answer?.trim() || "生成模型返回了空答案。本轮检索和上下文组装已完成，请检查服务日志或切换生成模型。";
  el("timingText").textContent = `${data.timings.total_seconds}s · ${data.sources.length} sources · context ${data.context.used_chars}/${data.context.max_chars}`;
  renderEffectSummary(data);
  renderPlan(data);
  renderSources(data.sources);
  renderContext(data.context);
  renderAudit(data.audit);
  renderMemory(data.memory, data.effective_query);
}

function renderEffectSummary(data) {
  const auditPass = data.audit ? Boolean(data.audit.quality_pass ?? data.audit.overall_pass) : null;
  const rulePass = data.audit?.rule_audit?.rule_pass;
  const coveragePass = data.audit?.coverage_audit?.coverage_pass;
  const coverageScore = data.audit?.coverage_audit?.llm_coverage?.coverage_score ?? data.audit?.coverage_audit?.rule_coverage?.coverage_score;
  const sourceCount = data.sources.length;
  const contextRatio = data.context.max_chars ? data.context.used_chars / data.context.max_chars : 0;
  const memoryTurns = data.memory?.turn_count || 0;
  const longStats = data.memory?.long_term?.stats || {};
  const longCount = longStats.memory_count || 0;
  const retrievedLong = data.memory?.long_term?.retrieved?.length || 0;
  const route = data.routing;
  const routeMode = route?.selected_mode || data.settings.retrieval_mode || "memory";
  const routeRequest = route?.requested_mode || data.settings.requested_retrieval_mode || "-";
  const routeScore = route ? `${route.complexity_score}/${route.threshold}` : "-";
  const planCategories = data.plan?.category_filters?.length ? data.plan.category_filters.join(", ") : "none";
  const aspectCount = data.plan?.aspects?.length || 0;
  const coverageLabel = aspectCount ? `${aspectCount} 个回答面` : data.plan ? "单一问题" : "直接检索";
  const auditClass = auditPass === null ? "" : auditPass ? "good" : "bad";
  const auditText = auditPass === null ? "未审计" : auditPass ? "可交付" : "需检查";
  const ruleText = rulePass === undefined ? "未运行规则" : rulePass ? "引用格式通过" : "引用格式失败";
  const coverageClass = coveragePass === undefined ? "" : coveragePass ? "good" : "bad";
  const coverageText = coveragePass === undefined ? "未运行" : coveragePass ? "通过" : "不完整";
  el("effectSummary").innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <div class="summary-label">检索路由</div>
        <div class="summary-value">${escapeHtml(routeMode)}</div>
        <div class="summary-note">${escapeHtml(routeRequest)} · score ${escapeHtml(routeScore)}</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">检索覆盖</div>
        <div class="summary-value">${sourceCount} 条来源</div>
        <div class="summary-note">${coverageLabel} · ${escapeHtml(planCategories)}</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">上下文组装</div>
        <div class="summary-value">${Math.round(contextRatio * 100)}%</div>
        <div class="summary-note">${data.context.used_chars}/${data.context.max_chars} chars</div>
      </div>
      <div class="summary-card ${coverageClass}">
        <div class="summary-label">覆盖审计</div>
        <div class="summary-value">${coverageText}</div>
        <div class="summary-note">score ${coverageScore ?? "-"}</div>
      </div>
      <div class="summary-card ${auditClass}">
        <div class="summary-label">综合质量</div>
        <div class="summary-value">${auditText}</div>
        <div class="summary-note">${ruleText}</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">对话记忆</div>
        <div class="summary-value">${memoryTurns}/${longCount}</div>
        <div class="summary-note">短期轮数 / 长期条数 · 本轮命中 ${retrievedLong}</div>
      </div>
    </div>
  `;
}

function renderPlan(data) {
  const plan = data.plan;
  const settings = data.settings;
  const routing = data.routing;
  const generation = data.generation || {};
  const providerLabel = generation.provider_path?.length
    ? generation.provider_path.join(" -> ")
    : generation.fallback_used
      ? `${generation.requested_provider || settings.llm_provider} -> ${generation.provider || "unknown"}`
      : (generation.provider || settings.llm_provider);
  const routeReasonLabels = {
    forced_by_user: "用户强制指定",
    complexity_threshold_reached: "复杂度达到规划阈值",
    simple_or_specific_query: "问题较聚焦，使用快速路径",
    estimated_planned_latency_exceeds_budget: "预计规划耗时超过延迟预算",
    multiple_answer_aspects: "包含多个必须回答的方面",
    multi_category_expansion_without_aspects: "需要跨多个知识类别扩展检索",
    aspect_detected_with_low_category_confidence: "识别到回答方面，但类别置信度较低",
    cross_category_comparison: "需要跨类别比较",
  };
  const html = [
    `<div class="insight">
      <div class="insight-title">本轮检索意图</div>
      <div class="meta">系统先把用户问题映射到知识类别，再用原始问题和扩展问题多路检索。</div>
    </div>`,
    `<div class="kv">
      <b>requested mode</b><span>${escapeHtml(settings.requested_retrieval_mode || settings.retrieval_mode)}</span>
      <b>selected mode</b><span>${escapeHtml(settings.retrieval_mode)}</span>
      <b>planned fusion</b><span>${escapeHtml(settings.planned_fusion_mode || "legacy")}</span>
      <b>channel</b><span>${escapeHtml(settings.retrieval_strategy || "dense")}</span>
      <b>effective query</b><span>${escapeHtml(data.effective_query)}</span>
      <b>provider</b><span>${escapeHtml(providerLabel)}</span>
      ${generation.fallback_reason ? `<b>fallback</b><span>${escapeHtml(generation.fallback_reason)}</span>` : ""}
      <b>rerank</b><span>${escapeHtml(settings.rerank_mode)}</span>
      ${settings.reranker ? `<b>reranker model</b><span>${escapeHtml(settings.reranker.model)}</span>` : ""}
      ${settings.reranker ? `<b>reranker runtime</b><span>${escapeHtml(`${settings.reranker.backend} / ${settings.reranker.device}`)}</span>` : ""}
      <b>top_k</b><span>${settings.top_k}</span>
      <b>candidate_k</b><span>${settings.candidate_k}</span>
      <b>latency budget</b><span>${settings.latency_budget_ms} ms</span>
    </div>`,
  ];
  if (routing) {
    const features = routing.features || {};
    html.push(`<div class="kv">
      <b>route score</b><span>${routing.complexity_score}/${routing.threshold}</span>
      <b>route reasons</b><span>${escapeHtml((routing.reasons || []).map((reason) => routeReasonLabels[reason] || reason).join("；"))}</span>
      <b>plan shape</b><span>${features.aspect_count || 0} aspects · ${features.sub_query_count || 0} sub-queries · ${features.category_count || 0} categories</span>
      <b>planned estimate</b><span>${routing.estimated_planned_latency_ms} ms</span>
      <b>plan confidence</b><span>${features.confidence ?? "-"}</span>
    </div>`);
  }
  if (plan) {
    html.push(`<div class="kv">
      <b>intent</b><span>${escapeHtml(plan.intent)}</span>
      <b>confidence</b><span>${plan.confidence}</span>
      <b>categories</b><span>${escapeHtml((plan.category_filters || []).join(", ") || "none")}</span>
    </div>`);
    if (plan.aspects?.length) {
      html.push(`<div class="aspect-list">${plan.aspects.map((aspect) => `
        <article class="aspect-card">
          <div class="source-title">${escapeHtml(aspect.name)}</div>
          <div class="meta">${escapeHtml((aspect.categories || []).join(", "))}</div>
          <div class="preview">${escapeHtml(aspect.question || "")}</div>
          ${aspect.search_query ? `<div class="meta">search query: ${escapeHtml(aspect.search_query)}</div>` : ""}
        </article>
      `).join("")}</div>`);
    }
    html.push(`<div class="preview">${escapeHtml((plan.sub_queries || []).map((q, i) => `${i + 1}. ${q}`).join("\n"))}</div>`);
  }
  el("planTab").innerHTML = html.join("");
}

function renderSources(sources) {
  const maxScore = Math.max(...sources.map((source) => Number(source.score) || 0), 0.0001);
  el("sourcesTab").innerHTML = sources.map((source) => `
    <article class="source-item">
      <div class="source-topline">
        <div class="source-title">[${source.index}] ${escapeHtml(source.title || "")}</div>
        <span class="badge">${escapeHtml(source.category || "")}</span>
      </div>
      <div class="meta">
        score ${formatNum(source.score)} · distance ${formatNum(source.distance)} · ${escapeHtml((source.retrieval_channels || ["dense"]).join("+"))}${source.aspect ? ` · aspect ${escapeHtml(source.aspect)}` : ""}
      </div>
      <div class="score-bar"><div class="score-fill" style="width: ${Math.max(6, Math.round((Number(source.score) || 0) / maxScore * 100))}%"></div></div>
      <div class="meta">${escapeHtml(source.heading_path || "")}</div>
      <div class="meta"><a href="${escapeAttr(source.url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(source.url || "")}</a></div>
      <div class="preview">${escapeHtml(source.preview || "")}</div>
      ${source.rerank_reason ? `<div class="meta">${escapeHtml(source.rerank_reason)}</div>` : ""}
    </article>
  `).join("");
}

function renderContext(context) {
  const sections = context.sections || [];
  const ratio = context.max_chars ? context.used_chars / context.max_chars : 0;
  const fillClass = ratio > 0.82 ? "warn" : "";
  el("contextTab").innerHTML = [
    `<div class="insight">
      <div class="insight-title">上下文组装结果</div>
      <div class="meta">对话记忆只用于理解问题，检索资料才允许作为事实来源引用。</div>
      <div class="budget-bar"><div class="budget-fill ${fillClass}" style="width: ${Math.min(100, Math.round(ratio * 100))}%"></div></div>
    </div>`,
    `<div class="kv">
      <b>used chars</b><span>${context.used_chars}</span>
      <b>max chars</b><span>${context.max_chars}</span>
    </div>`,
    ...sections.map((section) => `
      <article class="context-item">
        <div class="source-title">${escapeHtml(section.name)} <span class="badge">${escapeHtml(section.role)}</span></div>
        <div class="meta">${section.char_count} chars</div>
        <div class="preview">${escapeHtml(section.content.slice(0, 1600))}${section.content.length > 1600 ? "\n..." : ""}</div>
      </article>
    `),
  ].join("");
}

function renderAudit(audit) {
  if (!audit) {
    el("auditTab").innerHTML = `<span class="badge">audit disabled</span>`;
    return;
  }
  const overallPass = Boolean(audit.quality_pass ?? audit.overall_pass);
  const overall = overallPass ? "good" : "bad";
  const llm = audit.llm_audit || {};
  const coverage = audit.coverage_audit || {};
  const coveragePayload = coverage.llm_coverage || coverage.rule_coverage || {};
  el("auditTab").innerHTML = `
    <div class="insight ${overall}">
      <div class="insight-title">${overallPass ? "答案可以交付" : "答案需要复查"}</div>
      <div class="meta">faithfulness ${llm.faithfulness_score ?? "-"} · citation ${llm.citation_score ?? "-"} · relevance ${llm.relevance_score ?? "-"}</div>
      <div class="meta">coverage ${coveragePayload.coverage_score ?? "-"} · source coverage ${coveragePayload.source_coverage_score ?? "-"}</div>
    </div>
    <div>
      <span class="badge ${overall}">quality ${overallPass ? "pass" : "fail"}</span>
      <span class="badge ${audit.overall_pass ? "good" : "bad"}">faithfulness ${audit.overall_pass ? "pass" : "fail"}</span>
      ${coverage.coverage_pass !== undefined ? `<span class="badge ${coverage.coverage_pass ? "good" : "bad"}">coverage ${coverage.coverage_pass ? "pass" : "fail"}</span>` : ""}
      <span class="badge ${audit.rule_audit?.rule_pass ? "good" : "bad"}">rules ${audit.rule_audit?.rule_pass ? "pass" : "fail"}</span>
    </div>
    <pre class="jsonbox">${escapeHtml(JSON.stringify(audit, null, 2))}</pre>
  `;
}

function renderMemory(memory, effectiveQuery) {
  if (!memory) {
    el("memoryTab").innerHTML = "";
    return;
  }
  const turns = memory.turns || [];
  const longTerm = memory.long_term || {};
  const retrieved = longTerm.retrieved || [];
  const stored = longTerm.stored || [];
  const byKind = longTerm.stats?.by_kind || {};
  el("memoryTab").innerHTML = `
    <div class="kv">
      <b>session</b><span>${escapeHtml(memory.session_id)}</span>
      <b>turns</b><span>${memory.turn_count}</span>
      <b>effective query</b><span>${escapeHtml(effectiveQuery || "")}</span>
      <b>long namespace</b><span>${escapeHtml(longTerm.namespace || "")}</span>
      <b>long memories</b><span>${longTerm.stats?.memory_count ?? 0}</span>
      <b>by kind</b><span>${escapeHtml(Object.entries(byKind).map(([key, value]) => `${key}:${value}`).join(", ") || "none")}</span>
    </div>
    ${longTerm.error ? `<div class="error">${escapeHtml(longTerm.error)}</div>` : ""}
    <div class="insight">
      <div class="insight-title">本轮长期记忆命中</div>
      <div class="meta">长期记忆只用于理解偏好、项目状态和追问指代，不作为事实来源引用。</div>
    </div>
    ${retrieved.length ? retrieved.map((record) => renderMemoryRecord(record, "retrieved")).join("") : `<div class="empty-state">本轮没有命中长期记忆。</div>`}
    <div class="insight">
      <div class="insight-title">本轮新写入长期记忆</div>
    </div>
    ${stored.length ? stored.map((record) => renderMemoryRecord(record, "stored")).join("") : `<div class="empty-state">本轮没有写入新的长期记忆。</div>`}
    <div class="insight">
      <div class="insight-title">短期会话记忆</div>
    </div>
    <div class="preview">${escapeHtml(memory.context || "")}</div>
    ${turns.map((turn) => `
      <article class="memory-item">
        <div class="source-title">${escapeHtml(turn.user)}</div>
        <div class="preview">${escapeHtml(turn.answer_preview || "")}</div>
        <div class="meta">${escapeHtml((turn.source_titles || []).join(", "))}</div>
      </article>
    `).join("")}
  `;
}

function renderMemoryRecord(record, label) {
  return `
    <article class="memory-item">
      <div class="source-topline">
        <div class="source-title">${escapeHtml(record.kind || "")}</div>
        <span class="badge">${escapeHtml(label)}</span>
      </div>
      <div class="meta">importance ${formatNum(record.importance)} · score ${formatNum(record.score)} · access ${record.access_count ?? 0}</div>
      <div class="preview">${escapeHtml(record.content || "")}</div>
    </article>
  `;
}

function renderEmpty() {
  ["planTab", "sourcesTab", "contextTab", "auditTab", "memoryTab"].forEach((id) => {
    el(id).innerHTML = "";
  });
  el("timingText").textContent = "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

function formatNum(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(4) : "0.0000";
}
