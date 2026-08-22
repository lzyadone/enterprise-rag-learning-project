const state = {
  overview: null,
  currentCase: null,
  candidateIndex: 0,
  busy: false,
};

const elements = {
  datasetPath: document.querySelector("#datasetPath"),
  overallCount: document.querySelector("#overallCount"),
  overallBar: document.querySelector("#overallBar"),
  questionCount: document.querySelector("#questionCount"),
  questionList: document.querySelector("#questionList"),
  caseId: document.querySelector("#caseId"),
  questionText: document.querySelector("#questionText"),
  candidatePosition: document.querySelector("#candidatePosition"),
  previousCandidate: document.querySelector("#previousCandidate"),
  nextCandidate: document.querySelector("#nextCandidate"),
  sourceTitle: document.querySelector("#sourceTitle"),
  headingPath: document.querySelector("#headingPath"),
  sourceLink: document.querySelector("#sourceLink"),
  metadataStrip: document.querySelector("#metadataStrip"),
  documentText: document.querySelector("#documentText"),
  caseProgress: document.querySelector("#caseProgress"),
  chunkId: document.querySelector("#chunkId"),
  poolRank: document.querySelector("#poolRank"),
  retrievalChannels: document.querySelector("#retrievalChannels"),
  retrievalRank: document.querySelector("#retrievalRank"),
  queryAspect: document.querySelector("#queryAspect"),
  reviewNote: document.querySelector("#reviewNote"),
  clearJudgment: document.querySelector("#clearJudgment"),
  saveStatus: document.querySelector("#saveStatus"),
  errorBanner: document.querySelector("#errorBanner"),
};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  bindEvents();
  try {
    state.overview = await requestJson("/api/state");
    renderOverview();
    const firstIncomplete = state.overview.cases.find(
      (item) => item.progress.labeled < item.progress.total,
    );
    const firstCase = firstIncomplete || state.overview.cases[0];
    if (firstCase) await selectCase(firstCase.case_id);
  } catch (error) {
    showError(error.message);
  }
}

function bindEvents() {
  elements.previousCandidate.addEventListener("click", () => moveCandidate(-1));
  elements.nextCandidate.addEventListener("click", () => moveCandidate(1));
  elements.clearJudgment.addEventListener("click", clearJudgment);
  document.querySelectorAll(".grade").forEach((button) => {
    button.addEventListener("click", () => saveGrade(Number(button.dataset.grade)));
  });
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return;
    if (["0", "1", "2", "3"].includes(event.key)) saveGrade(Number(event.key));
    if (event.key === "ArrowLeft") moveCandidate(-1);
    if (event.key === "ArrowRight") moveCandidate(1);
  });
}

async function selectCase(caseId) {
  if (state.busy) return;
  clearError();
  try {
    state.currentCase = await requestJson(`/api/case?case_id=${encodeURIComponent(caseId)}`);
    const firstUnlabeled = state.currentCase.candidates.findIndex((item) => !item.judgment);
    state.candidateIndex = firstUnlabeled >= 0 ? firstUnlabeled : 0;
    renderOverview();
    renderCase();
  } catch (error) {
    showError(error.message);
  }
}

function renderOverview() {
  if (!state.overview) return;
  const { labeled, total } = state.overview.progress;
  const percent = total ? Math.round((labeled / total) * 100) : 0;
  elements.datasetPath.textContent = state.overview.qrels_path;
  elements.overallCount.textContent = `${labeled} / ${total}`;
  elements.overallBar.style.width = `${percent}%`;
  elements.questionCount.textContent = String(state.overview.cases.length);
  elements.questionList.replaceChildren(
    ...state.overview.cases.map((item, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "question-item";
      if (state.currentCase?.case_id === item.case_id) button.classList.add("active");
      const complete = item.progress.labeled === item.progress.total;
      if (complete) button.classList.add("complete");

      const number = document.createElement("span");
      number.className = "question-number";
      number.textContent = String(index + 1).padStart(2, "0");
      const content = document.createElement("span");
      content.className = "question-copy";
      const text = document.createElement("span");
      text.textContent = item.question;
      const progress = document.createElement("small");
      progress.textContent = `${item.progress.labeled}/${item.progress.total}`;
      content.append(text, progress);
      button.append(number, content);
      button.addEventListener("click", () => selectCase(item.case_id));
      return button;
    }),
  );
}

function renderCase() {
  const currentCase = state.currentCase;
  if (!currentCase) return;
  elements.caseId.textContent = currentCase.case_id;
  elements.questionText.textContent = currentCase.question;
  elements.caseProgress.textContent = `${currentCase.progress.labeled} / ${currentCase.progress.total}`;
  renderCandidate();
}

function renderCandidate() {
  const candidates = state.currentCase?.candidates || [];
  const candidate = candidates[state.candidateIndex];
  if (!candidate) return;

  elements.candidatePosition.textContent = `${state.candidateIndex + 1} / ${candidates.length}`;
  elements.previousCandidate.disabled = state.candidateIndex === 0;
  elements.nextCandidate.disabled = state.candidateIndex === candidates.length - 1;
  elements.sourceTitle.textContent = candidate.title;
  elements.headingPath.textContent = candidate.heading_path || "无章节路径";
  elements.sourceLink.href = candidate.url || "#";
  elements.sourceLink.hidden = !candidate.url;
  elements.documentText.textContent = candidate.document;
  elements.chunkId.textContent = candidate.chunk_id;
  elements.poolRank.textContent = String(candidate.pool_rank);
  elements.retrievalChannels.textContent = candidate.retrieval_channels.join(" + ") || "-";
  elements.retrievalRank.textContent = candidate.retrieval_rank == null ? "-" : String(candidate.retrieval_rank);
  elements.queryAspect.textContent = candidate.aspect || "-";
  elements.reviewNote.value = candidate.judgment?.note || "";
  elements.saveStatus.textContent = candidate.judgment
    ? `已标注为 ${candidate.judgment.relevance}`
    : "未标注";
  elements.saveStatus.classList.toggle("saved", Boolean(candidate.judgment));
  elements.clearJudgment.disabled = !candidate.judgment || state.busy;

  elements.metadataStrip.replaceChildren(
    makeTag(candidate.category),
    makeTag(candidate.source_type || "source"),
    makeTag(`${candidate.document.length} chars`),
  );
  document.querySelectorAll(".grade").forEach((button) => {
    button.classList.toggle("selected", Number(button.dataset.grade) === candidate.judgment?.relevance);
    button.disabled = state.busy;
  });
}

function makeTag(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span;
}

function moveCandidate(delta) {
  if (!state.currentCase || state.busy) return;
  const nextIndex = state.candidateIndex + delta;
  if (nextIndex < 0 || nextIndex >= state.currentCase.candidates.length) return;
  state.candidateIndex = nextIndex;
  renderCandidate();
}

async function saveGrade(relevance) {
  const candidate = currentCandidate();
  if (!candidate || state.busy) return;
  state.busy = true;
  renderCandidate();
  try {
    const response = await requestJson("/api/judgment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_id: state.currentCase.case_id,
        chunk_id: candidate.chunk_id,
        relevance,
        note: elements.reviewNote.value,
      }),
    });
    candidate.judgment = response.judgment;
    updateProgress(response.progress, response.case_progress);
    renderCandidate();
    window.setTimeout(moveToNextUnlabeled, 180);
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    renderCandidate();
  }
}

async function clearJudgment() {
  const candidate = currentCandidate();
  if (!candidate || !candidate.judgment || state.busy) return;
  state.busy = true;
  try {
    const response = await requestJson("/api/judgment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_id: state.currentCase.case_id,
        chunk_id: candidate.chunk_id,
        relevance: null,
      }),
    });
    candidate.judgment = null;
    updateProgress(response.progress, response.case_progress);
    renderCandidate();
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    renderCandidate();
  }
}

function updateProgress(overall, caseProgress) {
  state.overview.progress = overall;
  state.currentCase.progress = caseProgress;
  const overviewCase = state.overview.cases.find((item) => item.case_id === state.currentCase.case_id);
  if (overviewCase) overviewCase.progress = caseProgress;
  renderOverview();
  elements.caseProgress.textContent = `${caseProgress.labeled} / ${caseProgress.total}`;
}

function moveToNextUnlabeled() {
  const candidates = state.currentCase?.candidates || [];
  const nextIndex = candidates.findIndex((item, index) => index > state.candidateIndex && !item.judgment);
  if (nextIndex >= 0) {
    state.candidateIndex = nextIndex;
    renderCandidate();
    return;
  }
  const wrappedIndex = candidates.findIndex((item) => !item.judgment);
  if (wrappedIndex >= 0) {
    state.candidateIndex = wrappedIndex;
    renderCandidate();
  }
}

function currentCandidate() {
  return state.currentCase?.candidates[state.candidateIndex] || null;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}
