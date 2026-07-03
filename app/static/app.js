const API_STREAM_URL = "/api/v1/chat/stream";
const API_CHAT_URL = "/api/v1/chat";
const API_AGENT_URL = "/api/v1/agent/run";
const STORAGE_KEY = "ic-expert-agent-state";

const els = {
  chatLog: document.querySelector("#chatLog"),
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#promptInput"),
  sendButton: document.querySelector("#sendButton"),
  workspace: document.querySelector(".workspace"),
  agentPanel: document.querySelector("#agentPanel"),
  taskForm: document.querySelector("#taskForm"),
  taskInput: document.querySelector("#taskInput"),
  maxStepsInput: document.querySelector("#maxStepsInput"),
  runTaskButton: document.querySelector("#runTaskButton"),
  taskOutput: document.querySelector("#taskOutput"),
  conversationId: document.querySelector("#conversationId"),
  memoryDot: document.querySelector("#memoryDot"),
  memoryState: document.querySelector("#memoryState"),
  toolsPanel: document.querySelector("#toolsPanel"),
  sourcesPanel: document.querySelector("#sourcesPanel"),
  clearButton: document.querySelector("#clearButton"),
  copyButton: document.querySelector("#copyButton"),
  newSessionButton: document.querySelector("#newSessionButton"),
};

const state = {
  conversationId: null,
  messages: [],
  toolEvents: [],
  sources: [],
  activeMode: "chat",
  currentTask: null,
  busy: false,
  taskBusy: false,
};

function init() {
  restoreState();
  bindEvents();
  renderAll();
  refreshIcons();
}

function bindEvents() {
  els.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const prompt = els.input.value.trim();
    if (!prompt || state.busy) return;
    els.input.value = "";
    resizeInput();
    await sendMessage(prompt);
  });

  els.input.addEventListener("input", resizeInput);
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      els.input.value = button.dataset.prompt || "";
      resizeInput();
      els.input.focus();
    });
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activatePanel(tab.dataset.panel));
  });

  document.querySelectorAll(".mode-button").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode || "chat"));
  });

  els.taskForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const goal = els.taskInput.value.trim();
    if (!goal || state.taskBusy) return;
    await runAutonomousTask(goal);
  });

  els.clearButton.addEventListener("click", () => {
    state.messages = [];
    state.toolEvents = [];
    state.sources = [];
    persistState();
    renderAll();
  });

  els.newSessionButton.addEventListener("click", () => {
    state.conversationId = null;
    state.messages = [];
    state.toolEvents = [];
    state.sources = [];
    persistState();
    renderAll();
    els.input.focus();
  });

  els.copyButton.addEventListener("click", async () => {
    if (!state.conversationId || !navigator.clipboard) return;
    await navigator.clipboard.writeText(state.conversationId);
  });
}

async function sendMessage(content) {
  state.busy = true;
  state.toolEvents = [];
  state.sources = [];
  state.messages.push({ role: "user", content });
  const assistant = { role: "assistant", content: "" };
  state.messages.push(assistant);
  renderAll();
  setBusy(true);

  const payload = {
    conversation_id: state.conversationId,
    messages: state.messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .slice(0, -1),
  };

  try {
    const response = await fetch(API_STREAM_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok || !response.body) {
      throw new Error(`stream ${response.status}`);
    }

    await readEventStream(response, (eventName, data) => {
      handleStreamEvent(eventName, data, assistant);
    });

    if (!assistant.content.trim()) {
      await sendFallback(payload, assistant);
    }
  } catch (error) {
    assistant.content = `请求失败：${error.message || error}`;
  } finally {
    state.busy = false;
    setBusy(false);
    persistState();
    renderAll();
  }
}

async function runAutonomousTask(goal) {
  state.taskBusy = true;
  setTaskBusy(true);
  state.currentTask = {
    status: "running",
    goal,
    steps: [],
    answer_mode: "assisted_draft",
    final_answer: "",
    reflection: {},
  };
  renderTask();

  try {
    const response = await fetch(API_AGENT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        goal,
        conversation_id: state.conversationId,
        max_steps: Number(els.maxStepsInput.value || 6),
      }),
    });
    if (!response.ok) {
      throw new Error(`agent ${response.status}`);
    }
    const task = await response.json();
    state.currentTask = task;
    state.conversationId = task.session_id || state.conversationId;
    persistState();
  } catch (error) {
    state.currentTask = {
      status: "failed",
      goal,
      steps: [],
      answer_mode: "refusal",
      final_answer: `自主任务失败：${error.message || error}`,
      reflection: {},
    };
  } finally {
    state.taskBusy = false;
    setTaskBusy(false);
    renderAll();
  }
}

async function sendFallback(payload, assistant) {
  const response = await fetch(API_CHAT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`chat ${response.status}`);
  }
  const data = await response.json();
  assistant.content = data.answer || data.content || "";
  state.conversationId = data.conversation_id || state.conversationId;
  state.sources = Array.isArray(data.sources) ? data.sources : [];
  state.toolEvents = Array.isArray(data.tool_events) ? data.tool_events : [];
}

async function readEventStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const parsed = parseSseBlock(part);
      if (parsed) onEvent(parsed.event, parsed.data);
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseBlock(buffer);
    if (parsed) onEvent(parsed.event, parsed.data);
  }
}

function parseSseBlock(block) {
  const lines = block.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));
  if (!eventLine || dataLines.length === 0) return null;

  const event = eventLine.slice(6).trim();
  const raw = dataLines.map((line) => line.slice(5).trim()).join("\n");
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: {} };
  }
}

function handleStreamEvent(eventName, data, assistant) {
  if (eventName === "tool_call" || eventName === "tool_result") {
    state.toolEvents.push({ event: eventName, ...data });
  }
  if (eventName === "answer") {
    assistant.content += data.chunk || "";
  }
  if (eventName === "citation") {
    state.sources = Array.isArray(data.sources) ? data.sources : [];
  }
  if (eventName === "done") {
    state.conversationId = data.conversation_id || state.conversationId;
  }
  if (eventName === "error") {
    assistant.content = `请求失败：${data.error || "unknown error"}`;
  }
  persistState();
  renderAll();
}

function renderAll() {
  renderSession();
  renderMessages();
  renderTools();
  renderSources();
  renderTask();
  refreshIcons();
}

function renderSession() {
  els.conversationId.textContent = state.conversationId || "未开始";
  els.memoryDot.classList.toggle("ready", Boolean(state.conversationId));
  els.memoryState.textContent = state.conversationId ? "记忆已连接" : "记忆待连接";
}

function renderMessages() {
  if (state.messages.length === 0) {
    els.chatLog.innerHTML = `
      <article class="message system">
        <div class="message-head"><span>Ready</span></div>
        <div class="message-content">IC-Expert Agent</div>
      </article>
    `;
    return;
  }

  els.chatLog.innerHTML = state.messages
    .map((message) => {
      const role = escapeHtml(message.role);
      const label = role === "user" ? "You" : "Agent";
      return `
        <article class="message ${role}">
          <div class="message-head"><span>${label}</span></div>
          <div class="message-content">${escapeHtml(message.content || "")}</div>
        </article>
      `;
    })
    .join("");
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function renderTools() {
  if (!state.toolEvents.length) {
    els.toolsPanel.innerHTML = `<div class="empty-state">暂无工具调用</div>`;
    return;
  }

  els.toolsPanel.innerHTML = state.toolEvents
    .map((event) => {
      const isResult = event.event === "tool_result";
      const ok = event.ok !== false;
      const title = escapeHtml(event.tool || "tool");
      const confidence = event.confidence || "unknown";
      const flags = Array.isArray(event.review_flags) ? event.review_flags : [];
      const evidence = Array.isArray(event.evidence) ? event.evidence : [];
      const body = isResult
        ? escapeHtml([
            event.summary || "",
            `confidence: ${confidence}`,
            `evidence: ${evidence.length}`,
            flags.length ? `review: ${flags.join(", ")}` : "",
          ].filter(Boolean).join("\n"))
        : escapeHtml(JSON.stringify(event.arguments || {}, null, 2));
      return `
        <article class="tool-item">
          <div class="item-title">
            <span>${title}</span>
            <span class="badge ${ok ? "" : "error"}">${isResult ? (ok ? "OK" : "ERR") : "CALL"}</span>
          </div>
          <div class="item-body">${body}</div>
        </article>
      `;
    })
    .join("");
}

function renderSources() {
  if (!state.sources.length) {
    els.sourcesPanel.innerHTML = `<div class="empty-state">暂无引用来源</div>`;
    return;
  }

  els.sourcesPanel.innerHTML = state.sources
    .map((source, index) => {
      const title = escapeHtml(source.source || `Source ${index + 1}`);
      const page = escapeHtml(source.page || "页码未知");
      const content = escapeHtml(source.content || "");
      return `
        <article class="source-item">
          <div class="item-title">
            <span>${title}</span>
            <span class="badge">${page}</span>
          </div>
          <div class="item-body">${content}</div>
        </article>
      `;
    })
    .join("");
}

function renderTask() {
  const task = state.currentTask;
  if (!task) {
    els.taskOutput.innerHTML = `<div class="empty-state">暂无自主任务</div>`;
    return;
  }

  const reflection = task.reflection || {};
  const audit = task.audit_summary || {};
  const steps = Array.isArray(task.steps) ? task.steps : [];
  const statusClass = task.status === "failed" || task.status === "needs_review" ? "error" : "";
  const mode = task.answer_mode || "assisted_draft";
  const modeClass = mode === "strict_answer" ? "" : "error";
  els.taskOutput.innerHTML = `
    <article class="task-card">
      <div class="task-card-header">
        <h3>${escapeHtml(task.goal || "自主任务")}</h3>
        <span class="badge ${statusClass}">
          ${escapeHtml(task.status || "running")}
        </span>
      </div>
      <div class="task-card-body">
        <div class="step-list">
          ${steps.map(renderTaskStep).join("") || '<div class="empty-state">计划生成中</div>'}
        </div>
      </div>
    </article>
    <article class="task-card">
      <div class="task-card-header">
        <h3>最终交付</h3>
        <span class="badge ${modeClass}">
          ${escapeHtml(formatAnswerMode(mode))}
        </span>
      </div>
      <div class="task-card-body">
        <div class="audit-line">
          ${escapeHtml(`${task.confidence || "unknown"} · ${reflection.quality_score ?? "NA"} · ${formatAuditSummary(audit, task.review_flags)}`)}
        </div>
        <div class="final-answer">${escapeHtml(task.final_answer || "等待执行完成")}</div>
      </div>
    </article>
    <article class="task-card">
      <div class="task-card-header">
        <h3>反思审查</h3>
        <span class="badge ${reflection.likely_hallucination ? "error" : ""}">
          ${reflection.likely_hallucination ? "REVIEW" : "CHECK"}
        </span>
      </div>
      <div class="task-card-body">
        <div class="item-body">
          ${escapeHtml(reflection.summary || "暂无反思结果")}
          ${renderSuggestionList(reflection.suggestions)}
        </div>
      </div>
    </article>
  `;
}

function renderTaskStep(step) {
  const status = step.status || "pending";
  const badgeClass = status === "failed" ? "error" : "";
  const tool = step.tool_name ? ` · ${step.tool_name}` : "";
  const args = step.arguments && Object.keys(step.arguments).length
    ? `\n参数: ${JSON.stringify(step.arguments)}`
    : "";
  const evidence = Array.isArray(step.evidence) ? step.evidence : [];
  const flags = Array.isArray(step.review_flags) ? step.review_flags : [];
  const rationale = step.rationale ? `\n原因: ${step.rationale}` : "";
  const audit = `\n置信度: ${step.confidence || "unknown"}\n证据数: ${evidence.length}`;
  const reviews = flags.length ? `\n复核: ${flags.join(", ")}` : "";
  const error = step.error ? `\n错误: ${step.error}` : "";
  const observation = step.observation || "";
  return `
    <section class="step-item">
      <div class="step-title">
        <span>${escapeHtml(step.title || step.id || "步骤")}</span>
        <span class="badge ${badgeClass}">${escapeHtml(status)}</span>
      </div>
      <div class="step-meta">
        ${escapeHtml(`${step.action_type || "reasoning"}${tool}${rationale}${args}${audit}${reviews}${error}`)}
      </div>
      <div class="step-observation">${escapeHtml(observation)}</div>
    </section>
  `;
}

function formatAuditSummary(audit, flags) {
  const reviewFlags = Array.isArray(flags) ? flags : [];
  const parts = [
    `证据 ${audit.evidence_count ?? 0}`,
    `工具步骤 ${audit.tool_step_count ?? 0}`,
    `低置信步骤 ${audit.low_confidence_step_count ?? 0}`,
  ];
  if (reviewFlags.length) {
    parts.push(`复核 ${reviewFlags.join(", ")}`);
  }
  return parts.join(" · ");
}

function formatAnswerMode(mode) {
  if (mode === "strict_answer") return "Evidence Answer";
  if (mode === "refusal") return "Refusal";
  return "Assisted Draft · 需人工复核";
}

function renderSuggestionList(items) {
  if (!Array.isArray(items) || !items.length) return "";
  return escapeHtml(`\n\n建议:\n${items.map((item) => `- ${item}`).join("\n")}`);
}

function activatePanel(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.panel === name);
  });
  els.toolsPanel.classList.toggle("active", name === "tools");
  els.sourcesPanel.classList.toggle("active", name === "sources");
}

function setMode(mode) {
  state.activeMode = mode === "agent" ? "agent" : "chat";
  els.workspace.classList.toggle("agent-mode", state.activeMode === "agent");
  els.agentPanel.classList.toggle("active", state.activeMode === "agent");
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.activeMode);
  });
  persistState();
  refreshIcons();
}

function setBusy(value) {
  els.sendButton.disabled = value;
  els.input.disabled = value;
}

function setTaskBusy(value) {
  els.runTaskButton.disabled = value;
  els.taskInput.disabled = value;
}

function resizeInput() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(180, els.input.scrollHeight)}px`;
}

function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      conversationId: state.conversationId,
      messages: state.messages.slice(-24),
      toolEvents: state.toolEvents.slice(-20),
      sources: state.sources.slice(0, 12),
      activeMode: state.activeMode,
      currentTask: state.currentTask,
    }),
  );
}

function restoreState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    state.conversationId = saved.conversationId || null;
    state.messages = Array.isArray(saved.messages) ? saved.messages : [];
    state.toolEvents = Array.isArray(saved.toolEvents) ? saved.toolEvents : [];
    state.sources = Array.isArray(saved.sources) ? saved.sources : [];
    state.activeMode = saved.activeMode === "agent" ? "agent" : "chat";
    state.currentTask = saved.currentTask || null;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init();
setMode(state.activeMode);
