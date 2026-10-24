"use strict";

// Chat page of the AI Assistant plugin. Talks to the plugin's local server:
// POST /api/message, /api/action, /api/settings, /api/reset and a long-polled GET /api/events.

const $ = (sel) => document.querySelector(sel);
const timeline = $("#timeline");
const input = $("#input");
const sendBtn = $("#btn-send");
const attachBox = $("#attachments");

const MODE_NAMES = { classic: "3D", cylindrical: "5D Spiral", cylindrical_full: "5D Spiral Full" };
const VERDICTS = {
  printable: "можно печатать",
  printable_with_tuning: "печать с доработкой",
  not_printable: "печать невозможна",
};
const SOURCES = { agent: "агент", mcp: "MCP", plan: "план", user: "кнопка" };
// Tools that do not change the scene or the settings, i.e. keep the last slicing result valid.
const READ_ONLY_TOOLS = new Set(["get_state", "list_quality_profiles", "list_materials", "get_settings",
  "list_objects", "get_object_info", "get_slice_status", "export_gcode", "save_project"]);
const SEVERITY = { high: "высокий", medium: "средний", low: "низкий" };
const STAGES = {
  "1-2": "Разбор текста и анализ сетки", "1": "Разбор задания", "2": "Анализ сетки", "3": "Ориентации",
  "4": "Правила и выбор режима", "5": "Подбор материала", "6": "Рассуждение модели", "7": "Проверка плана",
  "done": "План готов",
};

let lastSeq = 0;
let busy = false;
let pending = [];            // attachments waiting to be sent
let liveBubble = null;       // assistant message being streamed
let typingEl = null;
const toolCards = {};        // call_id -> card element
let currentPlanCard = null;  // running plan_print card (receives pipeline stages)
let currentSliceCard = null; // running slice card (receives progress)
let saveBar = null;          // save buttons under the latest slicing result
let settings = {};

// ------------------------------------------------------------------ helpers
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inlineMd(s) {
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

// Paragraphs and "- " lists; everything is escaped first.
function renderText(text) {
  const wrap = el("div");
  const blocks = String(text || "").trim().split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l))) {
      const ul = el("ul");
      for (const l of lines) {
        const li = el("li");
        li.innerHTML = inlineMd(l.replace(/^\s*([-*•]|\d+[.)])\s+/, ""));
        ul.appendChild(li);
      }
      wrap.appendChild(ul);
    } else {
      const p = el("p");
      p.innerHTML = lines.map(inlineMd).join("<br>");
      wrap.appendChild(p);
    }
  }
  return wrap;
}

function scrollDown(force) {
  const nearBottom = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 160;
  if (force || nearBottom) timeline.scrollTop = timeline.scrollHeight;
}

function add(node) {
  const welcome = $("#welcome");
  if (welcome) welcome.remove();
  if (typingEl && typingEl.parentNode) timeline.insertBefore(node, typingEl);
  else timeline.appendChild(node);
  scrollDown();
  return node;
}

function fmtDuration(sec) {
  if (sec === null || sec === undefined) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  return h ? `${h} ч ${String(m).padStart(2, "0")} мин` : `${Math.max(m, 1)} мин`;
}

function fmtValue(v) {
  if (v === true) return "да";
  if (v === false) return "нет";
  if (Array.isArray(v) || (v && typeof v === "object")) return JSON.stringify(v);
  return String(v);
}

// ------------------------------------------------------------------ typing indicator
function setTyping(on) {
  if (on && !typingEl) {
    typingEl = el("div", "msg assistant");
    const b = el("div", "bubble typing");
    b.append(el("span"), el("span"), el("span"));
    typingEl.appendChild(b);
    timeline.appendChild(typingEl);
    scrollDown();
  } else if (!on && typingEl) {
    typingEl.remove();
    typingEl = null;
  }
}

function setBusy(on) {
  busy = on;
  sendBtn.disabled = on;
  $("#btn-new").disabled = on;
  if (!on) setTyping(false);
  updateSaveBar();
}

// ------------------------------------------------------------------ messages
function addUser(ev) {
  const row = el("div", "msg user");
  const bubble = el("div", "bubble");
  if (ev.text) bubble.appendChild(renderText(ev.text));
  if (ev.attachments && ev.attachments.length) {
    const files = el("div", "files");
    for (const a of ev.attachments) {
      const icon = { model: "▲", image: "🖼", text: "📄", error: "⚠" }[a.kind] || "•";
      files.appendChild(el("span", "file-chip", `${icon} ${a.name}`));
    }
    bubble.appendChild(files);
  }
  row.appendChild(bubble);
  add(row);
  scrollDown(true);
}

function addAssistant(text) {
  if (liveBubble) {
    liveBubble.bubble.replaceChildren(renderText(text));
    liveBubble = null;
    scrollDown();
    return;
  }
  if (!text) return;
  const row = el("div", "msg assistant");
  const bubble = el("div", "bubble");
  bubble.appendChild(renderText(text));
  row.appendChild(bubble);
  add(row);
}

function addDelta(text) {
  if (!liveBubble) {
    const row = el("div", "msg assistant");
    const bubble = el("div", "bubble");
    row.appendChild(bubble);
    add(row);
    liveBubble = { row, bubble, text: "" };
  }
  liveBubble.text += text;
  liveBubble.bubble.replaceChildren(renderText(liveBubble.text));
  scrollDown();
}

// ------------------------------------------------------------------ tool cards
function summarizeCall(name, args) {
  switch (name) {
    case "set_settings": return Object.entries(args.settings || {}).map(([k, v]) => `${k}=${fmtValue(v)}`).join(", ");
    case "set_quality_profile": return args.quality_type;
    case "set_material": return args.material_id;
    case "set_printing_mode": return MODE_NAMES[args.mode] || args.mode;
    case "load_model": return (args.path || "").split(/[\\/]/).pop();
    case "rotate_object": return `${args.axis} ${args.angle_deg}°`;
    case "apply_orientation": return args.orientation_id || "";
    case "plan_print": return (args.task || "").slice(0, 120);
    case "get_settings": return (args.keys || []).join(", ");
    default: {
      const s = JSON.stringify(args || {});
      return s === "{}" ? "" : s.slice(0, 120);
    }
  }
}

function summarizeResult(name, result) {
  if (result === null || result === undefined) return "";
  switch (name) {
    case "load_model": return (result.loaded || []).map((o) => `${o.name} ${(o.size_mm || []).join("×")} мм`).join("; ");
    case "set_settings": {
      const n = Object.keys(result.applied || {}).length;
      const r = Object.keys(result.rejected || {}).length;
      return `применено ${n}` + (r ? `, отклонено ${r}` : "");
    }
    case "slice": return `${fmtDuration(result.print_time_s)}, ${(result.material_length_m || [])[0] ?? "—"} м`;
    case "export_gcode":
    case "save_project": return result.path;
    case "plan_print": return `${result.mode_name}, ${result.quality_type}, ${result.material}`;
    case "get_state": return `${MODE_NAMES[result.printing_mode] || result.printing_mode}, ${result.quality_type}, ${result.material}`;
    default:
      if (typeof result === "string") return result;
      return "";
  }
}

function toolCall(ev) {
  if (!READ_ONLY_TOOLS.has(ev.name)) markSliceStale();
  const card = el("div", "tool" + (ev.source === "plan" ? " sub" : ""));
  const head = el("div", "tool-head");
  card._status = el("span", "status run");
  head.appendChild(card._status);
  head.appendChild(el("span", "tool-title", ev.title || ev.name));
  head.appendChild(el("span", "tool-name", ev.name));
  if (ev.source && ev.source !== "agent") head.appendChild(el("span", "badge src-" + ev.source, SOURCES[ev.source] || ev.source));
  card._summary = el("span", "tool-summary", summarizeCall(ev.name, ev.args || {}));
  head.appendChild(card._summary);
  card._time = el("span", "tool-time");
  head.appendChild(card._time);
  head.addEventListener("click", () => card.classList.toggle("open"));
  card.appendChild(head);
  card._body = el("div", "tool-body");
  card._args = ev.args;
  card._body.appendChild(el("pre", null, "Аргументы: " + JSON.stringify(ev.args || {}, null, 2)));
  card.appendChild(card._body);
  if (ev.name === "plan_print") {
    card._stages = el("div", "stages");
    card.appendChild(card._stages);
    currentPlanCard = card;
  }
  if (ev.name === "slice") {
    const bar = el("div", "progress");
    card._bar = el("div");
    bar.appendChild(card._bar);
    card.appendChild(bar);
    currentSliceCard = card;
  }
  toolCards[ev.call_id] = card;
  add(card);
}

function toolResult(ev) {
  const card = toolCards[ev.call_id];
  if (!card) return;
  card._status.className = "status " + (ev.ok ? "ok" : "err");
  if (ev.elapsed_s !== undefined) card._time.textContent = ev.elapsed_s >= 1 ? `${Math.round(ev.elapsed_s)} с` : "";
  const summary = ev.ok ? summarizeResult(ev.name, ev.result) : "";
  if (summary) card._summary.textContent = summary;
  card._body.appendChild(el("pre", null, (ev.ok ? "Результат: " : "Ошибка: ") +
    (ev.ok ? JSON.stringify(ev.result, null, 2) : ev.error)));
  if (!ev.ok) card.appendChild(el("div", "tool-error", ev.error));
  if (card._bar) {
    card._bar.style.width = ev.ok ? "100%" : card._bar.style.width;
    if (currentSliceCard === card) currentSliceCard = null;
  }
  if (ev.name === "plan_print") {
    if (currentPlanCard === card) currentPlanCard = null;
    if (ev.ok && ev.result) add(planCard(ev.result));
  }
  if (ev.name === "slice" && ev.ok && ev.source !== "plan") add(sliceCard(ev.result));
}

function pipelineEvent(ev) {
  if (!currentPlanCard) return;
  if (ev.stage === "6" && ev.status === "progress") {
    let line = currentPlanCard._stages.querySelector(".llm-progress");
    if (!line) {
      line = el("div", "llm-progress");
      currentPlanCard._stages.appendChild(line);
    }
    line.textContent = ev.summary;
  } else if (ev.status !== "start" || ev.stage === "6") {
    const prefix = STAGES[ev.stage] ? STAGES[ev.stage] + ": " : "";
    currentPlanCard._stages.appendChild(el("div", null, prefix + ev.summary));
  }
  scrollDown();
}

function sliceProgress(ev) {
  if (currentSliceCard && currentSliceCard._bar) currentSliceCard._bar.style.width = Math.round(ev.progress * 100) + "%";
}

// ------------------------------------------------------------------ plan and slicing cards
function kv(k, v) {
  const box = el("div", "kv");
  box.appendChild(el("div", "k", k));
  box.appendChild(el("div", "v", v));
  return box;
}

function details(title, open) {
  const d = el("details");
  if (open) d.open = true;
  d.appendChild(el("summary", null, title));
  return d;
}

function sliceSummary(s) {
  const box = el("div", "slice-result");
  const time = el("span");
  time.innerHTML = `Время печати: <b>${escapeHtml(fmtDuration(s.print_time_s))}</b>`;
  const len = el("span");
  const lengths = s.material_length_m || [];
  const weights = s.material_weight_g || [];
  len.innerHTML = `Материал: <b>${escapeHtml(lengths[0] ?? "—")} м</b>` + (weights[0] ? ` (${escapeHtml(weights[0])} г)` : "");
  box.append(time, len);
  return box;
}

function sliceCard(s) {
  const card = el("div", "plan");
  card.appendChild(el("h3", null, "Результат слайсинга"));
  card.appendChild(sliceSummary(s));
  card.appendChild(saveActions());
  return card;
}

// Save buttons: only under the latest slicing result, disabled while the assistant works
// and once the scene or the settings have changed after that slicing.
function saveActions() {
  if (saveBar) saveBar.remove();
  const bar = el("div", "save-actions");
  bar._buttons = [saveButton("Сохранить G-code", "export_gcode", bar),
    saveButton("Сохранить проект 3MF", "save_project", bar)];
  bar.append(...bar._buttons);
  bar._log = el("div", "save-log");
  bar.appendChild(bar._log);
  saveBar = bar;
  updateSaveBar();
  return bar;
}

function saveButton(label, tool, bar) {
  const button = el("button", "btn", label);
  button.type = "button";
  button.addEventListener("click", () => runAction(tool, bar, button));
  return button;
}

function updateSaveBar() {
  if (!saveBar) return;
  for (const b of saveBar._buttons) b.disabled = busy || saveBar._stale || saveBar._running;
}

function markSliceStale() {
  if (!saveBar || saveBar._stale) return;
  saveBar._stale = true;
  saveBar._log.appendChild(el("div", "save-hint", "Модель или настройки изменились — нарежьте заново, чтобы сохранить."));
  updateSaveBar();
}

async function runAction(tool, bar, button) {
  bar._running = true;
  updateSaveBar();
  const label = button.textContent;
  button.textContent = "Сохраняется…";   // a big model can take a while
  const what = tool === "export_gcode" ? "G-code" : "Проект 3MF";
  try {
    const res = await fetch("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tool }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.ok) {
      const line = el("div", "save-ok", `${what}: `);
      line.appendChild(el("code", null, data.result.path));
      bar._log.appendChild(line);
    } else {
      bar._log.appendChild(el("div", "save-error", `${what} не сохранён: ${data.error || "ошибка " + res.status}`));
    }
  } catch (e) {
    bar._log.appendChild(el("div", "save-error", "Сервер ассистента недоступен: " + e));
  } finally {
    button.textContent = label;
    bar._running = false;
    updateSaveBar();
  }
}

function planCard(p) {
  const card = el("div", "plan");
  const h = el("h3", null, "План печати" + (p.object ? ` · ${p.object.name}` : ""));
  h.appendChild(el("span", "verdict " + p.verdict, VERDICTS[p.verdict] || p.verdict));
  if (p.source === "fallback") h.appendChild(el("span", "badge", "по правилам"));
  card.appendChild(h);

  const grid = el("div", "plan-grid");
  grid.append(kv("Режим", p.mode_name), kv("Профиль", p.quality_type), kv("Материал", p.material),
    kv("Ориентация", p.orientation ? p.orientation.id : "—"));
  card.appendChild(grid);
  if (p.orientation && p.orientation.reason) card.appendChild(el("div", "note", "Ориентация: " + p.orientation.reason));

  if (p.slice && !p.slice.error) card.append(sliceSummary(p.slice), saveActions());
  if (p.slice && p.slice.error) card.appendChild(el("div", "notice error", "Слайсинг: " + p.slice.error));

  if (p.mode_decision) {
    const d = details(`Выбор режима: оценка ${p.mode_decision.score} при пороге ${p.mode_decision.threshold} (уверенность: ${p.mode_decision.confidence})`);
    const ul = el("ul");
    for (const e of p.mode_decision.evidence || []) ul.appendChild(el("li", null, e));
    if (!(p.mode_decision.evidence || []).length) ul.appendChild(el("li", null, "у детали нет оси вращения"));
    d.appendChild(ul);
    card.appendChild(d);
  }

  if (p.settings && p.settings.length) {
    const d = details(`Настройки (${p.settings.length})`);
    const table = el("table");
    for (const s of p.settings) {
      const tr = el("tr");
      tr.appendChild(el("td", "key", s.key));
      tr.appendChild(el("td", "val", fmtValue(s.value)));
      const why = el("td", "why", s.reason || "");
      if (s.source) why.appendChild(el("span", "src", s.source === "plan" ? " · изменено моделью" : ""));
      tr.appendChild(why);
      table.appendChild(tr);
    }
    d.appendChild(table);
    card.appendChild(d);
  }

  if (p.materials_ranking && p.materials_ranking.length) {
    const d = details("Рейтинг материалов");
    const ul = el("ul");
    for (const m of p.materials_ranking) ul.appendChild(el("li", null, `${m.id} — ${m.score}${m.chain ? " · " + m.chain : ""}`));
    for (const c of p.material_conflicts || []) ul.appendChild(el("li", null, "⚠ " + c));
    d.appendChild(ul);
    card.appendChild(d);
  }

  if (p.risks && p.risks.length) {
    const d = details(`Риски (${p.risks.length})`, true);
    for (const r of p.risks) {
      const item = el("div", "risk " + (r.severity || ""));
      item.appendChild(el("div", null, `${r.text}`));
      if (r.evidence) item.appendChild(el("div", "ev", `${SEVERITY[r.severity] || r.severity || ""} · ${r.evidence}`));
      d.appendChild(item);
    }
    card.appendChild(d);
  }

  const lists = [["Подсказки по конструкции", p.design_suggestions], ["Вопросы", p.questions_to_user],
    ["Отступления от рекомендаций", (p.deviations || []).map((x) => `${x.what}: ${x.reason}`)]];
  for (const [title, items] of lists) {
    if (!items || !items.length) continue;
    const d = details(title, title === "Вопросы");
    const ul = el("ul");
    for (const it of items) ul.appendChild(el("li", null, it));
    d.appendChild(ul);
    card.appendChild(d);
  }
  if (p.explanation) {
    const d = details("Объяснение плана");
    d.appendChild(renderText(p.explanation));
    card.appendChild(d);
  }
  return card;
}

// ------------------------------------------------------------------ events
function handle(ev) {
  switch (ev.kind) {
    case "user": addUser(ev); break;
    case "turn_start": setBusy(true); break;
    case "thinking": setTyping(true); break;
    case "assistant_delta": setTyping(false); addDelta(ev.text); break;
    case "assistant_note": setTyping(false); liveBubble = null; add(el("div", "note", ev.text)); break;
    case "assistant": setTyping(false); addAssistant(ev.text); break;
    case "tool_call": setTyping(false); liveBubble = null; toolCall(ev); break;
    case "tool_result": toolResult(ev); break;
    case "pipeline": pipelineEvent(ev); break;
    case "slice_progress": sliceProgress(ev); break;
    case "error": setTyping(false); add(el("div", "notice error", ev.text)); break;
    case "turn_done": setBusy(false); liveBubble = null; break;
    case "reset": resetView(); break;
    case "settings": applySettings(ev.settings); break;
    default: break;
  }
}

function resetView() {
  timeline.replaceChildren();
  liveBubble = null;
  typingEl = null;
  currentPlanCard = null;
  currentSliceCard = null;
  saveBar = null;
  const hello = el("section", "welcome");
  hello.id = "welcome";
  hello.appendChild(el("h1", null, "Новый диалог"));
  hello.appendChild(el("p", null, "Прикрепите модель и опишите задачу печати."));
  timeline.appendChild(hello);
}

async function poll() {
  for (;;) {
    try {
      const res = await fetch(`/api/events?since=${lastSeq}&wait=25`, { cache: "no-store" });
      const data = await res.json();
      for (const ev of data.events) {
        lastSeq = Math.max(lastSeq, ev.seq);
        handle(ev);
      }
    } catch (e) {
      await new Promise((r) => setTimeout(r, 2000));   // server restarting or slicer closed
    }
  }
}

// ------------------------------------------------------------------ sending
function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function addFiles(files) {
  for (const f of files) pending.push(f);
  renderAttachments();
}

function renderAttachments() {
  attachBox.replaceChildren();
  pending.forEach((f, i) => {
    const chip = el("span", "att", `${f.name} (${Math.max(1, Math.round(f.size / 1024))} КБ)`);
    const x = el("button", null, "×");
    x.type = "button";
    x.title = "Убрать";
    x.addEventListener("click", () => { pending.splice(i, 1); renderAttachments(); });
    chip.appendChild(x);
    attachBox.appendChild(chip);
  });
}

async function send() {
  const text = input.value.trim();
  if (busy || (!text && !pending.length)) return;
  sendBtn.disabled = true;
  try {
    const attachments = [];
    for (const f of pending) attachments.push({ name: f.name, data: await readFile(f) });
    const res = await fetch("/api/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, attachments }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      add(el("div", "notice error", data.error || `Ошибка ${res.status}`));
      sendBtn.disabled = busy;
      return;
    }
    input.value = "";
    autosize();
    pending = [];
    renderAttachments();
  } catch (e) {
    add(el("div", "notice error", "Сервер ассистента недоступен: " + e));
    sendBtn.disabled = false;
  }
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

// ------------------------------------------------------------------ settings
function applySettings(s) {
  settings = s || {};
  const chip = $("#llm-status");
  if (settings.has_api_key) {
    chip.className = "chip ok";
    chip.textContent = settings.model || "LLM";
  } else {
    chip.className = "chip warn";
    chip.textContent = "LLM не подключена";
  }
}

function openSettings() {
  const form = $("#settings-form");
  form.base_url.value = settings.base_url || "";
  form.model.value = settings.model || "";
  form.api_key.value = "";
  form.api_key.placeholder = settings.has_api_key
    ? (settings.api_key_from_env ? "задан переменной окружения" : `сохранён${settings.api_key_tail ? " (…" + settings.api_key_tail + ")" : ""}`)
    : "не задан";
  form.output_dir.value = settings.output_dir || "";
  form.port.value = settings.port || "";
  $("#mcp-hint").textContent = `MCP-сервер: POST ${location.origin}/mcp (JSON-RPC 2.0)`;
  $("#settings").hidden = false;
  form.api_key.focus();
}

async function saveSettings(e) {
  e.preventDefault();
  const form = $("#settings-form");
  const body = {
    base_url: form.base_url.value.trim(),
    model: form.model.value.trim(),
    output_dir: form.output_dir.value.trim(),
  };
  if (form.api_key.value.trim()) body.api_key = form.api_key.value.trim();
  if (form.port.value) body.port = parseInt(form.port.value, 10);
  const res = await fetch("/api/settings", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (res.ok) {
    applySettings(data.settings);
    $("#settings").hidden = true;
  } else {
    alert(data.error || "Не удалось сохранить настройки");
  }
}

// ------------------------------------------------------------------ wiring
sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
input.addEventListener("input", autosize);
$("#btn-attach").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
$("#btn-settings").addEventListener("click", openSettings);
$("#settings-cancel").addEventListener("click", () => { $("#settings").hidden = true; });
$("#settings-form").addEventListener("submit", saveSettings);
$("#btn-new").addEventListener("click", async () => {
  const res = await fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  if (!res.ok) add(el("div", "notice error", "Дождитесь окончания ответа"));
});
for (const b of document.querySelectorAll(".example")) {
  b.addEventListener("click", () => { input.value = b.textContent; autosize(); input.focus(); });
}
input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData ? e.clipboardData.files : [])];
  if (files.length) { e.preventDefault(); addFiles(files); }
});

let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; $("#drop-overlay").hidden = false; });
window.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("#drop-overlay").hidden = true; });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("#drop-overlay").hidden = true;
  if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});

(async function init() {
  try {
    const state = await (await fetch("/api/state", { cache: "no-store" })).json();
    applySettings(state.settings);
    if (!state.settings.has_api_key) {
      add(el("div", "note", "Ключ API не задан: ассистент будет строить план по правилам, без языковой модели. Укажите ключ в настройках (⚙)."));
    }
  } catch (e) {
    add(el("div", "notice error", "Сервер ассистента недоступен"));
  }
  poll();
})();
