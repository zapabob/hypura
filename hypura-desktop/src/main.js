import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/webview";

const RECENT_KEY = "hypura-desktop-recent-gguf";
const MAX_RECENT = 10;

const pathEl = document.getElementById("path");
const hostEl = document.getElementById("host");
const portEl = document.getElementById("port");
const apiKeyEl = document.getElementById("apiKey");
const statusEl = document.getElementById("status");
const urlEl = document.getElementById("url");
const recentEl = document.getElementById("recent");

function setStatus(msg) {
  statusEl.textContent = msg ?? "";
}

function baseFromInputs() {
  const host = hostEl.value.trim() || "127.0.0.1";
  const port = Number(portEl.value) || 8080;
  return { host, port, base: `http://${host}:${port}` };
}

function updateUrlDisplays(base) {
  urlEl.textContent = base;
  document.getElementById("u-kobold-gen").textContent = `${base}/api/v1/generate`;
  document.getElementById("u-kobold-model").textContent = `${base}/api/v1/model`;
  document.getElementById("u-lite").textContent = `${base}/kobold-lite`;
}

function loadRecent() {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveRecent(paths) {
  localStorage.setItem(RECENT_KEY, JSON.stringify(paths.slice(0, MAX_RECENT)));
}

function pushRecent(p) {
  const t = p.trim();
  if (!t) return;
  let list = loadRecent().filter((x) => x !== t);
  list.unshift(t);
  saveRecent(list);
  renderRecent();
}

function renderRecent() {
  recentEl.innerHTML = "";
  for (const p of loadRecent()) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = p.length > 80 ? `…${p.slice(-76)}` : p;
    a.title = p;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      pathEl.value = p;
      setStatus("");
    });
    li.appendChild(a);
    recentEl.appendChild(li);
  }
}

function copyText(text) {
  if (!text || text === "—") return;
  navigator.clipboard.writeText(text).catch(() => {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  });
  setStatus("Copied.");
}

document.querySelectorAll("[data-copy-target]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const id = btn.getAttribute("data-copy-target");
    copyText(document.getElementById(id).textContent);
  });
});
document.querySelectorAll("[data-copy-from]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const id = btn.getAttribute("data-copy-from");
    copyText(document.getElementById(id).textContent);
  });
});

document.getElementById("pick").addEventListener("click", async () => {
  setStatus("");
  try {
    const p = await invoke("pick_gguf");
    if (p) {
      pathEl.value = p;
      pushRecent(p);
    }
  } catch (e) {
    setStatus(String(e));
  }
});

document.getElementById("serve").addEventListener("click", async () => {
  setStatus("");
  const model = pathEl.value.trim();
  if (!model) {
    setStatus("Choose a GGUF first.");
    return;
  }
  const { host, port, base } = baseFromInputs();
  try {
    await invoke("spawn_hypura_serve", {
      modelPath: model,
      host,
      port,
    });
    pushRecent(model);
    updateUrlDisplays(base);
    setStatus("Started hypura serve. If load fails, check the hypura terminal/log.");
  } catch (e) {
    setStatus(String(e));
  }
});

document.getElementById("stop").addEventListener("click", async () => {
  setStatus("");
  try {
    await invoke("stop_hypura_serve");
    setStatus("Stopped managed hypura child process (if any).");
  } catch (e) {
    setStatus(String(e));
  }
});

document.getElementById("switch").addEventListener("click", async () => {
  setStatus("");
  const model = pathEl.value.trim();
  if (!model) {
    setStatus("Choose a GGUF first.");
    return;
  }
  const { host, port, base } = baseFromInputs();
  const apiKey = apiKeyEl.value.trim() || null;
  try {
    const res = await invoke("switch_model_http", {
      path: model,
      host,
      port,
      apiKey,
    });
    pushRecent(model);
    updateUrlDisplays(base);
    setStatus(`Model switch OK: ${String(res).slice(0, 200)}`);
  } catch (e) {
    setStatus(String(e));
  }
});

hostEl.addEventListener("change", () => updateUrlDisplays(baseFromInputs().base));
portEl.addEventListener("change", () => updateUrlDisplays(baseFromInputs().base));

const win = getCurrentWindow();
win.onDragDropEvent((event) => {
  if (event.payload.type !== "drop") return;
  const paths = event.payload.paths;
  const gguf = paths.find((p) => p.toLowerCase().endsWith(".gguf"));
  if (gguf) {
    pathEl.value = gguf;
    pushRecent(gguf);
    setStatus("GGUF path set from drop.");
  } else {
    setStatus("Drop a .gguf file (no .gguf in drop payload).");
  }
});

updateUrlDisplays(baseFromInputs().base);
renderRecent();
