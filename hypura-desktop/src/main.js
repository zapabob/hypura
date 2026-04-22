import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";

const RECENT_KEY = "hypura-desktop-recent-gguf";
const MAX_RECENT = 10;

const pathEl = document.getElementById("path");
const hostEl = document.getElementById("host");
const portEl = document.getElementById("port");
const assetRootEl = document.getElementById("assetRoot");
const statusEl = document.getElementById("status");
const featuresEl = document.getElementById("features");
const urlEl = document.getElementById("url");
const liteUrlEl = document.getElementById("liteUrl");

function setStatus(message) {
  statusEl.textContent = message ?? "";
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

function pushRecent(value) {
  const trimmed = value.trim();
  if (!trimmed) return;
  const next = loadRecent().filter((existing) => existing !== trimmed);
  next.unshift(trimmed);
  saveRecent(next);
}

function renderStatus(snapshot) {
  if (!snapshot) return;
  urlEl.textContent = snapshot.baseUrl ?? "-";
  liteUrlEl.textContent = snapshot.liteUrl ?? "-";
  setStatus(snapshot.statusText || (snapshot.running ? "Running." : "Idle."));
  featuresEl.textContent = JSON.stringify(
    {
      running: snapshot.running,
      assetRoot: snapshot.assetRoot,
      modelPath: snapshot.modelPath,
      embeddings: snapshot.embeddingsReady,
      transcribe: snapshot.transcribeReady,
      tts: snapshot.ttsReady,
      audio: snapshot.audioReady,
      pendingAssets: snapshot.pendingAssets,
      downloadedAssets: snapshot.downloadedAssets,
      lastError: snapshot.lastError,
    },
    null,
    2,
  );
}

async function refreshStatus() {
  try {
    const snapshot = await invoke("get_runtime_status");
    renderStatus(snapshot);
  } catch (error) {
    setStatus(String(error));
  }
}

document.getElementById("pick").addEventListener("click", async () => {
  setStatus("");
  try {
    const path = await invoke("pick_gguf");
    if (path) {
      pathEl.value = path;
      pushRecent(path);
    }
  } catch (error) {
    setStatus(String(error));
  }
});

document.getElementById("pickAssetRoot").addEventListener("click", async () => {
  setStatus("");
  try {
    const path = await invoke("pick_asset_root");
    if (path) {
      assetRootEl.value = path;
    }
  } catch (error) {
    setStatus(String(error));
  }
});

document.getElementById("launch").addEventListener("click", async () => {
  setStatus("");
  const model = pathEl.value.trim();
  if (!model) {
    setStatus("Choose a GGUF model first.");
    return;
  }
  const payload = {
    modelPath: model,
    host: hostEl.value.trim() || "127.0.0.1",
    port: Number(portEl.value) || 5001,
    assetRoot: assetRootEl.value.trim() || null,
  };
  try {
    const base = await invoke("launch_packaged_koboldcpp", { cmd: payload });
    pushRecent(model);
    urlEl.textContent = base;
    liteUrlEl.textContent = `${base}/kobold-lite`;
    setStatus("Started packaged koboldcpp runtime. Optional assets may continue downloading in the background.");
    await refreshStatus();
  } catch (error) {
    setStatus(String(error));
  }
});

document.getElementById("stop").addEventListener("click", async () => {
  setStatus("");
  try {
    await invoke("stop_managed_runtime");
    await refreshStatus();
  } catch (error) {
    setStatus(String(error));
  }
});

listen("packaged-status", (event) => {
  renderStatus(event.payload);
});

setInterval(refreshStatus, 1500);

const win = getCurrentWindow();
win.onDragDropEvent((event) => {
  if (event.payload.type !== "drop") return;
  const gguf = event.payload.paths.find((value) =>
    value.toLowerCase().endsWith(".gguf"),
  );
  if (!gguf) {
    setStatus("Drop a .gguf file.");
    return;
  }
  pathEl.value = gguf;
  pushRecent(gguf);
  setStatus("GGUF path set from drag-and-drop.");
});

refreshStatus();
