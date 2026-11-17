const form = document.getElementById("caption-form");
const modelSelect = document.getElementById("model");
const chunkSelect = document.getElementById("chunk-seconds");
const learningToggle = document.getElementById("learning-enabled");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const clearButton = document.getElementById("clear-button");
const statusMessage = document.getElementById("status-message");
const statusDot = document.getElementById("status-dot");

function send(type, payload = {}) {
  return chrome.runtime.sendMessage({ target: "service-worker", type, ...payload });
}

function render(state) {
  const busy = state.status === "connecting";
  statusMessage.textContent = state.message;
  statusDot.dataset.state = state.status === "idle" && state.serverReady ? "ready" : state.status;
  modelSelect.value = state.model || "wav2vec2_vietnamese";
  chunkSelect.value = String(state.chunkSeconds || 4);
  learningToggle.checked = state.learningEnabled !== false;
  startButton.disabled = Boolean(state.active || busy || !state.serverReady || !state.tabSupported);
  stopButton.disabled = !state.active;
  modelSelect.disabled = Boolean(state.active || busy);
  chunkSelect.disabled = Boolean(state.active || busy);
  learningToggle.disabled = Boolean(state.active || busy);
  startButton.setAttribute("aria-busy", busy ? "true" : "false");
  if (!state.serverReady) {
    statusMessage.textContent = "Server local chưa chạy tại 127.0.0.1:8765.";
    statusDot.dataset.state = "error";
  } else if (!state.tabSupported && !state.active) {
    statusMessage.textContent = "Hãy mở một video trên youtube.com.";
    statusDot.dataset.state = "connecting";
  }
}

async function update(type, payload = {}) {
  const response = await send(type, payload);
  if (response?.state) render(response.state);
  if (!response?.ok) throw new Error(response?.error || "Extension không phản hồi.");
  return response.state;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  startButton.disabled = true;
  startButton.setAttribute("aria-busy", "true");
  statusMessage.textContent = "Đang bắt đầu…";
  try {
    await update("start", {
      model: modelSelect.value,
      chunkSeconds: Number(chunkSelect.value),
      learningEnabled: learningToggle.checked,
    });
  } catch (error) {
    statusMessage.textContent = error.message;
    statusDot.dataset.state = "error";
  } finally {
    startButton.setAttribute("aria-busy", "false");
  }
});

stopButton.addEventListener("click", () => update("stop").catch((error) => {
  statusMessage.textContent = error.message;
  statusDot.dataset.state = "error";
}));

clearButton.addEventListener("click", () => update("clear").catch((error) => {
  statusMessage.textContent = error.message;
  statusDot.dataset.state = "error";
}));

for (const control of [modelSelect, chunkSelect, learningToggle]) {
  control.addEventListener("change", () => {
    send("save-preferences", {
      model: modelSelect.value,
      chunkSeconds: Number(chunkSelect.value),
      learningEnabled: learningToggle.checked,
    }).catch(() => {});
  });
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.target === "popup" && message.type === "state") render(message.state);
});

update("get-state").catch((error) => {
  statusMessage.textContent = error.message;
  statusDot.dataset.state = "error";
});
