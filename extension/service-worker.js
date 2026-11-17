const SERVER_HTTP = "http://127.0.0.1:8765";
const SERVER_WS = "ws://127.0.0.1:8765/ws/captions";
const OFFSCREEN_URL = "offscreen.html";
const DEFAULT_STATE = {
  active: false,
  status: "idle",
  message: "Sẵn sàng tạo phụ đề.",
  model: "wav2vec2_vietnamese",
  chunkSeconds: 4,
  learningEnabled: true,
  tabId: null,
  serverReady: false,
  tabSupported: false,
};

async function readState() {
  const stored = await chrome.storage.session.get("captionState");
  return { ...DEFAULT_STATE, ...(stored.captionState || {}) };
}

async function writeState(patch) {
  const state = { ...(await readState()), ...patch };
  await chrome.storage.session.set({ captionState: state });
  chrome.runtime.sendMessage({ target: "popup", type: "state", state }).catch(() => {});
  return state;
}

function isYouTubeTab(tab) {
  try {
    return Boolean(tab?.id && new URL(tab.url).hostname === "www.youtube.com");
  } catch {
    return false;
  }
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function serverIsReady() {
  try {
    const response = await fetch(`${SERVER_HTTP}/health`, { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

async function ensureOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_URL);
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [offscreenUrl],
  });
  if (contexts.length > 0) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["USER_MEDIA"],
    justification: "Thu âm thanh tab YouTube để tạo phụ đề tiếng Việt local.",
  });
}

async function sendToTab(tabId, message) {
  if (!tabId) return;
  try {
    await chrome.tabs.sendMessage(tabId, { target: "content", ...message });
  } catch {
    // The content script may be unavailable while YouTube is navigating.
  }
}

async function stateForPopup() {
  const [state, tab, ready] = await Promise.all([
    readState(),
    activeTab(),
    serverIsReady(),
  ]);
  return writeState({
    ...state,
    serverReady: ready,
    tabSupported: isYouTubeTab(tab),
  });
}

async function startCaptions(message) {
  const tab = await activeTab();
  if (!isYouTubeTab(tab)) {
    throw new Error("Hãy mở một video trên youtube.com trước khi bắt đầu.");
  }
  if (!(await serverIsReady())) {
    throw new Error("Server local chưa chạy tại 127.0.0.1:8765.");
  }
  await ensureOffscreenDocument();
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
  const model = message.model === "whisper_fine_tuned"
    ? "whisper_fine_tuned"
    : "wav2vec2_vietnamese";
  const chunkSeconds = [3, 4, 5, 6].includes(Number(message.chunkSeconds))
    ? Number(message.chunkSeconds)
    : 4;
  const learningEnabled = message.learningEnabled !== false;
  await writeState({
    active: true,
    status: "connecting",
    message: "Đang kết nối âm thanh tab…",
    model,
    chunkSeconds,
    learningEnabled,
    tabId: tab.id,
    serverReady: true,
    tabSupported: true,
  });
  await sendToTab(tab.id, { type: "session-starting" });
  await chrome.runtime.sendMessage({
    target: "offscreen",
    type: "start",
    streamId,
    tabId: tab.id,
    model,
    chunkSeconds,
    learningEnabled,
    serverUrl: SERVER_WS,
  });
  return readState();
}

async function stopCaptions() {
  const state = await readState();
  await chrome.runtime.sendMessage({ target: "offscreen", type: "stop" }).catch(() => {});
  await sendToTab(state.tabId, { type: "session-stopped" });
  return writeState({
    active: false,
    status: "idle",
    message: "Đã dừng phụ đề.",
  });
}

async function clearCaptions() {
  const state = await readState();
  await sendToTab(state.tabId, { type: "clear" });
  return state;
}

async function relayOffscreenMessage(message) {
  const state = await readState();
  if (message.eventType === "caption") {
    await sendToTab(state.tabId, message.event);
    return writeState({
      status: "listening",
      message: `Đang tạo phụ đề · đoạn ${message.event.sequence}`,
    });
  }
  if (message.eventType === "silence") {
    await sendToTab(state.tabId, message.event);
    return state;
  }
  if (message.eventType === "learning") {
    const total = Number(message.event?.total_samples || 0);
    return writeState({
      message: message.event?.accepted
        ? `Đang học có kiểm soát · đã lưu ${total} mẫu`
        : `Đang nghe · ${total} mẫu học đủ tin cậy`,
    });
  }
  if (message.eventType === "status") {
    await sendToTab(state.tabId, {
      type: "status",
      status: message.status,
      message: message.message,
    });
    return writeState({
      active: message.status !== "idle" && message.status !== "error",
      status: message.status,
      message: message.message,
    });
  }
  if (message.eventType === "error") {
    await sendToTab(state.tabId, {
      type: "error",
      message: message.message,
    });
    return writeState({ active: false, status: "error", message: message.message });
  }
  return state;
}

async function routeMessage(message) {
  switch (message.type) {
    case "get-state":
      return stateForPopup();
    case "start":
      return startCaptions(message);
    case "stop":
      return stopCaptions();
    case "clear":
      return clearCaptions();
    case "save-preferences":
      return writeState({
        model: message.model,
        chunkSeconds: Number(message.chunkSeconds),
        learningEnabled: message.learningEnabled !== false,
      });
    case "offscreen-event":
      return relayOffscreenMessage(message);
    default:
      throw new Error("Yêu cầu extension không hợp lệ.");
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.session.set({ captionState: DEFAULT_STATE });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target !== "service-worker") return false;
  routeMessage(message)
    .then((state) => sendResponse({ ok: true, state }))
    .catch(async (error) => {
      const state = await writeState({
        active: false,
        status: "error",
        message: error?.message || "Extension gặp lỗi không xác định.",
      });
      sendResponse({ ok: false, error: state.message, state });
    });
  return true;
});
