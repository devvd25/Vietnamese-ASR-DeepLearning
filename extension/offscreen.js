let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let workletNode = null;
let socket = null;
let packetParts = [];
let packetSamples = 0;
let packetTargetSamples = 0;
let stopping = false;

function notify(type, payload = {}) {
  return chrome.runtime.sendMessage({
    target: "service-worker",
    type: "offscreen-event",
    eventSource: "offscreen",
    eventType: type,
    ...payload,
  }).catch(() => {});
}

function notifyStatus(status, message) {
  return chrome.runtime.sendMessage({
    target: "service-worker",
    type: "offscreen-event",
    eventSource: "offscreen",
    eventType: "status",
    status,
    message,
  }).catch(() => {});
}

export function forwardServerPayload(payload) {
  if (payload.type === "caption" || payload.type === "silence") {
    return notify(payload.type, { event: payload });
  }
  if (payload.type === "learning") {
    return notify("learning", { event: payload });
  }
  if (payload.type === "error") {
    return notify("error", { message: payload.message });
  }
  return Promise.resolve();
}

export function buildCaptionSocketUrl(serverUrl, settings) {
  const params = new URLSearchParams({
    model: settings.model,
    sample_rate: String(settings.sampleRate),
    chunk_seconds: String(settings.chunkSeconds),
    learning: settings.learningEnabled ? "true" : "false",
  });
  return `${serverUrl}?${params}`;
}

function waitForSocketOpen(webSocket) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Server local không phản hồi.")), 5000);
    webSocket.addEventListener("open", () => {
      clearTimeout(timeout);
      resolve();
    }, { once: true });
    webSocket.addEventListener("error", () => {
      clearTimeout(timeout);
      reject(new Error("Không kết nối được server phụ đề local."));
    }, { once: true });
  });
}

function flushPacket() {
  if (!socket || socket.readyState !== WebSocket.OPEN || packetSamples === 0) return;
  const packet = new Float32Array(packetSamples);
  let offset = 0;
  for (const part of packetParts) {
    packet.set(part, offset);
    offset += part.length;
  }
  packetParts = [];
  packetSamples = 0;
  socket.send(packet.buffer);
}

async function stopCapture({ announce = true } = {}) {
  stopping = true;
  flushPacket();
  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
  }
  sourceNode?.disconnect();
  mediaStream?.getTracks().forEach((track) => track.stop());
  if (audioContext && audioContext.state !== "closed") await audioContext.close();
  if (socket && socket.readyState < WebSocket.CLOSING) socket.close(1000, "Stopped by user");
  mediaStream = null;
  audioContext = null;
  sourceNode = null;
  workletNode = null;
  socket = null;
  packetParts = [];
  packetSamples = 0;
  if (announce) await notifyStatus("idle", "Đã dừng phụ đề.");
  stopping = false;
}

async function startCapture(message) {
  await stopCapture({ announce: false });
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: message.streamId,
      },
    },
    video: false,
  });
  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("audio-worklet.js");
  packetTargetSamples = Math.max(2048, Math.round(audioContext.sampleRate * 0.25));
  socket = new WebSocket(buildCaptionSocketUrl(message.serverUrl, {
    model: message.model,
    sampleRate: audioContext.sampleRate,
    chunkSeconds: message.chunkSeconds,
    learningEnabled: message.learningEnabled,
  }));
  await waitForSocketOpen(socket);
  socket.addEventListener("message", async (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      await notifyStatus("error", "Server trả về dữ liệu không hợp lệ.");
      return;
    }
    await forwardServerPayload(payload);
  });
  socket.addEventListener("close", () => {
    if (!stopping) notifyStatus("error", "Mất kết nối với server phụ đề local.");
  });

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  workletNode.port.onmessage = (event) => {
    const samples = new Float32Array(event.data);
    packetParts.push(samples);
    packetSamples += samples.length;
    if (packetSamples >= packetTargetSamples) flushPacket();
  };
  sourceNode.connect(workletNode);
  workletNode.connect(audioContext.destination);
  await notifyStatus("listening", `Đang nghe âm thanh · đoạn ${message.chunkSeconds} giây`);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target !== "offscreen") return false;
  const operation = message.type === "start"
    ? startCapture(message)
    : message.type === "stop"
      ? stopCapture()
      : Promise.reject(new Error("Lệnh offscreen không hợp lệ."));
  operation
    .then(() => sendResponse({ ok: true }))
    .catch(async (error) => {
      await stopCapture({ announce: false });
      const text = error?.message || "Không thể thu âm thanh tab.";
      await chrome.runtime.sendMessage({
        target: "service-worker",
        type: "offscreen-event",
        eventSource: "offscreen",
        eventType: "error",
        message: text,
      }).catch(() => {});
      sendResponse({ ok: false, error: text });
    });
  return true;
});
