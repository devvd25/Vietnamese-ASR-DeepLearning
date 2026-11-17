import assert from "node:assert/strict";
import test from "node:test";

test("offscreen events are translated for the YouTube content script", async () => {
  let messageListener;
  let stored = {
    captionState: {
      active: true,
      status: "listening",
      message: "Đang nghe âm thanh…",
      model: "wav2vec2_vietnamese",
      chunkSeconds: 4,
      tabId: 42,
      serverReady: true,
      tabSupported: true,
    },
  };
  const tabMessages = [];
  globalThis.chrome = {
    runtime: {
      onInstalled: { addListener() {} },
      onMessage: { addListener(listener) { messageListener = listener; } },
      sendMessage: async () => undefined,
    },
    storage: {
      session: {
        async get() { return stored; },
        async set(value) { stored = { ...stored, ...value }; },
      },
    },
    tabs: {
      async sendMessage(tabId, message) { tabMessages.push({ tabId, message }); },
    },
  };
  await import(`../../extension/service-worker.js?test=${Date.now()}`);
  const preferenceResponse = await new Promise((resolve) => {
    messageListener(
      {
        target: "service-worker",
        type: "save-preferences",
        model: "whisper_fine_tuned",
        chunkSeconds: 4,
        learningEnabled: true,
      },
      {},
      resolve,
    );
  });
  assert.equal(preferenceResponse.state.learningEnabled, true);
  const event = {
    type: "caption",
    sequence: 3,
    text: "xin chào việt nam",
  };

  const response = await new Promise((resolve) => {
    const keepChannelOpen = messageListener(
      {
        target: "service-worker",
        type: "offscreen-event",
        eventType: "caption",
        event,
      },
      {},
      resolve,
    );
    assert.equal(keepChannelOpen, true);
  });

  assert.equal(response.ok, true);
  assert.deepEqual(tabMessages, [
    { tabId: 42, message: { target: "content", ...event } },
  ]);
  assert.equal(stored.captionState.message, "Đang tạo phụ đề · đoạn 3");

  const silence = {
    type: "silence",
    sequence: 4,
    start_seconds: 9,
    end_seconds: 12,
  };
  const silenceResponse = await new Promise((resolve) => {
    messageListener(
      {
        target: "service-worker",
        type: "offscreen-event",
        eventType: "silence",
        event: silence,
      },
      {},
      resolve,
    );
  });

  assert.equal(silenceResponse.ok, true);
  assert.deepEqual(tabMessages[1], {
    tabId: 42,
    message: { target: "content", ...silence },
  });

  const statusResponse = await new Promise((resolve) => {
    messageListener(
      {
        target: "service-worker",
        type: "offscreen-event",
        eventType: "status",
        status: "listening",
        message: "Đang nghe âm thanh · đoạn 4 giây",
      },
      {},
      resolve,
    );
  });

  assert.equal(statusResponse.ok, true);
  assert.deepEqual(tabMessages[2], {
    tabId: 42,
    message: {
      target: "content",
      type: "status",
      status: "listening",
      message: "Đang nghe âm thanh · đoạn 4 giây",
    },
  });

  const errorResponse = await new Promise((resolve) => {
    messageListener(
      {
        target: "service-worker",
        type: "offscreen-event",
        eventType: "error",
        message: "Mất kết nối server.",
      },
      {},
      resolve,
    );
  });
  assert.equal(errorResponse.ok, true);
  assert.deepEqual(tabMessages[3], {
    tabId: 42,
    message: {
      target: "content",
      type: "error",
      message: "Mất kết nối server.",
    },
  });
});
