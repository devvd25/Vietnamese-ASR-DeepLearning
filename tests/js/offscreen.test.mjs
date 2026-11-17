import assert from "node:assert/strict";
import test from "node:test";

test("caption socket opts into controlled YouTube learning explicitly", async () => {
  globalThis.chrome = {
    runtime: {
      onMessage: { addListener() {} },
      async sendMessage() {},
    },
  };
  const module = await import(`../../extension/offscreen.js?url-test=${Date.now()}`);

  assert.equal(
    module.buildCaptionSocketUrl("ws://127.0.0.1:8765/ws/captions", {
      model: "whisper_fine_tuned",
      sampleRate: 48_000,
      chunkSeconds: 4,
      learningEnabled: true,
    }),
    "ws://127.0.0.1:8765/ws/captions?model=whisper_fine_tuned&sample_rate=48000&chunk_seconds=4&learning=true",
  );
});

test("offscreen forwards no-speech windows so stale captions can be hidden", async () => {
  const messages = [];
  globalThis.chrome = {
    runtime: {
      onMessage: { addListener() {} },
      async sendMessage(message) { messages.push(message); },
    },
  };
  const module = await import(`../../extension/offscreen.js?test=${Date.now()}`);

  await module.forwardServerPayload({
    type: "silence",
    sequence: 7,
    start_seconds: 12,
    end_seconds: 15,
  });

  assert.deepEqual(messages, [{
    target: "service-worker",
    type: "offscreen-event",
    eventSource: "offscreen",
    eventType: "silence",
    event: {
      type: "silence",
      sequence: 7,
      start_seconds: 12,
      end_seconds: 15,
    },
  }]);
});
