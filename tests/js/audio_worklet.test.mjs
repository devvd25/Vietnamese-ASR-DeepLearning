import assert from "node:assert/strict";
import test from "node:test";

test("audio worklet downmixes every input channel instead of dropping speech", async () => {
  globalThis.AudioWorkletProcessor = class {};
  globalThis.registerProcessor = () => {};
  const module = await import(`../../extension/audio-worklet.js?test=${Date.now()}`);
  const left = new Float32Array([1, 0.5, -1]);
  const right = new Float32Array([-1, 0.5, 1]);

  const mono = module.downmixChannels([left, right]);

  assert.deepEqual(Array.from(mono), [0, 0.5, 0]);
});
