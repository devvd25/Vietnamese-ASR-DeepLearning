export function downmixChannels(channels) {
  if (!channels.length) return new Float32Array();
  const mono = new Float32Array(channels[0].length);
  for (const channel of channels) {
    for (let index = 0; index < mono.length; index += 1) {
      mono[index] += channel[index] / channels.length;
    }
  }
  return mono;
}

class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const inputChannels = inputs[0] || [];
    const outputChannels = outputs[0] || [];
    for (let index = 0; index < outputChannels.length; index += 1) {
      const source = inputChannels[index] || inputChannels[0];
      if (source) outputChannels[index].set(source);
      else outputChannels[index].fill(0);
    }
    if (inputChannels.length) {
      const mono = downmixChannels(inputChannels);
      this.port.postMessage(mono.buffer, [mono.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
