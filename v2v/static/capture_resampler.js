/* Streaming fallback for a browser AudioContext that is not already 16 kHz. */
class V2VStreamingResampler {
  constructor(sourceRate, targetRate) {
    this.sourceRate = sourceRate;
    this.targetRate = targetRate;
    this.step = sourceRate / targetRate;
    this.reset();
  }

  reset() {
    this.buffer = new Float32Array(0);
    this.bufferStart = 0;
    this.nextOutput = 0;
    this.totalInput = 0;
    this.totalOutput = 0;
  }

  process(input) {
    if (this.sourceRate === this.targetRate) {
      this.totalInput += input.length;
      this.totalOutput += input.length;
      return input;
    }
    if (!input.length) return new Float32Array(0);
    const merged = new Float32Array(this.buffer.length + input.length);
    merged.set(this.buffer);
    merged.set(input, this.buffer.length);
    this.buffer = merged;
    this.totalInput += input.length;
    const values = [];
    while (true) {
      const numerator = this.nextOutput * this.sourceRate;
      const sourceIndex = Math.floor(numerator / this.targetRate);
      if (sourceIndex + 1 >= this.totalInput) break;
      const fraction = (numerator % this.targetRate) / this.targetRate;
      const i = sourceIndex - this.bufferStart;
      values.push(this.buffer[i] + (this.buffer[i + 1] - this.buffer[i]) * fraction);
      this.nextOutput += 1;
    }
    this._discardConsumed();
    this.totalOutput += values.length;
    return Float32Array.from(values);
  }

  flush() {
    if (this.sourceRate === this.targetRate || !this.buffer.length) return new Float32Array(0);
    const values = [];
    while (true) {
      const numerator = this.nextOutput * this.sourceRate;
      const sourceIndex = Math.floor(numerator / this.targetRate);
      if (sourceIndex >= this.totalInput) break;
      const fraction = (numerator % this.targetRate) / this.targetRate;
      const i = sourceIndex - this.bufferStart;
      const a = this.buffer[Math.min(i, this.buffer.length - 1)];
      const b = this.buffer[Math.min(i + 1, this.buffer.length - 1)];
      values.push(a + (b - a) * fraction);
      this.nextOutput += 1;
    }
    this.buffer = new Float32Array(0);
    this.bufferStart = this.totalInput;
    this.totalOutput += values.length;
    return Float32Array.from(values);
  }

  _discardConsumed() {
    // Retain the sample immediately before the fractional read position so
    // interpolation stays continuous at the next worklet callback.
    const nextSource = Math.floor(this.nextOutput * this.sourceRate / this.targetRate);
    const retainFrom = Math.max(this.bufferStart, nextSource - 1);
    const consumed = retainFrom - this.bufferStart;
    if (!consumed) return;
    this.buffer = this.buffer.slice(consumed);
    this.bufferStart = retainFrom;
  }

  diagnostics() {
    return {
      source_rate: this.sourceRate,
      target_rate: this.targetRate,
      total_input_samples: this.totalInput,
      total_output_samples: this.totalOutput,
      output_ratio: this.totalInput ? this.totalOutput / this.totalInput : 0,
      buffered_source_samples: this.buffer.length,
    };
  }
}
