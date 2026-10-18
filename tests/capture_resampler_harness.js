const fs = require("fs");
const vm = require("vm");
const { performance } = require("perf_hooks");

vm.runInThisContext(fs.readFileSync("v2v/static/capture_resampler.js", "utf8"));

function signal(n, rate) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin(2 * Math.PI * 997 * i / rate);
  return out;
}
function processBlocks(rate, input, blocks) {
  const r = new V2VStreamingResampler(rate, 16000);
  const parts = [];
  let at = 0, index = 0;
  while (at < input.length) {
    const size = blocks[index++ % blocks.length];
    parts.push(r.process(input.subarray(at, Math.min(input.length, at + size))));
    at += size;
  }
  parts.push(r.flush());
  const n = parts.reduce((sum, x) => sum + x.length, 0);
  const out = new Float32Array(n); let dst = 0;
  for (const p of parts) { out.set(p, dst); dst += p.length; }
  return { out, diagnostics: r.diagnostics() };
}
function assert(condition, message) { if (!condition) throw new Error(message); }
function close(a, b, tolerance, message) {
  assert(a.length === b.length, `${message}: length ${a.length} != ${b.length}`);
  for (let i = 0; i < a.length; i++) assert(Math.abs(a[i] - b[i]) <= tolerance, `${message}: sample ${i}`);
}

const results = {};
for (const rate of [48000, 44100]) {
  for (const seconds of [1, 10, 60]) {
    const input = signal(rate * seconds, rate);
    const continuous = processBlocks(rate, input, [input.length]);
    const split = processBlocks(rate, input, [128, 256, 73, 511, 19]);
    close(continuous.out, split.out, 1e-6, `${rate}/${seconds}s split continuity`);
    assert(Math.abs(split.out.length - seconds * 16000) <= 1, `${rate}/${seconds}s output count`);
    assert(split.diagnostics.buffered_source_samples === 0, `${rate}/${seconds}s bounded flush`);
    results[`${rate}_${seconds}s`] = { samples: split.out.length, ratio: split.out.length / input.length };
  }
}
const native = signal(16000, 16000);
const nativeResult = processBlocks(16000, native, [128, 73, 256]);
close(native, nativeResult.out, 0, "16k native bypass");
assert(nativeResult.diagnostics.buffered_source_samples === 0, "native bypass buffer");
const resetCheck = new V2VStreamingResampler(48000, 16000);
resetCheck.process(signal(128, 48000)); resetCheck.reset();
assert(resetCheck.diagnostics().total_input_samples === 0 && resetCheck.diagnostics().buffered_source_samples === 0, "reset state");

const timingInput = signal(48000 * 10, 48000);
const timing = [];
for (let i = 0; i < timingInput.length; i += 128) {
  const r = new V2VStreamingResampler(48000, 16000);
  const start = performance.now(); r.process(timingInput.subarray(i, i + 128)); timing.push(performance.now() - start);
}
timing.sort((a, b) => a - b);
results.processing_ms = { median: timing[Math.floor(timing.length / 2)], p95: timing[Math.floor(timing.length * .95)] };
console.log(JSON.stringify(results));
