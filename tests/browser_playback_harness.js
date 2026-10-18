const fs = require("fs");

const sent = [];
const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      textContent: "",
      style: {},
      className: "",
      classList: { add() {}, remove() {} },
      value: "1.10",
    });
  }
  return elements.get(id);
}
global.document = { getElementById: element };
global.localStorage = { getItem() { return "[]"; }, setItem() {} };
global.location = { protocol: "http:", host: "127.0.0.1:8088", search: "" };

class FakeSource {
  constructor(ctx) { this.ctx = ctx; this.playbackRate = { value: 1 }; this.stopped = false; }
  connect() {}
  disconnect() {}
  start(at) { this.startedAt = at; }
  stop() { this.stopped = true; }
}
class FakeAudioContext {
  constructor() { this.currentTime = 0; this.sources = []; }
  createBuffer(_channels, length, sampleRate) {
    return { length, sampleRate, getChannelData() { return new Float32Array(length); } };
  }
  createBufferSource() { const source = new FakeSource(this); this.sources.push(source); return source; }
  resume() { return Promise.resolve(); }
}
let ctx;
global.AudioContext = class extends FakeAudioContext { constructor() { super(); ctx = this; } };
global.WebSocket = class {
  constructor() { this.readyState = 1; }
  send(value) { try { sent.push(JSON.parse(value)); } catch {} }
};
global.performance = { now: () => (ctx ? ctx.currentTime * 1000 : 0) };
global.navigator = { mediaDevices: {} };

const html = fs.readFileSync("v2v/static/index.html", "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
const exposed = `
globalThis.__schedulerHarness = {
  accept(gid) { activeGeneration = gid; resetPlayPipeline(gid); playOpen = true; playCtx = new AudioContext({ sampleRate: 24000 }); },
  packet(gid, durationMs = 160) { const n = Math.round(24000 * durationMs / 1000); const out = new ArrayBuffer(4 + n * 2); new DataView(out).setUint32(0, gid, true); playPcmPacket(out); },
  sourceCount(gid) { return (scheduledSources[gid] || []).length; },
  pendingCount(gid) { return pendingChunks.filter((c) => c.gen === gid).length; },
  scheduledAhead(gid) { return scheduledAheadMs(gid); },
  markComplete(gid) { markTtsComplete(gid); },
  state(gid) { return playbackState(gid); },
  sources() { return playCtx.sources; },
  advance(ms) { playCtx.currentTime += ms / 1000; },
  drain(gid) { while ((scheduledSources[gid] || []).length) { const next = [...scheduledSources[gid]].sort((a, b) => a.end - b.end)[0]; playCtx.currentTime = next.end; next.src.ended = true; next.src.onended(); } },
  scheduled(gid) { return [...(scheduledSources[gid] || [])]; },
  cancel(gid) { cancelPlayback(gid, "confirmed-user-speech"); },
  starveCount() { return starveCount; },
  gapMax() { return gapMax; },
  generationCount() { return Object.keys(playbackGenerations).length; },
};`;
eval(script.replace(/\nconnect\(\);\s*$/, "") + exposed);
const h = globalThis.__schedulerHarness;

h.accept(1);
const longStart = h.sources().length;
h.packet(1);
h.advance(100); // Model a second PCM packet arriving 100 ms after its subtitle.
for (let i = 1; i < 30; i++) h.packet(1);
const beforeCompleteSources = h.sourceCount(1);
const beforeCompletePending = h.pendingCount(1);
const scheduledBeforeComplete = h.scheduledAhead(1);
h.markComplete(1);
const stateAfterTtsComplete = h.state(1);
const completionImmediatelyAfterTtsComplete = stateAfterTtsComplete.completionSent;
const normalTtsCompleteStoppedSources = h.sources().slice(longStart).filter((source) => source.stopped).length;
const doneBeforeFinalEnd = sent.some((m) => m.type === "playback_done" && m.generation_id === 1);
h.drain(1);
const longSources = h.sources().slice(longStart);
const starts = longSources.map((s) => s.startedAt).sort((a, b) => a - b);
const ends = starts.map((start, i) => start + 0.16 / 1.10);
const contiguous = starts.slice(1).every((start, i) => Math.abs(start - ends[i]) < 0.004);
const state = h.state(1);

h.accept(2);
for (let i = 0; i < 10; i++) h.packet(2);
const cancelScheduled = h.scheduled(2);
h.cancel(2);
const cancelledSourcesStopped = cancelScheduled.every((item) => item.src.stopped);
const cancelledPending = h.pendingCount(2);

for (let gid = 3; gid <= 12; gid++) {
  h.accept(gid);
  h.markComplete(gid);
}
const boundedGenerationCount = h.generationCount();
const activeGenerationRetained = Boolean(h.state(12));

console.log(JSON.stringify({
  max_scheduled_ahead_ms: state.maxScheduledAheadMs,
  max_pending_buffer_ms: state.maxPendingBufferMs,
  scheduled_before_tts_complete: beforeCompleteSources,
  pending_before_tts_complete: beforeCompletePending,
  scheduled_ahead_before_tts_complete_ms: scheduledBeforeComplete,
  playback_complete_immediately_after_tts_complete: completionImmediatelyAfterTtsComplete,
  normal_tts_complete_stopped_sources: normalTtsCompleteStoppedSources,
  playback_done_before_last_source_ended: doneBeforeFinalEnd,
  last_source_ended: state.lastSourceEnded,
  playback_generation_complete: state.completionSent,
  all_final_audio_played: longSources.length === 30 && longSources.every((source) => source.ended),
  final_tail_chunks_played: longSources.slice(-7).every((source) => source.ended),
  no_gaps: contiguous,
  starvation_count: h.starveCount(),
  max_gap_ms: h.gapMax(),
  cancelled_sources_stopped: cancelledSourcesStopped,
  cancelled_pending_chunks: cancelledPending,
  cancelled_generation_complete_sent: sent.some((m) => m.type === "playback_done" && m.generation_id === 2),
  subtitle_to_first_audio_ms: longSources[0].startedAt * 1000,
  bounded_generation_count: boundedGenerationCount,
  active_generation_retained: activeGenerationRetained,
}));
