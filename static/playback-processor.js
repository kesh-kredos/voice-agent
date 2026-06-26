// AudioWorklet processor for streaming float32 PCM playback.
// Receives { type: 'samples', data: Float32Array, epoch } chunks via the
// message port and appends them to an internal ring buffer. The render loop
// drains the buffer into the output; on underrun it emits silence. A 'clear'
// message empties the buffer and re-arms the prebuffer gate so stale audio
// from a previous turn never leaks into the next one. When the buffer
// transitions from non-empty to empty mid-render, the processor posts
// { type: 'ended', epoch } — echoing the epoch of the most recent 'samples' —
// so the UI can drop its "agent speaking" state exactly when audio stops and
// ignore drain notifications stale enough to belong to a prior utterance.

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const prebufferSec =
      (options.processorOptions && options.processorOptions.prebufferSec) || 0.15;
    this.prebufferSamples = Math.floor(prebufferSec * sampleRate);

    // Ring buffer state
    this.capacity = Math.floor(1.5 * sampleRate); // 1.5s initial capacity
    this.buffer = new Float32Array(this.capacity);
    this.readPos = 0;
    this.writePos = 0;
    this.filled = 0;

    // We hold output until at least prebufferSamples have accumulated so the
    // first render callback doesn't immediately underrun.
    this.ready = false;

    // True once we've posted 'ended' for the current drain-to-empty, so we
    // don't re-post every render while the buffer stays empty. Re-armed when
    // new samples arrive or on 'clear'.
    this.endedReported = false;

    // Epoch stamped on each 'samples' message by the main thread; echoed back
    // in 'ended' so the UI can reject stale drain notifications.
    this.epoch = 0;

    this.port.onmessage = (e) => this._handleMessage(e.data);
  }

  _handleMessage(msg) {
    if (!msg || typeof msg.type !== 'string') return;

    if (msg.type === 'clear') {
      this.filled = 0;
      this.readPos = 0;
      this.writePos = 0;
      this.ready = false;
      this.endedReported = false;
      return;
    }

    if (msg.type === 'samples') {
      const chunk = msg.data;
      if (!chunk || chunk.length === 0) return;
      this._append(chunk);
      this.epoch = msg.epoch;
      this.endedReported = false;
      if (!this.ready && this.filled >= this.prebufferSamples) {
        this.ready = true;
      }
    }
  }

  _append(chunk) {
    const needed = this.filled + chunk.length;
    if (needed > this.capacity) {
      // Grow: linearise the existing data into a fresh buffer.
      const newCap = Math.max(needed, this.capacity * 2);
      const fresh = new Float32Array(newCap);
      if (this.filled > 0) {
        const tail = this.capacity - this.readPos;
        if (tail >= this.filled) {
          fresh.set(this.buffer.subarray(this.readPos, this.readPos + this.filled), 0);
        } else {
          fresh.set(this.buffer.subarray(this.readPos), 0);
          fresh.set(this.buffer.subarray(0, this.filled - tail), tail);
        }
      }
      this.buffer = fresh;
      this.capacity = newCap;
      this.readPos = 0;
      this.writePos = this.filled;
    }

    // Write chunk into the ring.
    for (let i = 0; i < chunk.length; i++) {
      this.buffer[this.writePos] = chunk[i];
      this.writePos = (this.writePos + 1) % this.capacity;
    }
    this.filled += chunk.length;
  }

  process(_inputs, outputs) {
    const output = outputs[0] && outputs[0][0];
    if (!output) return true;

    if (!this.ready || this.filled === 0) {
      output.fill(0);
      return true;
    }

    for (let i = 0; i < output.length; i++) {
      if (this.filled > 0) {
        output[i] = this.buffer[this.readPos];
        this.readPos = (this.readPos + 1) % this.capacity;
        this.filled--;
      } else {
        // Underrun — emit silence rather than looping old audio.
        output[i] = 0;
      }
    }

    // Playback just drained the last buffered sample: tell the UI the agent
    // stopped speaking. The early-return above handles the steady-state empty
    // case; this only fires on the non-empty → empty transition.
    if (this.filled === 0 && !this.endedReported) {
      this.port.postMessage({ type: 'ended', epoch: this.epoch });
      this.endedReported = true;
    }

    return true;
  }
}

registerProcessor('playback-processor', PlaybackProcessor);
