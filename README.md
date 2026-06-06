# llm-stream-accumulator

Accumulate streaming LLM chunks into a complete response.

When you consume an LLM API in streaming mode (for example, server-sent events
from an agent or chat completion endpoint), the response arrives as a sequence
of small deltas: text fragments, partial tool-use arguments, token-usage
updates, and a final stop signal. `llm-stream-accumulator` collects those deltas
and reassembles them into a coherent final result — concatenated text, complete
tool calls with reassembled JSON input, accumulated token counts, and the stop
reason.

The crate is dependency-free and built around a single `StreamAccumulator` type.

## Features

- Accumulate plain text deltas into one final string.
- Reassemble fragmented tool-use input (e.g. streamed JSON arguments) by tool id.
- Track multiple distinct tool calls in a single response.
- Accumulate input/output token counts across chunks.
- Capture the stop reason and completion state.
- Inspect state with helpers like `has_text`, `has_tool_calls`, and `is_complete`.
- `reset` to reuse an accumulator for the next response.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llm-stream-accumulator = "0.1"
```

## Usage

### Accumulating text

```rust
use llm_stream_accumulator::StreamAccumulator;

let mut acc = StreamAccumulator::new();
acc.push_text("Hello");
acc.push_text(", world");
acc.push_text("!");

assert_eq!(acc.text(), "Hello, world!");
assert!(!acc.is_complete());

acc.finish();
assert!(acc.is_complete());
```

### Reassembling tool calls

Tool-use arguments often stream in as JSON fragments. Push them with the same
tool id and the accumulator concatenates the fragments into complete input JSON:

```rust
use llm_stream_accumulator::StreamAccumulator;

let mut acc = StreamAccumulator::new();
acc.push_tool_use("t1", "search", r#"{"q": "#);
acc.push_tool_use("t1", "search", r#""rust"}"#);

let calls = acc.tool_calls();
assert_eq!(calls.len(), 1);
assert_eq!(calls[0].name, "search");
assert_eq!(calls[0].input_json, r#"{"q": "rust"}"#);
```

### Driving the accumulator with `Chunk`

If your transport already decodes events into a tagged type, push them through
the generic `Chunk` enum and let the accumulator route each variant:

```rust
use llm_stream_accumulator::{Chunk, StreamAccumulator};

let mut acc = StreamAccumulator::new();
acc.push(Chunk::Text("The answer is ".to_string()));
acc.push(Chunk::Text("42.".to_string()));
acc.push(Chunk::InputTokens(120));
acc.push(Chunk::OutputTokens(8));
acc.push(Chunk::Stop("end_turn".to_string()));

assert_eq!(acc.text(), "The answer is 42.");
assert_eq!(acc.input_tokens(), 120);
assert_eq!(acc.output_tokens(), 8);
assert_eq!(acc.stop_reason(), Some("end_turn"));
assert!(acc.is_complete());
```

A `Chunk::Stop` automatically marks the accumulator complete, so you don't need
a separate `finish()` call when the stream provides an explicit stop event.

## API overview

| Method | Description |
| --- | --- |
| `StreamAccumulator::new()` | Create an empty accumulator. |
| `push_text(delta)` | Append a plain text delta. |
| `push_tool_use(id, name, fragment)` | Append a tool-use input fragment, keyed by tool id. |
| `push(chunk)` | Route a `Chunk` to the appropriate handler. |
| `finish()` | Mark the stream complete. |
| `text()` | Return the concatenated text. |
| `tool_calls()` | Return the accumulated `ToolCall`s. |
| `input_tokens()` / `output_tokens()` | Accumulated token counts. |
| `stop_reason()` | The stop reason, if any. |
| `is_complete()` | Whether the stream has finished. |
| `has_text()` / `has_tool_calls()` | Presence checks. |
| `reset()` | Clear all accumulated state. |

## Tech stack

- Language: Rust (edition 2021)
- Dependencies: none (standard library only)
- License: MIT

## Development

```bash
cargo build
cargo test
```

## License

Licensed under the MIT License.
