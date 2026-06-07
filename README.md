# llm-stream-accumulator

[![CI](https://github.com/MukundaKatta/llm-stream-accumulator-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-stream-accumulator-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A tiny, dependency-free Rust library for accumulating the incremental chunks of
a streaming LLM response (e.g. Server-Sent Events from an agent or chat
completion endpoint) into a single, complete result.

When you consume an LLM streaming API you receive a sequence of small deltas:
text fragments, partial tool-call JSON, token counts, and a final stop event.
`StreamAccumulator` collects those deltas as they arrive and lets you read back
the finished text, the reassembled tool calls, token usage, and the stop reason.

## Features

- **Zero dependencies** — pure `std`, fast to compile, easy to audit.
- **Text accumulation** — concatenate streamed text deltas into one string.
- **Tool-call reassembly** — merge partial tool-use JSON fragments by id, even
  when fragments for different tool calls are interleaved.
- **Usage tracking** — sum streamed input/output token counts.
- **Completion state** — track the stop reason and whether the stream is done.
- **Reusable** — `reset()` the accumulator to use it across multiple turns.

## Installation

Add it to your `Cargo.toml`:

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

### Driving it from a stream of chunks

In a real client you map each event from the wire to a [`Chunk`] and feed it to
`push`:

```rust
use llm_stream_accumulator::{Chunk, StreamAccumulator};

let mut acc = StreamAccumulator::new();

// These would normally come from your SSE / streaming HTTP client.
let events = vec![
    Chunk::InputTokens(42),
    Chunk::Text("Let me check the weather. ".to_string()),
    Chunk::ToolUse {
        id: "call_1".to_string(),
        name: "get_weather".to_string(),
        input_fragment: r#"{"city":"#.to_string(),
    },
    Chunk::ToolUse {
        id: "call_1".to_string(),
        name: "get_weather".to_string(),
        input_fragment: r#""Paris"}"#.to_string(),
    },
    Chunk::OutputTokens(17),
    Chunk::Stop("tool_use".to_string()),
];

for event in events {
    acc.push(event);
}

assert_eq!(acc.text(), "Let me check the weather. ");
assert!(acc.is_complete());
assert_eq!(acc.stop_reason(), Some("tool_use"));

// The fragmented tool-call JSON is reassembled into one call.
let call = acc.tool_call("call_1").unwrap();
assert_eq!(call.name, "get_weather");
assert_eq!(call.input_json, r#"{"city":"Paris"}"#);

assert_eq!(acc.input_tokens(), 42);
assert_eq!(acc.output_tokens(), 17);
```

## API overview

### `StreamAccumulator`

| Method | Description |
| --- | --- |
| `new()` | Create an empty accumulator. |
| `push(chunk: Chunk)` | Apply any [`Chunk`] variant. |
| `push_text(delta: &str)` | Append a text delta. |
| `push_tool_use(id, name, input_fragment)` | Append a tool-use JSON fragment, merged by `id`. |
| `finish()` | Mark the stream as complete. |
| `text() -> String` | The concatenated text so far. |
| `tool_calls() -> &[ToolCall]` | All reassembled tool calls. |
| `tool_call(id: &str) -> Option<&ToolCall>` | Look up a tool call by id. |
| `current_tool_id() -> Option<&str>` | Id of the most recently seen tool block. |
| `input_tokens() -> u64` / `output_tokens() -> u64` | Accumulated token usage. |
| `stop_reason() -> Option<&str>` | The stop reason, once received. |
| `is_complete() -> bool` | Whether the stream has finished. |
| `has_text() -> bool` / `has_tool_calls() -> bool` | Quick presence checks. |
| `reset()` | Clear all state so the accumulator can be reused. |

### `Chunk`

A single streamed event:

- `Text(String)` — a text delta.
- `ToolUse { id, name, input_fragment }` — a tool-use JSON fragment.
- `InputTokens(u64)` / `OutputTokens(u64)` — token usage deltas.
- `Stop(String)` — terminal event carrying the stop reason.

### `ToolCall`

A reassembled tool call with public fields `id`, `name`, and `input_json`
(the concatenated JSON; parse it with your JSON library of choice).

## Development

```sh
cargo build
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## License

Licensed under the [MIT License](LICENSE).
