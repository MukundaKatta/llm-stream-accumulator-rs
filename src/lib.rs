/*!
llm-stream-accumulator: accumulate LLM streaming chunks into a final response.

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
*/

/// A single streamed chunk.
#[derive(Debug, Clone)]
pub enum Chunk {
    Text(String),
    ToolUse { id: String, name: String, input_fragment: String },
    InputTokens(u64),
    OutputTokens(u64),
    Stop(String),
}

/// Accumulated tool-use call.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input_json: String,
}

/// Accumulated result of a streaming LLM response.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    text_parts: Vec<String>,
    tool_calls: Vec<ToolCall>,
    /// Current in-progress tool use id (last seen).
    current_tool_id: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
    stop_reason: Option<String>,
    complete: bool,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a plain text delta.
    pub fn push_text(&mut self, delta: &str) {
        self.text_parts.push(delta.to_string());
    }

    /// Push a tool-use fragment.
    pub fn push_tool_use(&mut self, id: &str, name: &str, input_fragment: &str) {
        if self.current_tool_id.as_deref() != Some(id) {
            self.tool_calls.push(ToolCall { id: id.to_string(), name: name.to_string(), input_json: String::new() });
            self.current_tool_id = Some(id.to_string());
        }
        if let Some(tc) = self.tool_calls.iter_mut().find(|t| t.id == id) {
            tc.input_json.push_str(input_fragment);
        }
    }

    /// Push a generic chunk.
    pub fn push(&mut self, chunk: Chunk) {
        match chunk {
            Chunk::Text(s) => self.push_text(&s),
            Chunk::ToolUse { id, name, input_fragment } => self.push_tool_use(&id, &name, &input_fragment),
            Chunk::InputTokens(n) => self.input_tokens += n,
            Chunk::OutputTokens(n) => self.output_tokens += n,
            Chunk::Stop(r) => { self.stop_reason = Some(r); self.complete = true; }
        }
    }

    /// Mark stream as done.
    pub fn finish(&mut self) {
        self.complete = true;
    }

    /// Concatenated text.
    pub fn text(&self) -> String {
        self.text_parts.join("")
    }

    /// Completed tool calls.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn input_tokens(&self) -> u64 { self.input_tokens }
    pub fn output_tokens(&self) -> u64 { self.output_tokens }
    pub fn stop_reason(&self) -> Option<&str> { self.stop_reason.as_deref() }
    pub fn is_complete(&self) -> bool { self.complete }

    /// True if any text was accumulated.
    pub fn has_text(&self) -> bool { !self.text_parts.is_empty() }

    /// True if any tool calls were accumulated.
    pub fn has_tool_calls(&self) -> bool { !self.tool_calls.is_empty() }

    /// Reset all accumulated state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_text() {
        let mut acc = StreamAccumulator::new();
        acc.push_text("Hello");
        acc.push_text(", world");
        assert_eq!(acc.text(), "Hello, world");
    }

    #[test]
    fn finish_marks_complete() {
        let mut acc = StreamAccumulator::new();
        assert!(!acc.is_complete());
        acc.finish();
        assert!(acc.is_complete());
    }

    #[test]
    fn stop_chunk_marks_complete() {
        let mut acc = StreamAccumulator::new();
        acc.push(Chunk::Stop("end_turn".to_string()));
        assert!(acc.is_complete());
        assert_eq!(acc.stop_reason(), Some("end_turn"));
    }

    #[test]
    fn token_counts_accumulate() {
        let mut acc = StreamAccumulator::new();
        acc.push(Chunk::InputTokens(100));
        acc.push(Chunk::InputTokens(50));
        acc.push(Chunk::OutputTokens(200));
        assert_eq!(acc.input_tokens(), 150);
        assert_eq!(acc.output_tokens(), 200);
    }

    #[test]
    fn tool_use_accumulates_fragments() {
        let mut acc = StreamAccumulator::new();
        acc.push_tool_use("t1", "search", r#"{"q": "#);
        acc.push_tool_use("t1", "search", r#""rust"}"#);
        assert_eq!(acc.tool_calls().len(), 1);
        assert_eq!(acc.tool_calls()[0].input_json, r#"{"q": "rust"}"#);
    }

    #[test]
    fn multiple_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.push_tool_use("t1", "search", "{}");
        acc.push_tool_use("t2", "fetch", "{}");
        assert_eq!(acc.tool_calls().len(), 2);
    }

    #[test]
    fn has_text_and_has_tool_calls() {
        let mut acc = StreamAccumulator::new();
        assert!(!acc.has_text());
        assert!(!acc.has_tool_calls());
        acc.push_text("hi");
        assert!(acc.has_text());
        acc.push_tool_use("t1", "fn", "{}");
        assert!(acc.has_tool_calls());
    }

    #[test]
    fn reset_clears_state() {
        let mut acc = StreamAccumulator::new();
        acc.push_text("hello");
        acc.push(Chunk::InputTokens(100));
        acc.reset();
        assert!(acc.text().is_empty());
        assert_eq!(acc.input_tokens(), 0);
    }

    #[test]
    fn empty_text_by_default() {
        let acc = StreamAccumulator::new();
        assert_eq!(acc.text(), "");
    }

    #[test]
    fn push_text_chunk() {
        let mut acc = StreamAccumulator::new();
        acc.push(Chunk::Text("chunk".to_string()));
        assert_eq!(acc.text(), "chunk");
    }

    #[test]
    fn stop_reason_none_before_stop() {
        let acc = StreamAccumulator::new();
        assert_eq!(acc.stop_reason(), None);
    }

    #[test]
    fn tool_call_name_preserved() {
        let mut acc = StreamAccumulator::new();
        acc.push_tool_use("id1", "my_tool", "{}");
        assert_eq!(acc.tool_calls()[0].name, "my_tool");
    }

    #[test]
    fn interleaved_text_and_tool() {
        let mut acc = StreamAccumulator::new();
        acc.push_text("I'll search for ");
        acc.push_tool_use("t1", "search", r#"{"q": "foo"}"#);
        acc.push_text("Done.");
        assert_eq!(acc.text(), "I'll search for Done.");
        assert_eq!(acc.tool_calls().len(), 1);
    }
}
