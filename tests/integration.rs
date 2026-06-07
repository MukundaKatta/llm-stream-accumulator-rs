//! Integration tests exercising `StreamAccumulator` through its public API,
//! simulating realistic end-to-end streaming scenarios.

use llm_stream_accumulator::{Chunk, StreamAccumulator};

/// A full text-only response delivered as many small deltas, ending with a
/// stop event, the way an SSE stream typically arrives.
#[test]
fn full_text_stream_via_chunks() {
    let mut acc = StreamAccumulator::new();
    for delta in ["The ", "quick ", "brown ", "fox"] {
        acc.push(Chunk::Text(delta.to_string()));
    }
    acc.push(Chunk::OutputTokens(4));
    acc.push(Chunk::Stop("end_turn".to_string()));

    assert_eq!(acc.text(), "The quick brown fox");
    assert!(acc.is_complete());
    assert_eq!(acc.stop_reason(), Some("end_turn"));
    assert_eq!(acc.output_tokens(), 4);
    assert!(acc.has_text());
    assert!(!acc.has_tool_calls());
}

/// A response that emits text and then a tool call whose JSON arrives in
/// several fragments.
#[test]
fn text_then_streamed_tool_call() {
    let mut acc = StreamAccumulator::new();
    acc.push(Chunk::InputTokens(42));
    acc.push(Chunk::Text("Let me look that up. ".to_string()));
    acc.push(Chunk::ToolUse {
        id: "call_1".to_string(),
        name: "web_search".to_string(),
        input_fragment: r#"{"query":"#.to_string(),
    });
    acc.push(Chunk::ToolUse {
        id: "call_1".to_string(),
        name: "web_search".to_string(),
        input_fragment: r#""rust streaming"}"#.to_string(),
    });
    acc.push(Chunk::Stop("tool_use".to_string()));

    assert_eq!(acc.text(), "Let me look that up. ");
    assert_eq!(acc.tool_calls().len(), 1);
    let call = acc.tool_call("call_1").expect("tool call present");
    assert_eq!(call.name, "web_search");
    assert_eq!(call.input_json, r#"{"query":"rust streaming"}"#);
    assert_eq!(acc.input_tokens(), 42);
    assert_eq!(acc.stop_reason(), Some("tool_use"));
    assert!(acc.is_complete());
}

/// Two tool calls whose fragments are interleaved must each be reassembled
/// correctly without phantom duplicates.
#[test]
fn interleaved_tool_calls_reassemble_correctly() {
    let mut acc = StreamAccumulator::new();
    acc.push_tool_use("a", "get_weather", r#"{"city":"#);
    acc.push_tool_use("b", "get_time", r#"{"tz":"#);
    acc.push_tool_use("a", "get_weather", r#""Paris"}"#);
    acc.push_tool_use("b", "get_time", r#""UTC"}"#);

    assert_eq!(acc.tool_calls().len(), 2);
    assert_eq!(
        acc.tool_call("a").unwrap().input_json,
        r#"{"city":"Paris"}"#
    );
    assert_eq!(acc.tool_call("b").unwrap().input_json, r#"{"tz":"UTC"}"#);
}

/// The accumulator can be reused for a second turn after a reset.
#[test]
fn reuse_after_reset() {
    let mut acc = StreamAccumulator::new();
    acc.push_text("first turn");
    acc.push(Chunk::OutputTokens(10));
    acc.finish();
    assert!(acc.is_complete());

    acc.reset();
    assert!(!acc.is_complete());
    assert_eq!(acc.text(), "");
    assert_eq!(acc.output_tokens(), 0);
    assert!(acc.current_tool_id().is_none());

    acc.push_text("second turn");
    assert_eq!(acc.text(), "second turn");
}

/// Unicode deltas split mid-grapheme-cluster across chunks must concatenate
/// back to the original string.
#[test]
fn unicode_deltas_concatenate() {
    let mut acc = StreamAccumulator::new();
    acc.push_text("café ");
    acc.push_text("→ ");
    acc.push_text("🦀");
    assert_eq!(acc.text(), "café → 🦀");
}
