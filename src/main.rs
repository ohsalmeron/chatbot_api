use axum::{routing::get, Router, extract::Query, response::IntoResponse, http::header};
use std::{collections::HashMap, pin::Pin, net::SocketAddr, fs};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use reqwest::Client;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;
use regex::Regex;  // <- We'll use regex to strip control tokens

#[derive(Debug, Deserialize, Serialize)]
struct Personality {
    name: String,
    traits: HashMap<String, f32>,
    ethical_rules: Vec<String>,
    default_tone: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatStreamResponse {
    model: String,
    created_at: String,
    message: Option<ChatMessage>,
    done: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/chat", get(chat_handler));

    let addr: SocketAddr = "0.0.0.0:8000".parse().unwrap();
    println!("🚀 Chatbot running at http://{}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .await
        .unwrap();
}

async fn chat_handler(Query(params): Query<HashMap<String, String>>) -> impl IntoResponse {
    println!("Received request with params: {:?}", params);

    let prompt = params.get("prompt").cloned().unwrap_or_else(|| "Hello".to_string());

    // Load personality and modify the prompt
    let personality = load_personality();
    let modified_prompt = format!(
        "[Personality: {} | Traits: {:?} | Ethics: {:?}]\n{}",
        personality.name, personality.traits, personality.ethical_rules, prompt
    );

    println!("🔹 Sending to Ollama: {}", modified_prompt);

    let stream = chat_stream(modified_prompt).await;

    // We'll respond with "text/plain" to keep it super simple
    axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

/// Reads personality data from a JSON file.
fn load_personality() -> Personality {
    let data = fs::read_to_string("personality.json").expect("Failed to read personality file");
    serde_json::from_str(&data).expect("Invalid personality JSON format")
}

/// Helper to remove control tokens like `[control_8]`, `<unk>`, etc.
fn clean_content(raw: &str) -> String {
    // Regex that removes patterns like `[control_8]` or `[control_250]`
    let control_regex = Regex::new(r"\[control_\d+\]").unwrap();
    // Remove <unk> or <unk> with other brackets if present
    let unknown_regex = Regex::new(r"(<unk>|<unk>)").unwrap();
    // Some Ollama-specific placeholders you might want to remove
    let tool_regex = Regex::new(r"(\[TOOL_CALLS\]|\[TOOL_RESULTS\])").unwrap();

    let without_control = control_regex.replace_all(raw, "");
    let without_unk = unknown_regex.replace_all(&without_control, "");
    let without_tools = tool_regex.replace_all(&without_unk, "");

    // Optionally, you could trim or condense repeated spaces
    // For example, replace multiple spaces with one:
    let multi_space = Regex::new(r"\s+").unwrap();
    let final_str = multi_space.replace_all(&without_tools, " ");
    
    final_str.into_owned()
}

/// 🚀 **Partial Token Streaming (Cleaned)**
async fn chat_stream(prompt: String) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
    let client = Client::new();

    let request = ChatRequest {
        model: "mistral".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt,
        }],
        stream: true, // Enable streaming
    };

    let response = client
        .post("http://localhost:11434/api/chat")
        .json(&request)
        .send()
        .await
        .expect("Failed to call Ollama API");

    let (tx, rx) = mpsc::channel(20);

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    // Try to parse partial JSON from the chunk
                    if let Ok(parsed) = serde_json::from_str::<ChatStreamResponse>(&text) {
                        // Each chunk of text is in `parsed.message.content`
                        if let Some(msg) = parsed.message {
                            // Clean the partial chunk
                            let cleaned = clean_content(&msg.content);
                            if !cleaned.is_empty() {
                                // Stream it as raw text + flush
                                // (We add a slight space or newline to separate tokens)
                                let _ = tx.send(Ok(cleaned + " ")).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    )))
                    .await;
                }
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}
