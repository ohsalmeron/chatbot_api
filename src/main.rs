use axum::{
    routing::get,
    Router,
    extract::Query,
    response::{Html, IntoResponse, Response},
    http::{header, StatusCode},
};
use std::{collections::HashMap, pin::Pin, net::SocketAddr};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use reqwest::Client;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;
use regex::Regex;
use once_cell::sync::Lazy;
use tokio::fs;

/// Precompile regex for efficiency
static CONTROL_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[control_\d+\]").unwrap());
static UNKNOWN_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(<unk>|<unk>)").unwrap());
static TOOL_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\[TOOL_CALLS\]|\[TOOL_RESULTS\])").unwrap());
static MULTI_SPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

/// Structs for request & response handling
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
    let app = Router::new()
        .route("/", get(index_handler))  
        .route("/chat", get(chat_handler));

    let addr: SocketAddr = "0.0.0.0:8000".parse().unwrap();
    println!("🚀 Chatbot running at http://{}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Serve the index.html file
async fn index_handler() -> impl IntoResponse {
    match fs::read_to_string("index.html").await {
        Ok(content) => Html(content).into_response(),
        Err(_) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body("index.html not found".into())
            .unwrap(),
    }
}

/// Chat handler
async fn chat_handler(Query(params): Query<HashMap<String, String>>) -> impl IntoResponse {
    println!("Received request with params: {:?}", params);
    let prompt = params.get("prompt").cloned().unwrap_or_else(|| "Hello".to_string());

    println!("🔹 Sending to Ollama: {}", prompt);

    let stream = chat_stream(prompt).await;

    Response::builder()
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

/// **Fast cleaning of response content**
fn clean_content(raw: &str) -> String {
    let text = CONTROL_REGEX.replace_all(raw, "");
    let text = UNKNOWN_REGEX.replace_all(&text, "");
    let text = TOOL_REGEX.replace_all(&text, "");
    MULTI_SPACE.replace_all(&text, " ").into_owned()
}

/// **Chat stream with efficient processing**
async fn chat_stream(prompt: String) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
    let client = Client::new();
    let request = ChatRequest {
        model: "mistral".to_string(),
        messages: vec![Message { role: "user".to_string(), content: prompt }],
        stream: true,
    };

    let response = match client.post("http://localhost:11434/api/chat")
        .json(&request)
        .send()
        .await 
    {
        Ok(resp) => resp,
        Err(err) => {
            eprintln!("Error fetching response: {:?}", err);
            return Box::pin(tokio_stream::once(Err(std::io::Error::new(std::io::ErrorKind::Other, "API request failed"))));
        }
    };

    let (tx, rx) = mpsc::channel(20);

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if let Ok(bytes) = chunk {
                let text = String::from_utf8_lossy(&bytes);
                if let Ok(parsed) = serde_json::from_str::<ChatStreamResponse>(&text) {
                    if let Some(msg) = parsed.message {
                        let cleaned = clean_content(&msg.content);
                        if !cleaned.is_empty() {
                            let _ = tx.send(Ok(cleaned + " ")).await;
                        }
                    }
                }
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}
