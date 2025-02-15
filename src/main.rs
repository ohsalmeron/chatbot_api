use axum::{routing::get, Router, extract::Query, response::IntoResponse, http::{header, StatusCode}};
use std::{collections::HashMap, pin::Pin, net::SocketAddr, fs, sync::Arc};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use reqwest::Client;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;
use regex::Regex;
use rand::prelude::*;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Personality {
    name: String,
    traits: HashMap<String, f32>,
    ethical_rules: Vec<String>,
    default_tone: String,
    signature_phrases: Vec<String>,
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
    println!("🚀 Personality Chatbot running at http://{}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .await
        .unwrap();
}

async fn chat_handler(Query(params): Query<HashMap<String, String>>) -> impl IntoResponse {
    println!("📥 Received request: {:?}", params);

    let personality = load_personality();
    let prompt = params.get("prompt").cloned().unwrap_or_else(|| "Hello".to_string());

    // Ethical check
    if personality.ethical_rules.iter().any(|rule| 
        prompt.to_lowercase().contains(&rule.to_lowercase())
    ) {
        return axum::response::Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(axum::body::Body::from("🚫 Ethical constraint violated"))
            .unwrap();
    }

    // Structured system prompt
    let modified_prompt = format!(
        "[System: You are {} - {}.\nCore Traits: {}\nEthical Constraints: {}\nSignature Style: {}]\n\nUser: {}",
        personality.name,
        personality.default_tone,
        personality.traits.iter()
            .map(|(k, v)| format!("{} ({:.0}%)", k, *v * 100.0))
            .collect::<Vec<_>>()
            .join(", "),
        personality.ethical_rules.join(". "),
        personality.signature_phrases.join(" "),
        prompt
    );

    println!("🔹 Enhanced prompt:\n{}", modified_prompt);

    let stream = chat_stream(modified_prompt, personality).await;

    axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

fn load_personality() -> Personality {
    let data = fs::read_to_string("personality.json").expect("Failed to read personality file");
    serde_json::from_str(&data).expect("Invalid personality JSON format")
}

fn clean_content(raw: &str, personality: &Personality) -> String {
    let preserve_regex = Regex::new(r"\[System:.*?\]").unwrap();
    let preserved = preserve_regex.replace_all(raw, |caps: &regex::Captures| {
        caps.get(0).unwrap().as_str().to_string()
    });

    let control_regex = Regex::new(r"\[control_\d+\]|<unk>|\[TOOL_CALLS\]|\[TOOL_RESULTS\]").unwrap();
    let cleaned = control_regex.replace_all(&preserved, "");

    let mut rng = thread_rng();

    // Insert signature phrases randomly in sentences
    let mut words: Vec<&str> = cleaned.split_whitespace().collect();
    if !personality.signature_phrases.is_empty() && rng.gen_bool(0.3) {
        let index = rng.gen_range(1..words.len());
        let phrase = personality.signature_phrases.choose(&mut rng).unwrap();
        words.insert(index, phrase);
    }

    words.join(" ")
}

async fn enforce_personality(response: String, personality: &Personality) -> String {
    let mut modified = response;

    let mut rng = thread_rng();

    // Apply sarcasm
    if personality.traits.get("sarcasm").unwrap_or(&0.0) > &0.5 && rng.gen_bool(0.5) {
        modified.push_str(" Oh wow, genius move.");
    }

    // Add curiosity-driven follow-up
    if personality.traits.get("curiosity").unwrap_or(&0.0) > &0.7 && rng.gen_bool(0.4) {
        modified.push_str(" But wait—have you ever considered the opposite approach?");
    }

    modified
}

async fn chat_stream(prompt: String, personality: Personality) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
    let client = Client::new();
    let personality = Arc::new(personality);

    let request = ChatRequest {
        model: "mistral".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt,
        }],
        stream: true,
    };

    let response = client
        .post("http://localhost:11434/api/chat")
        .json(&request)
        .send()
        .await
        .expect("Failed to call Ollama API");

    let (tx, rx) = mpsc::channel(20);
    let personality_clone = Arc::clone(&personality);

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    if let Ok(parsed) = serde_json::from_str::<ChatStreamResponse>(&text) {
                        if let Some(msg) = parsed.message {
                            let cleaned = clean_content(&msg.content, &personality_clone);
                            let enforced = enforce_personality(cleaned, &personality_clone).await;
                            
                            if !enforced.is_empty() {
                                let _ = tx.send(Ok(enforced + " ")).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::new(std::io::ErrorKind::Other, e))).await;
                }
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}
