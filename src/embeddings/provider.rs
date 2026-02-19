use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(test)]
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::config::{parse_provider_model, EmbeddingsConfig};
use crate::error::{MomoError, Result};

enum EmbeddingBackend {
    Local {
        query_model: Arc<Mutex<TextEmbedding>>,
        ingest_model: Arc<Mutex<TextEmbedding>>,
        batch_size: usize,
        ingest_batch_size: usize,
        ingest_batch_pause_ms: u64,
    },
    #[cfg(test)]
    Mock,
}

pub struct EmbeddingProvider {
    backend: EmbeddingBackend,
    dimensions: usize,
}

impl EmbeddingProvider {
    /// Sync constructor for local models only.
    pub fn new(config: &EmbeddingsConfig) -> Result<Self> {
        let (provider, model_name) = parse_provider_model(&config.model);

        if provider != "local" {
            return Err(MomoError::Embedding(format!(
                "Unsupported embedding provider: {provider}. Local embeddings only.",
            )));
        }

        Self::new_local(config, model_name)
    }

    fn new_local(config: &EmbeddingsConfig, model_name: &str) -> Result<Self> {
        let embedding_model = resolve_embedding_model(model_name);

        let ingest_batch_size = std::env::var("EMBEDDING_INGEST_BATCH_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|size| *size > 0)
            .unwrap_or_else(|| config.batch_size.clamp(1, 32));
        let ingest_batch_pause_ms = std::env::var("EMBEDDING_INGEST_BATCH_PAUSE_MS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .unwrap_or(0);

        let dual_model = std::env::var("EMBEDDING_DUAL_MODEL")
            .ok()
            .and_then(|raw| parse_bool(&raw))
            .unwrap_or(true);

        let query_model = Arc::new(Mutex::new(build_model(embedding_model.clone())?));
        let ingest_model = if dual_model {
            Arc::new(Mutex::new(build_model(embedding_model)?))
        } else {
            Arc::clone(&query_model)
        };

        Ok(Self {
            backend: EmbeddingBackend::Local {
                query_model,
                ingest_model,
                batch_size: config.batch_size,
                ingest_batch_size,
                ingest_batch_pause_ms,
            },
            dimensions: config.dimensions,
        })
    }

    /// Deterministic mock embeddings for tests and offline validation.
    #[cfg(test)]
    pub fn new_mock(dimensions: usize) -> Self {
        Self {
            backend: EmbeddingBackend::Mock,
            dimensions: dimensions.max(1),
        }
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_with_mode(texts, EmbeddingMode::Query).await
    }

    async fn embed_with_mode(
        &self,
        texts: Vec<String>,
        mode: EmbeddingMode,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        match &self.backend {
            EmbeddingBackend::Local {
                query_model,
                ingest_model,
                batch_size,
                ..
            } => {
                let selected = match mode {
                    EmbeddingMode::Query => query_model,
                    EmbeddingMode::Ingest => ingest_model,
                };
                let model = Arc::clone(selected);
                let batch_size = *batch_size;
                tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().map_err(|e| {
                        MomoError::Embedding(format!("Embedding model lock poisoned: {e}"))
                    })?;
                    model
                        .embed(texts, Some(batch_size))
                        .map_err(|e| MomoError::Embedding(e.to_string()))
                })
                .await
                .map_err(|e| MomoError::Embedding(format!("Embedding worker failed: {e}")))?
            }
            #[cfg(test)]
            EmbeddingBackend::Mock => Ok(texts
                .into_iter()
                .map(|text| deterministic_embedding(&text, self.dimensions, mode))
                .collect()),
        }
    }

    pub async fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(vec![text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| MomoError::Embedding("No embedding generated".to_string()))
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        match &self.backend {
            EmbeddingBackend::Local { .. } => {
                // Local models use query: prefix
                let prefixed = format!("query: {query}");
                self.embed_single(&prefixed).await
            }
            #[cfg(test)]
            EmbeddingBackend::Mock => {
                let embeddings = self
                    .embed_with_mode(vec![query.to_string()], EmbeddingMode::Query)
                    .await?;
                embeddings
                    .into_iter()
                    .next()
                    .ok_or_else(|| MomoError::Embedding("No embedding generated".to_string()))
            }
        }
    }

    pub async fn embed_passage(&self, passage: &str) -> Result<Vec<f32>> {
        match &self.backend {
            EmbeddingBackend::Local { .. } => {
                // Local models use passage: prefix
                let prefixed = format!("passage: {passage}");
                self.embed_single(&prefixed).await
            }
            #[cfg(test)]
            EmbeddingBackend::Mock => {
                let embeddings = self
                    .embed_with_mode(vec![passage.to_string()], EmbeddingMode::Ingest)
                    .await?;
                embeddings
                    .into_iter()
                    .next()
                    .ok_or_else(|| MomoError::Embedding("No embedding generated".to_string()))
            }
        }
    }

    pub async fn embed_passages(&self, passages: Vec<String>) -> Result<Vec<Vec<f32>>> {
        match &self.backend {
            EmbeddingBackend::Local {
                ingest_batch_size,
                ingest_batch_pause_ms,
                ..
            } => {
                if passages.is_empty() {
                    return Ok(Vec::new());
                }

                let mut all_embeddings = Vec::with_capacity(passages.len());
                for batch in passages.chunks(*ingest_batch_size) {
                    let prefixed: Vec<String> =
                        batch.iter().map(|p| format!("passage: {p}")).collect();
                    let mut embedded = self
                        .embed_with_mode(prefixed, EmbeddingMode::Ingest)
                        .await?;
                    all_embeddings.append(&mut embedded);
                    tokio::task::yield_now().await;
                    if *ingest_batch_pause_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(
                            *ingest_batch_pause_ms,
                        ))
                        .await;
                    }
                }

                Ok(all_embeddings)
            }
            #[cfg(test)]
            EmbeddingBackend::Mock => self.embed_with_mode(passages, EmbeddingMode::Ingest).await,
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl Clone for EmbeddingProvider {
    fn clone(&self) -> Self {
        match &self.backend {
            EmbeddingBackend::Local {
                query_model,
                ingest_model,
                batch_size,
                ingest_batch_size,
                ingest_batch_pause_ms,
            } => Self {
                backend: EmbeddingBackend::Local {
                    query_model: Arc::clone(query_model),
                    ingest_model: Arc::clone(ingest_model),
                    batch_size: *batch_size,
                    ingest_batch_size: *ingest_batch_size,
                    ingest_batch_pause_ms: *ingest_batch_pause_ms,
                },
                dimensions: self.dimensions,
            },
            #[cfg(test)]
            EmbeddingBackend::Mock => Self {
                backend: EmbeddingBackend::Mock,
                dimensions: self.dimensions,
            },
        }
    }
}

#[derive(Clone, Copy)]
enum EmbeddingMode {
    Query,
    Ingest,
}

fn resolve_embedding_model(model_name: &str) -> EmbeddingModel {
    match model_name {
        "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        "all-MiniLM-L6-v2" | "sentence-transformers/all-MiniLM-L6-v2" => {
            EmbeddingModel::AllMiniLML6V2
        }
        "all-MiniLM-L12-v2" | "sentence-transformers/all-MiniLM-L12-v2" => {
            EmbeddingModel::AllMiniLML12V2
        }
        "nomic-embed-text-v1" | "nomic-ai/nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
        "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
            EmbeddingModel::NomicEmbedTextV15
        }
        _ => EmbeddingModel::BGESmallENV15,
    }
}

fn build_model(embedding_model: EmbeddingModel) -> Result<TextEmbedding> {
    let mut last_error: Option<String> = None;

    for attempt in 1..=3 {
        match TextEmbedding::try_new(
            InitOptions::new(embedding_model.clone()).with_show_download_progress(true),
        ) {
            Ok(model) => return Ok(model),
            Err(err) => {
                last_error = Some(err.to_string());
                if attempt < 3 {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
    }

    Err(MomoError::Embedding(format!(
        "Failed to initialize embedding model after retries: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )))
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
fn deterministic_embedding(text: &str, dimensions: usize, mode: EmbeddingMode) -> Vec<f32> {
    let mode_seed = match mode {
        EmbeddingMode::Query => 0_u64,
        EmbeddingMode::Ingest => 1_u64,
    };

    (0..dimensions)
        .map(|index| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            index.hash(&mut hasher);
            mode_seed.hash(&mut hasher);
            let bucket = (hasher.finish() % 2001) as i32;
            (bucket as f32 / 1000.0) - 1.0
        })
        .collect()
}
