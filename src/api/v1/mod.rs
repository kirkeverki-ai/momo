pub mod dto;
pub mod handlers;
pub mod middleware;
pub mod openapi;
pub mod response;
pub mod router;

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::api::routes::create_router;
    use crate::api::state::AppState;
    use crate::config::{
        Config, DatabaseConfig, EmbeddingsConfig, InferenceConfig, McpConfig, MemoryConfig,
        OcrConfig, ProcessingConfig, ServerConfig, TranscriptionConfig,
    };

    async fn test_state(api_keys: Vec<String>) -> AppState {
        let config = Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
                api_keys,
                allow_no_auth: false,
                allow_public_bind: false,
                allow_wide_cors: false,
                max_request_body_bytes: 30 * 1024 * 1024,
                cors_allowed_origins: vec![],
                enable_uploads: false,
                upload_max_file_size_bytes: 5 * 1024 * 1024,
                upload_allowed_content_types: vec![
                    "text/plain".to_string(),
                    "text/markdown".to_string(),
                ],
                allowed_containers: vec![
                    "openclaw_forever".to_string(),
                    "openclaw_vault".to_string(),
                ],
                reject_secrets: false,
                documents_batch_concurrency: 4,
            },
            mcp: McpConfig::default(),
            database: DatabaseConfig {
                url: "file::memory:".to_string(),
                auth_token: None,
                local_path: None,
            },
            embeddings: EmbeddingsConfig {
                model: "BAAI/bge-small-en-v1.5".to_string(),
                dimensions: 384,
                batch_size: 256,
            },
            processing: ProcessingConfig {
                chunk_size: 512,
                chunk_overlap: 50,
                allow_remote_urls: false,
                remote_url_allowlist: vec![],
                remote_url_max_bytes: 10 * 1024 * 1024,
            },
            memory: MemoryConfig {
                episode_decay_days: 30.0,
                episode_decay_factor: 0.9,
                episode_decay_threshold: 0.3,
                episode_forget_grace_days: 7,
                forgetting_check_interval_secs: 3600,
                profile_refresh_interval_secs: 86400,
                inference: InferenceConfig {
                    enabled: false,
                    interval_secs: 86400,
                    confidence_threshold: 0.7,
                    max_per_run: 50,
                    candidate_count: 5,
                    seed_limit: 50,
                    exclude_episodes: true,
                },
            },
            ocr: OcrConfig {
                model: "local/tesseract".to_string(),
                api_key: None,
                base_url: None,
                languages: "eng".to_string(),
                timeout_secs: 60,
                max_image_dimension: 4096,
                min_image_dimension: 50,
            },
            transcription: TranscriptionConfig::default(),
            llm: None,
            reranker: None,
        };

        let raw_db = crate::db::Database::new(&config.database).await.unwrap();
        let db_backend = crate::db::LibSqlBackend::new(raw_db);
        let db: std::sync::Arc<dyn crate::db::DatabaseBackend> = std::sync::Arc::new(db_backend);

        let embeddings =
            crate::embeddings::EmbeddingProvider::new_mock(config.embeddings.dimensions);
        let ocr = crate::ocr::OcrProvider::new(&config.ocr).unwrap();
        let transcription =
            crate::transcription::TranscriptionProvider::new(&config.transcription).unwrap();
        let llm = crate::llm::LlmProvider::new(config.llm.as_ref());

        AppState::new(
            config,
            db.clone(),
            db,
            embeddings,
            None,
            ocr,
            transcription,
            llm,
        )
    }

    async fn body_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn protected_route_requires_auth() {
        let app = create_router(test_state(vec!["test-key".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let json = body_json(response).await;
        assert_eq!(json["error"]["code"], "unauthorized");
    }

    #[tokio::test]
    async fn health_requires_auth_without_key() {
        let app = create_router(test_state(vec!["secret".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn health_is_accessible_with_auth() {
        let app = create_router(test_state(vec!["secret".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["data"]["ok"], true);
        assert_eq!(json["data"]["authEnabled"], true);
        assert_eq!(json["data"]["uploadsEnabled"], false);
        assert_eq!(json["data"]["inferenceEnabled"], false);
    }

    #[tokio::test]
    async fn openapi_json_is_valid_with_auth() {
        let app = create_router(test_state(vec!["secret".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/openapi.json")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let version = json["openapi"]
            .as_str()
            .expect("openapi field should be a string");
        assert!(
            version.starts_with("3"),
            "OpenAPI version should start with 3, got: {version}"
        );
    }

    #[tokio::test]
    async fn success_envelope_has_data_no_error() {
        let app = create_router(test_state(vec!["k".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .header("authorization", "Bearer k")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert!(json.get("data").is_some(), "success should have 'data' key");
        assert!(
            json.get("error").is_none(),
            "success should NOT have 'error' key"
        );
    }

    #[tokio::test]
    async fn error_envelope_has_error_no_data() {
        let app = create_router(test_state(vec!["key".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let json = body_json(response).await;
        assert!(
            json.get("error").is_some(),
            "error response should have 'error' key"
        );
        assert!(
            json.get("data").is_none(),
            "error response should NOT have 'data' key"
        );
        assert!(
            json["error"]["code"].is_string(),
            "error.code should be a string"
        );
        assert!(
            json["error"]["message"].is_string(),
            "error.message should be a string"
        );
    }

    #[tokio::test]
    async fn upload_route_disabled_by_default() {
        let app = create_router(test_state(vec!["key".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/documents:upload")
                    .header("authorization", "Bearer key")
                    .header("content-type", "multipart/form-data; boundary=---x")
                    .body(Body::from("---x--"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
