use axum::http::{
    header::{HeaderName, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    HeaderValue, Method, Request, StatusCode,
};
use axum::middleware::Next;
use axum::response::Response;
use axum::routing::{any, get};
use axum::Router;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::mcp;

use super::frontend;
use super::v1;
use super::AppState;

#[derive(Debug, Clone)]
struct RequestTraceId(String);

async fn api_not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

fn build_cors(origins: &[String], project_header: &str, allow_wide_cors: bool) -> CorsLayer {
    let mut allowed_headers = vec![AUTHORIZATION, CONTENT_TYPE, ACCEPT];
    match HeaderName::from_bytes(project_header.as_bytes()) {
        Ok(header_name) => allowed_headers.push(header_name),
        Err(error) => tracing::warn!(
            header = %project_header,
            error = %error,
            "Skipping invalid MCP project header for CORS"
        ),
    }

    let mut cors = CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers(allowed_headers);

    if origins.iter().any(|origin| origin == "*") {
        if allow_wide_cors {
            tracing::warn!("Using wildcard CORS origin due to MOMO_ALLOW_WIDE_CORS=1");
            return cors.allow_origin(Any);
        }
        tracing::warn!("Ignoring wildcard CORS origin without MOMO_ALLOW_WIDE_CORS=1");
        return cors;
    }

    let mut allowed = Vec::new();
    for origin in origins {
        match origin.parse::<HeaderValue>() {
            Ok(header_value) => allowed.push(header_value),
            Err(error) => tracing::warn!(
                origin = %origin,
                error = %error,
                "Skipping invalid CORS origin from MOMO_CORS_ORIGINS"
            ),
        }
    }

    if !allowed.is_empty() {
        cors = cors.allow_origin(AllowOrigin::list(allowed));
    }

    cors
}

async fn request_trace_middleware(mut request: Request<axum::body::Body>, next: Next) -> Response {
    let trace_id = request
        .headers()
        .get("x-trace-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty() && value.len() <= 128)
        .map(str::to_string)
        .unwrap_or_else(|| nanoid::nanoid!(16));

    request
        .extensions_mut()
        .insert(RequestTraceId(trace_id.clone()));

    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&trace_id) {
        response.headers_mut().insert("x-trace-id", value);
    }
    response
}

async fn security_headers_middleware(request: Request<axum::body::Body>, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
    headers.insert(
        "content-security-policy",
        HeaderValue::from_static("default-src 'none'; frame-ancestors 'none'; base-uri 'none'"),
    );
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    response
}

pub fn create_router(state: AppState) -> Router {
    let cors = build_cors(
        &state.config.server.cors_allowed_origins,
        &state.config.mcp.project_header,
        state.config.server.allow_wide_cors,
    );
    let body_limit = state.config.server.max_request_body_bytes.max(1024);
    let trace_layer = TraceLayer::new_for_http().make_span_with(|request: &Request<_>| {
        let trace_id = request
            .extensions()
            .get::<RequestTraceId>()
            .map(|value| value.0.as_str())
            .unwrap_or("unknown");
        tracing::info_span!(
            "http_request",
            trace_id = %trace_id,
            method = %request.method(),
            path = %request.uri().path(),
        )
    });

    // legacy v3/v4/admin routers removed — only v1 remains mounted
    let v1 = v1::router::v1_router(state.clone());
    let mcp = mcp::mcp_router(state.clone());

    Router::new()
        .merge(mcp)
        .nest("/api/v1", v1)
        .route("/api", any(api_not_found))
        .route("/api/{*path}", any(api_not_found))
        .route("/", get(frontend::serve_root))
        .route("/{*path}", get(frontend::serve_path))
        .layer(RequestBodyLimitLayer::new(body_limit))
        .layer(cors)
        .layer(trace_layer)
        .layer(axum::middleware::from_fn(request_trace_middleware))
        .layer(axum::middleware::from_fn(security_headers_middleware))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::api::state::AppState;
    use crate::config::{
        Config, DatabaseConfig, EmbeddingsConfig, InferenceConfig, McpConfig, MemoryConfig,
        OcrConfig, ProcessingConfig, ServerConfig, TranscriptionConfig,
    };

    async fn test_state(cors_allowed_origins: Vec<String>) -> AppState {
        let config = Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
                api_keys: vec!["test-key".to_string()],
                allow_no_auth: false,
                allow_public_bind: false,
                allow_wide_cors: false,
                max_request_body_bytes: 5 * 1024 * 1024,
                cors_allowed_origins,
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
                batch_size: 64,
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

    #[tokio::test]
    async fn cors_allows_allowlisted_origin_only() {
        let app = create_router(test_state(vec!["http://127.0.0.1:18888".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/v1/health")
                    .header("origin", "http://127.0.0.1:18888")
                    .header("access-control-request-method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://127.0.0.1:18888")
        );
    }

    #[tokio::test]
    async fn cors_blocks_non_allowlisted_origin() {
        let app = create_router(test_state(vec!["http://127.0.0.1:18888".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/v1/health")
                    .header("origin", "http://evil.example")
                    .header("access-control-request-method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response
                .headers()
                .get("access-control-allow-origin")
                .is_none(),
            "CORS header should be omitted for non-allowlisted origins"
        );
    }

    #[tokio::test]
    async fn security_headers_are_set() {
        let app = create_router(test_state(vec!["http://127.0.0.1:18888".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .header("authorization", "Bearer test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("x-content-type-options")
                .and_then(|v| v.to_str().ok()),
            Some("nosniff")
        );
        assert_eq!(
            response
                .headers()
                .get("x-frame-options")
                .and_then(|v| v.to_str().ok()),
            Some("DENY")
        );
        assert_eq!(
            response
                .headers()
                .get("referrer-policy")
                .and_then(|v| v.to_str().ok()),
            Some("no-referrer")
        );
        assert!(
            response.headers().get("x-trace-id").is_some(),
            "x-trace-id should be present on responses"
        );
    }
}
