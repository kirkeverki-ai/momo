use axum::extract::State;
use serde::Serialize;

use crate::api::state::AppState;
use crate::api::v1::response::ApiResponse;

/// Health data returned inside the v1 envelope.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct HealthData {
    pub ok: bool,
    pub version: String,
    pub auth_enabled: bool,
    pub uploads_enabled: bool,
    pub inference_enabled: bool,
}

/// `GET /api/v1/health`
#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "health",
    operation_id = "health.get",
    responses(
        (status = 200, description = "Service health status", body = HealthData),
        (status = 401, description = "Unauthorized", body = crate::api::v1::response::ApiError),
    )
)]
pub async fn health_check(State(state): State<AppState>) -> ApiResponse<HealthData> {
    ApiResponse::success(HealthData {
        ok: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
        auth_enabled: !state.config.server.api_keys.is_empty(),
        uploads_enabled: state.config.server.enable_uploads,
        inference_enabled: state.config.memory.inference.enabled,
    })
}
