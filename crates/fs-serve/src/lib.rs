//! fs-serve 库 —— HTTP daemon 逻辑（供 main + 集成测试共用）[v2 R3/A2/E5]

pub mod metrics;

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use fs_core::compact::run_compact;
use fs_core::vector::store::VectorIndex;

/// 写队列默认上限（10K 待写，PRD §3.3 A2）
pub const DEFAULT_QUEUE_CAP: usize = 10_000;
/// 背压告警阈值（队列 80% 满 → /health backpressure=true）
pub const BACKPRESSURE_THRESHOLD: usize = 8_000;
/// 默认端口（11463，R3）
pub const DEFAULT_PORT: u16 = 11463;

/// 写请求（KV put 入队，后台 worker 消费执行）
pub enum WriteReq {
    PutKv {
        key: Vec<u8>,
        value: Vec<u8>,
        done: tokio::sync::oneshot::Sender<bool>,
    },
}

/// 共享状态 —— 持 KV + 向量索引 + 写队列深度 + compact 标志 + 写队列 sender + 队列上限
pub struct AppState {
    pub kv: Arc<fs_core::KvStore>,
    pub vec_index: Arc<VectorIndex>,
    pub queue_depth: Arc<std::sync::atomic::AtomicUsize>,
    pub compact_in_progress: Arc<std::sync::atomic::AtomicBool>,
    pub write_tx: mpsc::Sender<WriteReq>,
    pub queue_cap: usize,
}

#[derive(Serialize)]
pub struct HealthResp {
    pub status: &'static str,
    pub store_open: bool,
    pub compact_in_progress: bool,
    pub backpressure: bool,
}

#[derive(Serialize)]
pub struct StatsResp {
    pub namespace: String,
    pub kv_used_bytes: u64,
    pub kv_quota: u64,
    pub vector_count: usize,
    pub vector_dim: usize,
    pub graph_memory_bytes: usize,
    pub queue_depth: usize,
}

#[derive(Deserialize)]
pub struct PutKvReq {
    pub key: String,
    pub value: String,
}

#[derive(Deserialize)]
pub struct KnnReq {
    pub query: Vec<f32>,
    pub top_k: usize,
}

#[derive(Serialize)]
pub struct KnnResp {
    pub results: Vec<KnnHit>,
}

#[derive(Serialize)]
pub struct KnnHit {
    pub id: u64,
    pub distance: f32,
}

#[derive(Serialize)]
pub struct CompactResp {
    pub live_vectors: usize,
    pub reclaimable_segs: usize,
}

/// 构建 axum Router（main + 测试共用）
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/metrics", get(metrics_endpoint))
        .route("/admin/compact", post(compact_endpoint))
        .route("/kv", post(put_kv_endpoint))
        .route("/knn", post(knn_endpoint))
        .with_state(state)
}

async fn health(State(s): State<Arc<AppState>>) -> Json<HealthResp> {
    let qd = s.queue_depth.load(std::sync::atomic::Ordering::Relaxed);
    let compact = s
        .compact_in_progress
        .load(std::sync::atomic::Ordering::Relaxed);
    let backpressure = qd >= BACKPRESSURE_THRESHOLD;
    let status = if backpressure || compact {
        "degraded"
    } else {
        "ok"
    };
    metrics::inc("health", "ok");
    Json(HealthResp {
        status,
        store_open: true,
        compact_in_progress: compact,
        backpressure,
    })
}

async fn stats(State(s): State<Arc<AppState>>) -> Result<Json<StatsResp>, AppError> {
    let kv_used = s.kv.used_bytes()?;
    let kv_quota = s.kv.quota_limit();
    let vector_count = s.vec_index.len();
    let graph_mem = s.vec_index.graph_memory_usage();
    let qd = s.queue_depth.load(std::sync::atomic::Ordering::Relaxed);
    metrics::inc("stats", "ok");
    Ok(Json(StatsResp {
        namespace: "default".into(),
        kv_used_bytes: kv_used,
        kv_quota,
        vector_count,
        vector_dim: s.vec_index.schema().dim,
        graph_memory_bytes: graph_mem,
        queue_depth: qd,
    }))
}

async fn metrics_endpoint() -> Response {
    let body = metrics::gather();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
        .into_response()
}

async fn compact_endpoint(State(s): State<Arc<AppState>>) -> Result<Json<CompactResp>, AppError> {
    // compact 期间持写锁，标记 compact_in_progress 供 /health 报 degraded
    let was = s
        .compact_in_progress
        .swap(true, std::sync::atomic::Ordering::SeqCst);
    if was {
        return Err(AppError(
            StatusCode::CONFLICT,
            "compact already in progress".to_string(),
        ));
    }
    let start = std::time::Instant::now();
    let idx = s.vec_index.clone();
    let res = tokio::task::spawn_blocking(move || run_compact(&idx))
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    s.compact_in_progress
        .store(false, std::sync::atomic::Ordering::SeqCst);
    let us = start.elapsed().as_micros() as u64;
    match res {
        Ok(r) => {
            metrics::observe("compact", "ok", us);
            tracing::info!(live = r.live_vectors, "compact endpoint done");
            Ok(Json(CompactResp {
                live_vectors: r.live_vectors,
                reclaimable_segs: r.reclaimable_segs.len(),
            }))
        }
        Err(e) => {
            metrics::observe("compact", "error", us);
            Err(AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn put_kv_endpoint(
    State(s): State<Arc<AppState>>,
    Json(req): Json<PutKvReq>,
) -> Result<StatusCode, AppError> {
    // 背压：队列深度先自检，再 try_send 满即返 503（A2，不无界堆积致上游 OOM）
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let wreq = WriteReq::PutKv {
        key: req.key.into_bytes(),
        value: req.value.into_bytes(),
        done: done_tx,
    };
    let qd = s
        .queue_depth
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if qd >= s.queue_cap {
        s.queue_depth
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        metrics::inc("put_kv", "backpressure_503");
        return Err(AppError(
            StatusCode::SERVICE_UNAVAILABLE,
            "backpressure: queue full".to_string(),
        ));
    }
    // try_send：mpsc 满则 Err(TrySendError::Full)，回退队列深度 + 503
    if let Err(mpsc::error::TrySendError::Full(_)) = s.write_tx.try_send(wreq) {
        s.queue_depth
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        metrics::inc("put_kv", "backpressure_503");
        return Err(AppError(
            StatusCode::SERVICE_UNAVAILABLE,
            "backpressure: queue full".to_string(),
        ));
    }
    metrics::set_queue_depth(qd + 1);
    // 等待 worker 完成
    match done_rx.await {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        Ok(false) => Err(AppError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "put_kv failed".to_string(),
        )),
        Err(_) => Err(AppError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "worker dropped".to_string(),
        )),
    }
}

async fn knn_endpoint(
    State(s): State<Arc<AppState>>,
    Json(req): Json<KnnReq>,
) -> Result<Json<KnnResp>, AppError> {
    let start = std::time::Instant::now();
    let idx = s.vec_index.clone();
    let q = req.query.clone();
    let top_k = req.top_k;
    let res = tokio::task::spawn_blocking(move || idx.search_knn(&q, top_k, None))
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    let us = start.elapsed().as_micros() as u64;
    metrics::observe("knn", "ok", us);
    Ok(Json(KnnResp {
        results: res
            .into_iter()
            .map(|(id, distance)| KnnHit { id, distance })
            .collect(),
    }))
}

/// axum 错误类型 —— 携 HTTP status + message
pub struct AppError(pub StatusCode, pub String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

impl From<fs_core::StoreError> for AppError {
    fn from(e: fs_core::StoreError) -> Self {
        AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

/// 刷新段使用率指标（KV used_bytes / quota_limit）
pub fn refresh_seg_metrics(kv: &fs_core::KvStore) {
    if let Ok(used) = kv.used_bytes() {
        metrics::set_seg_util(used, kv.quota_limit());
    }
}

/// 启动写 worker（main + 测试共用）—— 返回 join handle
pub fn spawn_write_worker(
    kv: Arc<fs_core::KvStore>,
    queue_depth: Arc<std::sync::atomic::AtomicUsize>,
    mut rx: mpsc::Receiver<WriteReq>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        while let Some(req) = rx.blocking_recv() {
            match req {
                WriteReq::PutKv { key, value, done } => {
                    let start = std::time::Instant::now();
                    let res = kv.put_kv(&key, &value, None);
                    let us = start.elapsed().as_micros() as u64;
                    let qd = queue_depth.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    metrics::set_queue_depth(qd.saturating_sub(1));
                    match res {
                        Ok(()) => {
                            metrics::observe("put_kv", "ok", us);
                            refresh_seg_metrics(&kv);
                            let _ = done.send(true);
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "put_kv worker failed");
                            metrics::observe("put_kv", "error", us);
                            let _ = done.send(false);
                        }
                    }
                }
            }
        }
    })
}
