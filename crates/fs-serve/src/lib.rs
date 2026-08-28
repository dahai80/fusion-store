//! fs-serve 库 —— HTTP daemon 逻辑（供 main + 集成测试共用）[v2 R3/A2/E5/R4/R10/E10/E11]
//!
//! A2 修复：所有入口持 Engine（聚合 KV+Vector+Columnar+WAL），不再裸持子模块。
//! 写路径经 Engine → WAL（F2 崩溃安全），读路径经 Engine 委托子模块零拷贝。
//! R3：多写 worker 池消费队列（非单线程串行）。
//! R4：KNN 端点经 Semaphore 限并发 blocking 线程，防突发 OOM。
//! R10：管理/写端点强制 Bearer Token 认证（FS_AUTH_TOKEN），无 token 拒 401。
//! E10：HTTP API 完整化（put_kv + insert_vector + put_columnar + knn）。
//! E11：compact 期间 /health 报 unavailable + HTTP 503，触发上游熔断切流。

pub mod metrics;

use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use fs_core::compact::run_compact;
use fs_core::{Engine, FusionStoreEngine};

/// 写队列默认上限（10K 待写，PRD §3.3 A2）
pub const DEFAULT_QUEUE_CAP: usize = 10_000;
/// 背压告警阈值（队列 80% 满 → /health backpressure=true）
pub const BACKPRESSURE_THRESHOLD: usize = 8_000;
/// 默认端口（11463，R3）
pub const DEFAULT_PORT: u16 = 11463;
/// R3：写 worker 池大小（并发消费写队列，破单线程串行瓶颈）
pub const WRITE_WORKERS: usize = 4;
/// R4：KNN 并发上限（blocking 线程数硬顶，防突发 1000 并发 OOM）
pub const KNN_CONCURRENCY: usize = 64;

/// 写请求（入队，后台 worker 池消费经 Engine 写 WAL + 落段）
/// E10：枚举补全 InsertVector + PutColumnar，HTTP API 与底座对齐
pub enum WriteReq {
    PutKv {
        key: Vec<u8>,
        value: Vec<u8>,
        done: tokio::sync::oneshot::Sender<bool>,
    },
    InsertVector {
        id: u64,
        vector: Vec<f32>,
        done: tokio::sync::oneshot::Sender<bool>,
    },
    PutColumnar {
        table_id: String,
        ipc: Vec<u8>,
        done: tokio::sync::oneshot::Sender<bool>,
    },
}

/// 共享状态 —— 持 Engine + 写队列深度 + compact 标志 + 写队列 sender + 队列上限
/// + KNN 信号量（R4）+ auth token（R10）
pub struct AppState {
    pub engine: Arc<Engine>,
    pub queue_depth: Arc<std::sync::atomic::AtomicUsize>,
    pub compact_in_progress: Arc<std::sync::atomic::AtomicBool>,
    pub write_tx: mpsc::Sender<WriteReq>,
    pub queue_cap: usize,
    /// R4：KNN 并发限流信号量
    pub knn_sem: Arc<tokio::sync::Semaphore>,
    /// R10：管理/写端点 Bearer Token（None=匿名放行，仅 127.0.0.1 + 强警告）
    pub auth_token: Option<String>,
}

#[derive(Serialize)]
pub struct HealthResp {
    pub status: &'static str,
    pub store_open: bool,
    pub compact_in_progress: bool,
    pub backpressure: bool,
}

// E11：status 三态字符串（ok/degraded/unavailable），HealthResp.status 用 &'static str
// 经 const 匹配，避免 String 序列化开销。

#[derive(Serialize)]
pub struct StatsResp {
    pub namespace: String,
    pub kv_used_bytes: u64,
    pub kv_quota: u64,
    pub kv_disk_bytes: u64,
    pub vec_disk_bytes: u64,
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

// E10：向量写入请求
#[derive(Deserialize)]
pub struct InsertVectorReq {
    pub id: u64,
    pub vector: Vec<f32>,
}

// E10：列式写入请求 —— IPC 字节以 base64 文本承 JSON（HTTP 友好，客户端用 arrow IPC 编码后 b64）
#[derive(Deserialize)]
pub struct PutColumnarReq {
    pub table_id: String,
    pub ipc_base64: String,
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
/// R10：管理/写端点（compact/kv/vector/columnar）挂 auth middleware；只读监控（health/stats/metrics/knn）放行。
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/metrics", get(metrics_endpoint))
        .route("/knn", post(knn_endpoint))
        .route("/admin/compact", post(compact_endpoint))
        .route("/kv", post(put_kv_endpoint))
        .route("/vector", post(insert_vector_endpoint))
        .route("/columnar", post(put_columnar_endpoint))
        .with_state(state)
}

// R10：Bearer Token 认证校验。auth_token=Some 时管理/写端点必须带 `Authorization: Bearer <token>`。
// None=匿名放行（仅 127.0.0.1 部署可接受，main 启动时强警告）。
fn require_auth(state: &AppState, headers: &HeaderMap) -> Result<(), AppError> {
    let Some(token) = &state.auth_token else {
        return Ok(());
    };
    let expected = format!("Bearer {}", token);
    match headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        Some(v) if v == expected => Ok(()),
        _ => Err(AppError(
            StatusCode::UNAUTHORIZED,
            "missing or invalid Authorization Bearer token".to_string(),
        )),
    }
}

async fn health(State(s): State<Arc<AppState>>) -> Response {
    let qd = s.queue_depth.load(std::sync::atomic::Ordering::Relaxed);
    let compact = s
        .compact_in_progress
        .load(std::sync::atomic::Ordering::Relaxed);
    let backpressure = qd >= BACKPRESSURE_THRESHOLD;
    // E11：compact 持写锁阻塞所有读 → unavailable + 503，上游熔断切流。
    // 旧实现报 degraded 误导上游"慢但能用"，持续发读请求全卡锁。背压仍报 degraded（降级可用）。
    let (status, code) = if compact {
        ("unavailable", StatusCode::SERVICE_UNAVAILABLE)
    } else if backpressure {
        ("degraded", StatusCode::OK)
    } else {
        ("ok", StatusCode::OK)
    };
    metrics::inc("health", status);
    (
        code,
        Json(HealthResp {
            status,
            store_open: true,
            compact_in_progress: compact,
            backpressure,
        }),
    )
        .into_response()
}

async fn stats(State(s): State<Arc<AppState>>) -> Result<Json<StatsResp>, AppError> {
    let kv_used = s.engine.kv().used_bytes()?;
    let kv_quota = s.engine.kv().quota_limit();
    let kv_disk = s.engine.kv().disk_bytes()?;
    let (vector_count, vector_dim, graph_mem, vec_disk) = {
        let g = s.engine.vec_index()?;
        let idx = g.as_ref().unwrap();
        (
            idx.len(),
            idx.schema().dim,
            idx.graph_memory_usage(),
            idx.disk_bytes()?,
        )
    };
    let qd = s.queue_depth.load(std::sync::atomic::Ordering::Relaxed);
    metrics::inc("stats", "ok");
    Ok(Json(StatsResp {
        namespace: "default".into(),
        kv_used_bytes: kv_used,
        kv_quota,
        kv_disk_bytes: kv_disk,
        vec_disk_bytes: vec_disk,
        vector_count,
        vector_dim,
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

async fn compact_endpoint(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<CompactResp>, AppError> {
    // R10：管理端点强制认证（无 token 拒 401，防远程触发 compact DoS）
    require_auth(&s, &headers)?;
    // compact 期间标记 compact_in_progress 供 /health 报 unavailable（E11）
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
    let engine = s.engine.clone();
    let res = tokio::task::spawn_blocking(move || {
        let g = engine.vec_index()?;
        let idx = g.as_ref().unwrap();
        run_compact(idx)
    })
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
    headers: HeaderMap,
    Json(req): Json<PutKvReq>,
) -> Result<StatusCode, AppError> {
    // R10：写端点强制认证
    require_auth(&s, &headers)?;
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let wreq = WriteReq::PutKv {
        key: req.key.into_bytes(),
        value: req.value.into_bytes(),
        done: done_tx,
    };
    try_enqueue(&s, wreq, "put_kv")?;
    await_write_done(done_rx, "put_kv").await
}

// E10：向量写入端点 —— 经写队列背压（同 KV），worker 池经 Engine.insert_vector 写 WAL+连边
async fn insert_vector_endpoint(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InsertVectorReq>,
) -> Result<StatusCode, AppError> {
    require_auth(&s, &headers)?;
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let wreq = WriteReq::InsertVector {
        id: req.id,
        vector: req.vector,
        done: done_tx,
    };
    try_enqueue(&s, wreq, "insert_vector")?;
    await_write_done(done_rx, "insert_vector").await
}

// E10：列式写入端点 —— 客户端发 Arrow IPC(base64)，worker 解码后经 Engine.put_columnar 写
async fn put_columnar_endpoint(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<PutColumnarReq>,
) -> Result<StatusCode, AppError> {
    require_auth(&s, &headers)?;
    let ipc = base64_decode(&req.ipc_base64)?;
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let wreq = WriteReq::PutColumnar {
        table_id: req.table_id,
        ipc,
        done: done_tx,
    };
    try_enqueue(&s, wreq, "put_columnar")?;
    await_write_done(done_rx, "put_columnar").await
}

// 背压入队公共逻辑（put_kv/insert_vector/put_columnar 共用）：
// 队列深度自检 + try_send 满即 503（A2，不无界堆积致上游 OOM）。
// 调用方先建 oneshot (done_tx,done_rx)，done_tx 入 wreq，done_rx 自留 await。
fn try_enqueue(s: &AppState, wreq: WriteReq, label: &str) -> Result<(), AppError> {
    let qd = s
        .queue_depth
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if qd >= s.queue_cap {
        s.queue_depth
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        metrics::inc(label, "backpressure_503");
        return Err(AppError(
            StatusCode::SERVICE_UNAVAILABLE,
            "backpressure: queue full".to_string(),
        ));
    }
    if let Err(mpsc::error::TrySendError::Full(_)) = s.write_tx.try_send(wreq) {
        s.queue_depth
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        metrics::inc(label, "backpressure_503");
        return Err(AppError(
            StatusCode::SERVICE_UNAVAILABLE,
            "backpressure: queue full".to_string(),
        ));
    }
    metrics::set_queue_depth(qd + 1);
    Ok(())
}

async fn await_write_done(
    done_rx: tokio::sync::oneshot::Receiver<bool>,
    label: &str,
) -> Result<StatusCode, AppError> {
    match done_rx.await {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        Ok(false) => Err(AppError(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{} failed", label),
        )),
        Err(_) => Err(AppError(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{} worker dropped", label),
        )),
    }
}

async fn knn_endpoint(
    State(s): State<Arc<AppState>>,
    Json(req): Json<KnnReq>,
) -> Result<Json<KnnResp>, AppError> {
    // R4：Semaphore 限并发 blocking 线程，防突发 1000 并发 OOM（每线程栈 2MB）。
    // 拿不到许可立即 503 背压（不无限堆积 spawn_blocking）。
    let _permit = match s.knn_sem.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            metrics::inc("knn", "concurrency_503");
            return Err(AppError(
                StatusCode::SERVICE_UNAVAILABLE,
                "knn concurrency limit reached".to_string(),
            ));
        }
    };
    let start = std::time::Instant::now();
    let engine = s.engine.clone();
    let q = req.query.clone();
    let top_k = req.top_k;
    let res = tokio::task::spawn_blocking(move || {
        let g = engine.vec_index()?;
        let idx = g.as_ref().unwrap();
        idx.search_knn(&q, top_k, None)
    })
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
pub fn refresh_seg_metrics(engine: &Engine) {
    let kv = engine.kv();
    if let Ok(used) = kv.used_bytes() {
        metrics::set_seg_util(used, kv.quota_limit());
    }
}

/// 启动写 worker 池（R3：N 个并发消费，破单线程串行瓶颈；main + 测试共用）
///
/// 返回 join handle（代表 worker 池；任一 worker 退出不影响其余，handle 在所有 worker 结束时完成）。
/// mpsc Receiver 单消费者，多 worker 需共享：用 Arc<Mutex<Receiver>> + try_recv + 1ms 退避。
/// 取锁仅在取任务瞬间（μs），实际 Engine 写（mmap/heed，μs 级）并发执行，故锁竞争非瓶颈。
/// E10：worker 按 WriteReq 类型分发 put_kv / insert_vector / put_columnar。
pub fn spawn_write_worker_pool(
    engine: Arc<Engine>,
    queue_depth: Arc<std::sync::atomic::AtomicUsize>,
    rx: mpsc::Receiver<WriteReq>,
    workers: usize,
) -> Vec<tokio::task::JoinHandle<()>> {
    let rx = Arc::new(std::sync::Mutex::new(rx));
    let mut handles = Vec::with_capacity(workers);
    for wid in 0..workers.max(1) {
        let engine = engine.clone();
        let queue_depth = queue_depth.clone();
        let rx = rx.clone();
        handles.push(tokio::task::spawn_blocking(move || loop {
            let req = {
                let mut guard = match rx.lock() {
                    Ok(g) => g,
                    Err(e) => e.into_inner(),
                };
                match guard.try_recv() {
                    Ok(r) => r,
                    Err(mpsc::error::TryRecvError::Empty) => {
                        drop(guard);
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            };
            handle_write_req(&engine, &queue_depth, req, wid);
        }));
    }
    handles
}

/// 处理单个写请求（worker 池共用）—— 按 WriteReq 类型分发，经 Engine 写 WAL + 落段（F2 崩溃安全）
fn handle_write_req(
    engine: &Engine,
    queue_depth: &std::sync::atomic::AtomicUsize,
    req: WriteReq,
    wid: usize,
) {
    let (label, start, res, done) = match req {
        WriteReq::PutKv { key, value, done } => {
            let start = std::time::Instant::now();
            let res = engine.put_kv(&key, &value, None);
            ("put_kv", start, res.map(|_| ()), done)
        }
        WriteReq::InsertVector { id, vector, done } => {
            let start = std::time::Instant::now();
            let res = engine.insert_vector(id, &vector, None);
            ("insert_vector", start, res.map(|_| ()), done)
        }
        WriteReq::PutColumnar {
            table_id,
            ipc,
            done,
        } => {
            let start = std::time::Instant::now();
            // E10：解码 IPC → RecordBatch → put_columnar
            let res = (|| -> fs_core::Result<()> {
                let batch = fs_core::engine_impl::decode_ipc(&ipc)?;
                engine.put_columnar(&table_id, &batch, None)
            })();
            ("put_columnar", start, res, done)
        }
    };
    let us = start.elapsed().as_micros() as u64;
    // 队列深度回退（任务已出队执行）
    let qd = queue_depth.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    metrics::set_queue_depth(qd.saturating_sub(1));
    match res {
        Ok(()) => {
            metrics::observe(label, "ok", us);
            if label == "put_kv" {
                refresh_seg_metrics(engine);
            }
            let _ = done.send(true);
        }
        Err(e) => {
            tracing::warn!(wid, label, error = %e, "write worker op failed");
            metrics::observe(label, "error", us);
            let _ = done.send(false);
        }
    }
}

/// base64 解码（E10 /columnar 端点用）
fn base64_decode(s: &str) -> Result<Vec<u8>, AppError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| AppError(StatusCode::BAD_REQUEST, format!("base64 decode: {}", e)))
}
