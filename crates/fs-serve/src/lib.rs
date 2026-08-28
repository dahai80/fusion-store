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

use std::collections::HashMap;
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
/// F-SEC-7：HTTP 请求体上限（16MB，覆盖 Arrow IPC 批 + 向量批量写，拒超大 payload DoS）。
/// 16MB 远超单批列式/向量写入常态；超限 axum 返 413 Payload Too Large，反序列化前拦截。
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

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

/// F-SEC-2：RBAC 角色 —— admin（全权，含 compact/管理）/ readwrite（KV+向量写，禁 compact）
/// / readonly（仅读 KNN/stats）。FS_AUTH_TOKEN 默认映射 admin（向后兼容单 token 场景）。
/// 角色经独立 env token 命名：FS_AUTH_TOKEN(admin) / FS_AUTH_TOKEN_RW / FS_AUTH_TOKEN_RO。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuthRole {
    Readonly = 0,
    Readwrite = 1,
    Admin = 2,
}

impl AuthRole {
    fn label(self) -> &'static str {
        match self {
            AuthRole::Admin => "admin",
            AuthRole::Readwrite => "readwrite",
            AuthRole::Readonly => "readonly",
        }
    }
}

/// F-SEC-2：认证配置 —— token→role 映射表。空表 = 匿名放行（仅环回部署安全）。
/// 任一 token 命中即返其角色；不命中或未带 token 在「需认证端点」拒 401。
/// F-SEC-3：token 来源经 main 从 env / FS_AUTH_TOKEN_FILE 载入，本结构不关心来源。
pub struct AuthConfig {
    /// token 字符串 → 角色。常量时间比较见 token_match。
    tokens: HashMap<String, AuthRole>,
}

impl AuthConfig {
    pub fn empty() -> Self {
        Self {
            tokens: HashMap::new(),
        }
    }

    /// 登记一个 token→role（main 启动时从 env 文件载入后调）
    pub fn add(&mut self, token: String, role: AuthRole) {
        self.tokens.insert(token, role);
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// 校验 Bearer header → 命中返角色，未配置返 None（匿名放行），配置但未命中/不符返 None + 标记需拒。
    /// 返 (matched_role, configured)：configured=false 表无 token 配置 → 匿名放行（admin 默认）。
    fn check(&self, headers: &HeaderMap) -> (Option<AuthRole>, bool) {
        if self.tokens.is_empty() {
            return (None, false);
        }
        let Some(raw) = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
        else {
            return (None, true);
        };
        let Some(tok) = raw.strip_prefix("Bearer ") else {
            return (None, true);
        };
        // 常量时间比较：逐 token 比对（token 集合小，遍历开销可忽略）。
        // 不用 HashMap::get（哈希短路泄露前缀），改为遍历 constant_eq。
        for (stored, role) in &self.tokens {
            if constant_eq(tok.as_bytes(), stored.as_bytes()) {
                return (Some(*role), true);
            }
        }
        (None, true)
    }
}

/// 常量时间字节比较（防时序侧信道泄露 token 前缀）
fn constant_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// 共享状态 —— 持 Engine + 写队列深度 + compact 标志 + 写队列 sender + 队列上限
/// + KNN 信号量（R4）+ auth（R10/F-SEC-2 RBAC）+ bind 是否环回（F-OPS-8 /stats 守门）
pub struct AppState {
    pub engine: Arc<Engine>,
    pub queue_depth: Arc<std::sync::atomic::AtomicUsize>,
    pub compact_in_progress: Arc<std::sync::atomic::AtomicBool>,
    pub write_tx: mpsc::Sender<WriteReq>,
    pub queue_cap: usize,
    /// R4：KNN 并发限流信号量
    pub knn_sem: Arc<tokio::sync::Semaphore>,
    /// R10/F-SEC-2：token→role 认证配置（空 = 匿名放行，仅环回安全）
    pub auth: Arc<AuthConfig>,
    /// F-OPS-8：bind 是否环回。true 时 /stats 匿名放行；false 时 /stats 需 ≥readonly 认证。
    pub bind_is_loopback: bool,
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
/// R10：管理/写端点（compact/kv/vector/columnar）挂 auth；只读监控（health/metrics/knn）放行。
/// F-OPS-8：/stats 仅非环回时挂 auth（stats_endpoint 内自判 bind_is_loopback）。
/// F-SEC-7：全局 DefaultBodyLimit 16MB，拒超大 payload 在反序列化前 DoS。
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
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

// R10/F-SEC-2：角色门禁校验。返授权角色（匿名放行时为 Admin，因仅环回无 token 场景安全）。
// configured=true（配了 token）但未命中/角色不足 → 401/403。
fn require_role(
    state: &AppState,
    headers: &HeaderMap,
    min: AuthRole,
) -> Result<AuthRole, AppError> {
    let (role, configured) = state.auth.check(headers);
    if !configured {
        // 未配 token = 匿名放行（main 已对非环回+无 token 强警告）。匿名视作 Admin。
        return Ok(AuthRole::Admin);
    }
    match role {
        Some(r) if r >= min => Ok(r),
        Some(r) => Err(AppError(
            StatusCode::FORBIDDEN,
            format!(
                "role {} insufficient, require >= {}",
                r.label(),
                min.label()
            ),
        )),
        None => Err(AppError(
            StatusCode::UNAUTHORIZED,
            "missing or invalid Authorization Bearer token".to_string(),
        )),
    }
}

/// F-SEC-2：compact 审计日志 —— 记谁（授权角色）触发了 compact + 起止 + 结果。
/// token 本身不入日志（敏感），仅记角色 + 时间戳 + 段计数。
fn audit_compact(role: AuthRole, live_vectors: usize, reclaimable: usize, elapsed_us: u64) {
    tracing::info!(
        actor_role = role.label(),
        live_vectors,
        reclaimable_segs = reclaimable,
        elapsed_us,
        "compact triggered (audit F-SEC-2)"
    );
}

/// F-SEC-2/F-SEC-3：从 env / 文件载入三角色 token 构 AuthConfig（main + fs-cli 共用）。
/// env 取明文，file 取首行（去空白，避免 ps/proc 明文）。两源都缺的角色不登记。文件读失败 = Err。
pub fn build_auth_from_env() -> Result<AuthConfig, BuildAuthError> {
    let mut auth = AuthConfig::empty();
    load_role_token(
        &mut auth,
        AuthRole::Admin,
        "FS_AUTH_TOKEN",
        "FS_AUTH_TOKEN_FILE",
    )?;
    load_role_token(
        &mut auth,
        AuthRole::Readwrite,
        "FS_AUTH_TOKEN_RW",
        "FS_AUTH_TOKEN_RW_FILE",
    )?;
    load_role_token(
        &mut auth,
        AuthRole::Readonly,
        "FS_AUTH_TOKEN_RO",
        "FS_AUTH_TOKEN_RO_FILE",
    )?;
    Ok(auth)
}

/// build_auth_from_env 错误（文件读失败或空文件）
#[derive(Debug)]
pub struct BuildAuthError(String);

impl std::fmt::Display for BuildAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "build auth: {}", self.0)
    }
}

impl std::error::Error for BuildAuthError {}

fn load_role_token(
    auth: &mut AuthConfig,
    role: AuthRole,
    env_key: &str,
    file_key: &str,
) -> Result<(), BuildAuthError> {
    if let Ok(tok) = std::env::var(env_key) {
        if !tok.is_empty() {
            auth.add(tok, role);
            tracing::info!(role = ?role, src = "env", "auth token registered");
            return Ok(());
        }
    }
    if let Ok(path) = std::env::var(file_key) {
        if !path.is_empty() {
            let raw = std::fs::read_to_string(&path)
                .map_err(|e| BuildAuthError(format!("read {} ({}): {}", file_key, path, e)))?;
            let tok = raw.trim().to_string();
            if tok.is_empty() {
                return Err(BuildAuthError(format!(
                    "{} points to empty token file: {}",
                    file_key, path
                )));
            }
            auth.add(tok, role);
            tracing::info!(role = ?role, src = "file", "auth token registered");
        }
    }
    Ok(())
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

async fn stats(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<StatsResp>, AppError> {
    // F-OPS-8：非环回绑定时 /stats 暴露 namespace 规模（向量数/磁盘占用），需 ≥readonly 认证。
    // 环回部署维持匿名放行（本地调试便利）。
    if !s.bind_is_loopback {
        require_role(&s, &headers, AuthRole::Readonly)?;
    }
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
    // R10/F-SEC-2：管理端点强制 admin 角色（无 token 拒 401，防远程触发 compact DoS）
    let role = require_role(&s, &headers, AuthRole::Admin)?;
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
            // F-SEC-2：审计日志（角色+结果+计数，不入 token 明文）
            audit_compact(role, r.live_vectors, r.reclaimable_segs.len(), us);
            Ok(Json(CompactResp {
                live_vectors: r.live_vectors,
                reclaimable_segs: r.reclaimable_segs.len(),
            }))
        }
        Err(e) => {
            metrics::observe("compact", "error", us);
            tracing::warn!(actor_role = role.label(), error = %e, "compact failed (audit F-SEC-2)");
            Err(AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn put_kv_endpoint(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<PutKvReq>,
) -> Result<StatusCode, AppError> {
    // R10/F-SEC-2：写端点需 ≥readwrite 角色
    require_role(&s, &headers, AuthRole::Readwrite)?;
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
    require_role(&s, &headers, AuthRole::Readwrite)?;
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
    require_role(&s, &headers, AuthRole::Readwrite)?;
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
                    Err(e) => {
                        // F-ERR-1：毒锁恢复 —— worker 持锁时 panic 致 Mutex 中毒。
                        // into_inner 取出底层 Receiver 强制续命（避免写队列全停），但状态可能不一致，
                        // 显式告警 + 计 metric 供运维感知线程 panic 事件。
                        tracing::error!(
                            wid,
                            "write worker rx lock poisoned — recovered via into_inner (a worker panicked)"
                        );
                        metrics::inc("poison_lock_recover", "rx");
                        e.into_inner()
                    }
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
