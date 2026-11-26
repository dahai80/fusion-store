//! fs-serve library — HTTP daemon logic (shared by main and integration tests) [v2 R3/A2/E5/R4/R10/E10/E11]
//!
//! A2 fix: all entry points hold Engine (aggregating KV+Vector+Columnar+WAL) instead of holding submodules directly.
//! Write path goes through Engine → WAL (F2 crash safety); read path goes through Engine delegating to submodules zero-copy.
//! R3: a pool of write workers consumes the queue (not single-threaded serial).
//! R4: KNN endpoint caps concurrent blocking threads via a Semaphore to prevent burst OOM.
//! R10: admin/write endpoints enforce Bearer Token auth (FS_AUTH_TOKEN); missing token → 401.
//! E10: HTTP API completed (put_kv + insert_vector + put_columnar + knn).
//! E11: during compact, /health reports unavailable + HTTP 503, tripping upstream circuit breakers.

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

/// Default write queue cap (10K pending writes, PRD §3.3 A2)
pub const DEFAULT_QUEUE_CAP: usize = 10_000;
/// Backpressure alert threshold (queue 80% full → /health backpressure=true)
pub const BACKPRESSURE_THRESHOLD: usize = 8_000;
/// Default port (11463, R3)
pub const DEFAULT_PORT: u16 = 11463;
/// R3: write worker pool size (concurrent queue consumers, removing the single-threaded serial bottleneck)
pub const WRITE_WORKERS: usize = 4;
/// R4: KNN concurrency cap (hard ceiling on blocking threads, prevents OOM from a burst of 1000 concurrent requests)
pub const KNN_CONCURRENCY: usize = 64;
/// F-SEC-7: HTTP request body limit (16MB, covers Arrow IPC batches + vector batch writes, rejects oversized payload DoS).
/// 16MB is far above normal single-batch columnar/vector writes; over-limit requests get 413 Payload Too Large from axum, intercepted before deserialization.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Write request (enqueued; the background worker pool consumes it and writes WAL + segments via Engine)
/// E10: enum completed with InsertVector + PutColumnar, aligning the HTTP API with the core layer
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

/// F-SEC-2: RBAC roles — admin (full access, incl. compact/admin) / readwrite (KV+vector writes, no compact)
/// / readonly (read-only KNN/stats). FS_AUTH_TOKEN maps to admin by default (back-compat for single-token setups).
/// Roles are provisioned via separate env tokens: FS_AUTH_TOKEN(admin) / FS_AUTH_TOKEN_RW / FS_AUTH_TOKEN_RO.
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

/// F-SEC-2: auth config — token→role mapping. Empty table = anonymous access (safe for loopback-only deploys).
/// Any token hit returns its role; a miss or missing token → 401 on auth-required endpoints.
/// F-SEC-3: tokens are loaded by main from env / FS_AUTH_TOKEN_FILE; this struct is agnostic to the source.
pub struct AuthConfig {
    /// token string → role. See token_match for the constant-time comparison.
    tokens: HashMap<String, AuthRole>,
}

impl AuthConfig {
    pub fn empty() -> Self {
        Self {
            tokens: HashMap::new(),
        }
    }

    /// Register a token→role pair (called by main after loading from env/file at startup)
    pub fn add(&mut self, token: String, role: AuthRole) {
        self.tokens.insert(token, role);
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Validate the Bearer header → returns the role on hit; None when unconfigured (anonymous access);
    /// configured but no match/mismatch → None + flagged for rejection.
    /// Returns (matched_role, configured): configured=false means no tokens are configured → anonymous access (admin default).
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
        // Constant-time comparison: check each token in turn (small token set, loop cost is negligible).
        // Not HashMap::get (hash short-circuiting leaks prefixes); loop over constant_eq instead.
        for (stored, role) in &self.tokens {
            if constant_eq(tok.as_bytes(), stored.as_bytes()) {
                return (Some(*role), true);
            }
        }
        (None, true)
    }
}

/// Constant-time byte comparison (prevents timing side-channel from leaking token prefixes)
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

/// Shared state — holds Engine + write queue depth + compact flag + write queue sender + queue cap
/// + KNN semaphore (R4) + auth (R10/F-SEC-2 RBAC) + whether the bind is loopback (F-OPS-8 /stats gating)
pub struct AppState {
    pub engine: Arc<Engine>,
    pub queue_depth: Arc<std::sync::atomic::AtomicUsize>,
    pub compact_in_progress: Arc<std::sync::atomic::AtomicBool>,
    pub write_tx: mpsc::Sender<WriteReq>,
    pub queue_cap: usize,
    /// R4: KNN concurrency-limiting semaphore
    pub knn_sem: Arc<tokio::sync::Semaphore>,
    /// R10/F-SEC-2: token→role auth config (empty = anonymous access, loopback-only safe)
    pub auth: Arc<AuthConfig>,
    /// F-OPS-8: whether the bind is loopback. true → /stats allows anonymous; false → /stats requires ≥readonly auth.
    pub bind_is_loopback: bool,
}

#[derive(Serialize)]
pub struct HealthResp {
    pub status: &'static str,
    pub store_open: bool,
    pub compact_in_progress: bool,
    pub backpressure: bool,
}

// E11: status is a three-state string (ok/degraded/unavailable); HealthResp.status uses &'static str
// via const matching, avoiding String serialization overhead.

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

// E10: vector write request
#[derive(Deserialize)]
pub struct InsertVectorReq {
    pub id: u64,
    pub vector: Vec<f32>,
}

// E10: columnar write request — IPC bytes carried as base64 text in JSON (HTTP-friendly; clients encode arrow IPC then b64)
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

/// Build the axum Router (shared by main and tests)
/// R10: admin/write endpoints (compact/kv/vector/columnar) mount auth; read-only monitoring (health/metrics/knn) stays open.
/// F-OPS-8: /stats mounts auth only when not loopback-bound (stats_endpoint checks bind_is_loopback itself).
/// F-SEC-7: global DefaultBodyLimit 16MB, rejecting oversized payload DoS before deserialization.
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

// R10/F-SEC-2: role gate check. Returns the authorized role (Admin when anonymous, safe since loopback-only no-token).
// configured=true (tokens set) but no match/insufficient role → 401/403.
fn require_role(
    state: &AppState,
    headers: &HeaderMap,
    min: AuthRole,
) -> Result<AuthRole, AppError> {
    let (role, configured) = state.auth.check(headers);
    if !configured {
        // No tokens configured = anonymous access (main already hard-warns on non-loopback + no token). Anonymous counts as Admin.
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

/// F-SEC-2: compact audit log — records who (authorized role) triggered compact + timing + result.
/// The token itself is never logged (sensitive); only role + timestamp + segment counts.
fn audit_compact(role: AuthRole, live_vectors: usize, reclaimable: usize, elapsed_us: u64) {
    tracing::info!(
        actor_role = role.label(),
        live_vectors,
        reclaimable_segs = reclaimable,
        elapsed_us,
        "compact triggered (audit F-SEC-2)"
    );
}

/// F-SEC-2/F-SEC-3: load three-role tokens from env / file into an AuthConfig (shared by main and fs-cli).
/// env takes plaintext, file takes the first line (trimmed, avoiding ps/proc plaintext leak). Roles missing from both sources are not registered. File read failure = Err.
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

/// F-SEC-1 / Rule 12 fail visibly: bind + auth policy check (shared by main and fs-cli serve, testable).
/// Ok(()) means pass; Err means refuse to start.
/// - loopback + no token → Ok (anonymous access, safe for local debugging) + caller logs the warn
/// - non-loopback + no token → Err (hard fail, admin/write endpoints never run unprotected)
/// - either + token → Ok
pub fn enforce_bind_auth_policy(
    bind: &str,
    bind_is_loopback: bool,
    auth_enabled: bool,
) -> Result<(), BindAuthPolicyError> {
    if !auth_enabled && !bind_is_loopback {
        return Err(BindAuthPolicyError(bind.to_string()));
    }
    Ok(())
}

/// enforce_bind_auth_policy error: non-loopback bind + no token refuses startup
#[derive(Debug)]
pub struct BindAuthPolicyError(pub String);

impl std::fmt::Display for BindAuthPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NON-LOOPBACK bind ({}) WITHOUT FS_AUTH_TOKEN — refusing to start. \
             Set FS_AUTH_TOKEN(_FILE) or bind 127.0.0.1.",
            self.0
        )
    }
}

impl std::error::Error for BindAuthPolicyError {}

/// build_auth_from_env error (file read failure or empty file)
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
    // E11: compact holds the write lock blocking all reads → unavailable + 503, tripping upstream circuit breakers.
    // The old impl reported degraded, misleading upstream into "slow but usable" while reads all stuck on the lock. Backpressure still reports degraded (reduced but usable).
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
    // F-OPS-8: when not loopback-bound, /stats exposes namespace scale (vector count/disk usage), requires ≥readonly auth.
    // Loopback deployments keep anonymous access (local debugging convenience).
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
    // R10/F-SEC-2: admin endpoint requires the admin role (no token → 401, prevents remote compact DoS)
    let role = require_role(&s, &headers, AuthRole::Admin)?;
    // During compact, set compact_in_progress so /health reports unavailable (E11)
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
            // F-SEC-2: audit log (role+result+counts, never the token plaintext)
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
    // R10/F-SEC-2: write endpoints require ≥readwrite role
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

// E10: vector write endpoint — through the write queue backpressure (same as KV); the worker pool writes WAL+links via Engine.insert_vector
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

// E10: columnar write endpoint — the client sends Arrow IPC (base64); the worker decodes and writes via Engine.put_columnar
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

// Backpressure enqueue shared logic (used by put_kv/insert_vector/put_columnar):
// self-checked queue depth + try_send; full → 503 (A2, no unbounded pile-up causing upstream OOM).
// The caller creates the oneshot (done_tx,done_rx) first; done_tx goes into the wreq, done_rx is kept for awaiting.
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
    // R4: the Semaphore caps concurrent blocking threads, preventing OOM from a burst of 1000 concurrent requests (2MB stack each).
    // If no permit is available, return 503 backpressure immediately (no unbounded spawn_blocking pile-up).
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

/// axum error type — carries HTTP status + message
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

/// Refresh segment usage metrics (KV used_bytes / quota_limit)
pub fn refresh_seg_metrics(engine: &Engine) {
    let kv = engine.kv();
    if let Ok(used) = kv.used_bytes() {
        metrics::set_seg_util(used, kv.quota_limit());
    }
}

/// Start the write worker pool (R3: N concurrent consumers, removing the single-threaded serial bottleneck; shared by main and tests)
///
/// Returns join handles (representing the worker pool; any worker exiting does not affect the rest, and the handles complete when all workers finish).
/// mpsc Receiver is single-consumer; multiple workers share it via Arc<Mutex<Receiver>> + try_recv + 1ms backoff.
/// The lock is held only while grabbing a task (μs); actual Engine writes (mmap/heed, μs-scale) run concurrently, so lock contention is not the bottleneck.
/// E10: workers dispatch put_kv / insert_vector / put_columnar by WriteReq type.
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
                        // F-ERR-1: poisoned lock recovery — a worker panicking while holding the lock poisons the Mutex.
                        // into_inner pulls out the underlying Receiver to keep it alive (avoiding a full write-queue stall), but state may be inconsistent,
                        // so warn explicitly + count a metric so ops can see the thread panic event.
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

/// Handle a single write request (shared by the worker pool) — dispatch by WriteReq type, write WAL + segments via Engine (F2 crash safety)
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
            // E10: decode IPC → RecordBatch → put_columnar
            let res = (|| -> fs_core::Result<()> {
                let batch = fs_core::engine_impl::decode_ipc(&ipc)?;
                engine.put_columnar(&table_id, &batch, None)
            })();
            ("put_columnar", start, res, done)
        }
    };
    let us = start.elapsed().as_micros() as u64;
    // Queue depth decrement (task dequeued and executing)
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

/// base64 decode (used by the E10 /columnar endpoint)
fn base64_decode(s: &str) -> Result<Vec<u8>, AppError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| AppError(StatusCode::BAD_REQUEST, format!("base64 decode: {}", e)))
}
