//! fs-serve endpoint integration tests — real HTTP bind + reqwest [v2 R3/A2/E5]
//!
//! Acceptance aligned with PRD M4:
//!   - /health works (status ok / store_open / backpressure)
//!   - /stats works (KV used / vector_count / queue_depth)
//!   - /metrics works (Prometheus text format, incl. fusion_store_* metrics)
//!   - /kv write works (backpressure full → 503, A2)
//!   - /knn read works (KNN recall after inserting vectors)
//!   - /admin/compact works (COW atomic swap, A3)
//!   - backpressure: queue full → 503 + /health reports backpressure=true (degraded, for upstream circuit breaker A2)

use std::sync::Arc;

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine};
use fs_serve::{
    build_router, metrics, spawn_write_worker_pool, AppState, AuthConfig, AuthRole, WriteReq,
    DEFAULT_QUEUE_CAP, KNN_CONCURRENCY, MAX_BODY_BYTES, WRITE_WORKERS,
};
use tempfile::tempdir;
use tokio::sync::mpsc;

/// Start a daemon bound to an ephemeral port, return base URL + AppState handle (cleaned up after the test).
/// auth: empty AuthConfig = anonymous access; non-empty = RBAC role gating (R10/F-SEC-2).
/// bind_is_loopback=true (tests default to loopback); the /stats gating test sets its own.
async fn spawn_daemon(dim: usize, queue_cap: usize) -> (String, Arc<AppState>) {
    spawn_daemon_with_auth(dim, queue_cap, AuthConfig::empty(), true).await
}

async fn spawn_daemon_with_auth(
    dim: usize,
    queue_cap: usize,
    auth: AuthConfig,
    bind_is_loopback: bool,
) -> (String, Arc<AppState>) {
    // leak the tempdir into 'static: neither DirHandle nor path is reclaimed (cleaned up when the test process exits)
    let dir = Box::leak(Box::new(tempdir().unwrap()));
    let ns_dir = dir.path();
    std::fs::create_dir_all(ns_dir).unwrap();

    let schema = VectorSchema::new(dim, MetricKind::L2);
    let engine = Arc::new(Engine::open(ns_dir, Some(schema), 0).unwrap());
    let (tx, rx) = mpsc::channel::<WriteReq>(queue_cap);
    let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let compact_in_progress = Arc::new(std::sync::atomic::AtomicBool::new(false));
    metrics::init();
    fs_serve::refresh_seg_metrics(&engine);
    // R3: write worker pool
    let _workers = spawn_write_worker_pool(engine.clone(), queue_depth.clone(), rx, WRITE_WORKERS);
    let state = Arc::new(AppState {
        engine,
        queue_depth: queue_depth.clone(),
        compact_in_progress,
        write_tx: tx,
        queue_cap,
        knn_sem: Arc::new(tokio::sync::Semaphore::new(KNN_CONCURRENCY)),
        auth: Arc::new(auth),
        bind_is_loopback,
    });
    let app = build_router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{}", addr);
    (url, state)
}

#[tokio::test]
async fn health_returns_ok_when_idle() {
    let (url, _state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    let resp: serde_json::Value = reqwest::get(format!("{}/health", url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["status"], "ok");
    assert_eq!(resp["store_open"], true);
    assert_eq!(resp["backpressure"], false);
    assert_eq!(resp["compact_in_progress"], false);
}

#[tokio::test]
async fn stats_reports_kv_and_vector_state() {
    let (url, state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    // Insert one vector (via Engine writing WAL + segments, F2)
    state
        .engine
        .insert_vector(1, &[1.0, 0.0, 0.0, 0.0], None)
        .unwrap();
    let resp: serde_json::Value = reqwest::get(format!("{}/stats", url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["vector_count"], 1);
    assert_eq!(resp["vector_dim"], 4);
    // Graph resident-RAM metering (§776 #9 eval infra): single-node graph, neighbors ≤ M, usage > 0
    let graph_mem = resp["graph_memory_bytes"].as_u64().unwrap();
    assert!(graph_mem > 0, "graph_memory_bytes > 0 for 1-node graph");
    assert_eq!(resp["queue_depth"], 0);
    // P3: actual segment file disk usage — 1 vector inserted, vec segment preallocates 128MB > 0
    let vec_disk = resp["vec_disk_bytes"].as_u64().unwrap();
    assert!(vec_disk > 0, "vec_disk_bytes > 0 (segment preallocated)");
    assert!(
        resp["kv_disk_bytes"].as_u64().is_some(),
        "kv_disk_bytes field present"
    );
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_format() {
    let (url, _state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    // Trigger one health request so ops_total has a count
    let _ = reqwest::get(format!("{}/health", url)).await.unwrap();
    let body = reqwest::get(format!("{}/metrics", url))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        body.contains("fusion_store_ops_total"),
        "metrics has ops counter"
    );
    assert!(
        body.contains("fusion_store_backpressure_queue_depth"),
        "metrics has backpressure gauge"
    );
}

#[tokio::test]
async fn kv_write_endpoint_roundtrips() {
    let (url, _state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/kv", url))
        .json(&serde_json::json!({"key": "k1", "value": "v1-fusion"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204, "put_kv returns 204 No Content");
    // /stats should show kv_used_bytes growth
    let stats: serde_json::Value = reqwest::get(format!("{}/stats", url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let used = stats["kv_used_bytes"].as_u64().unwrap();
    assert!(used > 0, "kv used bytes > 0 after put");
}

#[tokio::test]
async fn knn_endpoint_returns_hits() {
    let (url, state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    // Insert 5 vectors (via Engine)
    for i in 0..5u64 {
        state
            .engine
            .insert_vector(i, &[i as f32, 0.0, 0.0, 0.0], None)
            .unwrap();
    }
    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .post(format!("{}/knn", url))
        .json(&serde_json::json!({"query": [3.0, 0.0, 0.0, 0.0], "top_k": 2}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    // Nearest should be id=3
    assert_eq!(results[0]["id"].as_u64().unwrap(), 3);
}

#[tokio::test]
async fn compact_endpoint_runs_cow() {
    let (url, state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    for i in 0..5u64 {
        state
            .engine
            .insert_vector(i, &[i as f32, 0.0, 0.0, 0.0], None)
            .unwrap();
    }
    state.engine.delete_vector(2, None).unwrap();
    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .post(format!("{}/admin/compact", url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // 1 soft-deleted, 4 live
    assert_eq!(resp["live_vectors"].as_u64().unwrap(), 4);
    // During compact /health reports degraded (compact_in_progress briefly true; already finished here)
    let health: serde_json::Value = reqwest::get(format!("{}/health", url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["compact_in_progress"], false);
}

#[tokio::test]
async fn backpressure_returns_503_when_queue_full() {
    // Set the queue cap tiny (2), artificially push queue_depth to the cap to trigger 503 (A2 backpressure)
    // Real transient fullness comes from concurrent write races; deterministically simulate a full queue here: depth>=cap → 503
    let (url, state) = spawn_daemon(4, 2).await;
    state
        .queue_depth
        .store(2, std::sync::atomic::Ordering::Relaxed);
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/kv", url))
        .json(&serde_json::json!({"key": "k0", "value": "x".repeat(64)}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "queue full → 503 backpressure (A2)");
    // /health should report backpressure=true (depth ≥ 8K threshold? here cap=2, threshold still 8K —
    // verifying health's backpressure uses a separate large depth, see the test below)
    // Reset depth to avoid polluting later assertions
    state
        .queue_depth
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

#[tokio::test]
async fn health_reports_backpressure_when_depth_high() {
    // /health reports backpressure=true when queue_depth ≥ BACKPRESSURE_THRESHOLD(8K)
    let (url, state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    state.queue_depth.store(
        fs_serve::BACKPRESSURE_THRESHOLD,
        std::sync::atomic::Ordering::Relaxed,
    );
    let resp: serde_json::Value = reqwest::get(format!("{}/health", url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["status"], "degraded");
    assert_eq!(resp["backpressure"], true);
    state
        .queue_depth
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

// R10/F-SEC-2: admin endpoint with an admin token configured — no Bearer → 401, correct Bearer → pass
#[tokio::test]
async fn admin_endpoint_requires_bearer_token() {
    let mut auth = AuthConfig::empty();
    auth.add("s3cret".into(), AuthRole::Admin);
    let (url, _state) = spawn_daemon_with_auth(4, DEFAULT_QUEUE_CAP, auth, true).await;
    let client = reqwest::Client::new();
    // no token → 401
    let resp = client
        .post(format!("{}/admin/compact", url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "compact without token → 401 (R10)");
    // wrong token → 401
    let resp = client
        .post(format!("{}/admin/compact", url))
        .header("Authorization", "Bearer wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "compact with wrong token → 401 (R10)");
    // correct token → not 401 (compact on an empty store may be 200/500; the point is not being blocked by auth)
    let resp = client
        .post(format!("{}/admin/compact", url))
        .header("Authorization", "Bearer s3cret")
        .send()
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        401,
        "compact with correct token passes auth (R10)"
    );
}

// R10/F-SEC-2: write endpoint /kv is auth-protected once an admin token is configured (no token → 401)
#[tokio::test]
async fn kv_write_requires_bearer_when_auth_set() {
    let mut auth = AuthConfig::empty();
    auth.add("tok".into(), AuthRole::Admin);
    let (url, _state) = spawn_daemon_with_auth(4, DEFAULT_QUEUE_CAP, auth, true).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/kv", url))
        .json(&serde_json::json!({"key": "k", "value": "v"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "kv write without token → 401 (R10)");
}

// R10/F-SEC-2: read-only endpoints /health /stats are not auth-gated on loopback bind (readable without token)
#[tokio::test]
async fn read_endpoints_not_gated_by_auth() {
    let mut auth = AuthConfig::empty();
    auth.add("tok".into(), AuthRole::Admin);
    let (url, _state) = spawn_daemon_with_auth(4, DEFAULT_QUEUE_CAP, auth, true).await;
    let resp = reqwest::get(format!("{}/health", url)).await.unwrap();
    assert_eq!(
        resp.status(),
        200,
        "health readable without token (R10 read exempt)"
    );
    let resp = reqwest::get(format!("{}/stats", url)).await.unwrap();
    assert_eq!(
        resp.status(),
        200,
        "stats readable without token (R10 read exempt)"
    );
}

// E10: vector write endpoint /vector goes through the write queue backpressure + worker pool writing WAL + links
#[tokio::test]
async fn vector_write_endpoint_roundtrips() {
    let (url, _state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/vector", url))
        .json(&serde_json::json!({"id": 42, "vector": [1.0, 2.0, 3.0, 4.0]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204, "insert_vector returns 204 (E10)");
    // KNN should recall the just-written vector
    let resp: serde_json::Value = client
        .post(format!("{}/knn", url))
        .json(&serde_json::json!({"query": [1.0, 2.0, 3.0, 4.0], "top_k": 1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0]["id"].as_u64().unwrap(),
        42,
        "vector 42 retrievable via knn (E10)"
    );
}

// F-SEC-2: RBAC role gating — readonly token writing /kv → 403, admin token write passes
#[tokio::test]
async fn rbac_readonly_rejected_on_write() {
    let mut auth = AuthConfig::empty();
    auth.add("ro-tok".into(), AuthRole::Readonly);
    auth.add("adm-tok".into(), AuthRole::Admin);
    let (url, _state) = spawn_daemon_with_auth(4, DEFAULT_QUEUE_CAP, auth, true).await;
    let client = reqwest::Client::new();
    // readonly writing /kv → 403 (insufficient role)
    let resp = client
        .post(format!("{}/kv", url))
        .header("Authorization", "Bearer ro-tok")
        .json(&serde_json::json!({"key": "k", "value": "v"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        403,
        "readonly token write /kv → 403 (F-SEC-2 RBAC)"
    );
    // admin writing /kv → 204 (admin ≥ readwrite)
    let resp = client
        .post(format!("{}/kv", url))
        .header("Authorization", "Bearer adm-tok")
        .json(&serde_json::json!({"key": "k", "value": "v"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        204,
        "admin token write /kv → 204 (F-SEC-2 RBAC)"
    );
}

// F-SEC-2: readonly token triggering compact → 403 (compact requires admin)
#[tokio::test]
async fn rbac_readonly_rejected_on_compact() {
    let mut auth = AuthConfig::empty();
    auth.add("ro-tok".into(), AuthRole::Readonly);
    let (url, _state) = spawn_daemon_with_auth(4, DEFAULT_QUEUE_CAP, auth, true).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/admin/compact", url))
        .header("Authorization", "Bearer ro-tok")
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        403,
        "readonly token compact → 403 (F-SEC-2 RBAC)"
    );
}

// F-OPS-8: when not loopback-bound, /stats requires ≥readonly auth; no token → 401, readonly token → 200
#[tokio::test]
async fn stats_requires_auth_when_non_loopback() {
    let mut auth = AuthConfig::empty();
    auth.add("ro-tok".into(), AuthRole::Readonly);
    let (url, _state) = spawn_daemon_with_auth(4, DEFAULT_QUEUE_CAP, auth, false).await;
    let client = reqwest::Client::new();
    // no token → 401
    let resp = client.get(format!("{}/stats", url)).send().await.unwrap();
    assert_eq!(
        resp.status(),
        401,
        "stats non-loopback without token → 401 (F-OPS-8)"
    );
    // readonly token → 200
    let resp = client
        .get(format!("{}/stats", url))
        .header("Authorization", "Bearer ro-tok")
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "stats non-loopback with readonly token → 200 (F-OPS-8)"
    );
}

// F-SEC-1 / Rule 12 fail visibly: enforce_bind_auth_policy startup gate
// non-loopback + no token → Err (refuse to start); loopback + no token / either + token → Ok
#[tokio::test]
async fn bind_auth_policy_refuses_non_loopback_without_token() {
    // non-loopback + no token → Err (hard fail, never run unprotected)
    assert!(
        fs_serve::enforce_bind_auth_policy("0.0.0.0", false, false).is_err(),
        "non-loopback + no token → refuse start (F-SEC-1)"
    );
    // loopback + no token → Ok (anonymous access, local debugging)
    assert!(
        fs_serve::enforce_bind_auth_policy("127.0.0.1", true, false).is_ok(),
        "loopback + no token → allow (anonymous safe on loopback)"
    );
    // non-loopback + token → Ok (auth protected)
    assert!(
        fs_serve::enforce_bind_auth_policy("0.0.0.0", false, true).is_ok(),
        "non-loopback + token → allow (auth protected)"
    );
    // loopback + token → Ok
    assert!(
        fs_serve::enforce_bind_auth_policy("127.0.0.1", true, true).is_ok(),
        "loopback + token → allow"
    );
}

// F-SEC-7: body over 16MB → 413 Payload Too Large (intercepted before deserialization)
#[tokio::test]
async fn body_limit_rejects_oversized_payload() {
    let (url, _state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    let client = reqwest::Client::new();
    // Build an oversized value (MAX_BODY_BYTES + 1, larger after JSON encoding, triggers DefaultBodyLimit)
    let oversized = "x".repeat(MAX_BODY_BYTES + 1);
    let resp = client
        .post(format!("{}/kv", url))
        .json(&serde_json::json!({"key": "k", "value": oversized}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        413,
        "oversized body → 413 Payload Too Large (F-SEC-7)"
    );
}
