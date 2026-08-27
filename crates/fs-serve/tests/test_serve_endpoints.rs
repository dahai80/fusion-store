//! fs-serve 端点集成测试 —— 真实 HTTP bind + reqwest [v2 R3/A2/E5]
//!
//! 验收对齐 PRD M4：
//!   - /health 通（status ok / store_open / backpressure）
//!   - /stats 通（KV used / vector_count / queue_depth）
//!   - /metrics 通（Prometheus 文本格式，含 fusion_store_* 指标）
//!   - /kv 写通（背压满返 503 A2）
//!   - /knn 读通（插入向量后 KNN 召回）
//!   - /admin/compact 通（COW 原子切换 A3）
//!   - 背压：队列满 → 503 + /health 报 backpressure=true（degraded，供上游熔断 A2）

use std::sync::Arc;

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::store::VectorIndex;
use fs_core::KvStore;
use fs_serve::{build_router, metrics, spawn_write_worker, AppState, WriteReq, DEFAULT_QUEUE_CAP};
use tempfile::tempdir;
use tokio::sync::mpsc;

/// 起一个绑定临时端口的 daemon，返回 base URL + AppState 句柄（测后清理）
async fn spawn_daemon(dim: usize, queue_cap: usize) -> (String, Arc<AppState>) {
    // leak tempdir 进 'static：DirHandle + 路径均不回收（测试进程退出即清理）
    let dir = Box::leak(Box::new(tempdir().unwrap()));
    let ns_dir = dir.path();
    std::fs::create_dir_all(ns_dir).unwrap();

    let kv = Arc::new(KvStore::open(ns_dir, 0, 0).unwrap());
    let schema = VectorSchema::new(dim, MetricKind::L2);
    let vec_index = Arc::new(VectorIndex::open(ns_dir, schema, 0).unwrap());
    let (tx, rx) = mpsc::channel::<WriteReq>(queue_cap);
    let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let compact_in_progress = Arc::new(std::sync::atomic::AtomicBool::new(false));
    metrics::init();
    fs_serve::refresh_seg_metrics(&kv);
    let _worker = spawn_write_worker(kv.clone(), queue_depth.clone(), rx);
    let state = Arc::new(AppState {
        kv,
        vec_index,
        queue_depth: queue_depth.clone(),
        compact_in_progress,
        write_tx: tx,
        queue_cap,
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
    // 插一个向量
    state
        .vec_index
        .insert(1, &[1.0, 0.0, 0.0, 0.0], None)
        .unwrap();
    let resp: serde_json::Value = reqwest::get(format!("{}/stats", url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["vector_count"], 1);
    assert_eq!(resp["vector_dim"], 4);
    // 图常驻 RAM 计量（§776 #9 评估基建）：单节点图，邻居数 ≤ M，占用 > 0
    let graph_mem = resp["graph_memory_bytes"].as_u64().unwrap();
    assert!(graph_mem > 0, "graph_memory_bytes > 0 for 1-node graph");
    assert_eq!(resp["queue_depth"], 0);
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_format() {
    let (url, _state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    // 触一次 health 让 ops_total 有计数
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
    // /stats 应见 kv_used_bytes 增长
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
    // 插入 5 向量
    for i in 0..5u64 {
        state
            .vec_index
            .insert(i, &[i as f32, 0.0, 0.0, 0.0], None)
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
    // 最近应是 id=3
    assert_eq!(results[0]["id"].as_u64().unwrap(), 3);
}

#[tokio::test]
async fn compact_endpoint_runs_cow() {
    let (url, state) = spawn_daemon(4, DEFAULT_QUEUE_CAP).await;
    for i in 0..5u64 {
        state
            .vec_index
            .insert(i, &[i as f32, 0.0, 0.0, 0.0], None)
            .unwrap();
    }
    state.vec_index.delete(2, None).unwrap();
    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .post(format!("{}/admin/compact", url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // 软删 1 个，活 4
    assert_eq!(resp["live_vectors"].as_u64().unwrap(), 4);
    // compact 期间 /health 报 degraded（compact_in_progress 短暂 true，此处已结束）
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
    // 队列上限设极小（2），人为推高 queue_depth 到上限触发 503（A2 背压）
    // 真实瞬时满由并发写竞争达成；此处确定性模拟队列满态：depth>=cap → 503
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
    // /health 应报 backpressure=true（depth≥阈值 8K？此处 cap=2 阈值仍 8K——
    // 验证 health 的 backpressure 用单独大 depth，见下测）
    // 回退 depth 防污染后续
    state
        .queue_depth
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

#[tokio::test]
async fn health_reports_backpressure_when_depth_high() {
    // /health 在 queue_depth ≥ BACKPRESSURE_THRESHOLD(8K) 时报 backpressure=true
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
