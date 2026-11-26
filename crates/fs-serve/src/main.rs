//! fs-serve bin — fusion-store admin/monitoring HTTP daemon [v2 R3/A2/E5]
//!
//! Listens on 11463 (FUSION_STORE_PORT can override; R3, not 11453).
//! Logic lives in lib.rs (shared with integration tests); this bin only parses args + opens the store + serves.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::Engine;
use fs_serve::{
    build_router, metrics, spawn_write_worker_pool, AppState, DEFAULT_PORT, KNN_CONCURRENCY,
    WRITE_WORKERS,
};
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(
    name = "fs-serve",
    version,
    about = "fusion-store admin/monitoring HTTP daemon"
)]
struct Args {
    /// Listen port (default 11463, R3; FUSION_STORE_PORT env can override)
    #[arg(long, env = "FUSION_STORE_PORT", default_value_t = DEFAULT_PORT)]
    port: u16,
    /// Bind address (F-SEC-1, default 127.0.0.1 loopback; FS_BIND env can override).
    /// For containerized/network deployments set 0.0.0.0; always pair it with FS_AUTH_TOKEN + network-layer ACLs.
    #[arg(long, env = "FS_BIND", default_value = "127.0.0.1")]
    bind: String,
    /// Data root directory (default ~/.fusion-store)
    #[arg(long, env = "FS_HOME")]
    home: Option<PathBuf>,
    /// namespace (default "default")
    #[arg(long, default_value = "default")]
    namespace: String,
    /// Vector dimension (used on store creation; ignored if one already exists)
    #[arg(long, default_value_t = 768)]
    dim: usize,
    /// Write queue cap (backpressure trigger line, 0 = default 10K)
    #[arg(long, default_value_t = 0)]
    queue_cap: usize,
    /// Segment capacity bytes (0 = default; Engine internal default is 64MB; kept for CLI compatibility)
    #[arg(long, default_value_t = 0)]
    #[allow(dead_code)]
    seg_size: u64,
    /// KV quota limit bytes (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    quota: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("fs_serve=info,fs_core=info")
            }),
        )
        .init();
    let args = Args::parse();
    let home = args.home.unwrap_or_else(default_home);
    let ns_dir = home.join(&args.namespace);
    std::fs::create_dir_all(&ns_dir).context("create namespace dir")?;

    let schema = VectorSchema::new(args.dim, MetricKind::L2);
    let engine = Arc::new(
        Engine::open(&ns_dir, Some(schema), args.quota).context("open engine (kv+vec+wal)")?,
    );
    let queue_cap = if args.queue_cap == 0 {
        fs_serve::DEFAULT_QUEUE_CAP
    } else {
        args.queue_cap
    };
    let (tx, rx) = mpsc::channel::<fs_serve::WriteReq>(queue_cap);
    let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let compact_in_progress = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Initialize prometheus metrics + initial segment usage
    metrics::init();
    fs_serve::refresh_seg_metrics(&engine);

    // R3: write worker pool (multi-threaded concurrent queue consumers, removing the single-threaded serial bottleneck)
    let _workers = spawn_write_worker_pool(engine.clone(), queue_depth.clone(), rx, WRITE_WORKERS);

    // R4: KNN concurrency cap semaphore
    let knn_sem = Arc::new(tokio::sync::Semaphore::new(KNN_CONCURRENCY));

    // F-SEC-1: bind address via --bind/FS_BIND. Resolve bind_is_loopback first (used by F-OPS-8 /stats gating).
    let ip: std::net::IpAddr = args
        .bind
        .parse()
        .with_context(|| format!("invalid --bind/FS_BIND address: {}", args.bind))?;
    let bind_is_loopback = ip.is_loopback();

    // R10/F-SEC-2/F-SEC-3: auth config — token→role mapping.
    // Source priority: FS_AUTH_TOKEN_FILE (file, avoids ps/proc plaintext leak) > FS_AUTH_TOKEN (env, back-compat).
    // Three-role tokens: FS_AUTH_TOKEN/FILE=admin, FS_AUTH_TOKEN_RW=readwrite, FS_AUTH_TOKEN_RO=readonly.
    // All empty = anonymous access (loopback-only safe). Shared logic lives in fs_serve::build_auth_from_env.
    let auth = fs_serve::build_auth_from_env().context("load auth tokens from env/file")?;
    let auth_enabled = !auth.is_empty();
    if !auth_enabled && bind_is_loopback {
        // loopback + no token = anonymous access (local debugging convenience, loopback-only safe)
        tracing::warn!(
            "FS_AUTH_TOKEN(_FILE) not set: admin/write endpoints run WITHOUT auth (anonymous). \
             Only safe on 127.0.0.1 loopback; do NOT expose to network without a token."
        );
    }
    // F-SEC-1 / Rule 12 fail visibly: non-loopback + no token = refuse to start (hard fail).
    // The old impl only warned and kept listening → admin/write endpoints exposed. A guaranteed trap for container --bind 0.0.0.0.
    fs_serve::enforce_bind_auth_policy(&args.bind, bind_is_loopback, auth_enabled)
        .context("bind/auth policy")?;

    let state = Arc::new(AppState {
        engine,
        queue_depth,
        compact_in_progress,
        write_tx: tx,
        queue_cap,
        knn_sem,
        auth: Arc::new(auth),
        bind_is_loopback,
    });
    let app = build_router(state);

    let addr = SocketAddr::from((ip, args.port));
    tracing::info!(addr = %addr, namespace = %args.namespace, auth = auth_enabled, "fs-serve listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn default_home() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".fusion-store"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/.fusion-store"))
}
