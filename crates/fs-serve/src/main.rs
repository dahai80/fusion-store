//! fs-serve bin —— fusion-store 管理/监控 HTTP daemon [v2 R3/A2/E5]
//!
//! 监听 11463（FUSION_STORE_PORT 可覆盖，R3 非 11453）。
//! 逻辑在 lib.rs（供集成测试共用），本 bin 仅做 args 解析 + 开 store + serve。

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
    about = "fusion-store 管理/监控 HTTP daemon"
)]
struct Args {
    /// 监听端口（默认 11463，R3；FUSION_STORE_PORT 环境变量可覆盖）
    #[arg(long, env = "FUSION_STORE_PORT", default_value_t = DEFAULT_PORT)]
    port: u16,
    /// 监听地址（F-SEC-1，默认 127.0.0.1 环回；FS_BIND 环境变量可覆盖）。
    /// 容器化/网络部署设 0.0.0.0，务必同时配 FS_AUTH_TOKEN + 网络层 ACL。
    #[arg(long, env = "FS_BIND", default_value = "127.0.0.1")]
    bind: String,
    /// 数据根目录（默认 ~/.fusion-store）
    #[arg(long, env = "FS_HOME")]
    home: Option<PathBuf>,
    /// namespace（默认 default）
    #[arg(long, default_value = "default")]
    namespace: String,
    /// 向量维度（建库用，已存在则忽略）
    #[arg(long, default_value_t = 768)]
    dim: usize,
    /// 写队列上限（背压触发线，0=默认 10K）
    #[arg(long, default_value_t = 0)]
    queue_cap: usize,
    /// 段容量字节（0=默认，Engine 内部默认 64MB，此参数保留兼容 CLI）
    #[arg(long, default_value_t = 0)]
    #[allow(dead_code)]
    seg_size: u64,
    /// KV 配额上限字节（0=不限）
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

    // 初始化 prometheus 指标 + 段使用率初值
    metrics::init();
    fs_serve::refresh_seg_metrics(&engine);

    // R3：写 worker 池（多线程并发消费队列，破单线程串行瓶颈）
    let _workers = spawn_write_worker_pool(engine.clone(), queue_depth.clone(), rx, WRITE_WORKERS);

    // R4：KNN 并发上限信号量
    let knn_sem = Arc::new(tokio::sync::Semaphore::new(KNN_CONCURRENCY));

    // F-SEC-1：监听地址经 --bind/FS_BIND 配置。先解析定 bind_is_loopback（F-OPS-8 /stats 守门用）。
    let ip: std::net::IpAddr = args
        .bind
        .parse()
        .with_context(|| format!("invalid --bind/FS_BIND address: {}", args.bind))?;
    let bind_is_loopback = ip.is_loopback();

    // R10/F-SEC-2/F-SEC-3：认证配置 —— token→role 映射。
    // 来源优先级：FS_AUTH_TOKEN_FILE（文件，避免 ps/proc 明文泄露）> FS_AUTH_TOKEN（env，向后兼容）。
    // 三角色 token：FS_AUTH_TOKEN/FILE=admin，FS_AUTH_TOKEN_RW=readwrite，FS_AUTH_TOKEN_RO=readonly。
    // 全空 = 匿名放行（仅环回安全）。共享逻辑在 fs_serve::build_auth_from_env。
    let auth = fs_serve::build_auth_from_env().context("load auth tokens from env/file")?;
    let auth_enabled = !auth.is_empty();
    if !auth_enabled && bind_is_loopback {
        // 环回 + 无 token = 匿名放行（本地调试便利，仅环回安全）
        tracing::warn!(
            "FS_AUTH_TOKEN(_FILE) not set: admin/write endpoints run WITHOUT auth (anonymous). \
             Only safe on 127.0.0.1 loopback; do NOT expose to network without a token."
        );
    }
    // F-SEC-1 / Rule 12 fail visibly：非环回 + 无 token = 拒绝启动（hard fail）。
    // 旧实现仅 warn 继续 listen → admin/write 端点裸奔。容器 --bind 0.0.0.0 场景必踩。
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
