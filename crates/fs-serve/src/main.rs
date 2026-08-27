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
use fs_core::vector::store::VectorIndex;
use fs_core::KvStore;
use fs_serve::{build_router, metrics, spawn_write_worker, AppState, DEFAULT_PORT};
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
    /// 段容量字节（0=默认）
    #[arg(long, default_value_t = 0)]
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

    let kv = Arc::new(KvStore::open(&ns_dir, args.seg_size, args.quota).context("open kv store")?);
    let schema = VectorSchema::new(args.dim, MetricKind::L2);
    let vec_index =
        Arc::new(VectorIndex::open(&ns_dir, schema, args.seg_size).context("open vector index")?);
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
    fs_serve::refresh_seg_metrics(&kv);

    // 后台写 worker：消费队列执行 KV put（单写者，降 flock 竞争 R1）
    let _worker = spawn_write_worker(kv.clone(), queue_depth.clone(), rx);

    let state = Arc::new(AppState {
        kv,
        vec_index,
        queue_depth,
        compact_in_progress,
        write_tx: tx,
        queue_cap,
    });
    let app = build_router(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    tracing::info!(addr = %addr, namespace = %args.namespace, "fs-serve listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn default_home() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".fusion-store"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/.fusion-store"))
}
