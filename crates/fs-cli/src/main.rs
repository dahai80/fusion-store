//! fs-cli —— fusion-store 管理命令 [v2 M4d]
//!
//! 子命令（PRD-plan §2.9）：
//!   - init   初始化 namespace 目录（登记 VectorSchema dim/metric）
//!   - put    写 KV
//!   - get    读 KV（零拷贝读后打印）
//!   - stats  段/向量数/配额用量统计
//!   - compact 物理清理软删向量 + 重建图 snapshot（COW 原子切换 A3）
//!   - recover 手动触发崩溃恢复（幂等 §2.7）
//!   - serve  启管理 daemon（11463 R3，委托 fs-serve）
//!
//! 日志走 stderr，stdout 仅输出结果（供脚本/测试解析）。

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use fs_core::compact::run_compact;
use fs_core::mem::recover;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::store::VectorIndex;
use fs_core::KvStore;

#[derive(Parser)]
#[command(name = "fusion-store", version, about = "fusion-store 管理命令")]
struct Cli {
    /// 数据根目录（默认 ~/.fusion-store）
    #[arg(long, env = "FS_HOME", global = true)]
    home: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 初始化 namespace 目录（登记 VectorSchema）
    Init {
        #[arg(long, default_value = "default")]
        namespace: String,
        /// 段容量（字节，0=默认 64MB）
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
        /// namespace 配额上限（字节，0=不限）
        #[arg(long, default_value_t = 0)]
        quota: u64,
        /// 向量维度（建库用，已存在则忽略）
        #[arg(long, default_value_t = 768)]
        dim: usize,
    },
    /// 写 KV
    Put {
        #[arg(long, default_value = "default")]
        namespace: String,
        key: String,
        value: String,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
        #[arg(long, default_value_t = 0)]
        quota: u64,
    },
    /// 读 KV（输出 value，零拷贝读后打印）
    Get {
        #[arg(long, default_value = "default")]
        namespace: String,
        key: String,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
    },
    /// 段/向量数/配额用量统计
    Stats {
        #[arg(long, default_value = "default")]
        namespace: String,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
    },
    /// 物理清理软删向量 + 重建 HNSW 图 snapshot（COW 原子切换）
    Compact {
        #[arg(long, default_value = "default")]
        namespace: String,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
        /// compact 后立即回收旧段（默认等待安全期，此处 CLI 直接回收）
        #[arg(long, default_value_t = false)]
        reclaim_now: bool,
    },
    /// 手动触发崩溃恢复（幂等，读 WAL + 报告待重放条目 + 截断）
    Recover {
        #[arg(long, default_value = "default")]
        namespace: String,
        /// 仅报告不执行截断
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// 启管理 daemon（端口 11463，委托 fs-serve）
    Serve {
        #[arg(long, default_value = "default")]
        namespace: String,
        /// 监听端口（默认 11463 R3）
        #[arg(long, env = "FUSION_STORE_PORT", default_value_t = fs_serve::DEFAULT_PORT)]
        port: u16,
        /// 向量维度（建库用，已存在则忽略）
        #[arg(long, default_value_t = 768)]
        dim: usize,
        /// 写队列上限（0=默认 10K）
        #[arg(long, default_value_t = 0)]
        queue_cap: usize,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
        #[arg(long, default_value_t = 0)]
        quota: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // 日志走 stderr，stdout 仅输出结果（供脚本/测试解析）
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let home = cli.home.unwrap_or_else(dirs_or_default);

    match cli.cmd {
        Cmd::Init {
            namespace,
            seg_size,
            quota,
            dim,
        } => cmd_init(&home, &namespace, seg_size, quota, dim)?,
        Cmd::Put {
            namespace,
            key,
            value,
            seg_size,
            quota,
        } => cmd_put(&home, &namespace, &key, &value, seg_size, quota)?,
        Cmd::Get {
            namespace,
            key,
            seg_size,
        } => cmd_get(&home, &namespace, &key, seg_size)?,
        Cmd::Stats {
            namespace,
            seg_size,
        } => cmd_stats(&home, &namespace, seg_size)?,
        Cmd::Compact {
            namespace,
            seg_size,
            reclaim_now,
        } => cmd_compact(&home, &namespace, seg_size, reclaim_now)?,
        Cmd::Recover { namespace, dry_run } => cmd_recover(&home, &namespace, dry_run)?,
        Cmd::Serve {
            namespace,
            port,
            dim,
            queue_cap,
            seg_size,
            quota,
        } => cmd_serve(&home, &namespace, port, dim, queue_cap, seg_size, quota).await?,
    }
    Ok(())
}

// ---- 子命令实现（占位，下方 Edit 填充） ----

fn cmd_init(home: &Path, ns: &str, seg_size: u64, quota: u64, dim: usize) -> Result<()> {
    let dir = home.join(ns);
    std::fs::create_dir_all(&dir)?;
    let _store = KvStore::open(&dir, seg_size, quota)
        .with_context(|| format!("init kv namespace {}", ns))?;
    // 建向量索引（登记 VectorSchema dim/metric，§1.4）
    let schema = VectorSchema::new(dim, MetricKind::L2);
    let _idx = VectorIndex::open(&dir, schema, seg_size)
        .with_context(|| format!("init vector index namespace {}", ns))?;
    println!("initialized: {} (dim={})", dir.display(), dim);
    Ok(())
}

fn cmd_put(home: &Path, ns: &str, key: &str, value: &str, seg_size: u64, quota: u64) -> Result<()> {
    let dir = home.join(ns);
    let store =
        KvStore::open(&dir, seg_size, quota).with_context(|| format!("open namespace {}", ns))?;
    store
        .put_kv(
            key.as_bytes(),
            value.as_bytes(),
            Some(Duration::from_secs(5)),
        )
        .map_err(|e| anyhow::anyhow!("put failed: {e}"))?;
    println!("put ok: {} ({}B)", key, value.len());
    Ok(())
}

fn cmd_get(home: &Path, ns: &str, key: &str, seg_size: u64) -> Result<()> {
    let dir = home.join(ns);
    let store =
        KvStore::open(&dir, seg_size, 0).with_context(|| format!("open namespace {}", ns))?;
    match store.get_kv_zero_copy(key.as_bytes(), None) {
        Ok(Some(buf)) => {
            let owned = buf.to_owned_slice();
            println!("{}", String::from_utf8_lossy(&owned));
        }
        Ok(None) => {
            println!("(not found)");
            std::process::exit(2);
        }
        Err(e) => return Err(anyhow::anyhow!("get failed: {e}")),
    }
    Ok(())
}

fn cmd_stats(home: &Path, ns: &str, seg_size: u64) -> Result<()> {
    let dir = home.join(ns);
    let kv =
        KvStore::open(&dir, seg_size, 0).with_context(|| format!("open kv namespace {}", ns))?;
    let used = kv.used_bytes()?;
    let quota = kv.quota_limit();
    let idx = VectorIndex::reopen(&dir, seg_size)
        .with_context(|| format!("open vector index namespace {}", ns))?;
    println!("namespace: {}", ns);
    println!("kv_used_bytes: {}", used);
    println!("kv_quota: {}", quota);
    println!("vector_count: {}", idx.len());
    println!("vector_dim: {}", idx.schema().dim);
    println!("graph_memory_bytes: {}", idx.graph_memory_usage());
    Ok(())
}

fn cmd_compact(home: &Path, ns: &str, seg_size: u64, reclaim_now: bool) -> Result<()> {
    let dir = home.join(ns);
    // reopen 不需 schema，从持久化 snapshot 恢复
    let idx = VectorIndex::reopen(&dir, seg_size)
        .with_context(|| format!("open vector index namespace {}", ns))?;
    let res = run_compact(&idx)?;
    println!(
        "compact ok: live_vectors={}, reclaimable_segs={}",
        res.live_vectors,
        res.reclaimable_segs.len()
    );
    if reclaim_now && !res.reclaimable_segs.is_empty() {
        // CLI 直接回收（假定无 in-flight reader，单用户管理态）
        fs_core::compact::reclaim(&idx, &res.reclaimable_segs)?;
        println!("reclaimed {} old segment(s)", res.reclaimable_segs.len());
    } else if !res.reclaimable_segs.is_empty() {
        println!(
            "note: old segments pending reclaim (safety period). pass --reclaim-now to delete."
        );
    }
    Ok(())
}

fn cmd_recover(home: &Path, ns: &str, dry_run: bool) -> Result<()> {
    let wal_dir = home.join(ns).join("wal");
    let plan = recover::build_recover_plan(&wal_dir)
        .with_context(|| format!("build recover plan namespace {}", ns))?;
    println!(
        "recover plan: applied_seq={}, to_replay={}, registered_blocks={}",
        plan.applied_seq,
        plan.entries.len(),
        plan.registered_blocks.len()
    );
    for e in &plan.entries {
        println!("  replay seq={} op={:?}", e.seq, e.op);
    }
    if dry_run {
        println!("dry_run: WAL not truncated");
    } else if !plan.entries.is_empty() {
        // 幂等重放由引擎层在 open 时自动做；CLI 此处确认后截断 WAL 防无限增长
        let max_seq = plan
            .entries
            .iter()
            .map(|e| e.seq)
            .max()
            .unwrap_or(plan.applied_seq);
        recover::finalize_recover(&wal_dir, max_seq)?;
        println!("recover ok: WAL truncated to seq={}", max_seq);
    } else {
        println!("recover: nothing to replay, WAL already checkpointed");
    }
    Ok(())
}

async fn cmd_serve(
    home: &Path,
    ns: &str,
    port: u16,
    dim: usize,
    queue_cap: usize,
    seg_size: u64,
    quota: u64,
) -> Result<()> {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    let ns_dir = home.join(ns);
    std::fs::create_dir_all(&ns_dir).context("create namespace dir")?;
    let kv = Arc::new(KvStore::open(&ns_dir, seg_size, quota).context("open kv store")?);
    let schema = VectorSchema::new(dim, MetricKind::L2);
    let vec_index =
        Arc::new(VectorIndex::open(&ns_dir, schema, seg_size).context("open vector index")?);
    let cap = if queue_cap == 0 {
        fs_serve::DEFAULT_QUEUE_CAP
    } else {
        queue_cap
    };
    let (tx, rx) = mpsc::channel::<fs_serve::WriteReq>(cap);
    let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let compact_in_progress = Arc::new(std::sync::atomic::AtomicBool::new(false));
    fs_serve::metrics::init();
    fs_serve::refresh_seg_metrics(&kv);
    let _worker = fs_serve::spawn_write_worker(kv.clone(), queue_depth.clone(), rx);
    let state = Arc::new(fs_serve::AppState {
        kv,
        vec_index,
        queue_depth,
        compact_in_progress,
        write_tx: tx,
        queue_cap: cap,
    });
    let app = fs_serve::build_router(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    tracing::info!(addr = %addr, namespace = %ns, "fusion-store serve listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// 默认 home：~/.fusion-store
fn dirs_or_default() -> PathBuf {
    if let Some(h) = std::env::var_os("HOME") {
        PathBuf::from(h).join(".fusion-store")
    } else {
        PathBuf::from(".fusion-store")
    }
}
