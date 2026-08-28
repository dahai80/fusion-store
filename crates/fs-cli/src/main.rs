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

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use fs_core::compact::run_compact;
use fs_core::mem::recover::build_recover_plan;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine};

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
    /// 物理清理软删向量 + 重建 NSW 图 snapshot（COW 原子切换）
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
        /// 监听地址（F-SEC-1，默认 127.0.0.1 环回；FS_BIND 环境变量可覆盖）。
        /// 容器化/网络部署设 0.0.0.0，务必同时配 FS_AUTH_TOKEN + 网络层 ACL。
        #[arg(long, env = "FS_BIND", default_value = "127.0.0.1")]
        bind: String,
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
            bind,
            dim,
            queue_cap,
            seg_size,
            quota,
        } => {
            cmd_serve(
                &home, &namespace, port, bind, dim, queue_cap, seg_size, quota,
            )
            .await?
        }
    }
    Ok(())
}

// ---- 子命令实现（占位，下方 Edit 填充） ----

fn cmd_init(home: &Path, ns: &str, _seg_size: u64, quota: u64, dim: usize) -> Result<()> {
    let dir = home.join(ns);
    std::fs::create_dir_all(&dir)?;
    let schema = VectorSchema::new(dim, MetricKind::L2);
    let _engine = Engine::open(&dir, Some(schema), quota)
        .with_context(|| format!("init engine namespace {}", ns))?;
    println!("initialized: {} (dim={})", dir.display(), dim);
    Ok(())
}

fn cmd_put(
    home: &Path,
    ns: &str,
    key: &str,
    value: &str,
    _seg_size: u64,
    quota: u64,
) -> Result<()> {
    let dir = home.join(ns);
    let engine = Engine::open_kv_only(&dir, quota)
        .with_context(|| format!("open engine namespace {}", ns))?;
    engine
        .put_kv(key.as_bytes(), value.as_bytes(), None)
        .map_err(|e| anyhow::anyhow!("put failed: {e}"))?;
    println!("put ok: {} ({}B)", key, value.len());
    Ok(())
}

fn cmd_get(home: &Path, ns: &str, key: &str, _seg_size: u64) -> Result<()> {
    let dir = home.join(ns);
    let engine =
        Engine::open_kv_only(&dir, 0).with_context(|| format!("open engine namespace {}", ns))?;
    match engine.get_kv_zero_copy(key.as_bytes(), None) {
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

fn cmd_stats(home: &Path, ns: &str, _seg_size: u64) -> Result<()> {
    let dir = home.join(ns);
    let engine =
        Engine::open(&dir, None, 0).with_context(|| format!("open engine namespace {}", ns))?;
    let used = engine.kv().used_bytes()?;
    let quota = engine.kv().quota_limit();
    let kv_disk = engine.kv().disk_bytes()?;
    let g = engine.vec_index()?;
    let idx = g.as_ref().unwrap();
    let vec_disk = idx.disk_bytes()?;
    println!("namespace: {}", ns);
    println!("kv_used_bytes: {}", used);
    println!("kv_quota: {}", quota);
    println!("kv_disk_bytes: {}", kv_disk);
    println!("vec_disk_bytes: {}", vec_disk);
    println!("vector_count: {}", idx.len());
    println!("vector_dim: {}", idx.schema().dim);
    println!("graph_memory_bytes: {}", idx.graph_memory_usage());
    Ok(())
}

fn cmd_compact(home: &Path, ns: &str, _seg_size: u64, reclaim_now: bool) -> Result<()> {
    let dir = home.join(ns);
    let engine =
        Engine::open(&dir, None, 0).with_context(|| format!("open engine namespace {}", ns))?;
    let res = {
        let g = engine.vec_index()?;
        let idx = g.as_ref().unwrap();
        run_compact(idx)?
    };
    println!(
        "compact ok: live_vectors={}, reclaimable_segs={}",
        res.live_vectors,
        res.reclaimable_segs.len()
    );
    if reclaim_now && !res.reclaimable_segs.is_empty() {
        let g = engine.vec_index()?;
        let idx = g.as_ref().unwrap();
        let n = fs_core::compact::reclaim(idx, &res.reclaimable_segs)?;
        // E7：reclaim 内部按安全期校验，可能跳过未达下限的段——报实际回收数
        if n < res.reclaimable_segs.len() {
            println!(
                "reclaimed {} of {} old segment(s); {} skipped (sealed < safety period, retry later)",
                n,
                res.reclaimable_segs.len(),
                res.reclaimable_segs.len() - n
            );
        } else {
            println!("reclaimed {} old segment(s)", n);
        }
    } else if !res.reclaimable_segs.is_empty() {
        println!(
            "note: old segments pending reclaim (safety period). pass --reclaim-now to delete."
        );
    }
    Ok(())
}

fn cmd_recover(home: &Path, ns: &str, dry_run: bool) -> Result<()> {
    let dir = home.join(ns);
    let wal_dir = dir.join("wal");
    if dry_run {
        // dry_run 仅读 plan 报告，不 apply 不截断
        let plan = build_recover_plan(&wal_dir)
            .with_context(|| format!("build recover plan namespace {}", ns))?;
        println!(
            "recover plan: applied_seq={}, to_replay={}",
            plan.applied_seq,
            plan.entries.len()
        );
        for e in &plan.entries {
            println!("  replay seq={} op={:?}", e.seq, e.op);
        }
        println!("dry_run: WAL not applied, not truncated");
        return Ok(());
    }
    // 非 dry_run：Engine::open 内嵌自动 recover（A2：recover 是 open 副作用，非独立动作）
    let _engine = Engine::open(&dir, None, 0)
        .with_context(|| format!("engine open + auto-recover namespace {}", ns))?;
    // 重放后 WAL 已截断到 max_seq（finalize_recover 在 open 内调）
    // A1：Engine 持 Wal 排他 flock，不可再 build_recover_plan 重开 → 用 Engine 内省
    let plan_after = _engine.pending_recover_plan()?;
    if plan_after.entries.is_empty() {
        println!("recover ok: WAL replayed + truncated, nothing pending");
    } else {
        println!(
            "recover ok: {} entry(ies) still pending after open",
            plan_after.entries.len()
        );
    }
    Ok(())
}

// F-SEC-1：参数从 clap Serve 子命令透传，数量由 CLI flag 决定，非设计冗余。
#[allow(clippy::too_many_arguments)]
async fn cmd_serve(
    home: &Path,
    ns: &str,
    port: u16,
    bind: String,
    dim: usize,
    queue_cap: usize,
    _seg_size: u64,
    quota: u64,
) -> Result<()> {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    use fs_core::vector::schema::{MetricKind, VectorSchema};

    let ns_dir = home.join(ns);
    std::fs::create_dir_all(&ns_dir).context("create namespace dir")?;
    let schema = VectorSchema::new(dim, MetricKind::L2);
    let engine =
        Arc::new(Engine::open(&ns_dir, Some(schema), quota).context("open engine (kv+vec+wal)")?);
    let cap = if queue_cap == 0 {
        fs_serve::DEFAULT_QUEUE_CAP
    } else {
        queue_cap
    };
    let (tx, rx) = mpsc::channel::<fs_serve::WriteReq>(cap);
    let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let compact_in_progress = Arc::new(std::sync::atomic::AtomicBool::new(false));
    fs_serve::metrics::init();
    fs_serve::refresh_seg_metrics(&engine);
    let _workers = fs_serve::spawn_write_worker_pool(
        engine.clone(),
        queue_depth.clone(),
        rx,
        fs_serve::WRITE_WORKERS,
    );
    // R10/F-SEC-2/F-SEC-3：CLI serve 共享 fs-serve 的 token→role 载入逻辑（env + 文件）。
    // CLI 本地工具常匿名放行；若 FS_AUTH_TOKEN* 已设则按角色门禁（与 daemon 一致）。
    let auth = fs_serve::build_auth_from_env().with_context(|| "load auth tokens from env/file")?;
    let auth_enabled = !auth.is_empty();
    // F-SEC-1：监听地址经 --bind/FS_BIND 配置。先解析定 bind_is_loopback（F-OPS-8 /stats 守门用）。
    let ip: std::net::IpAddr = bind
        .parse()
        .with_context(|| format!("invalid --bind/FS_BIND address: {}", bind))?;
    let bind_is_loopback = ip.is_loopback();
    if !auth_enabled && bind_is_loopback {
        tracing::warn!("FS_AUTH_TOKEN not set: admin/write endpoints anonymous (loopback only)");
    }
    if !bind_is_loopback && !auth_enabled {
        // F-SEC-1 / Rule 12 fail visibly：非环回 + 无 token = 拒绝启动（hard fail）。
        // 与 fs-serve daemon 一致。容器 --bind 0.0.0.0 场景必踩。
        fs_serve::enforce_bind_auth_policy(&bind, bind_is_loopback, auth_enabled)
            .with_context(|| "bind/auth policy")?;
    }
    let state = Arc::new(fs_serve::AppState {
        engine,
        queue_depth,
        compact_in_progress,
        write_tx: tx,
        queue_cap: cap,
        knn_sem: Arc::new(tokio::sync::Semaphore::new(fs_serve::KNN_CONCURRENCY)),
        auth: Arc::new(auth),
        bind_is_loopback,
    });
    let app = fs_serve::build_router(state);
    let addr = SocketAddr::from((ip, port));
    tracing::info!(addr = %addr, namespace = %ns, auth = auth_enabled, "fusion-store serve listening");
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
