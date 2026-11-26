//! fs-cli — fusion-store management CLI [v2 M4d]
//!
//! Subcommands (PRD-plan §2.9):
//!   - init    initialize namespace dir (register VectorSchema dim/metric)
//!   - put     write KV
//!   - get     read KV (zero-copy read, then print)
//!   - stats   segment/vector count/quota usage stats
//!   - compact physically purge soft-deleted vectors + rebuild graph snapshot (COW atomic swap A3)
//!   - recover manually trigger crash recovery (idempotent §2.7)
//!   - serve   start management daemon (port 11463 R3, delegates to fs-serve)
//!
//! Logs go to stderr; stdout carries only results (parsed by scripts/tests).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use fs_core::compact::run_compact;
use fs_core::mem::recover::build_recover_plan;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine};

#[derive(Parser)]
#[command(name = "fusion-store", version, about = "fusion-store management CLI")]
struct Cli {
    /// Data root directory (default ~/.fusion-store)
    #[arg(long, env = "FS_HOME", global = true)]
    home: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Initialize namespace directory (register VectorSchema)
    Init {
        #[arg(long, default_value = "default")]
        namespace: String,
        /// Segment capacity (bytes, 0 = default 64MB)
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
        /// Namespace quota limit (bytes, 0 = unlimited)
        #[arg(long, default_value_t = 0)]
        quota: u64,
        /// Vector dimension (used on create; ignored if store already exists)
        #[arg(long, default_value_t = 768)]
        dim: usize,
    },
    /// Write KV
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
    /// Read KV (prints the value; zero-copy read)
    Get {
        #[arg(long, default_value = "default")]
        namespace: String,
        key: String,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
    },
    /// Segment/vector count/quota usage stats
    Stats {
        #[arg(long, default_value = "default")]
        namespace: String,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
    },
    /// Physically purge soft-deleted vectors + rebuild NSW graph snapshot (COW atomic swap)
    Compact {
        #[arg(long, default_value = "default")]
        namespace: String,
        #[arg(long, default_value_t = 0)]
        seg_size: u64,
        /// Reclaim old segments immediately after compact (default waits out the safety period; CLI reclaims directly)
        #[arg(long, default_value_t = false)]
        reclaim_now: bool,
    },
    /// Manually trigger crash recovery (idempotent: read WAL + report pending entries + truncate)
    Recover {
        #[arg(long, default_value = "default")]
        namespace: String,
        /// Report only, do not truncate
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Start management daemon (port 11463, delegates to fs-serve)
    Serve {
        #[arg(long, default_value = "default")]
        namespace: String,
        /// Listen port (default 11463 R3)
        #[arg(long, env = "FUSION_STORE_PORT", default_value_t = fs_serve::DEFAULT_PORT)]
        port: u16,
        /// Bind address (F-SEC-1, default 127.0.0.1 loopback; FS_BIND env can override).
        /// For containerized/network deployments set 0.0.0.0, and always pair it with
        /// FS_AUTH_TOKEN plus network-level ACLs.
        #[arg(long, env = "FS_BIND", default_value = "127.0.0.1")]
        bind: String,
        /// Vector dimension (used on create; ignored if store already exists)
        #[arg(long, default_value_t = 768)]
        dim: usize,
        /// Write queue capacity (0 = default 10K)
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
    // Logs go to stderr; stdout carries only results (parsed by scripts/tests)
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

// ---- subcommand implementations ----

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
        // E7: reclaim validates against the safety period internally and may skip segments
        // below the minimum age — report the actual reclaimed count
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
        // dry_run only builds and reports the plan; nothing applied, nothing truncated
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
    // Not dry_run: Engine::open embeds auto recover (A2: recover is an open side effect, not a standalone action)
    let _engine = Engine::open(&dir, None, 0)
        .with_context(|| format!("engine open + auto-recover namespace {}", ns))?;
    // After replay the WAL is truncated to max_seq (finalize_recover runs inside open)
    // A1: Engine holds the WAL's exclusive flock, so build_recover_plan cannot reopen it — use Engine introspection
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

// F-SEC-1: parameters are passed through from the clap Serve subcommand; the count is
// driven by CLI flags, not design redundancy.
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
    // R10/F-SEC-2/F-SEC-3: CLI serve shares fs-serve's token->role loading logic (env + file).
    // The CLI is often anonymous by default; if FS_AUTH_TOKEN* is set, role gating applies (same as the daemon).
    let auth = fs_serve::build_auth_from_env().with_context(|| "load auth tokens from env/file")?;
    let auth_enabled = !auth.is_empty();
    // F-SEC-1: bind address comes from --bind/FS_BIND. Parse first to fix bind_is_loopback
    // (used by the F-OPS-8 /stats gate).
    let ip: std::net::IpAddr = bind
        .parse()
        .with_context(|| format!("invalid --bind/FS_BIND address: {}", bind))?;
    let bind_is_loopback = ip.is_loopback();
    if !auth_enabled && bind_is_loopback {
        tracing::warn!("FS_AUTH_TOKEN not set: admin/write endpoints anonymous (loopback only)");
    }
    if !bind_is_loopback && !auth_enabled {
        // F-SEC-1 / Rule 12 fail visibly: non-loopback + no token = refuse to start (hard fail).
        // Consistent with the fs-serve daemon. Always hit in container --bind 0.0.0.0 scenarios.
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

/// Default home: ~/.fusion-store
fn dirs_or_default() -> PathBuf {
    if let Some(h) = std::env::var_os("HOME") {
        PathBuf::from(h).join(".fusion-store")
    } else {
        PathBuf::from(".fusion-store")
    }
}
