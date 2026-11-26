//! ShardedEngine — NSW scale-evolution Path A thin sharding layer [ROADMAP F-ARCH-3]
//!
//! A single-layer NSW has a soft ceiling of roughly 1-2M vectors per namespace. Path A
//! does not touch the `NswGraph` core algorithm; instead it adds a thin sharding layer
//! above Engine: one logical library split into K namespaces (each with its own
//! `Engine::open` directory + independent WAL + independent flock). Vectors are routed
//! to shards by id; search fans out and queries the K shards in parallel, then merges
//! the global top-k.
//!
//! Design points:
//! - **Thin layer, no core changes**: holds `Vec<Engine>` and dispatches via routing
//!   methods; never touches `NswGraph`/WAL/mmap.
//! - **Full impl of `FusionStoreEngine`**: transparent to consumers — one handle operates
//!   the entire sharded library.
//! - **Injectable routing strategy**: the `ShardRouter` trait, defaulting to `HashRouter`
//!   (id % K / key fnv % K). Consumers can inject business-key routing (e.g. pin shards
//!   by tenant to avoid fan-out).
//! - **Parallel fan-out**: `std::thread::scope` queries the K shards in parallel and
//!   merges the global top-k. Engine is synchronous mmap internally, no async; thread
//!   parallelism is the latency-optimal fan-out (Rule 5: deterministic dispatch, not
//!   through a model).
//!
//! A4 stance unchanged: a single `Engine` remains single-namespace. ShardedEngine is
//! consumer-side multi-Engine orchestration, not multi-namespace routing inside one
//! engine. As K grows, fd/mmap/process counts grow with it; hitting macOS system limits
//! is the consumer's concern (same documented constraint as A4).

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::engine_impl::Engine;
use crate::error::Result;
use crate::mem::mmap::ZeroCopyBuffer;
use crate::vector::schema::VectorSchema;
use crate::FusionStoreEngine;

#[cfg(feature = "columnar")]
use crate::engine::ZeroCopyArrowBatch;
#[cfg(feature = "columnar")]
use arrow::record_batch::RecordBatch as ArrowRecordBatch;

// —— Routing strategy ——

/// Shard routing strategy — decides which shard a vector id / KV key lands on.
pub trait ShardRouter: Send + Sync {
    fn shard_for_vector(&self, id: u64, num_shards: usize) -> usize;
    fn shard_for_key(&self, key: &[u8], num_shards: usize) -> usize;
}

/// Default routing: vectors by id modulo, KV by key fnv-1a hash modulo.
/// Deterministic, uniformly distributed, stateless (Rule 5).
pub struct HashRouter;

impl ShardRouter for HashRouter {
    fn shard_for_vector(&self, id: u64, num_shards: usize) -> usize {
        if num_shards == 0 {
            return 0;
        }
        (id as usize) % num_shards
    }

    fn shard_for_key(&self, key: &[u8], num_shards: usize) -> usize {
        if num_shards == 0 {
            return 0;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % num_shards
    }
}

// —— ShardedEngine ——

/// Sharded engine — holds K `Engine`s, dispatches by routing + merges fan-out search.
pub struct ShardedEngine {
    shards: Vec<Engine>,
    home: PathBuf,
    router: Arc<dyn ShardRouter>,
}

impl ShardedEngine {
    /// Open/create the sharded engine. Creates `shard_0`..`shard_{num_shards-1}`
    /// subdirectories under home, each opening its own `Engine::open`. schema=Some →
    /// build a vector index with the same schema per shard; None → reopen all.
    /// After opening, each shard automatically recovers (independent WAL replay).
    pub fn open(
        home: &Path,
        num_shards: usize,
        schema: Option<VectorSchema>,
        quota_limit: u64,
    ) -> Result<Self> {
        Self::open_with_router(home, num_shards, schema, quota_limit, Arc::new(HashRouter))
    }

    /// Build the sharded engine with an injected custom routing strategy.
    pub fn open_with_router(
        home: &Path,
        num_shards: usize,
        schema: Option<VectorSchema>,
        quota_limit: u64,
        router: Arc<dyn ShardRouter>,
    ) -> Result<Self> {
        if num_shards == 0 {
            return Err(crate::StoreError::Corrupt("num_shards must be >= 1".into()));
        }
        std::fs::create_dir_all(home)?;
        let mut shards = Vec::with_capacity(num_shards);
        for i in 0..num_shards {
            let shard_dir = home.join(format!("shard_{}", i));
            // schema=Some builds the same schema per shard (each cloned, independent indexes); None reopens all
            let s = schema.clone();
            let engine = Engine::open(&shard_dir, s, quota_limit)?;
            shards.push(engine);
        }
        tracing::info!(num_shards, home = ?home, "sharded engine opened");
        Ok(Self {
            shards,
            home: home.to_path_buf(),
            router,
        })
    }

    /// Number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Sharded engine root directory.
    pub fn home(&self) -> &Path {
        &self.home
    }

    /// Route a vector id to a shard reference.
    fn shard_for_vector(&self, id: u64) -> &Engine {
        let idx = self.router.shard_for_vector(id, self.shards.len());
        &self.shards[idx]
    }

    /// Route a KV key to a shard reference.
    fn shard_for_key(&self, key: &[u8]) -> &Engine {
        let idx = self.router.shard_for_key(key, self.shards.len());
        &self.shards[idx]
    }

    /// Sum of disk usage across all shards.
    pub fn disk_bytes(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for s in &self.shards {
            total += s.disk_bytes()?;
        }
        Ok(total)
    }

    /// Total vector count (sum of live ids across all shards). A shard without a vector
    /// index counts 0.
    pub fn vector_count(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for s in &self.shards {
            if let Ok(g) = s.vec_index() {
                if let Some(idx) = g.as_ref() {
                    total += idx.len() as u64;
                }
            }
        }
        Ok(total)
    }
}

impl FusionStoreEngine for ShardedEngine {
    fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()> {
        self.shard_for_key(key).put_kv(key, value, timeout)
    }

    fn get_kv_zero_copy(
        &self,
        key: &[u8],
        timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyBuffer>> {
        self.shard_for_key(key).get_kv_zero_copy(key, timeout)
    }

    fn delete_kv(&self, key: &[u8], timeout: Option<Duration>) -> Result<bool> {
        self.shard_for_key(key).delete_kv(key, timeout)
    }

    fn create_vector_index(&self, name: &str, schema: VectorSchema) -> Result<()> {
        // Build a same-schema vector index per shard (name is logging-only)
        for (i, s) in self.shards.iter().enumerate() {
            s.create_vector_index(name, schema.clone())?;
            tracing::debug!(shard = i, "sharded: vector index created on shard");
        }
        Ok(())
    }

    fn open_vector_index(&self, name: &str) -> Result<VectorSchema> {
        // All shards share the same schema (synchronized at creation); the first shard suffices
        self.shards[0].open_vector_index(name)
    }

    fn insert_vector(&self, id: u64, vector: &[f32], timeout: Option<Duration>) -> Result<()> {
        self.shard_for_vector(id).insert_vector(id, vector, timeout)
    }

    fn insert_vector_batch(
        &self,
        items: &[(u64, &[f32])],
        timeout: Option<Duration>,
    ) -> Result<()> {
        // Route by id into per-shard buckets, then each shard commits one
        // insert_vector_batch (group commit).
        // No cross-shard atomicity: individual inserts each WAL fsync, and after bucketing
        // each shard persists independently.
        // Cross-shard atomicity would need 2PC — out of scope for a thin layer (Rule 2);
        // consumers tolerate this per their business semantics.
        let mut buckets: Vec<Vec<(u64, &[f32])>> = vec![Vec::new(); self.shards.len()];
        for (id, v) in items {
            let idx = self.router.shard_for_vector(*id, self.shards.len());
            buckets[idx].push((*id, *v));
        }
        for (i, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            self.shards[i].insert_vector_batch(&bucket, timeout)?;
        }
        Ok(())
    }

    fn search_knn(
        &self,
        query_vector: &[f32],
        top_k: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(u64, f32)>> {
        // fan-out: query the K shards in parallel, each returning top_k candidates, then
        // merge into the global top_k.
        // Each shard returns candidates already sorted ascending by dist; the global merge
        // = collect K×top_k candidates, sort, take the top (Rule 2: simple and correct;
        // K×top_k is bounded, not massive). Thread-parallel fan-out cuts latency
        // (max(shard latencies) instead of sum).
        if self.shards.len() == 1 {
            return self.shards[0].search_knn(query_vector, top_k, timeout);
        }
        let mut results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.shards.len());
            for s in &self.shards {
                let q = query_vector;
                let handle = scope.spawn(move || s.search_knn(q, top_k, timeout));
                handles.push(handle);
            }
            let mut all = Vec::new();
            for h in handles {
                match h.join() {
                    Ok(Ok(part)) => all.extend(part),
                    Ok(Err(e)) => tracing::error!(error = ?e, "sharded knn: shard failed"),
                    Err(_) => tracing::error!("sharded knn: shard thread panicked"),
                }
            }
            all
        });
        // Sort ascending and take the top_k (smaller dist = nearer)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    fn delete_vector(&self, id: u64, timeout: Option<Duration>) -> Result<bool> {
        self.shard_for_vector(id).delete_vector(id, timeout)
    }

    fn get_vector(&self, id: u64, timeout: Option<Duration>) -> Result<Option<Vec<f32>>> {
        self.shard_for_vector(id).get_vector(id, timeout)
    }

    fn list_vector_ids(&self, _timeout: Option<Duration>) -> Result<Vec<u64>> {
        // Merge live ids across all shards. Single-shard fast path
        let mut all = Vec::new();
        for s in &self.shards {
            all.extend(s.list_vector_ids(_timeout)?);
        }
        Ok(all)
    }

    #[cfg(feature = "columnar")]
    fn put_columnar(
        &self,
        table_id: &str,
        batch: &ArrowRecordBatch,
        timeout: Option<Duration>,
    ) -> Result<()> {
        // Columnar routes by table_id (key semantics). table_id is the shard key so the
        // same table always lands on the same shard.
        let idx = self
            .router
            .shard_for_key(table_id.as_bytes(), self.shards.len());
        self.shards[idx].put_columnar(table_id, batch, timeout)
    }

    #[cfg(feature = "columnar")]
    fn get_columnar_zero_copy(
        &self,
        table_id: &str,
        columns: &[&str],
        timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyArrowBatch>> {
        let idx = self
            .router
            .shard_for_key(table_id.as_bytes(), self.shards.len());
        self.shards[idx].get_columnar_zero_copy(table_id, columns, timeout)
    }

    fn checkpoint(&self) -> Result<()> {
        // Checkpoint all shards sequentially. Any shard failure → return Err immediately
        // (partial checkpoint; the consumer retries)
        for (i, s) in self.shards.iter().enumerate() {
            s.checkpoint().map_err(|e| {
                tracing::error!(shard = i, error = ?e, "sharded checkpoint: shard failed");
                e
            })?;
        }
        tracing::info!(num_shards = self.shards.len(), "sharded checkpoint done");
        Ok(())
    }

    fn recover(&self) -> Result<()> {
        for (i, s) in self.shards.iter().enumerate() {
            s.recover().map_err(|e| {
                tracing::error!(shard = i, error = ?e, "sharded recover: shard failed");
                e
            })?;
        }
        Ok(())
    }

    fn close(&self) -> Result<()> {
        // Close all shards sequentially. Any failure → return Err (later shards unclosed;
        // the consumer must handle it)
        for (i, s) in self.shards.iter().enumerate() {
            s.close().map_err(|e| {
                tracing::error!(shard = i, error = ?e, "sharded close: shard failed");
                e
            })?;
        }
        tracing::info!(num_shards = self.shards.len(), "sharded engine closed");
        Ok(())
    }
}
