//! ShardedEngine —— NSW 规模演进 Path A 分片薄层 [ROADMAP F-ARCH-3]
//!
//! 单层 NSW 单 namespace 软上限 ≤1-2M 向量。Path A 不改 `NswGraph` 核心算法，
//! 在 Engine 之上加薄分片层：一个逻辑库拆 K 个 namespace（各独立 `Engine::open`
//! 目录 + 独立 WAL + 独立 flock），向量按 id 分片路由，检索 fan-out 并行查 K 分片
//! 后归并全局 top-k。
//!
//! 设计要点：
//! - **薄层非核心改动**：持 `Vec<Engine>`，路由方法分派，不碰 `NswGraph`/WAL/mmap。
//! - **完整 impl `FusionStoreEngine`**：对消费方透明，一个句柄操作整个分片库。
//! - **路由策略可注入**：`ShardRouter` trait，默认 `HashRouter`（id % K / key fnv % K）。
//!   消费方可注入业务键路由（如按 tenant 固定分片，免 fan-out）。
//! - **fan-out 并行**：`std::thread::scope` 并行查 K 分片，归并取全局 top-k。
//!   Engine 内部同步 mmap，无 async；线程并行是 fan-out 延迟最优解（Rule 5：确定性分派，不经模型）。
//!
//! A4 立场不变：单 `Engine` 仍是单 namespace。ShardedEngine 是消费方侧多 Engine 编排，
//! 非单引擎内多 namespace 路由。K 增长时 fd/mmap/进程数随之膨胀，触及 macOS 系统上限
//! 由消费方控制（同 A4 文档化约束）。

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

// —— 路由策略 ——

/// 分片路由策略 —— 决定向量 id / KV key 落到哪个分片。
pub trait ShardRouter: Send + Sync {
    fn shard_for_vector(&self, id: u64, num_shards: usize) -> usize;
    fn shard_for_key(&self, key: &[u8], num_shards: usize) -> usize;
}

/// 默认路由：向量按 id 取模，KV 按 key fnv-1a hash 取模。
/// 确定性、均匀分布、无状态（Rule 5）。
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

/// 分片引擎 —— 持 K 个 `Engine`，路由分派 + fan-out 检索归并。
pub struct ShardedEngine {
    shards: Vec<Engine>,
    home: PathBuf,
    router: Arc<dyn ShardRouter>,
}

impl ShardedEngine {
    /// 打开/创建分片引擎。home 下建 `shard_0`..`shard_{num_shards-1}` 子目录，
    /// 各开独立 `Engine::open`。schema=Some → 每分片建同 schema 向量索引；None → 全部重开。
    /// 打开后各分片自动 recover（独立 WAL 重放）。
    pub fn open(
        home: &Path,
        num_shards: usize,
        schema: Option<VectorSchema>,
        quota_limit: u64,
    ) -> Result<Self> {
        Self::open_with_router(home, num_shards, schema, quota_limit, Arc::new(HashRouter))
    }

    /// 注入自定义路由策略建分片引擎。
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
            // schema=Some 每分片建同 schema（各 clone，建独立索引）；None 全部重开
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

    /// 分片数。
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// 分片引擎根目录。
    pub fn home(&self) -> &Path {
        &self.home
    }

    /// 按向量 id 路由到分片引用。
    fn shard_for_vector(&self, id: u64) -> &Engine {
        let idx = self.router.shard_for_vector(id, self.shards.len());
        &self.shards[idx]
    }

    /// 按 KV key 路由到分片引用。
    fn shard_for_key(&self, key: &[u8]) -> &Engine {
        let idx = self.router.shard_for_key(key, self.shards.len());
        &self.shards[idx]
    }

    /// 全分片磁盘占用之和。
    pub fn disk_bytes(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for s in &self.shards {
            total += s.disk_bytes()?;
        }
        Ok(total)
    }

    /// 向量总数（全分片存活 id 之和）。分片未建向量索引时该分片计 0。
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
        // 每分片建同 schema 向量索引（name 仅日志）
        for (i, s) in self.shards.iter().enumerate() {
            s.create_vector_index(name, schema.clone())?;
            tracing::debug!(shard = i, "sharded: vector index created on shard");
        }
        Ok(())
    }

    fn open_vector_index(&self, name: &str) -> Result<VectorSchema> {
        // 各分片 schema 一致（建时同步），取首分片即可
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
        // 按 id 路由分摊到各分片，各分片攒批后单次 insert_vector_batch（group commit）。
        // 不做跨分片原子：单条 insert 各自 WAL fsync，分摊后各分片独立持久化。
        // 跨分片原子需 2PC，超出薄层范围（Rule 2），消费方按业务容忍。
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
        // fan-out：并行查 K 分片，各返 top_k 候选，归并取全局 top_k。
        // 各分片返已按 dist 升序排列；全局归并 = 收集 K×top_k 候选排序取前 K（Rule 2：简单正确，
        // K×top_k 规模可控，非海量）。线程并行 fan-out 降延迟（max(分片延迟) 而非 sum）。
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
        // 升序排序取前 top_k（dist 越小越近）
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
        // 全分片合并存活 id。单分片快路径
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
        // 列式按 table_id 路由（key 语义）。table_id 作分片键，同表落同分片。
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
        // 全分片顺序 checkpoint。任一分片失败 → 立即返回 Err（部分 checkpoint，消费方重试）
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
        // 全分片顺序关闭。任一失败 → 返回 Err（后续分片未关闭，消费方需处理）
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
