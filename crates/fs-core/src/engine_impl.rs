//! Engine —— 聚合 KV + Vector + Columnar + WAL 的引擎层 [F1/F2/A1]
//!
//! 落地 PRD §1.4 的 `FusionStoreEngine` trait：`Engine` 是唯一实现者，聚合
//! `KvStore` + `VectorIndex` + `ColumnarStore` + `Wal`。所有外部入口（fs-cli /
//! fs-serve / fs-ffi-c / fs-ffi-py）持有 `Engine`，不再裸持子模块。
//!
//! WAL 接入写路径（F2）：put_kv/insert/insert_batch/put_columnar/delete 先
//! `wal.append`（fsync，crash-safe 唯一同步点）→ 再调子模块落 mmap 段 + heed。
//! PRD H5：WAL 唯一同步点，mmap 段延迟刷，heed NO_SYNC。崩溃后 `Engine::open`
//! 调 `build_recover_plan` → 重放 seq > applied_seq 的 WalOp（逻辑 payload，
//! 幂等：insert 查 dup 跳过，put_kv 覆盖，delete 幂等）→ `finalize_recover` 截断。
//!
//! 单 namespace = 一个 Engine 实例。
//!
//! A4 诚实定位：Engine **不支持**单引擎内多 namespace 路由/多租户隔离。每个 namespace
//! 需独立 `Engine::open(<ns_dir>, ...)` 实例 = 独立目录 + 独立 WAL + 独立 heed env +
//! 独立 flock（A1/E13 跨进程互斥按 namespace 粒度）。多 namespace 协调（如 fusion-memory
//! 同时读写多 namespace）是**消费方职责**——应用层自行管理多 Engine 句柄。fusion-store
//! 不在单引擎内提供多租户配额共享或 namespace 路由层，也不承诺该能力；定位是
//! 「单 namespace 零拷贝存储引擎」，非多租户引擎。namespace 数量增长时 fd/mmap/进程数
//! 随之膨胀，触及 macOS 系统上限由消费方控制。

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use crate::columnar::store::ColumnarStore;
use crate::error::Result;
use crate::mem::mmap::ZeroCopyBuffer;
use crate::mem::wal::{Wal, WalOp};
use crate::store::KvStore;
use crate::vector::schema::VectorSchema;
use crate::vector::store::VectorIndex;
use crate::FusionStoreEngine;

#[cfg(feature = "columnar")]
use crate::engine::ZeroCopyArrowBatch;
#[cfg(feature = "columnar")]
use arrow::record_batch::RecordBatch as ArrowRecordBatch;

/// 引擎实例 —— 聚合三类存储 + WAL
pub struct Engine {
    kv: KvStore,
    vec: Mutex<Option<VectorIndex>>,
    col: Mutex<Option<ColumnarStore>>,
    wal: Mutex<Wal>,
    home: PathBuf,
    quota_limit: u64,
}

impl Engine {
    /// 打开/创建引擎。home = namespace 根目录。
    /// 子目录：kv/ + vec/ + col/(列式) + wal/。
    /// schema=Some → 新建向量索引（锁定 dim）；None → 重开已存在索引。
    /// 打开后自动 recover（重放未 checkpoint 的 WAL，幂等）。
    pub fn open(home: &Path, schema: Option<VectorSchema>, quota_limit: u64) -> Result<Self> {
        let home = home.to_path_buf();
        let kv_dir = home.join("kv");
        let vec_dir = home.join("vec");
        let wal_dir = home.join("wal");
        std::fs::create_dir_all(&home)?;
        // kv 段默认 64MB；配额 0=不限
        let kv = KvStore::open(&kv_dir, 0, quota_limit)?;
        // 向量索引：Some 建（锁 schema），None 重开（无已存在索引则 None）
        let vec_idx = match schema {
            Some(s) => Some(VectorIndex::open(&vec_dir, s, 0)?),
            None => {
                // vec_meta 存在才 reopen；否则不建索引
                if vec_dir.join("vec_meta").exists() {
                    Some(VectorIndex::reopen(&vec_dir, 0)?)
                } else {
                    None
                }
            }
        };
        // 列式按需开（首次 put_columnar 时建），此处 None
        let col = None;
        let wal = Wal::open(&wal_dir)?;
        let engine = Self {
            kv,
            vec: Mutex::new(vec_idx),
            col: Mutex::new(col),
            wal: Mutex::new(wal),
            home: home.clone(),
            quota_limit,
        };
        // 打开即 recover：重放未 checkpoint 的 WAL（A1）
        engine.recover_inner()?;
        tracing::info!(home = ?home, "engine opened + recovered");
        Ok(engine)
    }

    /// 仅打开 KV（无向量索引场景）
    pub fn open_kv_only(home: &Path, quota_limit: u64) -> Result<Self> {
        Self::open(home, None, quota_limit)
    }

    pub fn home(&self) -> &Path {
        &self.home
    }

    /// 读当前待重放 WAL 计划（经已开 Wal，不重开抢 flock）。
    /// A1：Wal::open 持阻塞排他 flock；外部 `build_recover_plan` 会重开 Wal →
    /// 与本 Engine 持有的 Wal 死锁。故持 Engine 时只能经此方法读 plan，
    /// 不能调裸 `build_recover_plan(&wal_dir)`。
    pub fn pending_recover_plan(&self) -> Result<crate::mem::RecoverPlan> {
        let wal = self
            .wal
            .lock()
            .map_err(|_| crate::StoreError::LockPoisoned)?;
        let applied_seq = match wal.read_marker()? {
            Some(m) => m.applied_seq,
            None => 0,
        };
        let entries: Vec<_> = wal
            .read_all()?
            .into_iter()
            .filter(|e| e.seq > applied_seq)
            .collect();
        Ok(crate::mem::RecoverPlan {
            entries,
            applied_seq,
        })
    }

    /// KV 子模块直引用（FFI/CLI 读配额/统计用）
    pub fn kv(&self) -> &KvStore {
        &self.kv
    }

    /// 全引擎实际磁盘占用 = kv 段 + vec 段 + col 段（P3：真实占用，含 padding/段尾空洞）。
    /// 各子模块按已开与否求和，未开的子模块计 0（不报错）。
    pub fn disk_bytes(&self) -> Result<u64> {
        let mut total: u64 = 0;
        total += self.kv.disk_bytes()?;
        if let Ok(g) = self.vec.lock().map_err(|_| crate::StoreError::LockPoisoned) {
            if let Some(idx) = g.as_ref() {
                total += idx.disk_bytes()?;
            }
        }
        if let Ok(g) = self.col.lock().map_err(|_| crate::StoreError::LockPoisoned) {
            if let Some(col) = g.as_ref() {
                total += col.disk_bytes()?;
            }
        }
        Ok(total)
    }

    /// 向量索引引用（FFI/CLI 读 schema/count 用）—— 持锁返回 guard
    pub fn vec_index(&self) -> Result<std::sync::MutexGuard<'_, Option<VectorIndex>>> {
        let g = self
            .vec
            .lock()
            .map_err(|_| crate::StoreError::LockPoisoned)?;
        if g.is_none() {
            return Err(crate::StoreError::NotFound);
        }
        Ok(g)
    }

    fn ensure_vec(&self) -> Result<std::sync::MutexGuard<'_, Option<VectorIndex>>> {
        let g = self
            .vec
            .lock()
            .map_err(|_| crate::StoreError::LockPoisoned)?;
        if g.is_none() {
            return Err(crate::StoreError::NotFound);
        }
        Ok(g)
    }

    fn ensure_col(&self) -> Result<std::sync::MutexGuard<'_, Option<ColumnarStore>>> {
        let g = self
            .col
            .lock()
            .map_err(|_| crate::StoreError::LockPoisoned)?;
        if g.is_none() {
            return Err(crate::StoreError::NotFound);
        }
        Ok(g)
    }

    /// recover 实现：build_recover_plan → 逐条幂等重放 → finalize_recover
    fn recover_inner(&self) -> Result<()> {
        // 复用已开 Wal（A1：Wal::open 持阻塞排他 flock，不可再开第二个实例，否则死锁）。
        // build_recover_plan/finalize_recover 会 Wal::open 重开 → 与 Engine 持有的 Wal 抢同
        // 一把 flock → 第二个 open 永久阻塞。故在此直接读已开 Wal，不走重开路径。
        let (applied_seq, entries) = {
            let wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            let applied_seq = match wal.read_marker()? {
                Some(m) => m.applied_seq,
                None => 0,
            };
            let entries: Vec<_> = wal
                .read_all()?
                .into_iter()
                .filter(|e| e.seq > applied_seq)
                .collect();
            (applied_seq, entries)
        };
        if entries.is_empty() {
            tracing::info!("recover: no pending WAL entries, skip");
            return Ok(());
        }
        let mut max_seq = applied_seq;
        for entry in &entries {
            self.replay_op(&entry.op)?;
            max_seq = max_seq.max(entry.seq);
        }
        // 重放成功后截断 WAL 到 max_seq（复用已开 Wal）
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.truncate_to(max_seq)?;
        }
        tracing::info!(
            replayed = entries.len(),
            max_seq,
            "recover: replay done + wal truncated"
        );
        Ok(())
    }

    /// 单条 WAL op 幂等重放。insert dup → 跳过（已 applied）；其余直接调子模块。
    fn replay_op(&self, op: &WalOp) -> Result<()> {
        match op {
            WalOp::PutKv { key, value } => self.kv.put_kv(key, value, None),
            WalOp::DeleteKv { key } => {
                let _ = self.kv.delete_kv(key, None)?;
                Ok(())
            }
            WalOp::InsertVector { id, vector } => {
                let g = self.ensure_vec()?;
                let vec = g.as_ref().unwrap();
                match vec.insert(*id, vector, None) {
                    Ok(()) => Ok(()),
                    Err(crate::StoreError::DuplicateVector(_)) => {
                        tracing::debug!(id, "replay insert: id exists, skip (idempotent)");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            WalOp::DeleteVector { id } => {
                let g = self.ensure_vec()?;
                let vec = g.as_ref().unwrap();
                let _ = vec.delete(*id, None)?;
                Ok(())
            }
            WalOp::PutColumnar { table_id, ipc } => {
                #[cfg(feature = "columnar")]
                {
                    let batch = decode_ipc(ipc)?;
                    let g = self.ensure_col()?;
                    let col = g.as_ref().unwrap();
                    col.put_columnar(table_id, &batch, None)
                }
                #[cfg(not(feature = "columnar"))]
                {
                    let _ = (table_id, ipc);
                    Err(crate::StoreError::Corrupt(
                        "columnar op in WAL but feature disabled".into(),
                    ))
                }
            }
        }
    }
}

impl FusionStoreEngine for Engine {
    fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()> {
        // F2：先记 WAL（fsync 唯一同步点）→ 再落 mmap 段 + heed
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.append(WalOp::PutKv {
                key: key.to_vec(),
                value: value.to_vec(),
            })?;
        }
        self.kv.put_kv(key, value, timeout)
    }

    fn get_kv_zero_copy(
        &self,
        key: &[u8],
        timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyBuffer>> {
        self.kv.get_kv_zero_copy(key, timeout)
    }

    fn delete_kv(&self, key: &[u8], timeout: Option<Duration>) -> Result<bool> {
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.append(WalOp::DeleteKv { key: key.to_vec() })?;
        }
        self.kv.delete_kv(key, timeout)
    }

    fn create_vector_index(&self, name: &str, schema: VectorSchema) -> Result<()> {
        // 单 namespace 单向量索引：name 仅作日志，实际建在 vec/ 下
        let _ = name;
        let vec_dir = self.home.join("vec");
        let idx = VectorIndex::open(&vec_dir, schema, 0)?;
        *self
            .vec
            .lock()
            .map_err(|_| crate::StoreError::LockPoisoned)? = Some(idx);
        tracing::info!("vector index created");
        Ok(())
    }

    fn open_vector_index(&self, name: &str) -> Result<VectorSchema> {
        let _ = name;
        let g = self.ensure_vec()?;
        Ok(g.as_ref().unwrap().schema().clone())
    }

    fn insert_vector(&self, id: u64, vector: &[f32], timeout: Option<Duration>) -> Result<()> {
        // F2：先 WAL（payload 完整 vector）→ 再落段连边
        // E13：跨进程安全由 Wal::open 持的 namespace 排他 flock 保证（A1）——
        // 第二进程无法 open 同 namespace，向量写不会与另一进程交错。
        // 进程内并发由 Wal Mutex（append 串行）+ VectorIndex RwLock（insert 写锁）串行。
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.append(WalOp::InsertVector {
                id,
                vector: vector.to_vec(),
            })?;
        }
        let g = self.ensure_vec()?;
        g.as_ref().unwrap().insert(id, vector, timeout)
    }

    fn insert_vector_batch(
        &self,
        items: &[(u64, &[f32])],
        timeout: Option<Duration>,
    ) -> Result<()> {
        // E2：整批单次 WAL append+fsync（group commit），替代旧逐条 fsync。
        // 批次原子性：fsync 前崩 → 整批丢（torn tail 容错）；fsync 后 → 整批持久。
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            let ops: Vec<WalOp> = items
                .iter()
                .map(|(id, v)| WalOp::InsertVector {
                    id: *id,
                    vector: v.to_vec(),
                })
                .collect();
            wal.append_batch(&ops)?;
        }
        let g = self.ensure_vec()?;
        g.as_ref().unwrap().insert_batch(items, timeout)
    }

    fn search_knn(
        &self,
        query_vector: &[f32],
        top_k: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(u64, f32)>> {
        let g = self.ensure_vec()?;
        g.as_ref().unwrap().search_knn(query_vector, top_k, timeout)
    }

    fn delete_vector(&self, id: u64, timeout: Option<Duration>) -> Result<bool> {
        // L7：向量删也记 WAL，recover 重放删除
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.append(WalOp::DeleteVector { id })?;
        }
        let g = self.ensure_vec()?;
        g.as_ref().unwrap().delete(id, timeout)
    }

    // —— 向量读取/枚举（#3，只读不经 WAL）——
    fn get_vector(&self, id: u64, _timeout: Option<Duration>) -> Result<Option<Vec<f32>>> {
        let g = self.ensure_vec()?;
        g.as_ref().unwrap().get_vector(id)
    }

    fn list_vector_ids(&self, _timeout: Option<Duration>) -> Result<Vec<u64>> {
        let g = self.ensure_vec()?;
        g.as_ref().unwrap().list_vector_ids()
    }

    #[cfg(feature = "columnar")]
    fn put_columnar(
        &self,
        table_id: &str,
        batch: &ArrowRecordBatch,
        timeout: Option<Duration>,
    ) -> Result<()> {
        // F2：WAL 记 IPC（payload）→ 落 col 段
        let ipc = encode_ipc(batch)?;
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.append(WalOp::PutColumnar {
                table_id: table_id.to_string(),
                ipc,
            })?;
        }
        // 按需建列式 store
        if self
            .col
            .lock()
            .map_err(|_| crate::StoreError::LockPoisoned)?
            .is_none()
        {
            let col_dir = self.home.join("col");
            let store = ColumnarStore::open(&col_dir, 0, self.quota_limit)?;
            *self
                .col
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)? = Some(store);
        }
        let g = self.ensure_col()?;
        g.as_ref().unwrap().put_columnar(table_id, batch, timeout)
    }

    #[cfg(feature = "columnar")]
    fn get_columnar_zero_copy(
        &self,
        table_id: &str,
        columns: &[&str],
        timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyArrowBatch>> {
        let g = self.ensure_col()?;
        g.as_ref()
            .unwrap()
            .get_columnar_zero_copy(table_id, columns, timeout)
    }

    fn checkpoint(&self) -> Result<()> {
        // A6：顺序严格——先 flush KV + Columnar active 段 + heed force_sync，
        // 再写向量 snapshot，最后写 WAL marker + 截断。
        // 若先截断 WAL 后崩溃在 flush 前 → KV 段落后于已截断 WAL → 丢数据。
        // flush 在前，截断在最后：截断后任一点崩溃，已 flush 的 KV/col 仍完整。
        let applied_seq = {
            let wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.next_seq().saturating_sub(1)
        };
        // 1. flush KV active 段 + heed force_sync（locators 落盘）
        self.kv.flush()?;
        // 2. flush Columnar active 段 + heed force_sync（若已开）
        if let Ok(g) = self.ensure_col() {
            if let Some(col) = g.as_ref() {
                col.flush()?;
            }
        }
        // 3. 向量索引 snapshot（图拓扑 + 软删标记）
        if let Ok(g) = self.ensure_vec() {
            g.as_ref().unwrap().checkpoint()?;
        }
        // 4. WAL marker：applied_seq 推进，recover 不再重放已 checkpoint 部分
        let snapshot_name = format!("nsw_checkpoint_{}.mmap", applied_seq);
        let marker = crate::mem::wal::CheckpointMarker {
            applied_seq,
            checkpoint_seq: applied_seq,
            graph_snapshot: Some(snapshot_name),
        };
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.write_marker(&marker)?;
            // 5. 最后才截断已 checkpoint 的 WAL 条目
            wal.truncate_to(applied_seq)?;
        }
        tracing::info!(
            applied_seq,
            "engine checkpoint done: kv/col flushed, vec snapshot, wal truncated"
        );
        Ok(())
    }

    fn recover(&self) -> Result<()> {
        self.recover_inner()
    }

    // A6/E5：有序关闭——checkpoint 已含 KV/col flush + vec snapshot + WAL marker/截断。
    // E5：任一步失败聚合返回 Err，绝不吞没（吞没 = 静默丢数据）。
    fn close(&self) -> Result<()> {
        tracing::info!("engine closing: checkpoint (flush + snapshot + wal)");
        // checkpoint 失败 → 数据未完整落盘，必须报错（E5：不再 warn 后 Ok）
        self.checkpoint()?;
        tracing::info!("engine closed: all segments flushed + heed synced + wal truncated");
        Ok(())
    }
}

// —— 列式 IPC 编解码（WAL payload 序列化 RecordBatch）——
// E10：decode_ipc 暴露为 pub，供 fs-serve HTTP /columnar 端点解码客户端 IPC 字节。
#[cfg(feature = "columnar")]
pub fn encode_ipc(batch: &ArrowRecordBatch) -> Result<Vec<u8>> {
    use arrow::ipc::writer::StreamWriter;
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(buf)
}

#[cfg(feature = "columnar")]
pub fn decode_ipc(ipc: &[u8]) -> Result<ArrowRecordBatch> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;
    let mut reader = StreamReader::try_new(Cursor::new(ipc), None)?;
    reader
        .next()
        .ok_or_else(|| crate::StoreError::Corrupt("empty IPC stream".into()))?
        .map_err(crate::StoreError::Arrow)
}
