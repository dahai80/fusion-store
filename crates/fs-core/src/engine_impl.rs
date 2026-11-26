//! Engine — the engine layer aggregating KV + Vector + Columnar + WAL [F1/F2/A1]
//!
//! Implements the PRD §1.4 `FusionStoreEngine` trait: `Engine` is the sole implementer,
//! aggregating `KvStore` + `VectorIndex` + `ColumnarStore` + `Wal`. All external entry
//! points (fs-cli / fs-serve / fs-ffi-c / fs-ffi-py) hold an `Engine` and never touch
//! submodules directly.
//!
//! WAL in the write path (F2): put_kv/insert/insert_batch/put_columnar/delete first
//! `wal.append` (fsync, the sole crash-safe sync point) → then call submodules to land
//! mmap segments + heed.
//! PRD H5: the WAL is the only sync point, mmap segments flush lazily, heed NO_SYNC.
//! After a crash, `Engine::open` calls `build_recover_plan` → replays WalOps with
//! seq > applied_seq (logical payloads, idempotent: insert skips dups, put_kv overwrites,
//! delete is idempotent) → `finalize_recover` truncates.
//!
//! One namespace = one Engine instance.
//!
//! A4 honest positioning: the Engine does **not** support multi-namespace routing or
//! multi-tenant isolation inside a single engine. Each namespace needs its own
//! `Engine::open(<ns_dir>, ...)` instance = independent directory + independent WAL +
//! independent heed env + independent flock (A1/E13 cross-process exclusion at namespace
//! granularity). Multi-namespace coordination (e.g. fusion-memory reading/writing several
//! namespaces) is the **consumer's responsibility** — the application manages multiple
//! Engine handles itself. fusion-store does not provide multi-tenant quota sharing or a
//! namespace routing layer inside one engine, and does not promise that capability; the
//! positioning is "single-namespace zero-copy storage engine", not a multi-tenant engine.
//! As namespace count grows, fd/mmap/process counts grow with it; hitting macOS system
//! limits is the consumer's concern.

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

/// Engine instance — aggregates the three storage types + WAL
pub struct Engine {
    kv: KvStore,
    vec: Mutex<Option<VectorIndex>>,
    col: Mutex<Option<ColumnarStore>>,
    wal: Mutex<Wal>,
    home: PathBuf,
    quota_limit: u64,
}

impl Engine {
    /// Open/create the engine. home = namespace root directory.
    /// Subdirectories: kv/ + vec/ + col/ (columnar) + wal/.
    /// schema=Some → create a new vector index (dim locked); None → reopen an existing one.
    /// Automatically recovers after open (replays un-checkpointed WAL, idempotent).
    pub fn open(home: &Path, schema: Option<VectorSchema>, quota_limit: u64) -> Result<Self> {
        let home = home.to_path_buf();
        let kv_dir = home.join("kv");
        let vec_dir = home.join("vec");
        let wal_dir = home.join("wal");
        std::fs::create_dir_all(&home)?;
        // kv segments default 64MB; quota 0 = unlimited
        let kv = KvStore::open(&kv_dir, 0, quota_limit)?;
        // Vector index: Some builds (schema locked), None reopens (None when no index exists)
        let vec_idx = match schema {
            Some(s) => Some(VectorIndex::open(&vec_dir, s, 0)?),
            None => {
                // Reopen only when vec_meta exists; otherwise no index is created
                if vec_dir.join("vec_meta").exists() {
                    Some(VectorIndex::reopen(&vec_dir, 0)?)
                } else {
                    None
                }
            }
        };
        // Columnar opens lazily (created on first put_columnar); None here
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
        // Recover right at open: replay the un-checkpointed WAL (A1)
        engine.recover_inner()?;
        tracing::info!(home = ?home, "engine opened + recovered");
        Ok(engine)
    }

    /// Open KV only (no vector index scenarios)
    pub fn open_kv_only(home: &Path, quota_limit: u64) -> Result<Self> {
        Self::open(home, None, quota_limit)
    }

    pub fn home(&self) -> &Path {
        &self.home
    }

    /// Read the current pending WAL replay plan (via the already-open Wal, no reopen that
    /// would race the flock).
    /// A1: Wal::open holds a blocking exclusive flock; the external `build_recover_plan`
    /// would reopen the Wal → deadlock with the Wal this Engine holds. While holding an
    /// Engine, read the plan only through this method — never call bare
    /// `build_recover_plan(&wal_dir)`.
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

    /// Direct reference to the KV submodule (FFI/CLI read quota/stats)
    pub fn kv(&self) -> &KvStore {
        &self.kv
    }

    /// Whole-engine actual disk usage = kv segments + vec segments + col segments
    /// (P3: real usage, including padding/segment tail holes).
    /// Sums each submodule that is open; unopened submodules count 0 (no error).
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

    /// Vector index reference (FFI/CLI read schema/count) — returns the guard under lock
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

    /// Recover implementation: build_recover_plan → idempotent replay entry by entry → finalize_recover
    fn recover_inner(&self) -> Result<()> {
        // Reuse the already-open Wal (A1: Wal::open holds a blocking exclusive flock;
        // a second instance would deadlock).
        // build_recover_plan/finalize_recover would Wal::open again → contend for the same
        // flock as the Engine's Wal → the second open blocks forever. So read the open Wal
        // directly here, never via the reopen path.
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
        // Truncate the WAL to max_seq after successful replay (reusing the open Wal)
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

    /// Idempotent replay of a single WAL op. insert dup → skip (already applied);
    /// everything else goes straight to the submodule.
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
        // F2: log the WAL first (fsync, the sole sync point) → then land mmap segment + heed
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
        // Single namespace, single vector index: name is logging-only; the index actually
        // lives under vec/
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
        // F2: WAL first (payload is the full vector) → then land segment + link edges
        // E13: cross-process safety comes from the namespace exclusive flock held by
        // Wal::open (A1) — a second process cannot open the same namespace, so vector
        // writes never interleave with another process.
        // In-process concurrency is serialized by the Wal Mutex (serial appends) + the
        // VectorIndex RwLock (insert write lock).
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
        // E2: the whole batch does one WAL append+fsync (group commit), replacing the old
        // per-entry fsync.
        // Batch atomicity: crash before fsync → whole batch lost (torn-tail tolerance);
        // crash after fsync → whole batch durable.
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
        // L7: vector deletes are WAL-logged too; recover replays the deletion
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

    // —— Vector read/enumeration (#3, read-only, bypasses the WAL) ——
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
        // F2: WAL records the IPC (payload) → land in col segments
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
        // Create the columnar store on demand
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
        // A6: strict ordering — first flush KV + Columnar active segments + heed force_sync,
        // then write the vector snapshot, and only last write the WAL marker + truncate.
        // If the WAL were truncated first and the process crashed before flush → KV segments
        // lag behind the truncated WAL → data loss.
        // Flush first, truncate last: crashing at any point after truncation still leaves
        // the already-flushed KV/col data complete.
        let applied_seq = {
            let wal = self
                .wal
                .lock()
                .map_err(|_| crate::StoreError::LockPoisoned)?;
            wal.next_seq().saturating_sub(1)
        };
        // 1. Flush the KV active segment + heed force_sync (locators durable)
        self.kv.flush()?;
        // 2. Flush the Columnar active segment + heed force_sync (if opened)
        if let Ok(g) = self.ensure_col() {
            if let Some(col) = g.as_ref() {
                col.flush()?;
            }
        }
        // 3. Vector index snapshot (graph topology + soft-delete marks)
        if let Ok(g) = self.ensure_vec() {
            g.as_ref().unwrap().checkpoint()?;
        }
        // 4. WAL marker: applied_seq advances so recover no longer replays checkpointed entries
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
            // 5. Only now truncate the checkpointed WAL entries
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

    // A6/E5: orderly shutdown — checkpoint already covers KV/col flush + vec snapshot +
    // WAL marker/truncation.
    // E5: any step failure aggregates into Err, never swallowed (swallowing = silent data loss).
    fn close(&self) -> Result<()> {
        tracing::info!("engine closing: checkpoint (flush + snapshot + wal)");
        // A failed checkpoint → data not fully durable, must error (E5: no warn-then-Ok)
        self.checkpoint()?;
        tracing::info!("engine closed: all segments flushed + heed synced + wal truncated");
        Ok(())
    }
}

// —— Columnar IPC codec (WAL payload serializing a RecordBatch) ——
// E10: decode_ipc is pub so the fs-serve HTTP /columnar endpoint can decode client IPC bytes.
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
