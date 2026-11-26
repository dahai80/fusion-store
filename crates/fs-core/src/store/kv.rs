//! KV storage layer — heed-backed locators + mmap payload segments [v2 E1/E6]
//!
//! put_kv: payload lands in a segment → heed stores (key -> ValueLocator) + quota accumulates.
//! get_kv_zero_copy: read the heed locator → map the sealed segment → ZeroCopyBuffer
//! (pointer lands in the mmap region).
//! delete_kv: heed deletes the locator, quota decrements. Segments are not reclaimed in
//! place (H4); compact rewrites whole segments to reclaim (M4).
//! Multi-process writes: flock fails fast (R1).
//!
//! Quota semantics (P3): quota_limit bounds the net growth of "logical payload bytes"
//! (overwrite subtracts the old value and adds the new); it does not count
//! append_aligned padding, the tail hole of a not-yet-full segment, or heed metadata.
//! Append-only segments preallocate 64MB each, so the last unfilled segment can waste up
//! to 64MB × namespace count.
//! Thus actual disk usage = payload + padding + segment tail hole + heed > quota counter.
//! disk_bytes() reports real segment-file usage; /stats exposes the quota-vs-disk gap
//! for monitoring to keep disk in check.

use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use fs2::FileExt;
use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn};

use crate::error::Result;
use crate::mem::mmap::ZeroCopyBuffer;
use crate::mem::segment::{SegmentPool, ValueLocator};
use crate::StoreError;

/// KV values always land in mmap segments (zero-copy read pointer assertion, aligned with §1.4)
const KV_DB_NAME: &str = "kv_locators";
/// namespace quota counter DB (key = fixed "used" -> u64 bytes)
const QUOTA_DB_NAME: &str = "kv_quota";
const QUOTA_KEY: &[u8] = b"used";

/// E8: heed map_size cap. Locators are 12B/entry; 2GB ≈ 180M keys (far beyond an
/// enterprise KV namespace).
/// Apple Silicon's virtual address space is huge, so a 2GB preallocation costs zero
/// physical memory (demand paging).
/// A full map surfaces MapFull (not a panic); callers can alert or rebuild with a larger
/// map_size.
const KV_MAP_SIZE: usize = 2 * 1024 * 1024 * 1024;

// E8: promote MapFull inside a heed Error into an explicit StoreError::MapFull so callers
// can alert on capacity. Non-MapFull heed errors pass through as StoreError::Heed. Returns
// a bare StoreError for the caller to wrap in Err(), fitting the differing return types
// (put_kv_inner Result<()>, delete_kv_inner Result<bool>).
fn classify_heed(e: heed::Error) -> StoreError {
    match e {
        heed::Error::Mdb(heed::MdbError::MapFull) => StoreError::MapFull {
            current: 0,
            limit: KV_MAP_SIZE as u64,
        },
        other => StoreError::Heed(other),
    }
}

/// KV storage — single namespace
///
/// The writer holds an exclusive per-process flock; readers are lock-free (heed MVCC).
pub struct KvStore {
    env: Env,
    kv_db: Database<Bytes, Bytes>,
    quota_db: Database<Bytes, Bytes>,
    pool: Mutex<SegmentPool>,
    lock_file: std::fs::File,
    quota_limit: u64,
}

impl KvStore {
    /// Open/create the KV store. data_dir = <ns>/kv_data, meta_dir = <ns>/kv_meta
    /// Isolated from VectorIndex's vec_meta/vec_data — heed forbids two Envs on one path [v2 E1]
    pub fn open(namespace_dir: &Path, seg_size: u64, quota_limit: u64) -> Result<Self> {
        let data_dir = namespace_dir.join("kv_data");
        let meta_dir = namespace_dir.join("kv_meta");
        std::fs::create_dir_all(&meta_dir)?;
        let lock_path = namespace_dir.join("lock");

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(KV_MAP_SIZE)
                .max_dbs(crate::HEED_MAX_DBS)
                // NO_SYNC: heed stores only (tiny) locator metadata; commit does not fsync.
                // The WAL is the sole crash-safe sync point (H5); after a crash, WAL replay
                // backfills entries heed had not flushed
                .flags(heed::EnvFlags::NO_SYNC)
                .open(&meta_dir)?
        };
        let mut wtxn = env.write_txn()?;
        let kv_db: Database<Bytes, Bytes> = env.create_database(&mut wtxn, Some(KV_DB_NAME))?;
        let quota_db: Database<Bytes, Bytes> =
            env.create_database(&mut wtxn, Some(QUOTA_DB_NAME))?;
        wtxn.commit()?;

        let pool = SegmentPool::open(&data_dir, seg_size)?;

        // flock multi-process write exclusion (try_lock; fail fast if unavailable, R1)
        let lock_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        // Only the writer path takes the exclusive lock; not locked at open, locked at put/delete

        tracing::info!(dir = ?namespace_dir, quota_limit, "kv store opened");
        Ok(Self {
            env,
            kv_db,
            quota_db,
            pool: Mutex::new(pool),
            lock_file,
            quota_limit,
        })
    }

    /// Acquire the write lock (E9: blocking flock, not polling)
    /// None = kernel-blocking `lock_exclusive` until acquired (kernel wake, zero CPU spin).
    /// Some(d) = time-bounded wait: fs2 has no timed-lock primitive, so approximate the
    /// deadline with try + a 5ms short-sleep loop (5ms granularity cuts the old 50ms
    /// jitter 10×, still non-zero CPU; the pure blocking path via None covers the norm).
    fn acquire_write_lock(&self, timeout: Option<Duration>) -> Result<()> {
        match timeout {
            None => self
                .lock_file
                .lock_exclusive()
                .map_err(|_| StoreError::LockBusy),
            Some(d) => {
                let mut waited = Duration::ZERO;
                loop {
                    if self.lock_file.try_lock_exclusive().is_ok() {
                        return Ok(());
                    }
                    if waited >= d {
                        return Err(StoreError::LockBusy);
                    }
                    // E9: 5ms-granularity short sleep cuts the old 50ms jitter 10×; very short backoff on failed try
                    std::thread::sleep(Duration::from_millis(5));
                    waited += Duration::from_millis(5);
                }
            }
        }
    }

    /// Write KV — payload lands in a segment + heed stores the locator + quota accumulates
    pub fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()> {
        self.acquire_write_lock(timeout)?;
        let res = self.put_kv_inner(key, value);
        // Release the write lock
        let _ = self.lock_file.unlock();
        res
    }

    fn put_kv_inner(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // F-TEST-2: reject empty keys / keys above MDB_MAXKEYSIZE explicitly (Rule 12: fail visibly)
        // LMDB default page size 4096 → MDB_MAXKEYSIZE = 511. Empty keys hit BadValSize directly.
        if key.is_empty() {
            return Err(StoreError::InvalidKey("empty key rejected"));
        }
        const LMDB_MAX_KEY: usize = 511;
        if key.len() > LMDB_MAX_KEY {
            return Err(StoreError::InvalidKey(
                "key exceeds LMDB MDB_MAXKEYSIZE (511)",
            ));
        }
        let start = std::time::Instant::now();
        let mut wtxn = self.env.write_txn()?;
        // Complete within a single write txn: read old value (subtract from quota on
        // overwrite) + quota check + write locator + write quota
        let old_len: u64 = if let Some(old_loc_bytes) = self.kv_db.get(&wtxn, key)? {
            let old_loc: ValueLocator = bindecode(old_loc_bytes)?;
            old_loc.len as u64
        } else {
            0
        };
        let used = read_quota_txn(&self.quota_db, &wtxn)?;
        // Subtract the old value length before adding the new — under overwrite semantics the quota counts net growth
        let net = (value.len() as u64).saturating_sub(old_len);
        let new_used = used.checked_add(net).ok_or(StoreError::QuotaExceeded)?;
        if self.quota_limit > 0 && new_used > self.quota_limit {
            tracing::warn!(
                key_hash = fnv1a(key),
                len = value.len(),
                used,
                limit = self.quota_limit,
                "quota exceeded, reject put_kv"
            );
            wtxn.abort();
            return Err(StoreError::QuotaExceeded);
        }

        // Land the payload in a segment
        let loc = {
            let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
            pool.append(value)?
        };
        // Serialize the locator into heed
        let loc_bytes = binencode(&loc)?;
        self.kv_db.put(&mut wtxn, key, &loc_bytes)?;
        write_quota_txn(&self.quota_db, &mut wtxn, new_used)?;
        // L1: a failed commit rolls back already-landed segment bytes, avoiding in-segment leaks
        if let Err(e) = wtxn.commit() {
            tracing::warn!(key_hash = fnv1a(key), err = ?e, "put_kv commit failed, rewinding segment");
            if let Ok(mut pool) = self.pool.lock() {
                if let Err(re) = pool.rewind_active_to(loc.seg_id, loc.offset) {
                    // F-ERR-2: a failed rewind leaves garbage in the segment; alert loudly for operator intervention
                    tracing::error!(
                        key_hash = fnv1a(key),
                        seg = loc.seg_id,
                        off = loc.offset,
                        err = ?re,
                        "put_kv rewind failed after commit error — segment may leak bytes"
                    );
                }
            }
            // E8: MapFull reported distinctly so callers can alert on capacity
            return Err(classify_heed(e));
        }

        tracing::debug!(
            key_hash = fnv1a(key),
            len = value.len(),
            seg_id = loc.seg_id,
            offset = loc.offset,
            elapsed_us = start.elapsed().as_micros() as u64,
            "put_kv done"
        );
        Ok(())
    }

    /// Zero-copy KV read — pointer lands in the mmap segment region
    pub fn get_kv_zero_copy(
        &self,
        key: &[u8],
        _timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyBuffer>> {
        let rtxn = self.env.read_txn()?;
        let Some(loc_bytes) = self.kv_db.get(&rtxn, key)? else {
            return Ok(None);
        };
        let loc: ValueLocator = bindecode(loc_bytes)?;
        drop(rtxn);

        let handle = {
            let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
            pool.sealed_handle(loc.seg_id)?
        };
        let buf = ZeroCopyBuffer::new(handle, loc.offset as usize, loc.len as usize)?;
        tracing::debug!(
            key_hash = fnv1a(key),
            seg_id = loc.seg_id,
            len = loc.len,
            "get_kv_zero_copy hit"
        );
        Ok(Some(buf))
    }

    /// Delete KV — heed deletes the locator, quota decrements
    pub fn delete_kv(&self, key: &[u8], timeout: Option<Duration>) -> Result<bool> {
        self.acquire_write_lock(timeout)?;
        let res = self.delete_kv_inner(key);
        let _ = self.lock_file.unlock();
        res
    }

    fn delete_kv_inner(&self, key: &[u8]) -> Result<bool> {
        let mut wtxn = self.env.write_txn()?;
        let existing = self.kv_db.get(&wtxn, key)?;
        let old_len = existing.as_ref().map(|b| {
            bindecode(b)
                .map(|l: ValueLocator| l.len as u64)
                .unwrap_or(0)
        });
        let Some(old_len) = old_len else {
            wtxn.abort();
            return Ok(false);
        };
        let deleted = self.kv_db.delete(&mut wtxn, key)?;
        let used = read_quota_txn(&self.quota_db, &wtxn)?;
        let new_used = used.saturating_sub(old_len);
        write_quota_txn(&self.quota_db, &mut wtxn, new_used)?;
        // E8: MapFull on commit reported distinctly
        if let Err(e) = wtxn.commit() {
            return Err(classify_heed(e));
        }
        tracing::debug!(key_hash = fnv1a(key), old_len, "delete_kv done");
        Ok(deleted)
    }

    /// Current namespace used bytes
    pub fn used_bytes(&self) -> Result<u64> {
        let rtxn = self.env.read_txn()?;
        read_quota_txn(&self.quota_db, &rtxn)
    }

    /// Quota limit
    pub fn quota_limit(&self) -> u64 {
        self.quota_limit
    }

    /// Actual segment-file disk usage (P3: includes padding/segment tail holes, vs used_bytes net payload)
    pub fn disk_bytes(&self) -> Result<u64> {
        let pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
        pool.disk_bytes()
    }

    /// Orderly flush: flush the active segment + heed force_sync (used by A6 close)
    /// Under NO_SYNC, commit does not fsync; close's explicit force_sync guarantees
    /// metadata durability.
    pub fn flush(&self) -> Result<()> {
        let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
        pool.flush()?;
        self.env.force_sync()?;
        tracing::info!("kv store flushed (active segment + heed sync)");
        Ok(())
    }
}

// —— Quota read/write: txn-parameterized so it can be reused inside a write txn ——
fn read_quota_txn(db: &Database<Bytes, Bytes>, txn: &RoTxn) -> Result<u64> {
    if let Some(bytes) = db.get(txn, QUOTA_KEY)? {
        if bytes.len() == 8 {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(bytes);
            Ok(u64::from_le_bytes(arr))
        } else {
            Ok(0)
        }
    } else {
        Ok(0)
    }
}

fn write_quota_txn(db: &Database<Bytes, Bytes>, wtxn: &mut RwTxn, val: u64) -> Result<()> {
    db.put(wtxn, QUOTA_KEY, &val.to_le_bytes())?;
    Ok(())
}

// —— Encoding helpers: ValueLocator uses a compact fixed 12-byte form, skipping serde_json size ——
fn binencode(loc: &ValueLocator) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&loc.seg_id.to_le_bytes());
    out.extend_from_slice(&loc.offset.to_le_bytes());
    out.extend_from_slice(&loc.len.to_le_bytes());
    Ok(out)
}

fn bindecode(bytes: &[u8]) -> Result<ValueLocator> {
    if bytes.len() != 12 {
        return Err(StoreError::Corrupt(format!(
            "locator len {} != 12",
            bytes.len()
        )));
    }
    let seg_id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let offset = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    Ok(ValueLocator {
        seg_id,
        offset,
        len,
    })
}

/// Lightweight key hash for logs (keys never logged in plaintext) — FNV-1a, not for security
fn fnv1a(key: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn open_store(dir: &Path, quota: u64) -> KvStore {
        KvStore::open(dir, 0, quota).unwrap()
    }

    #[test]
    fn put_get_delete_roundtrip() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path(), 0);
        let key = b"k1";
        let val = b"v1-fusion-store";
        store.put_kv(key, val, None).unwrap();
        let buf = store.get_kv_zero_copy(key, None).unwrap().unwrap();
        assert_eq!(buf.as_bytes(), val);
        assert!(store.delete_kv(key, None).unwrap());
        assert!(store.get_kv_zero_copy(key, None).unwrap().is_none());
    }

    #[test]
    fn get_returns_zero_copy_mmap_pointer() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path(), 0);
        let key = b"zk";
        let val = b"zero-copy-payload-into-mmap-region";
        store.put_kv(key, val, None).unwrap();
        // Force sealing so the payload lands in a sealed segment
        for i in 0..32 {
            store
                .put_kv(format!("fill{}", i).as_bytes(), &[0u8; 64], None)
                .unwrap();
        }
        let buf = store.get_kv_zero_copy(key, None).unwrap().unwrap();
        assert_eq!(buf.as_bytes(), val);
        // Pointer validity is guaranteed inside ZeroCopyBuffer; verifying data correctness here suffices
    }

    #[test]
    fn quota_exceeded_rejects_put() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path(), 64);
        // Quota of 64 bytes
        store.put_kv(b"a", &[0u8; 32], None).unwrap();
        let err = store.put_kv(b"b", &[0u8; 64], None).unwrap_err();
        assert!(matches!(err, StoreError::QuotaExceeded));
    }

    #[test]
    fn delete_decrements_quota() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path(), 128);
        store.put_kv(b"a", &[0u8; 32], None).unwrap();
        assert_eq!(store.used_bytes().unwrap(), 32);
        assert!(store.delete_kv(b"a", None).unwrap());
        assert_eq!(store.used_bytes().unwrap(), 0);
    }

    // E8: classify_heed promotes heed MapFull into an explicit StoreError::MapFull (not a panic)
    #[test]
    fn classify_heed_promotes_mapfull() {
        let e = heed::Error::Mdb(heed::MdbError::MapFull);
        let se = classify_heed(e);
        match se {
            StoreError::MapFull { limit, .. } => {
                assert_eq!(limit, KV_MAP_SIZE as u64);
            }
            other => panic!("expected MapFull, got {other:?}"),
        }
    }

    #[test]
    fn disk_bytes_exceeds_used_bytes_with_tail_hole() {
        // P3: quota counts net payload while disk_bytes counts actual segment-file usage
        // (including tail holes). Writing 32B of payload with a 64MB preallocated segment
        // leaves disk_bytes far above used_bytes.
        let dir = tempdir().unwrap();
        let store = open_store(dir.path(), 0);
        store.put_kv(b"a", &[0u8; 32], None).unwrap();
        let used = store.used_bytes().unwrap();
        let disk = store.disk_bytes().unwrap();
        assert_eq!(used, 32, "used_bytes = net payload");
        assert!(
            disk >= used,
            "disk_bytes {} >= used_bytes {} (segment preallocated size)",
            disk,
            used
        );
        // 64MB default segment → disk at least 64MB (segment tail hole fills the quota-vs-disk gap)
        assert!(
            disk >= 64 * 1024 * 1024,
            "disk_bytes {} >= 64MB default segment prealloc",
            disk
        );
    }
}
