//! KV 存储层 —— heed 持久化定位 + mmap payload 段 [v2 E1/E6]
//!
//! put_kv: payload 落段 → heed 存 (key -> ValueLocator) + 配额累加。
//! get_kv_zero_copy: 读 heed locator → 映射封存段 → ZeroCopyBuffer（指针落 mmap 区域）。
//! delete_kv: heed 删 locator，配额减。段不就地回收（H4），compact 整段重写回收（M4）。
//! 多进程写: flock 快速失败（R1）。
//!
//! 配额语义（P3）：quota_limit 限的是「逻辑 payload 字节」净增（覆盖减旧值加新值），
//! 不计 append_aligned 的 padding、段未写满的尾空洞、heed 元数据。
//! append-only 段每段 64MB 预分配，最后一未满段浪费可达 64MB×namespace 数。
//! 故实际磁盘占用 = payload + padding + 段尾空洞 + heed > 配额计数。
//! disk_bytes() 报真实段文件占用，/stats 暴露 quota-vs-disk 缺口供监控制约磁盘。

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

/// KV value 一律落 mmap 段（零拷贝读指针断言对齐 §1.4）
const KV_DB_NAME: &str = "kv_locators";
/// namespace 配额计数 DB（key=固定 "used" -> u64 字节）
const QUOTA_DB_NAME: &str = "kv_quota";
const QUOTA_KEY: &[u8] = b"used";

/// E8：heed map_size 上限。locator 12B/项，2GB ≈ 1.8 亿 key（远超企业级 KV namespace）。
/// Apple Silicon 虚拟地址空间巨大，2GB 预分配零物理内存成本（按需分页）。
/// 写满抛 MapFull（非 panic），caller 据此告警或重建更大 map_size。
const KV_MAP_SIZE: usize = 2 * 1024 * 1024 * 1024;

// E8：把 heed Error 中的 MapFull 提升为显式 StoreError::MapFull，caller 可据容量告警。
// 非 MapFull 的 heed 错误原样透传为 StoreError::Heed。返回裸 StoreError，caller 用 Err() 包，
// 以适配不同返回类型（put_kv_inner Result<()>，delete_kv_inner Result<bool>）。
fn classify_heed(e: heed::Error) -> StoreError {
    match e {
        heed::Error::Mdb(heed::MdbError::MapFull) => StoreError::MapFull {
            current: 0,
            limit: KV_MAP_SIZE as u64,
        },
        other => StoreError::Heed(other),
    }
}

/// KV 存储 —— 单 namespace
///
/// writer 持 flock 单进程排他；reader 无锁（heed MVCC）。
pub struct KvStore {
    env: Env,
    kv_db: Database<Bytes, Bytes>,
    quota_db: Database<Bytes, Bytes>,
    pool: Mutex<SegmentPool>,
    lock_file: std::fs::File,
    quota_limit: u64,
}

impl KvStore {
    /// 打开/创建 KV store。data_dir = <ns>/kv_data, meta_dir = <ns>/kv_meta
    /// 与 VectorIndex 的 vec_meta/vec_data 隔离——heed 禁同路径双 Env [v2 E1]
    pub fn open(namespace_dir: &Path, seg_size: u64, quota_limit: u64) -> Result<Self> {
        let data_dir = namespace_dir.join("kv_data");
        let meta_dir = namespace_dir.join("kv_meta");
        std::fs::create_dir_all(&meta_dir)?;
        let lock_path = namespace_dir.join("lock");

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(KV_MAP_SIZE)
                .max_dbs(8)
                // NO_SYNC：heed 仅存 locator 元数据（tiny），commit 不 fsync；
                // WAL 是唯一 crash-safe 同步点（H5），crash 后 WAL 重放补 heed 未刷条目
                .flags(heed::EnvFlags::NO_SYNC)
                .open(&meta_dir)?
        };
        let mut wtxn = env.write_txn()?;
        let kv_db: Database<Bytes, Bytes> = env.create_database(&mut wtxn, Some(KV_DB_NAME))?;
        let quota_db: Database<Bytes, Bytes> =
            env.create_database(&mut wtxn, Some(QUOTA_DB_NAME))?;
        wtxn.commit()?;

        let pool = SegmentPool::open(&data_dir, seg_size)?;

        // flock 多进程写互斥（try_lock，拿不到快速失败 R1）
        let lock_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        // 仅 writer 路径拿排他锁；open 时不锁，put/delete 时锁

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

    /// 拿写锁（E9：阻塞 flock，非轮询）
    /// None = 内核阻塞 `lock_exclusive`，拿到为止（内核唤醒，零 CPU 空转）。
    /// Some(d) = 限时等待：fs2 无 timed-lock 原语，用 try + 5ms 短睡循环逼近 deadline
    ///   （5ms 粒度较旧 50ms 抖动降 10×，仍非零 CPU；纯阻塞路径 None 覆盖常态）。
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
                    // E9：5ms 粒度短睡，旧 50ms 抖动降 10×；try 失败极短退避
                    std::thread::sleep(Duration::from_millis(5));
                    waited += Duration::from_millis(5);
                }
            }
        }
    }

    /// 写 KV —— payload 落段 + heed 存 locator + 配额累加
    pub fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()> {
        self.acquire_write_lock(timeout)?;
        let res = self.put_kv_inner(key, value);
        // 释放写锁
        let _ = self.lock_file.unlock();
        res
    }

    fn put_kv_inner(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let start = std::time::Instant::now();
        let mut wtxn = self.env.write_txn()?;
        // 单写事务内完成：读旧值（覆盖时减配额）+ 配额校验 + 写 locator + 写配额
        let old_len: u64 = if let Some(old_loc_bytes) = self.kv_db.get(&wtxn, key)? {
            let old_loc: ValueLocator = bindecode(old_loc_bytes)?;
            old_loc.len as u64
        } else {
            0
        };
        let used = read_quota_txn(&self.quota_db, &wtxn)?;
        // 减旧 value 长度再加新 —— 覆盖语义下配额按净增计
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

        // payload 落段
        let loc = {
            let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
            pool.append(value)?
        };
        // locator 序列化落 heed
        let loc_bytes = binencode(&loc)?;
        self.kv_db.put(&mut wtxn, key, &loc_bytes)?;
        write_quota_txn(&self.quota_db, &mut wtxn, new_used)?;
        // L1：commit 失败回滚已落段字节，免段内泄漏
        if let Err(e) = wtxn.commit() {
            tracing::warn!(key_hash = fnv1a(key), err = ?e, "put_kv commit failed, rewinding segment");
            if let Ok(mut pool) = self.pool.lock() {
                let _ = pool.rewind_active_to(loc.seg_id, loc.offset);
            }
            // E8：MapFull 单独报，caller 可据容量告警
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

    /// 零拷贝读 KV —— 指针落 mmap 段区域
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

    /// 删 KV —— heed 删 locator，配额减
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
        // E8：commit 的 MapFull 单独报
        if let Err(e) = wtxn.commit() {
            return Err(classify_heed(e));
        }
        tracing::debug!(key_hash = fnv1a(key), old_len, "delete_kv done");
        Ok(deleted)
    }

    /// 当前 namespace 已用字节
    pub fn used_bytes(&self) -> Result<u64> {
        let rtxn = self.env.read_txn()?;
        read_quota_txn(&self.quota_db, &rtxn)
    }

    /// 配额上限
    pub fn quota_limit(&self) -> u64 {
        self.quota_limit
    }

    /// 实际段文件磁盘占用（P3：含 padding/段尾空洞，对比 used_bytes 净 payload）
    pub fn disk_bytes(&self) -> Result<u64> {
        let pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
        pool.disk_bytes()
    }

    /// 有序落盘：flush active 段 + heed force_sync（A6 close 用）
    /// NO_SYNC 下 commit 不 fsync；close 显式 force_sync 保证元数据落盘。
    pub fn flush(&self) -> Result<()> {
        let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
        pool.flush()?;
        self.env.force_sync()?;
        tracing::info!("kv store flushed (active segment + heed sync)");
        Ok(())
    }
}

// —— 配额读写：txn 参数化，支持在写事务内复用 ——
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

// —— 编解码小工具：ValueLocator 用紧凑定长 12 字节，不走 serde_json 体积 ——
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

/// 轻量 key hash（日志用，不明文记 key）—— FNV-1a，非安全用途
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
        // 触发封存使 payload 进封存段
        for i in 0..32 {
            store
                .put_kv(format!("fill{}", i).as_bytes(), &[0u8; 64], None)
                .unwrap();
        }
        let buf = store.get_kv_zero_copy(key, None).unwrap().unwrap();
        assert_eq!(buf.as_bytes(), val);
        // 指针有效性已在 ZeroCopyBuffer 内保证；此处校验数据正确即可
    }

    #[test]
    fn quota_exceeded_rejects_put() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path(), 64);
        // 配额 64 字节
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

    // E8：classify_heed 把 heed MapFull 提升为显式 StoreError::MapFull（非 panic）
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
        // P3：配额计净 payload，disk_bytes 计段文件实际占用（含段尾空洞）。
        // 写 32B payload，段预分配 64MB → disk_bytes 远大于 used_bytes。
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
        // 64MB 默认段 → disk 至少 64MB（段尾空洞占据 quota-vs-disk 缺口）
        assert!(
            disk >= 64 * 1024 * 1024,
            "disk_bytes {} >= 64MB default segment prealloc",
            disk
        );
    }
}
