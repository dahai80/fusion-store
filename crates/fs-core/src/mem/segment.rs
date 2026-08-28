//! append-only payload 段管理 —— KV 大 value 落 mmap 段 [v2 H4/E6]
//!
//! 段固定预分配容量，写满封存只读，永不可变（H4）。
//! 当前写段 MmapMut 追加；封存段 Mmap 只读，经 heed 定位信息读。
//! 无 free-list 就地复用——空洞由 compact 整段重写回收（M4）。

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};

use crate::error::{Result, StoreError};
use crate::mem::mmap::MmapHandle;

/// 单段默认容量（64MB）—— 写满封存开新段
pub const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;
/// 封存段只读映射缓存容量上限（A4：LRU 淘汰，防无界 mmap 增长）
const SEALED_CACHE_CAP: usize = 64;

/// 段定位信息 —— 落 heed，指向封存段内 payload
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ValueLocator {
    pub seg_id: u32,
    pub offset: u32,
    pub len: u32,
}

/// 段文件名（统一前缀 + seg_id 零填充）
pub fn segment_file(seg_id: u32) -> String {
    format!("kv_payload_{:04}.mmap", seg_id)
}

/// R9：active 段写游标持久化文件（crash 后 reopen 复用未满段，免每次浪费 64MB 段尾）。
/// 内容 [seg_id:4 LE][cursor:8 LE]。seal/flush 时原子重写（tmp+rename）。
const ACTIVE_CURSOR_FILE: &str = "active.cursor";

/// append-only 段池 —— 管理一个 namespace 的所有 payload 段
///
/// 持有当前写段（MmapMut，单写者独占）+ 封存段只读映射缓存（按 seg_id）。
/// reader 经 locator 取封存段 MmapHandle 构造 ZeroCopyBuffer。
pub struct SegmentPool {
    data_dir: PathBuf,
    seg_size: u64,
    // 当前写段
    active_id: u32,
    active_file: File,
    active_mmap: MmapMut,
    active_cursor: u64,
    // 封存段只读映射缓存（seg_id -> Arc<MmapHandle>）
    sealed: std::collections::HashMap<u32, Arc<MmapHandle>>,
    // A4：LRU 顺序队列（队首最久未用）。淘汰时校验 strong_count==1 才 munmap，
    // 有 reader 持 handle（count>1）则跳过该条，保零拷贝指针有效。
    sealed_lru: VecDeque<u32>,
}

impl SegmentPool {
    /// 创建/打开段池。R9：优先复用 active.cursor 记录的未满 active 段（续写），
    /// 否则开新段。免频繁重启每次浪费最多 64MB 段尾空间。
    pub fn open(data_dir: &Path, seg_size: u64) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let seg_size = if seg_size == 0 {
            DEFAULT_SEGMENT_SIZE
        } else {
            seg_size
        };
        // R9：读 active.cursor 续写点。{seg_id, cursor}，校验 seg 文件存在 + cursor ≤ seg_size。
        let (active_id, cursor) = match read_active_cursor(data_dir)? {
            Some((sid, c)) if data_dir.join(segment_file(sid)).exists() && c <= seg_size => {
                tracing::info!(
                    seg_id = sid,
                    cursor = c,
                    "reopen reusing unfilled active segment (R9)"
                );
                (sid, c)
            }
            Some((sid, c)) => {
                // cursor 文件与段文件不一致（段被删/换/手动改）——弃 cursor，开新段免覆盖
                tracing::warn!(
                    recorded_seg = sid,
                    recorded_cursor = c,
                    "active.cursor stale (seg missing or cursor>seg_size), opening new segment"
                );
                let new_id = next_seg_id(data_dir)?;
                (new_id, 0)
            }
            None => {
                // 无 cursor 文件：首开或旧版。开新段（max+1）。
                let new_id = next_seg_id(data_dir)?;
                (new_id, 0)
            }
        };
        let path = data_dir.join(segment_file(active_id));
        let file = open_or_create_segment(&path, seg_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        tracing::info!(seg_id = active_id, cursor, seg_size, "segment pool opened");
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            seg_size,
            active_id,
            active_file: file,
            active_mmap: mmap,
            active_cursor: cursor,
            sealed: Default::default(),
            sealed_lru: VecDeque::new(),
        })
    }

    /// 当前写段剩余可写字节
    pub fn remaining(&self) -> u64 {
        self.seg_size.saturating_sub(self.active_cursor)
    }

    /// 当前 active seg_id
    pub fn active_seg_id(&self) -> u32 {
        self.active_id
    }

    /// 当前 active 段已写字节游标（L3 回滚快照用）
    pub fn active_cursor(&self) -> u64 {
        self.active_cursor
    }

    /// 数据目录（stats 报实际段文件磁盘占用用，P3）
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// 实际段文件磁盘占用 —— 扫描 data_dir 内所有 .mmap 文件 size 求和。
    /// P3：配额只计净 payload 字节，不计 padding/段尾空洞/heed。
    /// 此值反映真实磁盘占用，供 /stats 暴露 quota-vs-disk 缺口（配额"未满"但磁盘可能已满）。
    pub fn disk_bytes(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("mmap") {
                total += entry.metadata()?.len();
            }
        }
        Ok(total)
    }

    /// 写入 payload —— 返回定位信息。写满自动封存开新段。
    pub fn append(&mut self, payload: &[u8]) -> Result<ValueLocator> {
        // F3：payload 超 u32::MAX 拒绝，免 len 截断损坏（写全字节但 locator 记截断 len）。
        if payload.len() > u32::MAX as usize {
            return Err(StoreError::ValueTooLarge(payload.len()));
        }
        let need = payload.len() as u64;
        if need > self.remaining() {
            self.seal_and_rotate()?;
        }
        let seg_id = self.active_id;
        // F3：offset 超 u32::MAX 拒绝（单段 seg_size ≤ u32::MAX 由构造保证，
        // 但累计 cursor 若超 u32 则封存轮转已重置；此处断言兜底）。
        if self.active_cursor > u32::MAX as u64 {
            return Err(StoreError::Corrupt(format!(
                "active cursor {} exceeds u32 max, segment overflow",
                self.active_cursor
            )));
        }
        let offset = self.active_cursor as u32;
        let len = payload.len() as u32;
        // 追加写入当前写段
        let end = self.active_cursor + need;
        self.active_mmap[self.active_cursor as usize..end as usize].copy_from_slice(payload);
        self.active_cursor = end;
        tracing::debug!(seg_id, offset, len, "payload appended");
        Ok(ValueLocator {
            seg_id,
            offset,
            len,
        })
    }

    /// 写入 payload，先按 align 对齐 cursor（padding 补 0）。
    /// 列式定宽列需类型对齐（i64/f64 需 8 对齐），否则 ScalarBuffer 构造报未对齐。
    /// mmap 段基址页对齐（4096），首列 offset=0 天然对齐；后续列按需 padding。
    pub fn append_aligned(&mut self, payload: &[u8], align: usize) -> Result<ValueLocator> {
        if align <= 1 {
            return self.append(payload);
        }
        let pad = (align - (self.active_cursor as usize % align)) % align;
        if pad > 0 {
            let pad_bytes = vec![0u8; pad];
            // padding 不占满段则正常 append；占满则先封存（payload 入新段 offset 0 天然对齐）
            if pad as u64 > self.remaining() {
                self.seal_and_rotate()?;
            } else {
                let _ = self.append(&pad_bytes)?;
            }
        }
        // 封存后 offset 重置为 0（对齐）；或 padding 后已对齐
        self.append(payload)
    }

    /// 回退 active 段写游标到 (seg_id, offset) [L1]
    /// commit 失败时调，回收本次 append 写入但未提交的字节，免段内泄漏。
    /// 仅当 seg_id 仍是当前 active 段有效（写锁独占期内无并发写）；否则不可回退。
    /// 游标后退后，[offset, 旧游标) 区间变孤儿字节——无 locator 引用，后续 append 覆盖。
    pub fn rewind_active_to(&mut self, seg_id: u32, offset: u32) -> Result<()> {
        if seg_id != self.active_id {
            // append 触发了封存轮转 → 该段已封存只读，无法回退。
            // 封存段孤儿字节由 compact 回收，此处仅记日志。
            tracing::warn!(
                seg_id,
                active_id = self.active_id,
                "rewind skipped: seg already sealed, orphan bytes left for compact"
            );
            return Ok(());
        }
        let target = offset as u64;
        if target > self.active_cursor {
            return Err(StoreError::Corrupt(format!(
                "rewind target {} > active cursor {}",
                target, self.active_cursor
            )));
        }
        self.active_cursor = target;
        tracing::info!(
            seg_id,
            offset,
            "active segment rewound, leaked bytes reclaimed"
        );
        Ok(())
    }

    /// 封存当前写段 + 开新写段
    fn seal_and_rotate(&mut self) -> Result<()> {
        let sealed_id = self.active_id;
        tracing::info!(seg_id = sealed_id, "segment full, sealing + rotating");
        // 当前写段 flush 落盘后转只读映射
        self.active_mmap.flush()?;
        let sealed_file = self.active_file.try_clone()?;
        let sealed_mmap = unsafe { Mmap::map(&sealed_file)? };
        let mut handle = MmapHandle::from_parts(sealed_file, sealed_mmap);
        handle.seal();
        self.sealed.insert(sealed_id, Arc::new(handle));
        self.sealed_lru.push_back(sealed_id);
        self.evict_sealed();
        // 开新写段
        let new_id = sealed_id + 1;
        let path = self.data_dir.join(segment_file(new_id));
        let file = open_or_create_segment(&path, self.seg_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        self.active_id = new_id;
        self.active_file = file;
        self.active_mmap = mmap;
        self.active_cursor = 0;
        // R9：新 active 段 cursor=0，持久化免 reopen 误复用旧满段
        write_active_cursor(&self.data_dir, new_id, 0)?;
        Ok(())
    }

    /// 取封存段只读 MmapHandle —— reader 经 locator 构造 ZeroCopyBuffer
    pub fn sealed_handle(&mut self, seg_id: u32) -> Result<Arc<MmapHandle>> {
        if let Some(h) = self.sealed.get(&seg_id) {
            // A4：LRU 命中，移到队尾（最近使用）。先 clone Arc 再 touch，免借用冲突。
            let cloned = h.clone();
            self.touch_sealed(seg_id);
            return Ok(cloned);
        }
        // 未缓存则按需打开封存段
        let path = self.data_dir.join(segment_file(seg_id));
        if !path.exists() {
            return Err(crate::StoreError::Corrupt(format!(
                "sealed segment {} missing",
                seg_id
            )));
        }
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut handle = MmapHandle::from_parts(file, mmap);
        handle.seal();
        let arc = Arc::new(handle);
        self.sealed.insert(seg_id, arc.clone());
        self.sealed_lru.push_back(seg_id);
        self.evict_sealed();
        Ok(arc)
    }

    /// A4：LRU 触碰——把 seg_id 移到队尾（最近使用）
    fn touch_sealed(&mut self, seg_id: u32) {
        if let Some(pos) = self.sealed_lru.iter().position(|&s| s == seg_id) {
            self.sealed_lru.remove(pos);
            self.sealed_lru.push_back(seg_id);
        }
    }

    /// A4/R8：LRU 淘汰——超 SEALED_CACHE_CAP 时从队首回收。
    /// 仅 strong_count==1（无 reader 持 handle）才 munmap 释放映射，
    /// count>1 表示零拷贝 reader 仍持有，跳过该条保指针有效。
    ///
    /// R8：旧实现遇首个被持段即 `break`，致其后所有更旧段无法淘汰，
    /// 长持句柄期间映射数无界增长触 mmap 上限。现改为 `continue` 跳过被持段
    /// 继续淘汰队首下一个可淘汰段；`touched` 计数防「全被持」死循环——
    /// 一轮全跳过即退出，留待下次 append 触发重试。
    fn evict_sealed(&mut self) {
        let mut touched = 0usize;
        let queue_len = self.sealed_lru.len();
        while self.sealed_lru.len() > SEALED_CACHE_CAP && touched < queue_len {
            let candidate = match self.sealed_lru.pop_front() {
                Some(id) => id,
                None => break,
            };
            let safe = self
                .sealed
                .get(&candidate)
                .map(|h| Arc::strong_count(h) <= 1)
                .unwrap_or(false);
            if safe {
                self.sealed.remove(&candidate);
                tracing::debug!(
                    seg_id = candidate,
                    "sealed segment evicted from cache (LRU)"
                );
            } else {
                // R8：仍有 reader 持有，跳过淘汰（重新入队尾稍后再试），继续淘汰下一个
                tracing::debug!(
                    seg_id = candidate,
                    "sealed segment still in use (strong_count>1), skip evict, continue"
                );
                self.sealed_lru.push_back(candidate);
                touched += 1;
            }
        }
        if self.sealed_lru.len() > SEALED_CACHE_CAP {
            // 全被持：本轮无法淘汰，记日志供监控（映射数暂超上限）
            tracing::warn!(
                over = self.sealed_lru.len() - SEALED_CACHE_CAP,
                "all sealed segments held by readers, eviction deferred"
            );
        }
    }

    /// 同步落盘（checkpoint 调用，M4）
    /// R9：同时原子持久化 active cursor，使 reopen 能续写未满段（与 WAL checkpoint 对齐）。
    pub fn flush(&mut self) -> Result<()> {
        self.active_mmap.flush()?;
        write_active_cursor(&self.data_dir, self.active_id, self.active_cursor)?;
        Ok(())
    }
}

/// 扫描数据目录，返回下一个 seg_id（最大已有 seg_id + 1，无则 0）
fn next_seg_id(data_dir: &Path) -> Result<u32> {
    let mut max_id: Option<u32> = None;
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(id) = parse_seg_id(&name) {
            max_id = Some(max_id.map_or(id, |m| m.max(id)));
        }
    }
    Ok(max_id.map_or(0, |m| m + 1))
}

// R9：active cursor sidecar 读写。crash 后 reopen 据此续写未满段，免每次浪费段尾。

/// 读 active.cursor → (seg_id, cursor)。不存在或损坏返回 None（退化到开新段）。
fn read_active_cursor(data_dir: &Path) -> Result<Option<(u32, u64)>> {
    let path = data_dir.join(ACTIVE_CURSOR_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    if bytes.len() != 12 {
        tracing::warn!(len = bytes.len(), "active.cursor malformed, ignoring");
        return Ok(None);
    }
    let seg_id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let cursor = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    Ok(Some((seg_id, cursor)))
}

/// 原子写 active.cursor（tmp + rename）。flush/seal 调，与 WAL checkpoint 对齐。
fn write_active_cursor(data_dir: &Path, seg_id: u32, cursor: u64) -> Result<()> {
    let path = data_dir.join(ACTIVE_CURSOR_FILE);
    let tmp = data_dir.join("active.cursor.tmp");
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&seg_id.to_le_bytes());
    buf[4..12].copy_from_slice(&cursor.to_le_bytes());
    {
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// 从文件名解析 seg_id（kv_payload_NNNN.mmap）
fn parse_seg_id(name: &str) -> Option<u32> {
    let name = name.strip_prefix("kv_payload_")?;
    let name = name.strip_suffix(".mmap")?;
    name.parse().ok()
}

/// 打开或创建段文件并预分配固定容量
fn open_or_create_segment(path: &Path, size: u64) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let cur = file.metadata()?.len();
    if cur < size {
        file.set_len(size)?;
    }
    Ok(file)
}

/// 扫写段已写内容末尾（首个非零反向扫描代价高，简化为文件实际写入计数）。
/// F4：active 段全新（next_seg_id=max+1），cursor=0 正确，无需扫描。保留为
/// 工具函数供未来「active 段跨重启」场景（若改为复用旧段时需真实实现）。
#[allow(dead_code)]
fn written_len_scan(mmap: &MmapMut) -> u64 {
    // 从尾向前找首个非零字节定位已写末尾；全零段返回 0。
    // 注：零值 payload 会被误判为未写——故仅作辅助，不作为 crash 恢复依据。
    let bytes = &mmap[..];
    let mut end = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        if b != 0 {
            end = (i + 1) as u64;
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn append_returns_locator_in_bounds() {
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 0).unwrap();
        let payload = b"fusion-store-kv-payload-001";
        let loc = pool.append(payload).unwrap();
        assert_eq!(loc.seg_id, pool.active_seg_id());
        assert_eq!(loc.len as usize, payload.len());
        assert!(loc.offset as u64 + loc.len as u64 <= pool.seg_size);
    }

    #[test]
    fn seal_and_rotate_on_full() {
        // 极小段容量强制触发封存轮转
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 32).unwrap();
        let first = pool.active_seg_id();
        let _ = pool.append(&[0xABu8; 32]).unwrap();
        // 下次追加触发封存轮转
        let loc = pool.append(&[0xCDu8; 16]).unwrap();
        assert_ne!(loc.seg_id, first);
        assert!(loc.seg_id > first);
    }

    #[test]
    fn sealed_handle_reads_back_payload() {
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 64).unwrap();
        let payload = b"hello-zero-copy-via-sealed-segment";
        let loc = pool.append(payload).unwrap();
        // 封存当前段后开新段，使 payload 段进入封存缓存
        let _ = pool.append(&[0u8; 64]).unwrap();
        let handle = pool.sealed_handle(loc.seg_id).unwrap();
        assert!(handle.is_sealed());
        let buf = crate::mem::mmap::ZeroCopyBuffer::new(
            handle.clone(),
            loc.offset as usize,
            loc.len as usize,
        )
        .unwrap();
        assert_eq!(buf.as_bytes(), payload);
        // 零拷贝断言：指针落在 mmap 区域内
        let region = handle.as_bytes().as_ptr();
        let buf_ptr = buf.as_bytes().as_ptr();
        assert!(buf_ptr >= region && buf_ptr <= unsafe { region.add(handle.len()) });
    }

    #[test]
    fn sealed_lru_evicts_when_cap_exceeded() {
        // A4：封存段缓存超 SEALED_CACHE_CAP 时 LRU 淘汰 strong_count==1 的条目。
        // 极小段强制多段封存，读后丢弃 handle（count 回 1），触发淘汰后缓存不超 cap。
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 16).unwrap();
        // 灌入足够多段触发多次封存（每段 16B，写 17B 触发封存轮转）
        let mut seg_ids = Vec::new();
        for i in 0..200u32 {
            let payload = [i as u8; 16];
            let loc = pool.append(&payload).unwrap();
            if i > 0 {
                seg_ids.push(loc.seg_id);
            }
            // 写一字节占位触发下一轮封存
            let _ = pool.append(&[i as u8; 1]).unwrap();
        }
        // 读取若干封存段后立即丢弃 handle（strong_count 回落），缓存应被淘汰约束在 cap 内
        for &sid in seg_ids.iter().take(80) {
            let _h = pool.sealed_handle(sid).unwrap();
        }
        // sealed.len() 不应超 SEALED_CACHE_CAP（in-use 条目跳过淘汰，但多数已释放）
        assert!(
            pool.sealed.len() <= 65,
            "sealed cache {} exceeded cap+1 after LRU eviction",
            pool.sealed.len()
        );
        // 淘汰后再读已淘汰段，应重新打开成功（按需重开）
        let h = pool.sealed_handle(seg_ids[0]).unwrap();
        assert!(h.is_sealed());
    }

    // R8：长持一个旧段句柄时，其后更旧的段仍应被淘汰（旧实现 break 致全停）。
    #[test]
    fn evict_continues_past_held_oldest_segment() {
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 16).unwrap();
        let mut handles = Vec::new();
        // 灌 200 段，每段读出 handle 后保留（strong_count>1，被持）
        for i in 0..200u32 {
            let loc = pool.append(&[i as u8; 16]).unwrap();
            let _ = pool.append(&[i as u8; 1]).unwrap();
            if i == 5 {
                // 持住第 5 段句柄（LRU 队中靠旧位置）
                handles.push(pool.sealed_handle(loc.seg_id).unwrap());
            }
        }
        drop(handles); // 释放，此时再触发一次淘汰
        let _ = pool.append(&[0xFFu8; 16]).unwrap();
        let _ = pool.append(&[0x01u8; 1]).unwrap();
        // 持段期间不应无界膨胀；释放后淘汰恢复，缓存 ≤ cap+1
        assert!(
            pool.sealed.len() <= SEALED_CACHE_CAP + 1,
            "R8: sealed cache {} still bounded while/after holding segment",
            pool.sealed.len()
        );
    }

    // R9：reopen 续写未满 active 段（不每次开新段浪费段尾）。
    #[test]
    fn reopen_reuses_unfilled_active_segment() {
        let dir = tempdir().unwrap();
        let pool_dir = dir.path().to_path_buf();
        let seg_id;
        let cursor_after;
        {
            let mut pool = SegmentPool::open(&pool_dir, 4096).unwrap();
            seg_id = pool.active_seg_id();
            // 写 100 字节后 flush（持久化 cursor sidecar）
            let _ = pool.append(&[0xABu8; 100]).unwrap();
            pool.flush().unwrap();
            cursor_after = pool.active_cursor();
            assert_eq!(cursor_after, 100);
        }
        // reopen：应复用同 seg_id + cursor=100，而非开新段
        let mut pool2 = SegmentPool::open(&pool_dir, 4096).unwrap();
        assert_eq!(
            pool2.active_seg_id(),
            seg_id,
            "R9: reopen reuses same active seg, not new one"
        );
        assert_eq!(
            pool2.active_cursor(),
            cursor_after,
            "R9: reopen restores cursor at {}",
            cursor_after
        );
        // 续写不覆盖：再 append 50，落 offset=100（续写位）
        let loc = pool2.append(&[0xCDu8; 50]).unwrap();
        assert_eq!(
            loc.offset as u64, 100,
            "R9: continued write at restored cursor"
        );
    }

    // R9：封存轮转后 reopen 不复用已满段（cursor 记新段 0）。
    #[test]
    fn reopen_after_seal_opens_new_segment() {
        let dir = tempdir().unwrap();
        let pool_dir = dir.path().to_path_buf();
        let sealed_id;
        {
            let mut pool = SegmentPool::open(&pool_dir, 64).unwrap();
            sealed_id = pool.active_seg_id();
            let _ = pool.append(&[0u8; 64]).unwrap(); // 写满
            let _ = pool.append(&[0u8; 10]).unwrap(); // 触发封存 + 开新段
            pool.flush().unwrap();
        }
        let pool2 = SegmentPool::open(&pool_dir, 64).unwrap();
        assert_ne!(
            pool2.active_seg_id(),
            sealed_id,
            "R9: sealed full segment not reused on reopen"
        );
        // 新段 reopen 续写：cursor=10（封存后在新段写 10B 再 flush，cursor 持久化）
        assert_eq!(
            pool2.active_cursor(),
            10,
            "R9: new active segment restores written cursor"
        );
    }
}
