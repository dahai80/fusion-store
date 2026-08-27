//! append-only payload 段管理 —— KV 大 value 落 mmap 段 [v2 H4/E6]
//!
//! 段固定预分配容量，写满封存只读，永不可变（H4）。
//! 当前写段 MmapMut 追加；封存段 Mmap 只读，经 heed 定位信息读。
//! 无 free-list 就地复用——空洞由 compact 整段重写回收（M4）。

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};

use crate::error::Result;
use crate::mem::mmap::MmapHandle;

/// 单段默认容量（64MB）—— 写满封存开新段
pub const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

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
}

impl SegmentPool {
    /// 创建/打开段池。打开时定位最后一个未封存段为 active，或开新段。
    pub fn open(data_dir: &Path, seg_size: u64) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let seg_size = if seg_size == 0 {
            DEFAULT_SEGMENT_SIZE
        } else {
            seg_size
        };
        let active_id = next_seg_id(data_dir)?;
        let path = data_dir.join(segment_file(active_id));
        let file = open_or_create_segment(&path, seg_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        // cursor 落已写字节末尾（追加点）；打开时扫文件 size
        let cursor = written_len(&mmap);
        tracing::info!(seg_id = active_id, cursor, seg_size, "segment pool opened");
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            seg_size,
            active_id,
            active_file: file,
            active_mmap: mmap,
            active_cursor: cursor,
            sealed: Default::default(),
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

    /// 写入 payload —— 返回定位信息。写满自动封存开新段。
    pub fn append(&mut self, payload: &[u8]) -> Result<ValueLocator> {
        let need = payload.len() as u64;
        if need > self.remaining() {
            self.seal_and_rotate()?;
        }
        let seg_id = self.active_id;
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
        // 开新写段
        let new_id = sealed_id + 1;
        let path = self.data_dir.join(segment_file(new_id));
        let file = open_or_create_segment(&path, self.seg_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        self.active_id = new_id;
        self.active_file = file;
        self.active_mmap = mmap;
        self.active_cursor = 0;
        Ok(())
    }

    /// 取封存段只读 MmapHandle —— reader 经 locator 构造 ZeroCopyBuffer
    pub fn sealed_handle(&mut self, seg_id: u32) -> Result<Arc<MmapHandle>> {
        if let Some(h) = self.sealed.get(&seg_id) {
            return Ok(h.clone());
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
        Ok(arc)
    }

    /// 同步落盘（checkpoint 调用，M4）
    pub fn flush(&mut self) -> Result<()> {
        self.active_mmap.flush()?;
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
/// M1 段打开时 cursor 由 caller 不持久化，重启后 active 段视为空——
/// 配合 WAL 幂等重放（M4）恢复 cursor。此处返回 0 占位。
fn written_len(_mmap: &MmapMut) -> u64 {
    // M1 简化：新开 active 段从 0 起。重启 active cursor 恢复依赖 M4 WAL。
    0
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
        );
        assert_eq!(buf.as_bytes(), payload);
        // 零拷贝断言：指针落在 mmap 区域内
        let region = handle.as_bytes().as_ptr();
        let buf_ptr = buf.as_bytes().as_ptr();
        assert!(buf_ptr >= region && buf_ptr <= unsafe { region.add(handle.len()) });
    }
}
