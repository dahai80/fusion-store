//! mmap 内存管理层 —— 段 append-only 不可变 (H2/H4)
//!
//! 核心约束：每个 mmap 段写满封存即只读，永不 truncate/grow/就地改。
//! reader 持 ZeroCopyBuffer 期间段永不被改大小，根治 SIGBUS 与并发撕裂。

use std::fs::File;
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};

use crate::error::{Result, StoreError};

/// mmap 段句柄 —— sealed=true 后只读、永不改大小 [v2 H4]
///
/// Arc<MmapHandle> 保活即指针永久有效。
pub struct MmapHandle {
    // file 保活底层文件，防 truncate；不直接读
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
    base: usize,
    sealed: bool,
}

// 不可变只读段可跨线程共享
unsafe impl Send for MmapHandle {}
unsafe impl Sync for MmapHandle {}

impl MmapHandle {
    /// 打开只读映射（封存段）
    pub fn open_read(file: File) -> Result<Self> {
        let mmap = unsafe { Mmap::map(&file)? };
        let len = mmap.len();
        tracing::debug!(len, "mmap read segment opened");
        Ok(Self {
            file,
            mmap,
            base: 0,
            sealed: true,
        })
    }

    /// 从已有 mmap 映射构造（segment.rs 封存段复用已 flush 的映射）
    pub fn from_parts(file: File, mmap: Mmap) -> Self {
        Self {
            file,
            mmap,
            base: 0,
            sealed: false,
        }
    }

    /// 打开读写映射（当前写段，单写者独占）
    pub fn open_write(file: File) -> Result<Self> {
        let mmap_mut = unsafe { MmapMut::map_mut(&file)? };
        tracing::debug!(len = mmap_mut.len(), "mmap write segment opened");
        // F7：旧实现 transmute MmapMut→Mmap 是 UB（两类型布局无保证）。
        // memmap2 0.9 make_read_only 消费 MmapMut 转 Mmap，安全且语义对（写路径在
        // segment.rs 单独持 MmapMut 写活跃段，此处仅开只读快照供读侧）。
        let mmap = mmap_mut.make_read_only()?;
        Ok(Self {
            file,
            mmap,
            base: 0,
            sealed: false,
        })
    }

    /// 封存：写满后转只读，永不再写 [v2 H4]
    pub fn seal(&mut self) {
        self.sealed = true;
        tracing::debug!(base = self.base, "segment sealed (read-only forever)");
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// 返回段内字节切片（零拷贝，段不可变保有效）
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }
}

/// 零拷贝 Buffer —— 持有 Arc<MmapHandle> 保活映射 [v2 H4]
///
/// 段 append-only 不可变 → 指针永久有效，无 SIGBUS。
/// 需 owned 数据时显式 to_owned_slice()。
pub struct ZeroCopyBuffer {
    ptr: *const u8,
    len: usize,
    // mmap_handle 保活映射段，段不可变保指针有效；借用不可见故 allow dead_code
    #[allow(dead_code)]
    mmap_handle: Arc<MmapHandle>,
}

/// 零拷贝 f32 向量切片 —— 持 Arc<MmapHandle> 保活映射 [v2 H1/H4]
///
/// 向量距离热路径用：每跳读 mmap 段算距离，零分配零拷贝（非 Vec<f32>）。
/// 段 append-only 不可变保指针永久有效，Arc 保活映射。
pub struct ZeroCopyVec {
    ptr: *const f32,
    len: usize,
    // mmap_handle 保活映射段，段不可变保指针有效
    #[allow(dead_code)]
    mmap_handle: Arc<MmapHandle>,
}

unsafe impl Send for ZeroCopyVec {}
unsafe impl Sync for ZeroCopyVec {}

impl ZeroCopyVec {
    // F6：bounds 校验免越界。返回 Result，offset+len 超 mmap 长度 → Corrupt。
    pub fn new(mmap_handle: Arc<MmapHandle>, offset: usize, len: usize) -> Result<Self> {
        let region_len = mmap_handle.len();
        // offset 对齐 f32（段 append 按 byte_len=dim*4 对齐，offset 始终 4 倍数）
        let byte_len = len
            .checked_mul(4)
            .ok_or_else(|| StoreError::Corrupt(format!("vector len {} overflow byte size", len)))?;
        let end = offset.checked_add(byte_len).ok_or_else(|| {
            StoreError::Corrupt(format!(
                "vec region offset {} + {} overflow usize",
                offset, byte_len
            ))
        })?;
        if end > region_len {
            tracing::error!(
                offset,
                byte_len,
                region_len,
                "zero-copy vec out of mmap bounds"
            );
            return Err(StoreError::Corrupt(format!(
                "zero-copy vec [{}..{}] out of mmap region len {}",
                offset, end, region_len
            )));
        }
        let base = mmap_handle.as_bytes().as_ptr();
        let ptr = unsafe { base.add(offset) as *const f32 };
        Ok(Self {
            ptr,
            len,
            mmap_handle,
        })
    }

    /// 零拷贝读 f32 切片 —— SIMD 距离直接用，无 Vec 分配
    pub fn as_f32(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

unsafe impl Send for ZeroCopyBuffer {}
unsafe impl Sync for ZeroCopyBuffer {}

impl ZeroCopyBuffer {
    // F6：bounds 校验免越界。返回 Result，offset+len 超 mmap 长度 → Corrupt。
    pub fn new(mmap_handle: Arc<MmapHandle>, offset: usize, len: usize) -> Result<Self> {
        let region_len = mmap_handle.len();
        let end = offset.checked_add(len).ok_or_else(|| {
            StoreError::Corrupt(format!(
                "buffer region offset {} + {} overflow usize",
                offset, len
            ))
        })?;
        if end > region_len {
            tracing::error!(
                offset,
                len,
                region_len,
                "zero-copy buffer out of mmap bounds"
            );
            return Err(StoreError::Corrupt(format!(
                "zero-copy buffer [{}..{}] out of mmap region len {}",
                offset, end, region_len
            )));
        }
        let base = mmap_handle.as_bytes().as_ptr();
        let ptr = unsafe { base.add(offset) };
        Ok(Self {
            ptr,
            len,
            mmap_handle,
        })
    }

    /// 零拷贝读字节切片
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 显式拷贝为 owned —— 需 owned 数据时调用
    pub fn to_owned_slice(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn zero_copy_buffer_reads_mmap_region() {
        let mut tmp = NamedTempFile::new().unwrap();
        let payload = b"hello-fusion-store-zero-copy";
        std::io::Write::write_all(&mut tmp, payload).unwrap();
        let handle = Arc::new(MmapHandle::open_read(tmp.into_file()).unwrap());
        let buf = ZeroCopyBuffer::new(handle.clone(), 0, payload.len()).unwrap();
        assert_eq!(buf.as_bytes(), payload);
        assert!(!buf.is_empty());
        assert_eq!(buf.len(), payload.len());
        // 零拷贝断言：指针落在 mmap 区域内
        let region = handle.as_bytes().as_ptr();
        let buf_ptr = buf.as_bytes().as_ptr();
        assert!(buf_ptr >= region && buf_ptr <= unsafe { region.add(handle.len()) });
    }

    #[test]
    fn sealed_flag_is_set() {
        let mut tmp = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut tmp, b"data").unwrap();
        let mut handle = MmapHandle::open_write(tmp.into_file()).unwrap();
        assert!(!handle.is_sealed());
        handle.seal();
        assert!(handle.is_sealed());
    }
}
