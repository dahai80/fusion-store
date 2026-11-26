//! mmap memory management layer — segments are append-only and immutable (H2/H4)
//!
//! Core constraint: every mmap segment becomes read-only once full and sealed, never
//! truncated/grown/modified in place. While a reader holds a ZeroCopyBuffer the segment
//! is never resized, eliminating SIGBUS and concurrent tearing at the root.

use std::fs::File;
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};

use crate::error::{Result, StoreError};

/// mmap segment handle — read-only and never resized once sealed=true [v2 H4]
///
/// Keeping Arc<MmapHandle> alive keeps pointers valid forever.
pub struct MmapHandle {
    // file keeps the underlying file alive to prevent truncation; never read directly
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
    base: usize,
    sealed: bool,
}

// Immutable read-only segments can be shared across threads
unsafe impl Send for MmapHandle {}
unsafe impl Sync for MmapHandle {}

impl MmapHandle {
    /// Open a read-only mapping (sealed segment)
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

    /// Build from an existing mmap mapping (segment.rs reuses already-flushed mappings for sealed segments)
    pub fn from_parts(file: File, mmap: Mmap) -> Self {
        Self {
            file,
            mmap,
            base: 0,
            sealed: false,
        }
    }

    /// Open a read-write mapping (active write segment, exclusive single writer)
    pub fn open_write(file: File) -> Result<Self> {
        let mmap_mut = unsafe { MmapMut::map_mut(&file)? };
        tracing::debug!(len = mmap_mut.len(), "mmap write segment opened");
        // F7: the old implementation transmuted MmapMut→Mmap, which is UB (no layout
        // guarantee between the two types). memmap2 0.9 make_read_only consumes MmapMut
        // into Mmap — safe and semantically correct (the write path in segment.rs keeps
        // its own MmapMut for the active segment; this only opens a read-only snapshot
        // for readers).
        let mmap = mmap_mut.make_read_only()?;
        Ok(Self {
            file,
            mmap,
            base: 0,
            sealed: false,
        })
    }

    /// Seal: flip to read-only once full, never written again [v2 H4]
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

    /// Return the byte slice inside the segment (zero-copy, valid while the segment stays immutable)
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }
}

/// Zero-copy buffer — holds Arc<MmapHandle> to keep the mapping alive [v2 H4]
///
/// Segments are append-only and immutable, so pointers stay valid forever — no SIGBUS.
/// Call to_owned_slice() explicitly when owned data is needed.
pub struct ZeroCopyBuffer {
    ptr: *const u8,
    len: usize,
    // mmap_handle keeps the mapped segment alive; immutability keeps pointers valid.
    // Invisible to borrows, hence allow dead_code
    #[allow(dead_code)]
    mmap_handle: Arc<MmapHandle>,
}

/// Zero-copy f32 vector slice — holds Arc<MmapHandle> to keep the mapping alive [v2 H1/H4]
///
/// Used on the vector-distance hot path: each hop reads the mmap segment to compute
/// distances with zero allocation and zero copy (not a Vec<f32>).
/// Append-only immutable segments keep pointers valid forever; Arc keeps the mapping alive.
pub struct ZeroCopyVec {
    ptr: *const f32,
    len: usize,
    // mmap_handle keeps the mapped segment alive; immutability keeps pointers valid
    #[allow(dead_code)]
    mmap_handle: Arc<MmapHandle>,
}

unsafe impl Send for ZeroCopyVec {}
unsafe impl Sync for ZeroCopyVec {}

impl ZeroCopyVec {
    // F6: bounds validation prevents out-of-bounds access. Returns Result; offset+len
    // beyond the mmap length yields Corrupt.
    pub fn new(mmap_handle: Arc<MmapHandle>, offset: usize, len: usize) -> Result<Self> {
        let region_len = mmap_handle.len();
        // offset is f32-aligned (segment append uses byte_len=dim*4, so offset is always a multiple of 4)
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

    /// Zero-copy read of the f32 slice — used directly for SIMD distance, no Vec allocation
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
    // F6: bounds validation prevents out-of-bounds access. Returns Result; offset+len
    // beyond the mmap length yields Corrupt.
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

    /// Zero-copy read of the byte slice
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Explicitly copy to owned — call when owned data is needed
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
        // Zero-copy assertion: pointer lands inside the mmap region
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
