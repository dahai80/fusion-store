//! Append-only payload segment management — KV large values land in mmap segments [v2 H4/E6]
//!
//! Segments have a fixed preallocated capacity; a full segment is sealed read-only and
//! never mutated (H4).
//! The active write segment appends via MmapMut; sealed segments are read-only Mmap,
//! read via heed locator info.
//! No free-list in-place reuse — holes are reclaimed by compact rewriting whole
//! segments (M4).

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};

use crate::error::{Result, StoreError};
use crate::mem::mmap::MmapHandle;

/// Default per-segment capacity (64MB) — full segments seal and a new one opens
pub const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;
/// Sealed-segment read-only mapping cache capacity cap (A4: LRU eviction prevents unbounded mmap growth)
const SEALED_CACHE_CAP: usize = 64;

/// Segment locator — persisted in heed, pointing at payload inside a sealed segment
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ValueLocator {
    pub seg_id: u32,
    pub offset: u32,
    pub len: u32,
}

/// Segment filename (uniform prefix + zero-padded seg_id)
pub fn segment_file(seg_id: u32) -> String {
    format!("kv_payload_{:04}.mmap", seg_id)
}

/// R9: persisted write-cursor file for the active segment (after a crash, reopen reuses
/// the unfilled segment instead of wasting up to 64MB of segment tail every time).
/// Content: [seg_id:4 LE][cursor:8 LE]. Atomically rewritten on seal/flush (tmp+rename).
const ACTIVE_CURSOR_FILE: &str = "active.cursor";

/// Append-only segment pool — manages all payload segments of one namespace
///
/// Holds the active write segment (MmapMut, exclusive single writer) + a read-only
/// mapping cache of sealed segments (keyed by seg_id).
/// Readers take a sealed segment's MmapHandle via the locator and build a ZeroCopyBuffer.
pub struct SegmentPool {
    data_dir: PathBuf,
    seg_size: u64,
    // Active write segment
    active_id: u32,
    active_file: File,
    active_mmap: MmapMut,
    active_cursor: u64,
    // Sealed-segment read-only mapping cache (seg_id -> Arc<MmapHandle>)
    sealed: std::collections::HashMap<u32, Arc<MmapHandle>>,
    // A4: LRU order queue (front = least recently used). Eviction only munmaps when
    // strong_count==1; entries still held by a reader (count>1) are skipped, keeping
    // zero-copy pointers valid.
    sealed_lru: VecDeque<u32>,
}

impl SegmentPool {
    /// Create/open the segment pool. R9: prefer resuming the unfilled active segment
    /// recorded in active.cursor (continue writing); otherwise open a new segment.
    /// Avoids wasting up to 64MB of segment tail on every frequent restart.
    pub fn open(data_dir: &Path, seg_size: u64) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let seg_size = if seg_size == 0 {
            DEFAULT_SEGMENT_SIZE
        } else {
            seg_size
        };
        // R9: read the active.cursor resume point {seg_id, cursor}; validate the seg file
        // exists + cursor <= seg_size.
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
                // Cursor file inconsistent with the segment file (segment deleted/swapped/
                // manually modified) — discard the cursor and open a new segment to avoid overwrite
                tracing::warn!(
                    recorded_seg = sid,
                    recorded_cursor = c,
                    "active.cursor stale (seg missing or cursor>seg_size), opening new segment"
                );
                let new_id = next_seg_id(data_dir)?;
                (new_id, 0)
            }
            None => {
                // No cursor file: first open or legacy layout. Open a new segment (max+1).
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

    /// Remaining writable bytes in the active write segment
    pub fn remaining(&self) -> u64 {
        self.seg_size.saturating_sub(self.active_cursor)
    }

    /// Current active seg_id
    pub fn active_seg_id(&self) -> u32 {
        self.active_id
    }

    /// Current active segment written-byte cursor (used for L3 rollback snapshots)
    pub fn active_cursor(&self) -> u64 {
        self.active_cursor
    }

    /// Data directory (used by stats to report actual segment-file disk usage, P3)
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Actual segment-file disk usage — scans data_dir and sums all .mmap file sizes.
    /// P3: quota counts only net payload bytes, not padding/segment tail holes/heed.
    /// This value reflects real disk usage, letting /stats expose the quota-vs-disk gap
    /// (quota "not full" while disk may already be full).
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

    /// Write a payload — returns the locator. Seals and rotates automatically when full.
    pub fn append(&mut self, payload: &[u8]) -> Result<ValueLocator> {
        // F3: reject payloads above u32::MAX to prevent len-truncation corruption (all bytes
        // written but the locator records a truncated len).
        if payload.len() > u32::MAX as usize {
            return Err(StoreError::ValueTooLarge(payload.len()));
        }
        let need = payload.len() as u64;
        if need > self.remaining() {
            self.seal_and_rotate()?;
        }
        let seg_id = self.active_id;
        // F3: reject offsets above u32::MAX (per-segment seg_size <= u32::MAX is guaranteed
        // by construction, and rotation resets the cursor if it ever approached u32; this
        // assertion is the final backstop).
        if self.active_cursor > u32::MAX as u64 {
            return Err(StoreError::Corrupt(format!(
                "active cursor {} exceeds u32 max, segment overflow",
                self.active_cursor
            )));
        }
        let offset = self.active_cursor as u32;
        let len = payload.len() as u32;
        // Append into the active write segment
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

    /// Write a payload, first aligning the cursor to `align` (zero padding).
    /// Columnar fixed-width columns need type alignment (i64/f64 need 8) or ScalarBuffer
    /// construction fails on misalignment.
    /// The mmap segment base is page-aligned (4096), so the first column at offset=0 is
    /// naturally aligned; later columns pad as needed.
    pub fn append_aligned(&mut self, payload: &[u8], align: usize) -> Result<ValueLocator> {
        if align <= 1 {
            return self.append(payload);
        }
        let pad = (align - (self.active_cursor as usize % align)) % align;
        if pad > 0 {
            let pad_bytes = vec![0u8; pad];
            // Padding that fits the segment appends normally; if it would fill the segment,
            // seal first (payload enters a new segment at offset 0, naturally aligned)
            if pad as u64 > self.remaining() {
                self.seal_and_rotate()?;
            } else {
                let _ = self.append(&pad_bytes)?;
            }
        }
        // After sealing, offset resets to 0 (aligned); or already aligned after padding
        self.append(payload)
    }

    /// Rewind the active segment write cursor to (seg_id, offset) [L1]
    /// Called on commit failure to reclaim bytes appended this round but never committed,
    /// avoiding in-segment leaks.
    /// Only valid while seg_id is still the current active segment (no concurrent writes
    /// within the exclusive write lock); otherwise it cannot rewind.
    /// After rewinding, the [offset, old cursor) range becomes orphan bytes — no locator
    /// references them; later appends overwrite.
    pub fn rewind_active_to(&mut self, seg_id: u32, offset: u32) -> Result<()> {
        if seg_id != self.active_id {
            // The append triggered a seal rotation → the segment is sealed read-only and
            // cannot rewind. Orphan bytes in sealed segments are reclaimed by compact;
            // log only here.
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

    /// Seal the active write segment + open a new one
    fn seal_and_rotate(&mut self) -> Result<()> {
        let sealed_id = self.active_id;
        tracing::info!(seg_id = sealed_id, "segment full, sealing + rotating");
        // Flush the active write segment to disk, then flip to a read-only mapping
        self.active_mmap.flush()?;
        let sealed_file = self.active_file.try_clone()?;
        let sealed_mmap = unsafe { Mmap::map(&sealed_file)? };
        let mut handle = MmapHandle::from_parts(sealed_file, sealed_mmap);
        handle.seal();
        self.sealed.insert(sealed_id, Arc::new(handle));
        self.sealed_lru.push_back(sealed_id);
        self.evict_sealed();
        // Open the new write segment
        let new_id = sealed_id + 1;
        let path = self.data_dir.join(segment_file(new_id));
        let file = open_or_create_segment(&path, self.seg_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        self.active_id = new_id;
        self.active_file = file;
        self.active_mmap = mmap;
        self.active_cursor = 0;
        // R9: new active segment has cursor=0; persist it so reopen never mistakes the old full segment as resumable
        write_active_cursor(&self.data_dir, new_id, 0)?;
        Ok(())
    }

    /// Take a sealed segment's read-only MmapHandle — readers build ZeroCopyBuffer via the locator
    pub fn sealed_handle(&mut self, seg_id: u32) -> Result<Arc<MmapHandle>> {
        if let Some(h) = self.sealed.get(&seg_id) {
            // A4: LRU hit, move to the back (most recently used). Clone the Arc before
            // touching to avoid a borrow conflict.
            let cloned = h.clone();
            self.touch_sealed(seg_id);
            return Ok(cloned);
        }
        // Not cached: open the sealed segment on demand
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

    /// A4: LRU touch — move seg_id to the back (most recently used)
    fn touch_sealed(&mut self, seg_id: u32) {
        if let Some(pos) = self.sealed_lru.iter().position(|&s| s == seg_id) {
            self.sealed_lru.remove(pos);
            self.sealed_lru.push_back(seg_id);
        }
    }

    /// A4/R8: LRU eviction — reclaim from the front once SEALED_CACHE_CAP is exceeded.
    /// Only munmap when strong_count==1 (no reader holds the handle); count>1 means a
    /// zero-copy reader still holds it, so skip that entry to keep pointers valid.
    ///
    /// R8: the old implementation `break`ed on the first held segment, so all older
    /// segments behind it could never be evicted and the mapping count grew without
    /// bound while a handle was held long-term, hitting the mmap limit. Now `continue`
    /// skips held segments and keeps evicting the next evictable one at the front; the
    /// `touched` counter prevents a "everything held" infinite loop — one fully skipped
    /// pass exits, and the next append triggers a retry.
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
                // R8: still held by a reader; skip eviction (re-queue at the back for a
                // later retry) and continue with the next
                tracing::debug!(
                    seg_id = candidate,
                    "sealed segment still in use (strong_count>1), skip evict, continue"
                );
                self.sealed_lru.push_back(candidate);
                touched += 1;
            }
        }
        if self.sealed_lru.len() > SEALED_CACHE_CAP {
            // All held: no eviction possible this round; log for monitoring (mapping count temporarily above cap)
            tracing::warn!(
                over = self.sealed_lru.len() - SEALED_CACHE_CAP,
                "all sealed segments held by readers, eviction deferred"
            );
        }
    }

    /// Synchronous flush (called by checkpoint, M4)
    /// R9: also atomically persists the active cursor so reopen can resume the unfilled
    /// segment (aligned with the WAL checkpoint).
    pub fn flush(&mut self) -> Result<()> {
        self.active_mmap.flush()?;
        write_active_cursor(&self.data_dir, self.active_id, self.active_cursor)?;
        Ok(())
    }
}

/// Scan the data directory for the next seg_id (max existing seg_id + 1, 0 when none)
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

// R9: active cursor sidecar read/write. After a crash, reopen resumes the unfilled
// segment from it instead of wasting the segment tail every time.

/// Read active.cursor → (seg_id, cursor). Returns None when absent or corrupt
/// (degrades to opening a new segment).
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

/// Atomically write active.cursor (tmp + rename). Called by flush/seal, aligned with the WAL checkpoint.
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

/// Parse seg_id from a filename (kv_payload_NNNN.mmap)
fn parse_seg_id(name: &str) -> Option<u32> {
    let name = name.strip_prefix("kv_payload_")?;
    let name = name.strip_suffix(".mmap")?;
    name.parse().ok()
}

/// Open or create a segment file and preallocate its fixed capacity
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

/// Scan the write segment for the end of written content (a reverse scan for the first
/// non-zero byte is costly; simplified to a count of actually written bytes).
/// F4: a fresh active segment (next_seg_id=max+1) has cursor=0 correctly, no scan needed.
/// Kept as a utility for a future "active segment survives restart" scenario (a real
/// implementation is needed if resuming old segments is ever introduced).
#[allow(dead_code)]
fn written_len_scan(mmap: &MmapMut) -> u64 {
    // Find the written end as the first non-zero byte scanning from the tail; an all-zero
    // segment returns 0.
    // Note: zero-valued payloads would be misjudged as unwritten — auxiliary only, never
    // a crash-recovery basis.
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
        // Tiny segment capacity forces a seal rotation
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 32).unwrap();
        let first = pool.active_seg_id();
        let _ = pool.append(&[0xABu8; 32]).unwrap();
        // The next append triggers the seal rotation
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
        // Seal the current segment by opening a new one so the payload segment enters the sealed cache
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
        // Zero-copy assertion: pointer lands inside the mmap region
        let region = handle.as_bytes().as_ptr();
        let buf_ptr = buf.as_bytes().as_ptr();
        assert!(buf_ptr >= region && buf_ptr <= unsafe { region.add(handle.len()) });
    }

    #[test]
    fn sealed_lru_evicts_when_cap_exceeded() {
        // A4: when the sealed cache exceeds SEALED_CACHE_CAP, LRU evicts entries with strong_count==1.
        // Tiny segments force many seals; after reading, drop the handle (count back to 1),
        // then verify eviction keeps the cache under the cap.
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 16).unwrap();
        // Fill enough segments to trigger multiple seals (16B per segment; writing 17B triggers rotation)
        let mut seg_ids = Vec::new();
        for i in 0..200u32 {
            let payload = [i as u8; 16];
            let loc = pool.append(&payload).unwrap();
            if i > 0 {
                seg_ids.push(loc.seg_id);
            }
            // Write one placeholder byte to trigger the next seal
            let _ = pool.append(&[i as u8; 1]).unwrap();
        }
        // Read some sealed segments then immediately drop the handles (strong_count falls
        // back), so eviction should keep the cache bounded within the cap
        for &sid in seg_ids.iter().take(80) {
            let _h = pool.sealed_handle(sid).unwrap();
        }
        // sealed.len() should not exceed SEALED_CACHE_CAP (in-use entries skip eviction, but most are released)
        assert!(
            pool.sealed.len() <= 65,
            "sealed cache {} exceeded cap+1 after LRU eviction",
            pool.sealed.len()
        );
        // After eviction, reading an evicted segment should reopen successfully (on-demand reopen)
        let h = pool.sealed_handle(seg_ids[0]).unwrap();
        assert!(h.is_sealed());
    }

    // R8: while an old segment handle is held long-term, even older segments must still be
    // evicted (the old implementation's break stopped eviction entirely).
    #[test]
    fn evict_continues_past_held_oldest_segment() {
        let dir = tempdir().unwrap();
        let mut pool = SegmentPool::open(dir.path(), 16).unwrap();
        let mut handles = Vec::new();
        // Fill 200 segments, reading out a handle for each and keeping them (strong_count>1, held)
        for i in 0..200u32 {
            let loc = pool.append(&[i as u8; 16]).unwrap();
            let _ = pool.append(&[i as u8; 1]).unwrap();
            if i == 5 {
                // Hold segment 5's handle (old position in the LRU queue)
                handles.push(pool.sealed_handle(loc.seg_id).unwrap());
            }
        }
        drop(handles); // release, then trigger another eviction round
        let _ = pool.append(&[0xFFu8; 16]).unwrap();
        let _ = pool.append(&[0x01u8; 1]).unwrap();
        // No unbounded growth while held; after release eviction resumes, cache <= cap+1
        assert!(
            pool.sealed.len() <= SEALED_CACHE_CAP + 1,
            "R8: sealed cache {} still bounded while/after holding segment",
            pool.sealed.len()
        );
    }

    // R9: reopen resumes the unfilled active segment (no new-segment-per-restart tail waste).
    #[test]
    fn reopen_reuses_unfilled_active_segment() {
        let dir = tempdir().unwrap();
        let pool_dir = dir.path().to_path_buf();
        let seg_id;
        let cursor_after;
        {
            let mut pool = SegmentPool::open(&pool_dir, 4096).unwrap();
            seg_id = pool.active_seg_id();
            // Write 100 bytes then flush (persists the cursor sidecar)
            let _ = pool.append(&[0xABu8; 100]).unwrap();
            pool.flush().unwrap();
            cursor_after = pool.active_cursor();
            assert_eq!(cursor_after, 100);
        }
        // Reopen: should reuse the same seg_id with cursor=100, not open a new segment
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
        // Continued write does not overwrite: append 50 more, landing at offset=100 (the resume point)
        let loc = pool2.append(&[0xCDu8; 50]).unwrap();
        assert_eq!(
            loc.offset as u64, 100,
            "R9: continued write at restored cursor"
        );
    }

    // R9: after a seal rotation, reopen must not reuse the full segment (cursor records the new segment at 0).
    #[test]
    fn reopen_after_seal_opens_new_segment() {
        let dir = tempdir().unwrap();
        let pool_dir = dir.path().to_path_buf();
        let sealed_id;
        {
            let mut pool = SegmentPool::open(&pool_dir, 64).unwrap();
            sealed_id = pool.active_seg_id();
            let _ = pool.append(&[0u8; 64]).unwrap(); // fill the segment
            let _ = pool.append(&[0u8; 10]).unwrap(); // trigger seal + open new segment
            pool.flush().unwrap();
        }
        let pool2 = SegmentPool::open(&pool_dir, 64).unwrap();
        assert_ne!(
            pool2.active_seg_id(),
            sealed_id,
            "R9: sealed full segment not reused on reopen"
        );
        // New segment reopen resumes: cursor=10 (10B written into the new segment after
        // sealing, then flushed, cursor persisted)
        assert_eq!(
            pool2.active_cursor(),
            10,
            "R9: new active segment restores written cursor"
        );
    }
}
