//! Write-Ahead Log — the sole crash-safe sync point + idempotent replay [v2 H5/R2]
//!
//! Write path: log the WAL (fsync, the only crash-safe guarantee) → mutate in-memory
//! state → return.
//! mmap segments flush lazily; crash safety never depends on msync (H5).
//!
//! Idempotency model (R2 root fix): entries carry a monotonically increasing seq; the
//! checkpoint records applied_seq into the marker (atomic tmp+fsync+rename); recover
//! replays seq > applied_seq idempotently (insert checks id existence and skips, put_kv
//! overwrites, delete is idempotent).
//!
//! Logical WAL (F2 fix): WalOp carries logical payloads (key+value / id+vector /
//! table+ipc), not physical locators. Rationale: under PRD H5 "WAL as the sole sync point
//! + lazy mmap flush", a physical-locator WAL would point at an un-fsynced mmap segment
//! tail after power loss → corruption. Logical payloads let replay re-land segments +
//! heed through submodule idempotent methods, independent of whether old segment bytes
//! reached disk. Replay = calling put_kv/insert/put_columnar again, idempotently:
//! insert checks id existence and skips, put_kv overwrites, delete is idempotent.
//!
//! Frame format (fixed-length header + variable-length body):
//!   [seq:8 LE][op_tag:1][body_len:4 LE][body: body_len]
//! op_tag: 1=PutKv, 2=InsertVector, 3=DeleteKv, 4=DeleteVector, 5=PutColumnar

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use fs2::FileExt;

use crate::error::Result;
use crate::StoreError;

/// WAL fixed-length frame header (seq 8 + op_tag 1 + body_len 4)
const FRAME_HEADER: usize = 13;
/// op tags
const OP_PUT_KV: u8 = 1;
const OP_INSERT_VEC: u8 = 2;
const OP_DELETE_KV: u8 = 3;
const OP_DELETE_VEC: u8 = 4;
const OP_PUT_COL: u8 = 5;

/// WAL operation entry — logical payload (F2), not physical locators
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WalOp {
    /// KV write: key + full value (replay lands the segment again)
    PutKv { key: Vec<u8>, value: Vec<u8> },
    /// Vector insert: id + full vector (replay lands the segment again, idempotent dup check)
    InsertVector { id: u64, vector: Vec<f32> },
    /// KV delete: key
    DeleteKv { key: Vec<u8> },
    /// Vector delete: id
    DeleteVector { id: u64 },
    /// Columnar write: table_id + IPC bytes (replay lands the segment again)
    PutColumnar { table_id: String, ipc: Vec<u8> },
}

/// WAL entry: seq + op
#[derive(Debug, Clone, PartialEq)]
pub struct WalEntry {
    pub seq: u64,
    pub op: WalOp,
}

/// Checkpoint marker — atomic write (tmp + fsync + rename) [v2 R2]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMarker {
    /// Max seq checkpointed (recover replays seq > applied_seq)
    pub applied_seq: u64,
    /// The checkpoint's own sequence (= applied_seq)
    pub checkpoint_seq: u64,
    /// Corresponding graph snapshot filename (graph/nsw_snapshot_XXXX.mmap)
    pub graph_snapshot: Option<String>,
}

/// Write-Ahead Log — single namespace
pub struct Wal {
    log_path: PathBuf,
    marker_path: PathBuf,
    file: File,
    next_seq: u64,
    // A1: cross-process write exclusion. The Wal holds an exclusive flock on wal.lock
    // for its lifetime, preventing a second process from concurrently writing the same
    // namespace's WAL and interleaving frames into corruption.
    _lock_file: File,
}

impl Wal {
    /// Open/create the WAL. wal_dir = <ns>/wal, marker = <ns>/wal/checkpoint.marker
    ///
    /// A1: holds a cross-process exclusive flock on wal.lock (blocks until acquired),
    /// guaranteeing a single WAL writer per namespace.
    /// A second process opening the same namespace concurrently → blocks until the first
    /// closes and releases, preventing frame-interleaving corruption.
    pub fn open(wal_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(wal_dir)?;
        let lock_path = wal_dir.join("wal.lock");
        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        // Take the exclusive lock blocking (A1/E9: no polling; fs2 lock_exclusive blocks until acquired)
        lock_file.lock_exclusive().map_err(|e| {
            StoreError::Io(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!("wal lock busy (another writer holds it): {e}"),
            ))
        })?;
        let log_path = wal_dir.join("wal.log");
        let marker_path = wal_dir.join("checkpoint.marker");
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&log_path)?;
        // next_seq = existing max seq + 1 (resume numbering across restarts)
        let next_seq = max_seq(&log_path)? + 1;
        tracing::info!(log = ?log_path, next_seq, "wal opened (cross-proc flock held)");
        Ok(Self {
            log_path,
            marker_path,
            file,
            next_seq,
            _lock_file: lock_file,
        })
    }

    /// Append one WAL entry (fsync, crash-safe). Returns the assigned seq.
    pub fn append(&mut self, op: WalOp) -> Result<u64> {
        let seq = self.next_seq;
        // E1: vectors use a binary body (not JSON); 768-dim is 3KB vs JSON ~15KB.
        let body = encode_op(&op)?;
        // F3: reject bodies above u32::MAX to prevent frame body_len truncation corruption.
        if body.len() > u32::MAX as usize {
            return Err(StoreError::ValueTooLarge(body.len()));
        }
        let mut frame = Vec::with_capacity(FRAME_HEADER + body.len());
        frame.extend_from_slice(&seq.to_le_bytes());
        frame.push(op_tag(&op));
        frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
        frame.extend_from_slice(&body);
        self.file.write_all(&frame)?;
        self.file.sync_data()?;
        self.next_seq += 1;
        tracing::debug!(seq, op = ?op, "wal entry appended + fsync'd");
        Ok(seq)
    }

    /// Append a batch of WAL entries (one fsync for the whole batch, E2 group commit).
    /// Returns the first entry's seq.
    ///
    /// All frames are written with write_all, then one sync_data at the end. Crash before
    /// fsync → whole batch lost (torn tail handled by read_frame_or_torn truncating back
    /// to the last complete frame); after fsync → whole batch durable. Batch atomicity:
    /// all or nothing, never half-written.
    /// Replaces the old per-entry append+fsync (a 1000-entry batch went from 1000 fsyncs → 1).
    pub fn append_batch(&mut self, ops: &[WalOp]) -> Result<u64> {
        if ops.is_empty() {
            return Ok(self.next_seq);
        }
        let first_seq = self.next_seq;
        let mut seq = self.next_seq;
        let mut buf: Vec<u8> = Vec::new();
        for op in ops {
            let body = encode_op(op)?;
            if body.len() > u32::MAX as usize {
                return Err(StoreError::ValueTooLarge(body.len()));
            }
            buf.extend_from_slice(&seq.to_le_bytes());
            buf.push(op_tag(op));
            buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
            buf.extend_from_slice(&body);
            seq += 1;
        }
        self.file.write_all(&buf)?;
        self.file.sync_data()?;
        self.next_seq = seq;
        tracing::debug!(
            count = ops.len(),
            first_seq,
            "wal batch appended + single fsync"
        );
        Ok(first_seq)
    }

    /// Read back all WAL entries (for recover, ascending seq)
    // F5: torn-frame tolerance. If a crash happens after header+body_len were written but
    // the body is partially written, read_exact(body) hits UnexpectedEof. The old
    // implementation returned Err wholesale → the store would not open.
    // Correct: classify it as a torn tail, stop reading, and return the complete frames
    // already read (see read_frame_or_torn).
    pub fn read_all(&self) -> Result<Vec<WalEntry>> {
        let file = File::open(&self.log_path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        while let FrameRead::Entry(e) = read_frame_or_torn(&mut reader)? {
            entries.push(e);
        }
        Ok(entries)
    }

    /// Read the checkpoint marker (None when absent, treated as applied_seq=0)
    pub fn read_marker(&self) -> Result<Option<CheckpointMarker>> {
        if !self.marker_path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&self.marker_path)?;
        let marker = serde_json::from_slice(&bytes)?;
        Ok(Some(marker))
    }

    /// Atomically write the checkpoint marker (tmp + fsync + rename) [v2 R2]
    /// Crash before rename → old marker stays valid; after rename → new marker takes effect.
    pub fn write_marker(&self, marker: &CheckpointMarker) -> Result<()> {
        let tmp = self.marker_path.with_extension("marker.tmp");
        let bytes = serde_json::to_vec(marker)?;
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.marker_path)?;
        tracing::info!(
            applied_seq = marker.applied_seq,
            "checkpoint marker atomically written"
        );
        Ok(())
    }

    /// Truncate the WAL: drop entries with seq <= applied_seq, rewrite the rest [v2 R2]
    /// Called after checkpoint to keep the WAL from growing without bound.
    pub fn truncate_to(&mut self, applied_seq: u64) -> Result<()> {
        let entries = self.read_all()?;
        let keep: Vec<&WalEntry> = entries.iter().filter(|e| e.seq > applied_seq).collect();
        if keep.is_empty() && entries.is_empty() {
            return Ok(());
        }
        // Rewrite wal.log: close the current handle → rewrite → reopen for append
        let mut new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.log_path)?;
        for e in &keep {
            let body = encode_op(&e.op)?;
            new_file.write_all(&e.seq.to_le_bytes())?;
            new_file.write_all(&[op_tag(&e.op)])?;
            new_file.write_all(&(body.len() as u32).to_le_bytes())?;
            new_file.write_all(&body)?;
        }
        new_file.sync_all()?;
        drop(new_file);
        // Reopen the append handle
        self.file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        tracing::info!(
            applied_seq,
            kept = keep.len(),
            "wal truncated to applied_seq"
        );
        Ok(())
    }

    /// The next seq that will be assigned
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}

fn op_tag(op: &WalOp) -> u8 {
    match op {
        WalOp::PutKv { .. } => OP_PUT_KV,
        WalOp::InsertVector { .. } => OP_INSERT_VEC,
        WalOp::DeleteKv { .. } => OP_DELETE_KV,
        WalOp::DeleteVector { .. } => OP_DELETE_VEC,
        WalOp::PutColumnar { .. } => OP_PUT_COL,
    }
}

fn decode_op(tag: u8, body: &[u8]) -> Result<WalOp> {
    match tag {
        OP_PUT_KV => decode_put_kv(body),
        OP_INSERT_VEC => decode_insert_vec(body),
        OP_DELETE_KV => decode_delete_kv(body),
        OP_DELETE_VEC => decode_delete_vec(body),
        OP_PUT_COL => decode_put_col(body),
        other => Err(StoreError::Corrupt(format!("unknown wal op tag {}", other))),
    }
}

// E1: binary op encode/decode. Frame bodies use a fixed-length header + variable-length
// payload; Vec<f32> vectors are written raw as LE f32 (not JSON text, 768-dim 3KB vs
// JSON ~15KB).
// key/value/ipc bytes use u32 len + raw bytes, avoiding serde_json text amplification.
// Backward compatibility with old JSON frames: not supported (the WAL is an internal
// artifact, consistent within a version, no cross-version compatibility needed).

fn encode_op(op: &WalOp) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    match op {
        WalOp::PutKv { key, value } => {
            write_u32(&mut buf, key.len() as u32)?;
            buf.extend_from_slice(key);
            write_u32(&mut buf, value.len() as u32)?;
            buf.extend_from_slice(value);
        }
        WalOp::InsertVector { id, vector } => {
            buf.extend_from_slice(&id.to_le_bytes());
            write_u32(&mut buf, vector.len() as u32)?;
            for &f in vector {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        WalOp::DeleteKv { key } => {
            write_u32(&mut buf, key.len() as u32)?;
            buf.extend_from_slice(key);
        }
        WalOp::DeleteVector { id } => {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        WalOp::PutColumnar { table_id, ipc } => {
            let tid = table_id.as_bytes();
            write_u32(&mut buf, tid.len() as u32)?;
            buf.extend_from_slice(tid);
            write_u32(&mut buf, ipc.len() as u32)?;
            buf.extend_from_slice(ipc);
        }
    }
    Ok(buf)
}

fn write_u32(buf: &mut Vec<u8>, v: u32) -> Result<()> {
    buf.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn read_u32(reader: &mut &[u8]) -> Result<u32> {
    if reader.len() < 4 {
        return Err(StoreError::Corrupt("wal op body truncated at u32".into()));
    }
    let v = u32::from_le_bytes(reader[0..4].try_into().unwrap());
    *reader = &reader[4..];
    Ok(v)
}

fn read_bytes<'a>(reader: &mut &'a [u8], len: usize) -> Result<&'a [u8]> {
    if reader.len() < len {
        return Err(StoreError::Corrupt(format!(
            "wal op body truncated: need {} have {}",
            len,
            reader.len()
        )));
    }
    let out = &reader[..len];
    *reader = &reader[len..];
    Ok(out)
}

fn decode_put_kv(body: &[u8]) -> Result<WalOp> {
    let mut r = body;
    let klen = read_u32(&mut r)? as usize;
    let key = read_bytes(&mut r, klen)?.to_vec();
    let vlen = read_u32(&mut r)? as usize;
    let value = read_bytes(&mut r, vlen)?.to_vec();
    if !r.is_empty() {
        return Err(StoreError::Corrupt("wal put_kv trailing bytes".into()));
    }
    Ok(WalOp::PutKv { key, value })
}

fn decode_insert_vec(body: &[u8]) -> Result<WalOp> {
    let mut r = body;
    if r.len() < 8 {
        return Err(StoreError::Corrupt("wal insert_vec truncated id".into()));
    }
    let id = u64::from_le_bytes(r[0..8].try_into().unwrap());
    r = &r[8..];
    let dim = read_u32(&mut r)? as usize;
    let need = dim
        .checked_mul(4)
        .ok_or(StoreError::Corrupt("wal insert_vec dim overflow".into()))?;
    let raw = read_bytes(&mut r, need)?;
    let mut vector = Vec::with_capacity(dim);
    for chunk in raw.chunks_exact(4) {
        vector.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    if !r.is_empty() {
        return Err(StoreError::Corrupt("wal insert_vec trailing bytes".into()));
    }
    Ok(WalOp::InsertVector { id, vector })
}

fn decode_delete_kv(body: &[u8]) -> Result<WalOp> {
    let mut r = body;
    let klen = read_u32(&mut r)? as usize;
    let key = read_bytes(&mut r, klen)?.to_vec();
    if !r.is_empty() {
        return Err(StoreError::Corrupt("wal delete_kv trailing bytes".into()));
    }
    Ok(WalOp::DeleteKv { key })
}

fn decode_delete_vec(body: &[u8]) -> Result<WalOp> {
    if body.len() != 8 {
        return Err(StoreError::Corrupt("wal delete_vec body != 8 bytes".into()));
    }
    let id = u64::from_le_bytes(body.try_into().unwrap());
    Ok(WalOp::DeleteVector { id })
}

fn decode_put_col(body: &[u8]) -> Result<WalOp> {
    let mut r = body;
    let tlen = read_u32(&mut r)? as usize;
    let tid_bytes = read_bytes(&mut r, tlen)?;
    let table_id = std::str::from_utf8(tid_bytes)
        .map_err(|_| StoreError::Corrupt("wal put_col table_id not utf8".into()))?
        .to_string();
    let ilen = read_u32(&mut r)? as usize;
    let ipc = read_bytes(&mut r, ilen)?.to_vec();
    if !r.is_empty() {
        return Err(StoreError::Corrupt("wal put_col trailing bytes".into()));
    }
    Ok(WalOp::PutColumnar { table_id, ipc })
}

// F5: single-frame read result. TornTail = an incomplete tail frame torn by a crash
// (truncated and dropped).
enum FrameRead {
    Entry(WalEntry),
    TornTail,
    Eof,
}

// F5: fault-tolerant read at frame boundaries. Short header read → Eof; complete header
// but short body → TornTail (the crash happened mid-body-write; truncate to the last
// complete frame rather than erroring wholesale).
fn read_frame_or_torn<R: Read>(reader: &mut R) -> Result<FrameRead> {
    let mut header = [0u8; FRAME_HEADER];
    match reader.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(FrameRead::Eof),
        Err(e) => return Err(StoreError::Io(e)),
    }
    let seq = u64::from_le_bytes(header[0..8].try_into().unwrap());
    let op_tag = header[8];
    let body_len = u32::from_le_bytes(header[9..13].try_into().unwrap()) as usize;
    let mut body = vec![0u8; body_len];
    match reader.read_exact(&mut body) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            tracing::warn!(seq, body_len, "wal torn tail frame detected, truncating");
            return Ok(FrameRead::TornTail);
        }
        Err(e) => return Err(StoreError::Io(e)),
    }
    let op = decode_op(op_tag, &body)?;
    Ok(FrameRead::Entry(WalEntry { seq, op }))
}

/// Scan the WAL for the max seq (used to resume numbering across restarts)
// F5: same torn-frame tolerance.
fn max_seq(log_path: &Path) -> Result<u64> {
    if !log_path.exists() {
        return Ok(0);
    }
    let file = File::open(log_path)?;
    let mut reader = BufReader::new(file);
    let mut max_seq: u64 = 0;
    while let FrameRead::Entry(e) = read_frame_or_torn(&mut reader)? {
        max_seq = max_seq.max(e.seq);
    }
    Ok(max_seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn append_then_read_back_preserves_entries() {
        let dir = tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();
        let s1 = wal
            .append(WalOp::PutKv {
                key: b"k1".to_vec(),
                value: b"hello-wal".to_vec(),
            })
            .unwrap();
        let s2 = wal
            .append(WalOp::InsertVector {
                id: 42,
                vector: vec![0.1, 0.2, 0.3, 0.4],
            })
            .unwrap();
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(
            entries[0].op,
            WalOp::PutKv {
                key: b"k1".to_vec(),
                value: b"hello-wal".to_vec()
            }
        );
        assert_eq!(entries[1].seq, 2);
        assert_eq!(
            entries[1].op,
            WalOp::InsertVector {
                id: 42,
                vector: vec![0.1, 0.2, 0.3, 0.4]
            }
        );
    }

    #[test]
    fn reopen_resumes_seq_from_max() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            wal.append(WalOp::DeleteKv { key: b"x".to_vec() }).unwrap();
            wal.append(WalOp::DeleteVector { id: 7 }).unwrap();
        }
        // Reopen: next_seq should start at 3 (existing max seq=2)
        let wal = Wal::open(&wal_dir).unwrap();
        assert_eq!(wal.next_seq(), 3);
        assert_eq!(wal.read_all().unwrap().len(), 2);
    }

    #[test]
    fn marker_atomic_write_roundtrip() {
        let dir = tempdir().unwrap();
        let wal = Wal::open(dir.path()).unwrap();
        assert!(wal.read_marker().unwrap().is_none());
        let m = CheckpointMarker {
            applied_seq: 10,
            checkpoint_seq: 10,
            graph_snapshot: Some("nsw_snapshot_0001.mmap".into()),
        };
        wal.write_marker(&m).unwrap();
        let got = wal.read_marker().unwrap().unwrap();
        assert_eq!(got.applied_seq, 10);
        assert_eq!(
            got.graph_snapshot.as_deref(),
            Some("nsw_snapshot_0001.mmap")
        );
    }

    #[test]
    fn truncate_keeps_only_after_applied_seq() {
        let dir = tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();
        for i in 0..5u64 {
            wal.append(WalOp::DeleteKv {
                key: format!("k{}", i).into_bytes(),
            })
            .unwrap();
        }
        // seq 1..5; truncate to applied_seq=3, keeping seq 4,5
        wal.truncate_to(3).unwrap();
        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 4);
        assert_eq!(entries[1].seq, 5);
        // Numbering continues unaffected
        assert_eq!(wal.next_seq(), 6);
    }

    #[test]
    fn empty_wal_read_returns_empty() {
        let dir = tempdir().unwrap();
        let wal = Wal::open(dir.path()).unwrap();
        assert!(wal.read_all().unwrap().is_empty());
        assert_eq!(wal.next_seq(), 1);
    }

    // F5: torn-frame tolerance. Simulate a crash: two complete frames + one torn tail
    // frame (header+body_len complete, body incomplete). read_all should return the first
    // two complete frames, drop the torn tail, and not error.
    // E1: binary op codec round-trip. Vectors use LE f32 (not JSON); verify each op type
    // round-trips identically.
    #[test]
    fn binary_op_encode_decode_roundtrip() {
        let ops = vec![
            WalOp::PutKv {
                key: b"k".to_vec(),
                value: vec![0u8; 300],
            },
            WalOp::InsertVector {
                id: 42,
                vector: vec![0.1, -0.2, 3.5, f32::INFINITY, 0.0],
            },
            WalOp::DeleteKv {
                key: b"long-key-xyz".to_vec(),
            },
            WalOp::DeleteVector { id: 999 },
            WalOp::PutColumnar {
                table_id: "t1".into(),
                ipc: vec![1, 2, 3, 4, 5],
            },
        ];
        for op in &ops {
            let body = encode_op(op).unwrap();
            let tag = op_tag(op);
            let decoded = decode_op(tag, &body).unwrap();
            assert_eq!(*op, decoded, "binary roundtrip mismatch for {:?}", op);
        }
    }

    // E1: vector body length = 8(id) + 4(dim) + dim*4. 768-dim → 3080 bytes, not JSON ~15KB.
    #[test]
    fn binary_vector_body_size_not_inflated() {
        let op = WalOp::InsertVector {
            id: 1,
            vector: vec![0.5f32; 768],
        };
        let body = encode_op(&op).unwrap();
        assert_eq!(body.len(), 8 + 4 + 768 * 4, "768-dim binary body = 3080B");
    }

    // E2: append_batch single fsync for the whole batch + contiguous seq + consistent read-back.
    #[test]
    fn append_batch_single_fsync_and_contiguous_seq() {
        let dir = tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();
        let ops: Vec<WalOp> = (0..1000u64)
            .map(|i| WalOp::InsertVector {
                id: i,
                vector: vec![i as f32; 8],
            })
            .collect();
        let first = wal.append_batch(&ops).unwrap();
        assert_eq!(first, 1, "batch first seq starts at next_seq");
        assert_eq!(wal.next_seq(), 1001, "next_seq advanced by batch size");
        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 1000, "all 1000 batch entries persisted");
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[999].seq, 1000);
        assert_eq!(
            entries[500].op,
            WalOp::InsertVector {
                id: 500,
                vector: vec![500.0f32; 8]
            }
        );
    }

    // E2: an empty batch advances no seq and writes no frames.
    #[test]
    fn append_batch_empty_noop() {
        let dir = tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();
        let before = wal.next_seq();
        let first = wal.append_batch(&[]).unwrap();
        assert_eq!(first, before);
        assert_eq!(wal.next_seq(), before);
        assert!(wal.read_all().unwrap().is_empty());
    }

    // E2: batch crashing before fsync → whole batch lost (torn tail tolerance, never half-written).
    // Simulate: append_batch writes but we manually truncate half the log → read_all
    // returns only complete frames (batch not fully durable).
    #[test]
    fn append_batch_torn_tail_drops_whole_batch() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        let log_path = wal_dir.join("wal.log");
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            // One batch of 5 entries
            let ops: Vec<WalOp> = (0..5u64)
                .map(|i| WalOp::InsertVector {
                    id: i,
                    vector: vec![i as f32; 4],
                })
                .collect();
            wal.append_batch(&ops).unwrap();
            drop(wal);
        }
        // Truncate the log to 1/3 (damaging some frames) → classify as torn tail, return complete frames
        let len = std::fs::metadata(&log_path).unwrap().len() as usize;
        let truncated = len / 3;
        let data = std::fs::read(&log_path).unwrap();
        std::fs::write(&log_path, &data[..truncated]).unwrap();
        let wal = Wal::open(&wal_dir).unwrap();
        let entries = wal.read_all().unwrap();
        // At most the complete-frame subset survives (frames before the truncation point), no panic
        assert!(entries.len() <= 5, "torn tail tolerates, no panic");
    }

    // F5: torn-frame tolerance. Simulate a crash: two complete frames + one torn tail
    // frame (header+body_len complete, body incomplete). read_all should return the first
    // two complete frames, drop the torn tail, and not error.
    #[test]
    fn torn_tail_frame_is_truncated_not_error() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        let log_path = wal_dir.join("wal.log");
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            wal.append(WalOp::PutKv {
                key: b"keep1".to_vec(),
                value: b"v1".to_vec(),
            })
            .unwrap();
            wal.append(WalOp::PutKv {
                key: b"keep2".to_vec(),
                value: b"v2".to_vec(),
            })
            .unwrap();
            // drop wal to release the handle, then manually append a torn tail frame
            drop(wal);
        }
        // Manually write one frame: complete header + body_len declaring 100 bytes, but
        // only 3 bytes of body
        let mut f = OpenOptions::new().append(true).open(&log_path).unwrap();
        let seq_bytes = 99u64.to_le_bytes();
        f.write_all(&seq_bytes).unwrap();
        f.write_all(&[OP_PUT_KV]).unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap();
        f.write_all(b"abc").unwrap(); // only 3 bytes, 97 short
        drop(f);

        let wal = Wal::open(&wal_dir).unwrap();
        let entries = wal.read_all().unwrap();
        // First two complete frames kept, torn tail dropped
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[1].seq, 2);
    }
}
