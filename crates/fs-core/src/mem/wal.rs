//! Write-Ahead Log —— 唯一 crash-safe 同步点 + 幂等重放 [v2 H5/R2]
//!
//! 写路径：记 WAL（fsync，crash-safe 唯一保证）→ 改内存状态 → 返回。
//! mmap 段延迟刷，不依赖 msync 保 crash-safe（H5）。
//!
//! 幂等模型（R2 根治）：条目带单调递增 seq；checkpoint 记 applied_seq 到 marker
//! （原子 tmp+fsync+rename）；recover 重放 seq > applied_seq，重放幂等
//! （insert 查 id 存在性跳过，put_kv 覆盖，delete 幂等）。
//!
//! 帧格式（定长头 + 变长 body）：
//!   [seq:8 LE][op_tag:1][body_len:4 LE][body: body_len]
//! op_tag: 1=PutKv, 2=InsertVector, 3=DeleteKv, 4=DeleteVector

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::mem::segment::ValueLocator;
use crate::StoreError;

/// WAL 帧头定长（seq 8 + op_tag 1 + body_len 4）
const FRAME_HEADER: usize = 13;
/// op 标签
const OP_PUT_KV: u8 = 1;
const OP_INSERT_VEC: u8 = 2;
const OP_DELETE_KV: u8 = 3;
const OP_DELETE_VEC: u8 = 4;

/// WAL 操作条目
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WalOp {
    /// KV 写：key + payload 定位
    PutKv { key: Vec<u8>, loc: ValueLocator },
    /// 向量插入：id + payload 定位
    InsertVector { id: u64, loc: ValueLocator },
    /// KV 删：key
    DeleteKv { key: Vec<u8> },
    /// 向量删：id
    DeleteVector { id: u64 },
}

/// WAL 条目：seq + op
#[derive(Debug, Clone, PartialEq)]
pub struct WalEntry {
    pub seq: u64,
    pub op: WalOp,
}

/// checkpoint marker —— 原子写（tmp + fsync + rename）[v2 R2]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMarker {
    /// 已 checkpoint 的最大 seq（recover 重放 seq > applied_seq）
    pub applied_seq: u64,
    /// checkpoint 自身序号（= applied_seq）
    pub checkpoint_seq: u64,
    /// 对应图 snapshot 文件名（graph/hnsw_snapshot_XXXX.mmap）
    pub graph_snapshot: Option<String>,
}

/// Write-Ahead Log —— 单 namespace
pub struct Wal {
    log_path: PathBuf,
    marker_path: PathBuf,
    file: File,
    next_seq: u64,
}

impl Wal {
    /// 打开/创建 WAL。wal_dir = <ns>/wal，marker = <ns>/wal/checkpoint.marker
    pub fn open(wal_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(wal_dir)?;
        let log_path = wal_dir.join("wal.log");
        let marker_path = wal_dir.join("checkpoint.marker");
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&log_path)?;
        // next_seq = 已有最大 seq + 1（重启续号）
        let next_seq = max_seq(&log_path)? + 1;
        tracing::info!(log = ?log_path, next_seq, "wal opened");
        Ok(Self {
            log_path,
            marker_path,
            file,
            next_seq,
        })
    }

    /// 追加一条 WAL（fsync，crash-safe）。返回分配的 seq。
    pub fn append(&mut self, op: WalOp) -> Result<u64> {
        let seq = self.next_seq;
        let body = serde_json::to_vec(&op)?;
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

    /// 读回全部 WAL 条目（recover 用，按 seq 升序）
    pub fn read_all(&self) -> Result<Vec<WalEntry>> {
        let file = File::open(&self.log_path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        loop {
            let mut header = [0u8; FRAME_HEADER];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(StoreError::Io(e)),
            }
            let seq = u64::from_le_bytes(header[0..8].try_into().unwrap());
            let op_tag = header[8];
            let body_len = u32::from_le_bytes(header[9..13].try_into().unwrap()) as usize;
            let mut body = vec![0u8; body_len];
            reader.read_exact(&mut body)?;
            let op = decode_op(op_tag, &body)?;
            entries.push(WalEntry { seq, op });
        }
        Ok(entries)
    }

    /// 读 checkpoint marker（不存在则返回 None，视为 applied_seq=0）
    pub fn read_marker(&self) -> Result<Option<CheckpointMarker>> {
        if !self.marker_path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&self.marker_path)?;
        let marker = serde_json::from_slice(&bytes)?;
        Ok(Some(marker))
    }

    /// 原子写 checkpoint marker（tmp + fsync + rename）[v2 R2]
    /// crash 在 rename 前 → 旧 marker 仍有效；rename 后 → 新 marker 生效。
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

    /// 截断 WAL：删除 applied_seq 及之前的条目，重写剩余 [v2 R2]
    /// checkpoint 后调，防 WAL 无限增长。
    pub fn truncate_to(&mut self, applied_seq: u64) -> Result<()> {
        let entries = self.read_all()?;
        let keep: Vec<&WalEntry> = entries.iter().filter(|e| e.seq > applied_seq).collect();
        if keep.is_empty() && entries.is_empty() {
            return Ok(());
        }
        // 重写 wal.log：关闭当前句柄 → 重写 → 重开 append
        let mut new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.log_path)?;
        for e in &keep {
            let body = serde_json::to_vec(&e.op)?;
            new_file.write_all(&e.seq.to_le_bytes())?;
            new_file.write_all(&[op_tag(&e.op)])?;
            new_file.write_all(&(body.len() as u32).to_le_bytes())?;
            new_file.write_all(&body)?;
        }
        new_file.sync_all()?;
        drop(new_file);
        // 重开 append 句柄
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

    /// 下一条将分配的 seq
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
    }
}

fn decode_op(tag: u8, body: &[u8]) -> Result<WalOp> {
    match tag {
        OP_PUT_KV | OP_INSERT_VEC | OP_DELETE_KV | OP_DELETE_VEC => {
            Ok(serde_json::from_slice(body)?)
        }
        _ => Err(StoreError::Corrupt(format!("unknown wal op tag {}", tag))),
    }
}

/// 扫 WAL 取最大 seq（重启续号用）
fn max_seq(log_path: &Path) -> Result<u64> {
    if !log_path.exists() {
        return Ok(0);
    }
    let file = File::open(log_path)?;
    let mut reader = BufReader::new(file);
    let mut max_seq: u64 = 0;
    loop {
        let mut header = [0u8; FRAME_HEADER];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(StoreError::Io(e)),
        }
        let seq = u64::from_le_bytes(header[0..8].try_into().unwrap());
        max_seq = max_seq.max(seq);
        let body_len = u32::from_le_bytes(header[9..13].try_into().unwrap()) as usize;
        let mut body = vec![0u8; body_len];
        reader.read_exact(&mut body)?;
    }
    Ok(max_seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn loc(seg: u32, off: u32, len: u32) -> ValueLocator {
        ValueLocator {
            seg_id: seg,
            offset: off,
            len,
        }
    }

    #[test]
    fn append_then_read_back_preserves_entries() {
        let dir = tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();
        let s1 = wal
            .append(WalOp::PutKv {
                key: b"k1".to_vec(),
                loc: loc(0, 0, 8),
            })
            .unwrap();
        let s2 = wal
            .append(WalOp::InsertVector {
                id: 42,
                loc: loc(0, 8, 16),
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
                loc: loc(0, 0, 8)
            }
        );
        assert_eq!(entries[1].seq, 2);
        assert_eq!(
            entries[1].op,
            WalOp::InsertVector {
                id: 42,
                loc: loc(0, 8, 16)
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
        // 重开：next_seq 应从 3 起（已有 max seq=2）
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
            graph_snapshot: Some("hnsw_snapshot_0001.mmap".into()),
        };
        wal.write_marker(&m).unwrap();
        let got = wal.read_marker().unwrap().unwrap();
        assert_eq!(got.applied_seq, 10);
        assert_eq!(
            got.graph_snapshot.as_deref(),
            Some("hnsw_snapshot_0001.mmap")
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
        // seq 1..5；截断到 applied_seq=3，保留 seq 4,5
        wal.truncate_to(3).unwrap();
        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 4);
        assert_eq!(entries[1].seq, 5);
        // 续号不受影响
        assert_eq!(wal.next_seq(), 6);
    }

    #[test]
    fn empty_wal_read_returns_empty() {
        let dir = tempdir().unwrap();
        let wal = Wal::open(dir.path()).unwrap();
        assert!(wal.read_all().unwrap().is_empty());
        assert_eq!(wal.next_seq(), 1);
    }
}
