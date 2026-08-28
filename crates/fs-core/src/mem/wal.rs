//! Write-Ahead Log —— 唯一 crash-safe 同步点 + 幂等重放 [v2 H5/R2]
//!
//! 写路径：记 WAL（fsync，crash-safe 唯一保证）→ 改内存状态 → 返回。
//! mmap 段延迟刷，不依赖 msync 保 crash-safe（H5）。
//!
//! 幂等模型（R2 根治）：条目带单调递增 seq；checkpoint 记 applied_seq 到 marker
//! （原子 tmp+fsync+rename）；recover 重放 seq > applied_seq，重放幂等
//! （insert 查 id 存在性跳过，put_kv 覆盖，delete 幂等）。
//!
//! 逻辑 WAL（F2 修复）：WalOp 携带逻辑 payload（key+value / id+vector / table+ipc），
//! 非物理 locator。理由：PRD H5「WAL 唯一同步点 + mmap 延迟刷」下，物理 locator WAL
//! 在断电时指向未 fsync 的 mmap 段尾 → 损坏。逻辑 payload 让 replay 经子模块幂等方法
//! 重新落段 + 落 heed，独立于旧段字节是否刷盘。重放 = 再调 put_kv/insert/put_columnar，
//! 幂等：insert 查 id 存在性跳过，put_kv 覆盖，delete 幂等。
//!
//! 帧格式（定长头 + 变长 body）：
//!   [seq:8 LE][op_tag:1][body_len:4 LE][body: body_len]
//! op_tag: 1=PutKv, 2=InsertVector, 3=DeleteKv, 4=DeleteVector, 5=PutColumnar

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use fs2::FileExt;

use crate::error::Result;
use crate::StoreError;

/// WAL 帧头定长（seq 8 + op_tag 1 + body_len 4）
const FRAME_HEADER: usize = 13;
/// op 标签
const OP_PUT_KV: u8 = 1;
const OP_INSERT_VEC: u8 = 2;
const OP_DELETE_KV: u8 = 3;
const OP_DELETE_VEC: u8 = 4;
const OP_PUT_COL: u8 = 5;

/// WAL 操作条目 —— 逻辑 payload（F2），非物理 locator
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WalOp {
    /// KV 写：key + 完整 value（replay 重新落段）
    PutKv { key: Vec<u8>, value: Vec<u8> },
    /// 向量插入：id + 完整 vector（replay 重新落段，幂等查 dup）
    InsertVector { id: u64, vector: Vec<f32> },
    /// KV 删：key
    DeleteKv { key: Vec<u8> },
    /// 向量删：id
    DeleteVector { id: u64 },
    /// 列式写：table_id + IPC 字节（replay 重新落段）
    PutColumnar { table_id: String, ipc: Vec<u8> },
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
    /// 对应图 snapshot 文件名（graph/nsw_snapshot_XXXX.mmap）
    pub graph_snapshot: Option<String>,
}

/// Write-Ahead Log —— 单 namespace
pub struct Wal {
    log_path: PathBuf,
    marker_path: PathBuf,
    file: File,
    next_seq: u64,
    // A1：跨进程写互斥锁。Wal 生命周期内持 wal.lock 排他 flock，
    // 阻止第二进程同 namespace 并发写 WAL 致帧交错损坏。
    _lock_file: File,
}

impl Wal {
    /// 打开/创建 WAL。wal_dir = <ns>/wal，marker = <ns>/wal/checkpoint.marker
    ///
    /// A1：持 wal.lock 跨进程排他 flock（阻塞到拿到），保证单 namespace 单 WAL 写者。
    /// 第二进程并发 open 同 namespace → 阻塞直到前者关闭释放，免帧交错损坏。
    pub fn open(wal_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(wal_dir)?;
        let lock_path = wal_dir.join("wal.lock");
        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        // 阻塞取排他锁（A1/E9：非轮询，fs2 lock_exclusive 阻塞到拿到）
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
        // next_seq = 已有最大 seq + 1（重启续号）
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

    /// 追加一条 WAL（fsync，crash-safe）。返回分配的 seq。
    pub fn append(&mut self, op: WalOp) -> Result<u64> {
        let seq = self.next_seq;
        // E1：向量走二进制 body（非 JSON），768 维 3KB vs JSON ~15KB。
        let body = encode_op(&op)?;
        // F3：body 超 u32::MAX 拒绝，免 frame body_len 截断损坏。
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

    /// 追加一批 WAL（整批单次 fsync，E2 group commit）。返回首条 seq。
    ///
    /// 全部帧 write_all 后末尾一次 sync_data。崩在 fsync 前 → 整批丢
    /// （torn tail 由 read_frame_or_torn 容错截断到上一完整帧）；
    /// fsync 后 → 整批持久。批次原子性：全成或全丢，无半写。
    /// 替代旧逐条 append+fsync（1000 条批量原 1000 次 fsync → 1 次）。
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

    /// 读回全部 WAL 条目（recover 用，按 seq 升序）
    // F5：torn-frame 容错。崩溃在 header+body_len 已写、body 只写一部分时，
    // read_exact(body) 会 UnexpectedEof。旧实现整体返 Err → store 打不开。
    // 正确：判定为 torn tail，停止读取，返回已读完整帧（见 read_frame_or_torn）。
    pub fn read_all(&self) -> Result<Vec<WalEntry>> {
        let file = File::open(&self.log_path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        while let FrameRead::Entry(e) = read_frame_or_torn(&mut reader)? {
            entries.push(e);
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
            let body = encode_op(&e.op)?;
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

// E1：二进制 op 编/解码。frame body 走定长头 + 变长 payload，
// 向量 Vec<f32> 按 LE f32 原样写（非 JSON 文本，768 维 3KB vs JSON ~15KB）。
// key/value/ipc 字节用 u32 len + raw bytes，避免 serde_json 文本放大。
// 向后兼容旧 JSON 帧：不支持（WAL 为内部产物，版本内一致，无需跨版本兼容）。

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

// F5：单帧读取结果。TornTail = 崩溃撕裂的不完整尾帧（截断丢弃）。
enum FrameRead {
    Entry(WalEntry),
    TornTail,
    Eof,
}

// F5：按帧边界容错读。header 读不足 → Eof；header 完整但 body 不足 → TornTail
// （崩溃发生在 body 写一半，截断到上一完整帧，非整体报错）。
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

/// 扫 WAL 取最大 seq（重启续号用）
// F5：同样 torn-frame 容错。
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

    // F5：torn-frame 容错。模拟崩溃：写两完整帧 + 一撕裂尾帧（header+body_len 完整，
    // body 不完整）。read_all 应返回前两完整帧，丢弃撕裂尾，不报错。
    // E1：二进制 op 编解码 round-trip。向量走 LE f32（非 JSON），校验各 op 类型往返一致。
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

    // E1：向量 body 长度 = 8(id) + 4(dim) + dim*4。768 维 → 3080 字节，非 JSON ~15KB。
    #[test]
    fn binary_vector_body_size_not_inflated() {
        let op = WalOp::InsertVector {
            id: 1,
            vector: vec![0.5f32; 768],
        };
        let body = encode_op(&op).unwrap();
        assert_eq!(body.len(), 8 + 4 + 768 * 4, "768-dim binary body = 3080B");
    }

    // E2：append_batch 整批单 fsync + 续号连续 + read_back 一致。
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

    // E2：空 batch 不推进 seq，不写帧。
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

    // E2：批量崩在 fsync 前 → 整批丢（torn tail 容错，无半写）。
    // 模拟：append_batch 写入但人为只截断一半 → read_all 返回 0（整批未完整）。
    #[test]
    fn append_batch_torn_tail_drops_whole_batch() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        let log_path = wal_dir.join("wal.log");
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            // 一批 5 条
            let ops: Vec<WalOp> = (0..5u64)
                .map(|i| WalOp::InsertVector {
                    id: i,
                    vector: vec![i as f32; 4],
                })
                .collect();
            wal.append_batch(&ops).unwrap();
            drop(wal);
        }
        // 截断 log 到 1/3（破坏部分帧）→ 应整体判 torn tail，返回已完整帧
        let len = std::fs::metadata(&log_path).unwrap().len() as usize;
        let truncated = len / 3;
        let data = std::fs::read(&log_path).unwrap();
        std::fs::write(&log_path, &data[..truncated]).unwrap();
        let wal = Wal::open(&wal_dir).unwrap();
        let entries = wal.read_all().unwrap();
        // 最多保留完整帧子集（截断点之前的完整帧），无 panic
        assert!(entries.len() <= 5, "torn tail tolerates, no panic");
    }

    // F5：torn-frame 容错。模拟崩溃：写两完整帧 + 一撕裂尾帧（header+body_len 完整，
    // body 不完整）。read_all 应返回前两完整帧，丢弃撕裂尾，不报错。
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
            // drop wal 释放句柄，手工追加撕裂尾帧
            drop(wal);
        }
        // 手工写一帧：完整 header + body_len 声明 100 字节，但只写 3 字节 body
        let mut f = OpenOptions::new().append(true).open(&log_path).unwrap();
        let seq_bytes = 99u64.to_le_bytes();
        f.write_all(&seq_bytes).unwrap();
        f.write_all(&[OP_PUT_KV]).unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap();
        f.write_all(b"abc").unwrap(); // 只 3 字节，缺 97
        drop(f);

        let wal = Wal::open(&wal_dir).unwrap();
        let entries = wal.read_all().unwrap();
        // 前两完整帧保留，撕裂尾丢弃
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[1].seq, 2);
    }
}
