//! Recover —— WAL 幂等重放 [v2 §2.7/R2]
//!
//! recover 流程（PRD §2.7）：
//!   1. 读 checkpoint.marker.applied_seq
//!   2. 读 WAL，取 seq > applied_seq 的条目（F5：torn-tail 容错）
//!   3. 重放：幂等（insert 查 id 存在性跳过，put_kv 覆盖，delete 幂等）
//!   4. 重放完截断 WAL
//!
//! 逻辑 WAL（F2）后不再做物理块登记回滚：重放经子模块幂等方法重新落段，
//! 旧段空字节的回收由 compact 负责，而非 recover 回滚未登记块。recover 仅产
//! 待重放条目 plan，引擎层消费执行重放（Rule 3 外科式：不入侵现有 store）。

use std::path::Path;

use crate::error::Result;
use crate::mem::wal::{Wal, WalEntry};

/// recover 产出的重放计划 —— 引擎层消费
#[derive(Debug, Default)]
pub struct RecoverPlan {
    /// 待重放条目（seq 升序，已过滤 seq > applied_seq）
    pub entries: Vec<WalEntry>,
    /// checkpoint 的 applied_seq
    pub applied_seq: u64,
}

/// 执行 recover 前半段：读 marker + 过滤待重放条目。
/// 不截断 WAL（截断在引擎确认重放成功后调 finalize_recover）。
pub fn build_recover_plan(wal_dir: &Path) -> Result<RecoverPlan> {
    let wal = Wal::open(wal_dir)?;
    let applied_seq = match wal.read_marker()? {
        Some(m) => m.applied_seq,
        None => 0,
    };
    let all = wal.read_all()?;
    let entries: Vec<WalEntry> = all
        .iter()
        .filter(|e| e.seq > applied_seq)
        .cloned()
        .collect();
    tracing::info!(
        applied_seq,
        total = all.len(),
        to_replay = entries.len(),
        "recover plan built"
    );
    Ok(RecoverPlan {
        entries,
        applied_seq,
    })
}

/// 重放完确认后截断 WAL（防无限增长）
pub fn finalize_recover(wal_dir: &Path, applied_seq: u64) -> Result<()> {
    let mut wal = Wal::open(wal_dir)?;
    wal.truncate_to(applied_seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::wal::{CheckpointMarker, Wal, WalOp};
    use tempfile::tempdir;

    #[test]
    fn plan_replays_only_after_applied_seq() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            // seq 1..4
            wal.append(WalOp::PutKv {
                key: b"a".to_vec(),
                value: b"va".to_vec(),
            })
            .unwrap();
            wal.append(WalOp::InsertVector {
                id: 1,
                vector: vec![0.1, 0.2],
            })
            .unwrap();
            wal.append(WalOp::PutKv {
                key: b"b".to_vec(),
                value: b"vb".to_vec(),
            })
            .unwrap();
            wal.append(WalOp::InsertVector {
                id: 2,
                vector: vec![0.3, 0.4],
            })
            .unwrap();
            // checkpoint applied_seq=2
            wal.write_marker(&CheckpointMarker {
                applied_seq: 2,
                checkpoint_seq: 2,
                graph_snapshot: None,
            })
            .unwrap();
        }
        let plan = build_recover_plan(&wal_dir).unwrap();
        assert_eq!(plan.applied_seq, 2);
        // 只重放 seq 3,4
        assert_eq!(plan.entries.len(), 2);
        assert_eq!(plan.entries[0].seq, 3);
        assert_eq!(plan.entries[1].seq, 4);
    }

    #[test]
    fn plan_no_marker_replays_all() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            wal.append(WalOp::DeleteKv { key: b"x".to_vec() }).unwrap();
            wal.append(WalOp::DeleteVector { id: 9 }).unwrap();
        }
        let plan = build_recover_plan(&wal_dir).unwrap();
        assert_eq!(plan.applied_seq, 0);
        assert_eq!(plan.entries.len(), 2);
    }

    #[test]
    fn finalize_truncates_wal_after_recover() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            for i in 0..6u64 {
                wal.append(WalOp::InsertVector {
                    id: i,
                    vector: vec![i as f32],
                })
                .unwrap();
            }
            wal.write_marker(&CheckpointMarker {
                applied_seq: 4,
                checkpoint_seq: 4,
                graph_snapshot: None,
            })
            .unwrap();
        }
        let plan = build_recover_plan(&wal_dir).unwrap();
        assert_eq!(plan.entries.len(), 2); // seq 5,6
        finalize_recover(&wal_dir, 4).unwrap();
        let wal = Wal::open(&wal_dir).unwrap();
        let remaining = wal.read_all().unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].seq, 5);
    }

    #[test]
    fn double_recover_plan_is_idempotent() {
        // R2：重复 recover plan 完全一致（纯读，无累积）
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            wal.append(WalOp::PutKv {
                key: b"k".to_vec(),
                value: b"v".to_vec(),
            })
            .unwrap();
        }
        let p1 = build_recover_plan(&wal_dir).unwrap();
        let p2 = build_recover_plan(&wal_dir).unwrap();
        assert_eq!(p1.entries.len(), p2.entries.len());
        assert_eq!(p1.applied_seq, p2.applied_seq);
    }
}
