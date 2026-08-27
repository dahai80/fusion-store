//! Recover —— WAL 幂等重放 + 块泄漏防护 [v2 §2.7/R2]
//!
//! recover 流程（PRD §2.7）：
//!   1. 读 checkpoint.marker.applied_seq
//!   2. 读 WAL，取 seq > applied_seq 的条目
//!   3. 重放：幂等（insert 查 id 存在性跳过，put_kv 覆盖，delete 幂等）
//!   4. 块泄漏防护：每条记 (seg_id, offset)，recover 回滚未登记块
//!   5. 重放完截断 WAL
//!
//! 本模块只产 RecoverPlan（待重放条目 + 待回滚块），不直接改 kv/vector 状态 ——
//! 引擎层消费 plan 执行重放（Rule 3 外科式：不入侵现有 store）。幂等判定由引擎层
//! 在 apply 时做（insert 查 id 存在性），recover 提供 plan + 块回滚登记表。

use std::collections::HashSet;
use std::path::Path;

use crate::error::Result;
use crate::mem::segment::ValueLocator;
use crate::mem::wal::{Wal, WalEntry, WalOp};

/// 块泄漏防护：记录已登记的 (seg_id, offset)
pub type RegisteredBlocks = HashSet<(u32, u32)>;

/// recover 产出的重放计划 —— 引擎层消费
#[derive(Debug, Default)]
pub struct RecoverPlan {
    /// 待重放条目（seq 升序，已过滤 seq > applied_seq）
    pub entries: Vec<WalEntry>,
    /// checkpoint 的 applied_seq
    pub applied_seq: u64,
    /// 所有 WAL 条目登记的块 (seg_id, offset) —— 引擎层对照段已用范围回滚未登记块
    pub registered_blocks: RegisteredBlocks,
}

/// 执行 recover 前半段：读 marker + 过滤待重放条目 + 收集登记块。
/// 不截断 WAL（截断在 checkpoint 后做，或引擎确认重放成功后调 truncate）。
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
    let mut registered = RegisteredBlocks::new();
    for e in &all {
        if let Some(loc) = entry_loc(&e.op) {
            registered.insert((loc.seg_id, loc.offset));
        }
    }
    tracing::info!(
        applied_seq,
        total = all.len(),
        to_replay = entries.len(),
        registered_blocks = registered.len(),
        "recover plan built"
    );
    Ok(RecoverPlan {
        entries,
        applied_seq,
        registered_blocks: registered,
    })
}

/// 重放完确认后截断 WAL（防无限增长）
pub fn finalize_recover(wal_dir: &Path, applied_seq: u64) -> Result<()> {
    let mut wal = Wal::open(wal_dir)?;
    wal.truncate_to(applied_seq)
}

fn entry_loc(op: &WalOp) -> Option<&ValueLocator> {
    match op {
        WalOp::PutKv { loc, .. } | WalOp::InsertVector { loc, .. } => Some(loc),
        WalOp::DeleteKv { .. } | WalOp::DeleteVector { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::wal::{CheckpointMarker, Wal, WalOp};
    use tempfile::tempdir;

    fn loc(seg: u32, off: u32, len: u32) -> ValueLocator {
        ValueLocator {
            seg_id: seg,
            offset: off,
            len,
        }
    }

    #[test]
    fn plan_replays_only_after_applied_seq() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            // seq 1..4
            wal.append(WalOp::PutKv {
                key: b"a".to_vec(),
                loc: loc(0, 0, 4),
            })
            .unwrap();
            wal.append(WalOp::InsertVector {
                id: 1,
                loc: loc(0, 4, 8),
            })
            .unwrap();
            wal.append(WalOp::PutKv {
                key: b"b".to_vec(),
                loc: loc(0, 12, 4),
            })
            .unwrap();
            wal.append(WalOp::InsertVector {
                id: 2,
                loc: loc(0, 16, 8),
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
        // 登记块含全部 4 条 Put/Insert 的 (seg,off)
        assert!(plan.registered_blocks.contains(&(0, 0)));
        assert!(plan.registered_blocks.contains(&(0, 4)));
        assert!(plan.registered_blocks.contains(&(0, 12)));
        assert!(plan.registered_blocks.contains(&(0, 16)));
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
        // Delete 无 loc，登记块为空
        assert!(plan.registered_blocks.is_empty());
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
                    loc: loc(0, (i * 8) as u32, 8),
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
    fn double_recover_no_duplicate_blocks() {
        // R2：重复 recover 不产重复向量/块泄漏 —— plan 本身幂等（纯读）
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            wal.append(WalOp::PutKv {
                key: b"k".to_vec(),
                loc: loc(0, 0, 4),
            })
            .unwrap();
        }
        let p1 = build_recover_plan(&wal_dir).unwrap();
        let p2 = build_recover_plan(&wal_dir).unwrap();
        // 两次 plan 完全一致，无累积/无重复
        assert_eq!(p1.entries.len(), p2.entries.len());
        assert_eq!(p1.applied_seq, p2.applied_seq);
        assert_eq!(p1.registered_blocks.len(), p2.registered_blocks.len());
    }
}
