//! Recover — idempotent WAL replay [v2 §2.7/R2]
//!
//! Recover flow (PRD §2.7):
//!   1. Read checkpoint.marker.applied_seq
//!   2. Read the WAL and take entries with seq > applied_seq (F5: torn-tail tolerance)
//!   3. Replay idempotently (insert skips existing ids, put_kv overwrites, delete is idempotent)
//!   4. Truncate the WAL after replay
//!
//! Since the logical WAL (F2), no physical block-registration rollback happens: replay
//! re-lands data through submodule idempotent methods, and reclaiming freed bytes in old
//! segments is compact's job, not recover rolling back unregistered blocks. Recover only
//! produces the pending-replay plan; the engine layer consumes and executes it
//! (Rule 3 surgical: no invasion of the existing store).

use std::path::Path;

use crate::error::Result;
use crate::mem::wal::{Wal, WalEntry};

/// Replay plan produced by recover — consumed by the engine layer
#[derive(Debug, Default)]
pub struct RecoverPlan {
    /// Entries pending replay (ascending seq, filtered to seq > applied_seq)
    pub entries: Vec<WalEntry>,
    /// applied_seq recorded at the checkpoint
    pub applied_seq: u64,
}

/// Run the first half of recover: read the marker and filter pending entries.
/// Does not truncate the WAL (truncation happens via finalize_recover after the
/// engine confirms replay succeeded).
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

/// Truncate the WAL after replay is confirmed (prevents unbounded growth)
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
        // replay only seq 3,4
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
        // R2: repeated recover plans are fully identical (pure read, no accumulation)
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
