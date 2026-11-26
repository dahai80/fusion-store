//! Compact — COW atomic switch orchestration + deferred reclamation of old segments [v2 A3/H4]
//!
//! Protocol (PRD-plan §443):
//!   1. VectorIndex::compact relocates live vectors into new segments + rebuilds the
//!      graph + atomically switches the heed pointer
//!   2. After the switch, old segments get no new readers, but in-flight readers still
//!      hold Arc<MmapHandle>
//!   3. Old segment files are physically deleted after a safety period (default 60s) —
//!      not deleted immediately (one of the H4 SIGBUS root causes)
//!   4. compact holds the write lock; reads are never blocked (MVCC)
//!
//! This module orchestrates the phases and deferred reclamation; the relocation logic
//! lives in VectorIndex::compact (needs access to private pool/graph).

use std::time::Duration;

use crate::error::Result;
use crate::vector::store::VectorIndex;

/// Default deferred-reclaim safety period (callers are advised to wait this long after
/// compact before reclaiming) [v2 A3]
///
/// This is an "advisory grace period" for callers — reclaim_segments enforces its own
/// hard floor `MIN_RECLAIM_SAFETY` (see vector/store.rs); a segment sealed less than
/// that floor ago with strong_count>1 is skipped and kept by reclaim. This default is
/// above the hard floor to leave enough room for in-flight readers to drop their Arc.
/// E7: single source of truth for both constants — a compile-time assertion locks this
/// value to be >= MIN_RECLAIM_SAFETY, preventing drift.
pub const DEFAULT_RECLAIM_SAFETY: Duration = Duration::from_secs(60);

// E7: compile-time guarantee that the advisory grace period >= the hard floor,
// otherwise panic (drift protection)
// Duration comparison is not const, so compare seconds (both values are whole seconds)
const _: () =
    assert!(DEFAULT_RECLAIM_SAFETY.as_secs() >= crate::vector::store::MIN_RECLAIM_SAFETY.as_secs());

/// Compact result — reported by fs-serve / fs-cli + used to schedule deferred reclamation
#[derive(Debug)]
pub struct CompactResult {
    /// Live vector count (valid vectors after compact)
    pub live_vectors: usize,
    /// Old segment ids pending reclamation (deleted after the safety period)
    pub reclaimable_segs: Vec<u32>,
}

/// Run the compact COW atomic switch [v2 A3/H4]
/// Returns the result; the caller decides when to call reclaim_segments to delete old
/// segments (after the default safety period).
pub fn run_compact(index: &VectorIndex) -> Result<CompactResult> {
    let (reclaimable, live) = index.compact()?;
    tracing::info!(
        live,
        reclaimable = reclaimable.len(),
        "compact orchestrated: COW atomic switch done, old segs pending reclaim"
    );
    Ok(CompactResult {
        live_vectors: live,
        reclaimable_segs: reclaimable,
    })
}

/// Reclaim old segment files after the safety period [v2 A3 + A5 safety-period enforcement]
/// A5: reclaim_segments already enforces the safety-period check internally (sealed longer
/// than RECLAIM_SAFETY OR strong_count==1); unsafe segments are skipped and kept. Returns
/// the number of segments reclaimed. Callers may retry un-reclaimed segments later.
pub fn reclaim(index: &VectorIndex, seg_ids: &[u32]) -> Result<usize> {
    index.reclaim_segments(seg_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::schema::{MetricKind, VectorSchema};
    use crate::vector::store::VectorIndex;
    use tempfile::tempdir;

    fn make_index(dir: &std::path::Path, dim: usize) -> VectorIndex {
        let schema = VectorSchema::new(dim, MetricKind::L2);
        VectorIndex::open(dir, schema, 0).unwrap()
    }

    #[test]
    fn compact_reclaims_soft_deleted_vectors() {
        // After soft-delete + compact, KNN no longer contains deleted vectors and the
        // live count is correct
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 8);
        for i in 0..10u64 {
            let mut v = vec![0.0f32; 8];
            v[0] = i as f32;
            idx.insert(i, &v, None).unwrap();
        }
        // Soft-delete 3,7
        idx.delete(3, None).unwrap();
        idx.delete(7, None).unwrap();
        assert_eq!(idx.len(), 8, "8 live after 2 soft-delete");
        let res = run_compact(&idx).unwrap();
        assert_eq!(res.live_vectors, 8, "compact keeps 8 live");
        // KNN excludes deleted vectors
        let q = vec![7.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let knn = idx.search_knn(&q, 3, None).unwrap();
        assert!(!knn.iter().any(|(id, _)| *id == 7));
        assert!(!knn.iter().any(|(id, _)| *id == 3));
        // Live count still 8 (after graph rebuild)
        assert_eq!(idx.len(), 8);
    }

    #[test]
    fn compact_preserves_knn_correctness() {
        // KNN recall after compact matches pre-compact (graph rebuilt correctly)
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 4);
        for i in 0..20u64 {
            let mut v = vec![0.0f32; 4];
            v[0] = i as f32;
            idx.insert(i, &v, None).unwrap();
        }
        let q = vec![5.0f32, 0.0, 0.0, 0.0];
        let before = idx.search_knn(&q, 5, None).unwrap();
        run_compact(&idx).unwrap();
        let after = idx.search_knn(&q, 5, None).unwrap();
        assert_eq!(before[0].0, after[0].0, "nearest preserved after compact");
        assert_eq!(before.len(), after.len());
    }

    #[test]
    fn compact_empty_index_noop() {
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 4);
        let res = run_compact(&idx).unwrap();
        assert_eq!(res.live_vectors, 0);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn compact_then_checkpoint_then_reopen_restores() {
        // compact → checkpoint → reopen: graph restored from snapshot, live count correct
        let dir = tempdir().unwrap();
        let ns_dir = dir.path().to_path_buf();
        {
            let idx = make_index(&ns_dir, 4);
            for i in 0..10u64 {
                let mut v = vec![0.0f32; 4];
                v[0] = i as f32;
                idx.insert(i, &v, None).unwrap();
            }
            idx.delete(2, None).unwrap();
            run_compact(&idx).unwrap();
            idx.checkpoint().unwrap();
        }
        let idx2 = make_index(&ns_dir, 4);
        assert_eq!(idx2.len(), 9, "9 live after compact+checkpoint+reopen");
        let q = vec![2.0f32, 0.0, 0.0, 0.0];
        let knn = idx2.search_knn(&q, 3, None).unwrap();
        assert!(!knn.iter().any(|(id, _)| *id == 2));
    }

    #[test]
    fn reclaim_deletes_old_segment_files() {
        // compact produces old segment ids and reclaim deletes the files; verified
        // against real vec_data (not fabricated paths)
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("vec_data");
        let idx = make_index(dir.path(), 4);
        for i in 0..5u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.insert(i, &v, None).unwrap();
        }
        // seg 0 file exists before compact
        assert!(data_dir.join("vec_payload_0000.mmap").exists());
        let res = run_compact(&idx).unwrap();
        assert!(!res.reclaimable_segs.is_empty());
        // reclaim deletes old segment files (A5: no in-flight reader → strong_count==1 → safe delete)
        let n = reclaim(&idx, &res.reclaimable_segs).unwrap();
        assert_eq!(
            n,
            res.reclaimable_segs.len(),
            "all old segs reclaimed (no reader)"
        );
        // Old segment files are gone (old segments physically deleted after COW)
        for sid in &res.reclaimable_segs {
            let p = data_dir.join(format!("vec_payload_{:04}.mmap", sid));
            assert!(!p.exists(), "old seg {} file reclaimed", sid);
        }
        // New segment files still exist (holding live vectors); KNN still works
        assert!(idx.search_knn(&[0.0, 0.0, 0.0, 0.0], 1, None).is_ok());
    }

    #[test]
    fn reclaim_idempotent_on_deleted_segs() {
        // A5: reclaiming already-deleted segments returns 0 (idempotent, no error)
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 4);
        for i in 0..5u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.insert(i, &v, None).unwrap();
        }
        let res = run_compact(&idx).unwrap();
        assert!(!res.reclaimable_segs.is_empty());
        let n = reclaim(&idx, &res.reclaimable_segs).unwrap();
        assert_eq!(n, res.reclaimable_segs.len(), "first reclaim deletes all");
        let n2 = reclaim(&idx, &res.reclaimable_segs).unwrap();
        assert_eq!(n2, 0, "second reclaim idempotent (segs gone)");
    }

    #[test]
    fn compact_read_during_compact_not_blocked() {
        // A3/H4: reads are not blocked during compact. Single-threaded tests cannot run
        // true concurrency, so verify read consistency after compact. True concurrent
        // read non-blocking is guaranteed by MVCC (heed RoTxn + old-segment Arcs);
        // this checks result correctness.
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 4);
        for i in 0..10u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.insert(i, &v, None).unwrap();
        }
        let _ = run_compact(&idx).unwrap();
        let q = vec![5.0f32, 0.0, 0.0, 0.0];
        let res = idx.search_knn(&q, 5, None).unwrap();
        assert_eq!(res.len(), 5);
        assert_eq!(res[0].0, 5, "nearest preserved after compact");
    }
}
