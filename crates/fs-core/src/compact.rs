//! Compact —— COW 原子切换编排 + 旧段延迟回收 [v2 A3/H4]
//!
//! 协议（PRD-plan §443）：
//!   1. VectorIndex::compact 重排有效向量到新段 + 重建图 + 原子切 heed
//!   2. 切换后旧段无新 reader 引用，但 in-flight reader 仍持 Arc<MmapHandle>
//!   3. 安全期（默认 60s）后物理删旧段文件 —— 不立即删（H4 SIGBUS 根因之一）
//!   4. compact 期间持写锁，读不阻塞（MVCC）
//!
//! 本模块编排阶段 + 延迟回收；重排逻辑在 VectorIndex::compact（访问私有 pool/graph）。

use std::time::Duration;

use crate::error::Result;
use crate::vector::store::VectorIndex;

/// 默认延迟回收安全期（建议 caller 在 compact 后等待此时长再 reclaim）[v2 A3]
///
/// 这是给 caller 的「建议宽限」——reclaim_segments 内部另有硬下限
/// `MIN_RECLAIM_SAFETY`（见 vector/store.rs），段封存未过该下限且 strong_count>1
/// 时 reclaim 会跳过保留。此默认值 > 硬下限，留足 in-flight reader 释放 Arc 的余量。
/// E7：两个常量单一来源——此值经编译期断言锁定 ≥ MIN_RECLAIM_SAFETY，杜绝漂移。
pub const DEFAULT_RECLAIM_SAFETY: Duration = Duration::from_secs(60);

// E7：编译期保证建议宽限 ≥ 硬下限，否则 panic（drift 防护）
// Duration 比较非 const，故比较秒数（两值均整秒）
const _: () =
    assert!(DEFAULT_RECLAIM_SAFETY.as_secs() >= crate::vector::store::MIN_RECLAIM_SAFETY.as_secs());

/// compact 结果 —— 供 fs-serve / fs-cli 上报 + 调度延迟回收
#[derive(Debug)]
pub struct CompactResult {
    /// 活向量数（compact 后有效向量）
    pub live_vectors: usize,
    /// 待回收旧段 id（安全期后删）
    pub reclaimable_segs: Vec<u32>,
}

/// 执行 compact COW 原子切换 [v2 A3/H4]
/// 返回结果；caller 决定何时调 reclaim_segments 删旧段（默认安全期后）。
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

/// 安全期后回收旧段文件 [v2 A3 + A5 安全期强制]
/// A5：reclaim_segments 内部已强制安全期校验（封存超 RECLAIM_SAFETY OR strong_count==1），
/// 不安全的段会被跳过保留，返回已回收段数。caller 可对未回收段稍后重试。
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
        // 软删向量后 compact，KNN 不再含删向量，活向量数正确
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 8);
        for i in 0..10u64 {
            let mut v = vec![0.0f32; 8];
            v[0] = i as f32;
            idx.insert(i, &v, None).unwrap();
        }
        // 软删 3,7
        idx.delete(3, None).unwrap();
        idx.delete(7, None).unwrap();
        assert_eq!(idx.len(), 8, "8 live after 2 soft-delete");
        let res = run_compact(&idx).unwrap();
        assert_eq!(res.live_vectors, 8, "compact keeps 8 live");
        // KNN 不含删向量
        let q = vec![7.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let knn = idx.search_knn(&q, 3, None).unwrap();
        assert!(!knn.iter().any(|(id, _)| *id == 7));
        assert!(!knn.iter().any(|(id, _)| *id == 3));
        // 活向量数仍 8（图重建后）
        assert_eq!(idx.len(), 8);
    }

    #[test]
    fn compact_preserves_knn_correctness() {
        // compact 后 KNN 召回与 compact 前一致（图重建正确）
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
        // compact → checkpoint → 重开，图从 snapshot 恢复，活向量正确
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
        // compact 产旧段 id，reclaim 删文件；用真实 vec_data 校验（非伪造路径）
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("vec_data");
        let idx = make_index(dir.path(), 4);
        for i in 0..5u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.insert(i, &v, None).unwrap();
        }
        // compact 前 seg 0 文件存在
        assert!(data_dir.join("vec_payload_0000.mmap").exists());
        let res = run_compact(&idx).unwrap();
        assert!(!res.reclaimable_segs.is_empty());
        // reclaim 删旧段文件（A5：无 in-flight reader → strong_count==1 → 安全删除）
        let n = reclaim(&idx, &res.reclaimable_segs).unwrap();
        assert_eq!(
            n,
            res.reclaimable_segs.len(),
            "all old segs reclaimed (no reader)"
        );
        // 旧段文件不存在（COW 后旧段物理删除）
        for sid in &res.reclaimable_segs {
            let p = data_dir.join(format!("vec_payload_{:04}.mmap", sid));
            assert!(!p.exists(), "old seg {} file reclaimed", sid);
        }
        // 新段文件仍存在（活向量所在），KNN 仍可用
        assert!(idx.search_knn(&[0.0, 0.0, 0.0, 0.0], 1, None).is_ok());
    }

    #[test]
    fn reclaim_idempotent_on_deleted_segs() {
        // A5：已删段二次 reclaim 返回 0（幂等，不报错）
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
        // A3/H4：compact 期间读不阻塞。单线程无法真并发，验证 compact 后读一致即可。
        // 真并发读不阻塞由 MVCC（ heed RoTxn + 旧段 Arc）保证，此处校验结果正确性。
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
