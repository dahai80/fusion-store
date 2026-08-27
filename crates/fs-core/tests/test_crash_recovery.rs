//! Crash recovery 集成测试 —— 对齐 PRD M4 验收 [v2 §2.7/R2/H5]
//!
//! 模拟 kill-9：进程崩溃 = 结构体 drop（无显式 checkpoint/flush）。
//! WAL 已 fsync，重开可读回全部条目。验收：
//!   - 崩溃后重开，recover plan 含全部未 checkpoint 条目
//!   - 重复 recover 幂等（无累积、无重复块）—— R2 根治
//!   - 块泄漏防护：登记块集完整（Put/Insert 的 (seg_id,offset) 全覆盖）
//!   - checkpoint 后崩溃：marker 生效，plan 只重放 seq > applied_seq
//!   - finalize 截断后 WAL 不增长

use fs_core::mem::{build_recover_plan, finalize_recover};
use fs_core::mem::{CheckpointMarker, Wal, WalOp};
use tempfile::tempdir;

fn loc(seg: u32, off: u32, len: u32) -> fs_core::mem::ValueLocator {
    fs_core::mem::ValueLocator {
        seg_id: seg,
        offset: off,
        len,
    }
}

/// kill-9 模拟：写 WAL → 无 checkpoint 直接 drop（crash）→ 重开 recover
#[test]
fn crash_without_checkpoint_recovers_all_entries() {
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("ns/wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    // phase 1: 写 5 条 WAL，模拟 crash（无 marker，结构体 drop）
    {
        let mut wal = Wal::open(&wal_dir).unwrap();
        for i in 0..5u64 {
            wal.append(WalOp::InsertVector {
                id: i,
                loc: loc(0, (i * 16) as u32, 16),
            })
            .unwrap();
            wal.append(WalOp::PutKv {
                key: format!("k{}", i).into_bytes(),
                loc: loc(1, (i * 8) as u32, 8),
            })
            .unwrap();
        }
        // 不 write_marker，直接 drop —— 模拟 kill-9
    }

    // phase 2: 重开 recover
    let plan = build_recover_plan(&wal_dir).unwrap();
    assert_eq!(plan.applied_seq, 0, "no checkpoint -> applied_seq=0");
    assert_eq!(
        plan.entries.len(),
        10,
        "all 10 entries replayed after crash"
    );
    // 块泄漏防护：登记块含全部 10 条
    assert_eq!(plan.registered_blocks.len(), 10);
    assert!(plan.registered_blocks.contains(&(0, 0)));
    assert!(plan.registered_blocks.contains(&(0, 64)));
    assert!(plan.registered_blocks.contains(&(1, 0)));
    assert!(plan.registered_blocks.contains(&(1, 32)));
}

/// R2 核心：重复 recover 幂等，无累积、无重复块
#[test]
fn repeated_recover_is_idempotent() {
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("ns/wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    {
        let mut wal = Wal::open(&wal_dir).unwrap();
        for i in 0..20u64 {
            wal.append(WalOp::InsertVector {
                id: i,
                loc: loc(0, (i * 4) as u32, 4),
            })
            .unwrap();
        }
    }
    // 连续 recover 5 次，结果恒等
    let baseline = build_recover_plan(&wal_dir).unwrap();
    for _ in 0..5 {
        let p = build_recover_plan(&wal_dir).unwrap();
        assert_eq!(p.entries.len(), baseline.entries.len());
        assert_eq!(p.applied_seq, baseline.applied_seq);
        assert_eq!(p.registered_blocks.len(), baseline.registered_blocks.len());
        // seq 升序不变
        assert_eq!(p.entries[0].seq, 1);
        assert_eq!(p.entries[19].seq, 20);
    }
}

/// checkpoint 后 crash：marker 生效，只重放 seq > applied_seq
#[test]
fn crash_after_checkpoint_replays_only_tail() {
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("ns/wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    {
        let mut wal = Wal::open(&wal_dir).unwrap();
        for i in 0..8u64 {
            wal.append(WalOp::InsertVector {
                id: i,
                loc: loc(0, (i * 4) as u32, 4),
            })
            .unwrap();
        }
        // checkpoint applied_seq=6（已落 seq 1..8 中的前 6），然后继续写 9,10，再 crash
        wal.write_marker(&CheckpointMarker {
            applied_seq: 6,
            checkpoint_seq: 6,
            graph_snapshot: Some("hnsw_snapshot_0006.mmap".into()),
        })
        .unwrap();
        wal.append(WalOp::PutKv {
            key: b"tail1".to_vec(),
            loc: loc(2, 0, 5),
        })
        .unwrap();
        wal.append(WalOp::PutKv {
            key: b"tail2".to_vec(),
            loc: loc(2, 5, 5),
        })
        .unwrap();
        // crash（drop）
    }
    let plan = build_recover_plan(&wal_dir).unwrap();
    assert_eq!(plan.applied_seq, 6);
    // seq 7,8,9,10 未 checkpoint，重放 4 条
    assert_eq!(plan.entries.len(), 4, "seq 7..10 replayed after checkpoint");
    assert_eq!(plan.entries[0].seq, 7);
    assert_eq!(plan.entries[3].seq, 10);
}

/// finalize 截断后：WAL 只剩 seq > applied_seq，再 crash 不丢
#[test]
fn finalize_then_crash_preserves_tail() {
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("ns/wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    {
        let mut wal = Wal::open(&wal_dir).unwrap();
        for i in 0..6u64 {
            wal.append(WalOp::InsertVector {
                id: i,
                loc: loc(0, (i * 4) as u32, 4),
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
    // recover + finalize
    let plan = build_recover_plan(&wal_dir).unwrap();
    assert_eq!(plan.entries.len(), 2); // seq 5,6
    finalize_recover(&wal_dir, 4).unwrap();
    // 再写一条，crash
    {
        let mut wal = Wal::open(&wal_dir).unwrap();
        wal.append(WalOp::PutKv {
            key: b"post".to_vec(),
            loc: loc(3, 0, 4),
        })
        .unwrap();
    }
    let plan2 = build_recover_plan(&wal_dir).unwrap();
    // applied_seq 仍 4（marker 没动），重放 seq 5,6,7（被截断的 5,6 仍在，因 marker=4）
    assert_eq!(plan2.entries.len(), 3);
    assert_eq!(plan2.entries[2].seq, 7);
    // 续号正确
    let wal = Wal::open(&wal_dir).unwrap();
    assert_eq!(wal.next_seq(), 8);
}

/// 块泄漏防护：Delete 操作不登记块，不影响块集
#[test]
fn delete_ops_register_no_blocks() {
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("ns/wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    {
        let mut wal = Wal::open(&wal_dir).unwrap();
        wal.append(WalOp::InsertVector {
            id: 1,
            loc: loc(0, 0, 8),
        })
        .unwrap();
        wal.append(WalOp::DeleteVector { id: 1 }).unwrap();
        wal.append(WalOp::PutKv {
            key: b"k".to_vec(),
            loc: loc(0, 8, 4),
        })
        .unwrap();
        wal.append(WalOp::DeleteKv { key: b"k".to_vec() }).unwrap();
    }
    let plan = build_recover_plan(&wal_dir).unwrap();
    assert_eq!(plan.entries.len(), 4);
    // 只有 Insert + Put 登记块，2 个 Delete 不登记
    assert_eq!(plan.registered_blocks.len(), 2);
    assert!(plan.registered_blocks.contains(&(0, 0)));
    assert!(plan.registered_blocks.contains(&(0, 8)));
}

/// 续号：crash 后重开，新 seq 紧接 max+1（不回卷、不重复 seq）
#[test]
fn crash_resume_seq_continues_monotonic() {
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("ns/wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    {
        let mut wal = Wal::open(&wal_dir).unwrap();
        for i in 0..100u64 {
            wal.append(WalOp::DeleteKv { key: vec![i as u8] }).unwrap();
        }
    }
    let wal = Wal::open(&wal_dir).unwrap();
    assert_eq!(wal.next_seq(), 101, "seq continues from max+1 after crash");
    let plan = build_recover_plan(&wal_dir).unwrap();
    assert_eq!(plan.entries.last().unwrap().seq, 100);
}
