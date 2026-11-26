//! Crash recovery end-to-end integration tests — M1 rewrite [v2 §2.7/R2/H5/F2]
//!
//! Writes go through the real Engine (WAL fsync as the single sync point + mmap segments flushed lazily, H5/F2),
//! simulates kill-9 (drop Engine without checkpoint), reopens with Engine::open auto-recover,
//! and asserts data is persisted and readable back. Replaces the old loc/registered_blocks unit tests
//! (physical block registration after the logical WAL is deprecated; recover replays via submodule idempotent
//! methods and no longer rolls back unregistered blocks).
//!
//! Acceptance:
//!   - Reopen after crash: all KV + vectors recovered and readable
//!   - Repeated open-recover is idempotent (no accumulation, insert dup skipped)
//!   - Crash after checkpoint: marker takes effect, only the tail is replayed (checkpointed portion not replayed)
//!   - Deletes replayed via WAL: key/id actually gone after recover
//!   - WAL truncated after recover: plan empty, no replay on next open
//!   - seq continues monotonically

use fs_core::mem::build_recover_plan;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine};
use tempfile::tempdir;

fn schema(dim: usize) -> VectorSchema {
    VectorSchema::new(dim, MetricKind::L2)
}

// ---- Placeholder: each test filled in via Edit ----

/// kill-9: write KV+vectors → drop (no checkpoint) → reopen and recover → all readable
#[test]
fn crash_without_checkpoint_recovers_all_writes() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 4usize;

    // phase 1: write via Engine (WAL fsync), no checkpoint, drop simulates crash
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..5u64 {
            let k = format!("k{}", i);
            let v = format!("val{}", i);
            engine.put_kv(k.as_bytes(), v.as_bytes(), None).unwrap();
            let mut vec = vec![0.0f32; dim];
            vec[0] = i as f32;
            engine.insert_vector(100 + i, &vec, None).unwrap();
        }
        // No checkpoint, drop directly — kill-9
    }

    // phase 2: reopen Engine::open auto-recovers, assert data readable
    let engine = Engine::open(&home, None, 0).unwrap();
    for i in 0..5u64 {
        let k = format!("k{}", i);
        let buf = engine.get_kv_zero_copy(k.as_bytes(), None).unwrap();
        assert!(buf.is_some(), "kv k{} recovered after crash", i);
        let owned = buf.unwrap().to_owned_slice();
        let expect = format!("val{}", i);
        assert_eq!(&owned, expect.as_bytes(), "kv k{} value recovered", i);
    }
    // Vector recover: all 5 searchable
    let g = engine.vec_index().unwrap();
    let idx = g.as_ref().unwrap();
    assert_eq!(idx.len(), 5, "5 vectors recovered");
    let q = vec![3.0f32, 0.0, 0.0, 0.0];
    let hits = idx.search_knn(&q, 1, None).unwrap();
    assert_eq!(hits[0].0, 103, "nearest id=103 recovered");
}

/// R2: repeated open-recover is idempotent (insert dup skipped, put_kv overwrites, no accumulation errors)
#[test]
fn repeated_reopen_is_idempotent() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // phase 1: write 3 vectors, crash
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..3u64 {
            engine.insert_vector(i, &[i as f32, 0.0], None).unwrap();
        }
    }

    // Reopen 3 times in a row, each recover idempotent: vector count stays 3, no panic, no accumulation
    for round in 0..3 {
        let engine = Engine::open(&home, None, 0).unwrap();
        let g = engine.vec_index().unwrap();
        let idx = g.as_ref().unwrap();
        assert_eq!(idx.len(), 3, "round {}: 3 vectors, no accumulation", round);
    }

    // After the last round the WAL is truncated (finalize_recover called inside open), plan empty
    let plan = build_recover_plan(&home.join("wal")).unwrap();
    assert!(plan.entries.is_empty(), "WAL truncated, no pending replay");
}

/// Crash after checkpoint: marker takes effect, only the tail is replayed
#[test]
fn crash_after_checkpoint_replays_only_tail() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // phase 1: write 4 vectors + checkpoint (marker applied_seq=4, WAL truncated)
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..4u64 {
            engine.insert_vector(i, &[i as f32, 0.0], None).unwrap();
        }
        engine.checkpoint().unwrap();
        // After checkpoint, keep writing the tail (2 ops), then crash
        engine.insert_vector(4, &[4.0, 0.0], None).unwrap();
        engine.put_kv(b"tail_k", b"tail_v", None).unwrap();
    }

    // phase 2: reopen after crash, plan should only contain the tail (marker applied_seq already advanced)
    // Note: open already recovered + truncated, so the plan here should be empty
    let engine = Engine::open(&home, None, 0).unwrap();
    let plan_after = engine.pending_recover_plan().unwrap();
    assert!(
        plan_after.entries.is_empty(),
        "tail replayed + truncated on open, no pending"
    );
    // Checkpointed 4 vectors + tail vectors all present
    let g = engine.vec_index().unwrap();
    let idx = g.as_ref().unwrap();
    assert_eq!(idx.len(), 5, "4 checkpointed + 1 tail vector recovered");
    let buf = engine.get_kv_zero_copy(b"tail_k", None).unwrap();
    assert_eq!(
        buf.unwrap().to_owned_slice(),
        b"tail_v",
        "tail kv recovered"
    );
}

/// Deletes replayed via WAL: KV key + vector id actually gone after recover
#[test]
fn delete_ops_replayed_on_recover() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // phase 1: write KV+vectors, then delete, crash (deletes recorded in WAL, not checkpointed)
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        engine.put_kv(b"del_k", b"del_v", None).unwrap();
        engine.insert_vector(7, &[7.0, 0.0], None).unwrap();
        engine.delete_kv(b"del_k", None).unwrap();
        engine.delete_vector(7, None).unwrap();
        engine.put_kv(b"keep_k", b"keep_v", None).unwrap();
        engine.insert_vector(8, &[8.0, 0.0], None).unwrap();
    }

    // phase 2: reopen and recover; deleted gone, kept present
    let engine = Engine::open(&home, None, 0).unwrap();
    let del_buf = engine.get_kv_zero_copy(b"del_k", None).unwrap();
    assert!(del_buf.is_none(), "deleted kv key gone after recover");
    let keep_buf = engine.get_kv_zero_copy(b"keep_k", None).unwrap();
    assert_eq!(
        keep_buf.unwrap().to_owned_slice(),
        b"keep_v",
        "kept kv survived"
    );
    let g = engine.vec_index().unwrap();
    let idx = g.as_ref().unwrap();
    assert_eq!(idx.len(), 1, "1 vector kept (id=8), deleted id=7 gone");
    let hits = idx.search_knn(&[8.0, 0.0], 1, None).unwrap();
    assert_eq!(hits[0].0, 8, "kept vector id=8 recovered");
}

/// WAL truncated after recover: plan empty on next open
#[test]
fn recover_truncates_wal_no_replay_after() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // Write 3 entries, crash
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..3u64 {
            engine.insert_vector(i, &[i as f32, 0.0], None).unwrap();
        }
    }
    // Before the crash the WAL holds 3 pending entries
    let plan_before = build_recover_plan(&home.join("wal")).unwrap();
    assert_eq!(
        plan_before.entries.len(),
        3,
        "3 entries pending before reopen"
    );

    // Reopen: recover replays + finalize_recover truncates
    let _engine = Engine::open(&home, None, 0).unwrap();
    let plan_after = _engine.pending_recover_plan().unwrap();
    assert!(
        plan_after.entries.is_empty(),
        "WAL truncated after recover, no pending replay"
    );
}

/// seq continues monotonically: after crash + reopen, new writes pick up at max+1
#[test]
fn crash_resume_seq_continues_monotonic() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // Write 100 KV entries, crash
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..100u64 {
            let k = vec![i as u8];
            engine.put_kv(&k, b"v", None).unwrap();
        }
    }
    // After crash + reopen: new writes continue the sequence (recover never rewinds seq)
    let engine = Engine::open(&home, None, 0).unwrap();
    engine.put_kv(b"new", b"v", None).unwrap();
    // Recover plan should be empty (open already truncated), but the new write has landed in the WAL
    let plan = engine.pending_recover_plan().unwrap();
    // After checkpoint the marker applied_seq advances; no checkpoint here so applied_seq=0,
    // but open already finalized_recover truncated to max_seq(100), so the new write has seq=101 and the plan holds 1 entry
    assert_eq!(plan.entries.len(), 1, "only the new write seq=101 pending");
    assert_eq!(
        plan.entries[0].seq, 101,
        "new write seq continues from max+1"
    );
}
