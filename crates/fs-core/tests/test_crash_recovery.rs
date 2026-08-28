//! Crash recovery 端到端集成测试 —— M1 重写 [v2 §2.7/R2/H5/F2]
//!
//! 真实经 Engine 写（WAL fsync 唯一同步点 + mmap 段延迟刷，H5/F2），
//! 模拟 kill-9（drop Engine 无 checkpoint），重开 Engine::open 自动 recover，
//! 断言数据落盘可读回。替代旧 loc/registered_blocks 单元测试（逻辑 WAL 后物理
//! 块登记已废弃，recover 经子模块幂等方法重放，不再回滚未登记块）。
//!
//! 验收：
//!   - 崩溃后重开：KV + 向量全部 recover 可读
//!   - 重复 open recover 幂等（无累积、insert dup 跳过）
//!   - checkpoint 后崩溃：marker 生效，仅 tail 重放（已 checkpoint 部分不重放）
//!   - delete 经 WAL 重放：recover 后 key/id 确实删
//!   - recover 后 WAL 截断：plan 空，再开无重放
//!   - seq 单调续号

use fs_core::mem::build_recover_plan;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine};
use tempfile::tempdir;

fn schema(dim: usize) -> VectorSchema {
    VectorSchema::new(dim, MetricKind::L2)
}

// ---- 占位：各测试用 Edit 填充 ----

/// kill-9：写 KV+向量 → drop（无 checkpoint）→ 重开 recover → 全部可读
#[test]
fn crash_without_checkpoint_recovers_all_writes() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 4usize;

    // phase 1：经 Engine 写（WAL fsync），不 checkpoint，drop 模拟 crash
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
        // 不 checkpoint，直接 drop —— kill-9
    }

    // phase 2：重开 Engine::open 自动 recover，断言数据可读
    let engine = Engine::open(&home, None, 0).unwrap();
    for i in 0..5u64 {
        let k = format!("k{}", i);
        let buf = engine.get_kv_zero_copy(k.as_bytes(), None).unwrap();
        assert!(buf.is_some(), "kv k{} recovered after crash", i);
        let owned = buf.unwrap().to_owned_slice();
        let expect = format!("val{}", i);
        assert_eq!(&owned, expect.as_bytes(), "kv k{} value recovered", i);
    }
    // 向量 recover：5 条全可检索
    let g = engine.vec_index().unwrap();
    let idx = g.as_ref().unwrap();
    assert_eq!(idx.len(), 5, "5 vectors recovered");
    let q = vec![3.0f32, 0.0, 0.0, 0.0];
    let hits = idx.search_knn(&q, 1, None).unwrap();
    assert_eq!(hits[0].0, 103, "nearest id=103 recovered");
}

/// R2：重复 open recover 幂等（insert dup 跳过、put_kv 覆盖，无累积错误）
#[test]
fn repeated_reopen_is_idempotent() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // phase 1：写 3 向量，crash
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..3u64 {
            engine.insert_vector(i, &[i as f32, 0.0], None).unwrap();
        }
    }

    // 连续 reopen 3 次，每次 recover 幂等：向量数恒 3，无 panic 无累积
    for round in 0..3 {
        let engine = Engine::open(&home, None, 0).unwrap();
        let g = engine.vec_index().unwrap();
        let idx = g.as_ref().unwrap();
        assert_eq!(idx.len(), 3, "round {}: 3 vectors, no accumulation", round);
    }

    // 末次后 WAL 已截断（finalize_recover 在 open 内调），plan 空
    let plan = build_recover_plan(&home.join("wal")).unwrap();
    assert!(plan.entries.is_empty(), "WAL truncated, no pending replay");
}

/// checkpoint 后 crash：marker 生效，仅 tail 重放
#[test]
fn crash_after_checkpoint_replays_only_tail() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // phase 1：写 4 向量 + checkpoint（marker applied_seq=4，截断 WAL）
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..4u64 {
            engine.insert_vector(i, &[i as f32, 0.0], None).unwrap();
        }
        engine.checkpoint().unwrap();
        // checkpoint 后继续写 tail（2 条），再 crash
        engine.insert_vector(4, &[4.0, 0.0], None).unwrap();
        engine.put_kv(b"tail_k", b"tail_v", None).unwrap();
    }

    // phase 2：crash 后重开，plan 应只含 tail（marker applied_seq 推进过）
    // 注意：open 内已 recover+truncate，故此处 plan 应空
    let engine = Engine::open(&home, None, 0).unwrap();
    let plan_after = engine.pending_recover_plan().unwrap();
    assert!(
        plan_after.entries.is_empty(),
        "tail replayed + truncated on open, no pending"
    );
    // checkpoint 的 4 向量 + tail 2 向量全在
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

/// delete 经 WAL 重放：recover 后 KV key + 向量 id 确实删
#[test]
fn delete_ops_replayed_on_recover() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // phase 1：写 KV+向量，再删，crash（删已记 WAL 未 checkpoint）
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        engine.put_kv(b"del_k", b"del_v", None).unwrap();
        engine.insert_vector(7, &[7.0, 0.0], None).unwrap();
        engine.delete_kv(b"del_k", None).unwrap();
        engine.delete_vector(7, None).unwrap();
        engine.put_kv(b"keep_k", b"keep_v", None).unwrap();
        engine.insert_vector(8, &[8.0, 0.0], None).unwrap();
    }

    // phase 2：重开 recover，删的确实删，留的确实在
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

/// recover 后 WAL 截断：再开 plan 空
#[test]
fn recover_truncates_wal_no_replay_after() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // 写 3 条，crash
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..3u64 {
            engine.insert_vector(i, &[i as f32, 0.0], None).unwrap();
        }
    }
    // crash 前 WAL 有 3 条待重放
    let plan_before = build_recover_plan(&home.join("wal")).unwrap();
    assert_eq!(
        plan_before.entries.len(),
        3,
        "3 entries pending before reopen"
    );

    // 重开：recover 重放 + finalize_recover 截断
    let _engine = Engine::open(&home, None, 0).unwrap();
    let plan_after = _engine.pending_recover_plan().unwrap();
    assert!(
        plan_after.entries.is_empty(),
        "WAL truncated after recover, no pending replay"
    );
}

/// seq 单调续号：crash 后重开新写紧接 max+1
#[test]
fn crash_resume_seq_continues_monotonic() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("ns");
    let dim = 2usize;

    // 写 100 KV，crash
    {
        let engine = Engine::open(&home, Some(schema(dim)), 0).unwrap();
        for i in 0..100u64 {
            let k = vec![i as u8];
            engine.put_kv(&k, b"v", None).unwrap();
        }
    }
    // crash 后重开：新写应紧接续号（recover 不回卷 seq）
    let engine = Engine::open(&home, None, 0).unwrap();
    engine.put_kv(b"new", b"v", None).unwrap();
    // recover plan 应空（open 已截断），但新写已落 WAL next 条
    let plan = engine.pending_recover_plan().unwrap();
    // checkpoint 后 marker applied_seq 推进；此处无 checkpoint 故 applied_seq=0，
    // 但 open 已 finalize_recover 截断到 max_seq(100)，故新写 seq=101，plan 含 1 条
    assert_eq!(plan.entries.len(), 1, "only the new write seq=101 pending");
    assert_eq!(
        plan.entries[0].seq, 101,
        "new write seq continues from max+1"
    );
}
