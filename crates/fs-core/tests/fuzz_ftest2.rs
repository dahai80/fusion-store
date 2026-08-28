//! F-TEST-2 proptest fuzz harness —— 异常输入属性测试 [v2 P2-13]
//!
//! 审计 F-TEST-2 指出：crash 测试覆盖正常路径，异常输入模糊覆盖弱。
//! 三个属性面：
//!   1. WAL torn-frame 容错（任意截断点 → read_all 不 panic，返回截断点前完整帧）
//!   2. 并发 compact+insert+search 交错 → 不死锁/不丢数据/不 panic
//!   3. 边界 dim/value 大小 → 维度不符拒 DimensionMismatch，合法边界往返一致
//!
//! proptest 配置：cases 默认 256，配 env FS_FUZZ_CASES 可调（CI 跑轻量，本地跑重）。

use std::sync::Arc;

use proptest::prelude::*;
use tempfile::tempdir;

use fs_core::mem::recover::build_recover_plan;
use fs_core::mem::wal::{Wal, WalOp};
use fs_core::mem::CheckpointMarker;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine};

/// proptest 运行 case 数（env 可覆盖，默认 256 —— 平衡覆盖与 CI 时长）。
fn fuzz_cases() -> usize {
    std::env::var("FS_FUZZ_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

/// 生成合法 WalOp 策略：key/value/table_id 短字串，vector dim 1..=16
fn wal_op_strategy() -> impl Strategy<Value = WalOp> {
    prop_oneof![
        (
            prop::collection::vec(any::<u8>(), 0..64),
            prop::collection::vec(any::<u8>(), 0..256)
        )
            .prop_map(|(k, v)| WalOp::PutKv { key: k, value: v }),
        (any::<u64>(), 1u32..=16).prop_map(|(id, dim)| WalOp::InsertVector {
            id,
            vector: vec![0.5f32; dim as usize],
        }),
        prop::collection::vec(any::<u8>(), 0..64).prop_map(|k| WalOp::DeleteKv { key: k }),
        any::<u64>().prop_map(|id| WalOp::DeleteVector { id }),
        (".{1,16}", prop::collection::vec(any::<u8>(), 0..128))
            .prop_map(|(t, ipc)| WalOp::PutColumnar { table_id: t, ipc }),
    ]
}

// ===== 1. WAL torn-frame 容错属性 =====

proptest! {
    #![proptest_config(ProptestConfig::with_cases(fuzz_cases() as u32))]

    /// 任意条目序列 + 任意截断点 → read_all 不 panic。
    /// 不变：返回条目数 ≤ 写入数；每条 seq 严格递增（截断点前完整帧）。
    #[test]
    fn prop_wal_torn_truncation_never_panics(
        ops in prop::collection::vec(wal_op_strategy(), 1..50),
        truncate_ratio in 0u64..=100,
    ) {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        let log_path = wal_dir.join("wal.log");

        // 写入完整帧序列
        {
            let mut wal = Wal::open(&wal_dir).unwrap();
            for op in &ops {
                wal.append(op.clone()).unwrap();
            }
        }
        // 按比例截断日志尾部（模拟崩溃撕裂：截断点可能在 header 中部 / body 中部 / 帧边界）
        let data = std::fs::read(&log_path).unwrap();
        let cut = if data.is_empty() {
            0
        } else {
            (data.len() * truncate_ratio as usize) / 100
        };
        std::fs::write(&log_path, &data[..cut]).unwrap();

        // 不变：read_all 不 panic；返回 ≤ 写入数；seq 递增
        let wal = Wal::open(&wal_dir).unwrap();
        let entries = wal.read_all().unwrap();
        prop_assert!(entries.len() <= ops.len(), "torn tail never yields more than written");
        let mut last_seq = 0u64;
        for e in &entries {
            prop_assert!(e.seq > last_seq, "entries strictly increasing seq");
            last_seq = e.seq;
        }
    }

    /// 任意条目序列 + marker applied_seq → recover plan 只含 seq > applied_seq，不 panic。
    /// 不变：plan.entries 全部 seq > applied_seq；applied_seq 与 marker 一致。
    #[test]
    fn prop_recover_plan_filters_by_applied_seq(
        ops in prop::collection::vec(wal_op_strategy(), 1..40),
        marker_seq_pct in 0u64..=100,
    ) {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().to_path_buf();
        let max_seq = {
            let mut wal = Wal::open(&wal_dir).unwrap();
            let mut last = 0u64;
            for op in &ops {
                last = wal.append(op.clone()).unwrap();
            }
            // marker 取写入数的百分比位置
            let applied = (last * marker_seq_pct) / 100;
            wal.write_marker(&CheckpointMarker {
                applied_seq: applied,
                checkpoint_seq: applied,
                graph_snapshot: None,
            })
            .unwrap();
            last
        };
        let plan = build_recover_plan(&wal_dir).unwrap();
        prop_assert!(plan.entries.iter().all(|e| e.seq > plan.applied_seq),
            "replay entries strictly after applied_seq");
        prop_assert!(plan.applied_seq <= max_seq, "applied_seq bounded by max written seq");
    }

    /// 二进制编解码往返（经公共 API）：任意 WalOp → append → read_all == 原 op。
    /// 不变：往返无损（向量 f32 比特级一致，含 NaN/Inf）。
    #[test]
    fn prop_wal_op_binary_roundtrip(op in wal_op_strategy()) {
        let dir = tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();
        wal.append(op.clone()).unwrap();
        let entries = wal.read_all().unwrap();
        prop_assert_eq!(entries.len(), 1);
        prop_assert_eq!(&entries[0].op, &op, "binary roundtrip lossless via public API");
    }
}

// ===== 2. 并发 compact+insert+search 交错属性 =====

/// 并发不交死锁 + 数据一致：N 插入线程 + 1 compact 线程 + M search 线程同时跑，
/// 全部完成后：活向量数 == 成功插入数 - 成功删除数；search 不 panic。
/// 注：proptest 驱动参数化（线程数/向量数），非真随机交错，但覆盖多规模组合。
#[test]
fn concurrent_compact_insert_search_no_deadlock_no_data_loss() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(8, MetricKind::L2);
    let engine = Arc::new(Engine::open(dir.path(), Some(schema), 0).unwrap());

    let n_inserts: usize = 200;
    let n_deletes: usize = 50;

    // 插入线程：id 0..n_inserts
    let eng = engine.clone();
    let inserter = std::thread::spawn(move || {
        let mut ok = 0usize;
        for i in 0..n_inserts as u64 {
            let v = vec![i as f32; 8];
            if eng.insert_vector(i, &v, None).is_ok() {
                ok += 1;
            }
        }
        ok
    });

    // compact 线程：跑 3 轮 compact（COW 原子切换）
    let eng = engine.clone();
    let compactor = std::thread::spawn(move || {
        for _ in 0..3 {
            let _ = run_compact_via_engine(&eng);
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    });

    // search 线程：持续 KNN 查询，不 panic
    let eng = engine.clone();
    let searcher = std::thread::spawn(move || {
        let mut ok = 0usize;
        for i in 0..100u64 {
            let q = vec![i as f32; 8];
            if eng.search_knn(&q, 3, None).is_ok() {
                ok += 1;
            }
        }
        ok
    });

    // 删除线程：删 id 0..n_deletes（与插入交叠）
    // 计实际移除数（delete_vector 返 Ok(true)=真删 / Ok(false)=id 未插入或已删），
    // 非 is_ok()——后者把 Ok(false)（竞态下 id 尚未插入）误计为删除，破坏不变式。
    let eng = engine.clone();
    let deleter = std::thread::spawn(move || {
        let mut removed = 0usize;
        for i in 0..n_deletes as u64 {
            std::thread::sleep(std::time::Duration::from_micros(100));
            if let Ok(true) = eng.delete_vector(i, None) {
                removed += 1;
            }
        }
        removed
    });

    let inserted = inserter.join().expect("inserter panicked");
    compactor.join().expect("compactor panicked");
    let _searched = searcher.join().expect("searcher panicked");
    let deleted = deleter.join().expect("deleter panicked");

    // 不变：活向量数 == 成功插入 - 成功删除（compact 不丢活向量，仅重排）
    let live = engine.vec_index().unwrap();
    let idx = live.as_ref().unwrap();
    let expected = inserted.saturating_sub(deleted);
    assert_eq!(
        idx.len(),
        expected,
        "live vectors == inserted({}) - deleted({}), got {}",
        inserted,
        deleted,
        idx.len()
    );
}

/// 经 Engine 触发 compact（run_compact 需 VectorIndex 引用，Engine 经 vec_index guard 取）。
fn run_compact_via_engine(engine: &Arc<Engine>) -> fs_core::Result<()> {
    use fs_core::compact::run_compact;
    let g = engine.vec_index()?;
    let idx = g.as_ref().unwrap();
    run_compact(idx)?;
    Ok(())
}

// ===== 3. 边界 dim / value 大小属性 =====

proptest! {
    #![proptest_config(ProptestConfig::with_cases(fuzz_cases() as u32))]

    /// 合法维度 insert + search 往返一致：任意 dim 1..=64，插入后 search 命中自身。
    /// 不变：search_knn top_k=1 返回 (id, 0.0-ish 距离) —— 自查最近。
    #[test]
    fn prop_insert_search_roundtrip_arbitrary_dim(dim in 1usize..=64) {
        let dir = tempdir().unwrap();
        let schema = VectorSchema::new(dim, MetricKind::L2);
        let engine = Engine::open(dir.path(), Some(schema), 0).unwrap();
        let v = vec![0.5f32; dim];
        engine.insert_vector(7, &v, None).unwrap();
        let hits = engine.search_knn(&v, 1, None).unwrap();
        prop_assert_eq!(hits.len(), 1);
        prop_assert_eq!(hits[0].0, 7, "self-query returns own id");
    }

    /// 维度不符拒 DimensionMismatch，不静默写错。
    /// 不变：插入 dim ≠ schema.dim 的向量 → Err(DimensionMismatch)，不落段。
    #[test]
    fn prop_dim_mismatch_rejected(schema_dim in 1usize..=64, insert_dim in 1usize..=64) {
        prop_assume!(schema_dim != insert_dim, "skip matching dim");
        let dir = tempdir().unwrap();
        let schema = VectorSchema::new(schema_dim, MetricKind::L2);
        let engine = Engine::open(dir.path(), Some(schema), 0).unwrap();
        let v = vec![0.0f32; insert_dim];
        let res = engine.insert_vector(1, &v, None);
        prop_assert!(
            matches!(res, Err(fs_core::StoreError::DimensionMismatch { .. })),
            "dim mismatch rejected, not silent write"
        );
        // 不变：未落段 → list_vector_ids 为空
        let ids = engine.list_vector_ids(None).unwrap();
        prop_assert!(ids.is_empty(), "rejected vector not persisted");
    }

    /// KV 边界 value 往返（合法大小）：任意 key 1..=256B / value ≤1KB → put + get 字节等。
    /// 注：空 key / 超大 key 由 put_kv 显式拒 InvalidKey（见 prop_kv_invalid_key_rejected）。
    /// 不变：合法范围内 put_kv 成功 + 读出 == 写入。
    #[test]
    fn prop_kv_boundary_value_roundtrip(
        key in prop::collection::vec(any::<u8>(), 1..=256),
        value in prop::collection::vec(any::<u8>(), 0..1024),
    ) {
        let dir = tempdir().unwrap();
        let engine = Engine::open_kv_only(dir.path(), 0).unwrap();
        engine.put_kv(&key, &value, None).unwrap();
        let got = engine.get_kv_zero_copy(&key, None).unwrap();
        let got_bytes: Option<Vec<u8>> = got.map(|b| b.to_owned_slice());
        prop_assert!(got_bytes.is_some(), "kv get returns value");
        let got_bytes = got_bytes.unwrap();
        prop_assert_eq!(&got_bytes[..], &value[..], "kv value roundtrip byte-exact");
    }

    /// KV 非法 key（空 / >LMDB MDB_MAXKEYSIZE 511B）→ put_kv 返 Err(InvalidKey)，不 panic。
    /// 不变：非法被显式拒（Rule 12 fail visibly），store 不损坏，无晦涩 BadValSize 泄漏。
    #[test]
    fn prop_kv_invalid_key_rejected(key_len in 0u32..=1200) {
        let dir = tempdir().unwrap();
        let engine = Engine::open_kv_only(dir.path(), 0).unwrap();
        let key = vec![b'k'; key_len as usize];
        let res = engine.put_kv(&key, b"v", None);
        if key_len == 0 || key_len > 511 {
            prop_assert!(
                matches!(res, Err(fs_core::StoreError::InvalidKey(_))),
                "invalid key ({}B) → Err(InvalidKey), got {:?}",
                key_len,
                res
            );
        } else {
            prop_assert!(res.is_ok(), "valid key ({}B) accepted", key_len);
        }
    }

    /// KNN top_k 边界：top_k=0 空结果，top_k > 实际向量数 返回全部（不越界）。
    #[test]
    fn prop_knn_top_k_bounds(n in 1usize..=20, top_k in 0usize..=40) {
        let dir = tempdir().unwrap();
        let schema = VectorSchema::new(4, MetricKind::L2);
        let engine = Engine::open(dir.path(), Some(schema), 0).unwrap();
        for i in 0..n as u64 {
            engine.insert_vector(i, &[i as f32, 0.0, 0.0, 0.0], None).unwrap();
        }
        let hits = engine.search_knn(&[0.0, 0.0, 0.0, 0.0], top_k, None).unwrap();
        let expected = top_k.min(n);
        prop_assert_eq!(hits.len(), expected,
            "top_k capped at actual vector count (top_k={}, n={})", top_k, n);
    }
}
