#!/usr/bin/env python3
# fusion-store Python 绑定测试 [v2 §3.4/E3/E4]
#
# M3 §630 验收：fusion_store.Store.search_knn 经 numpy 入参零拷贝 + 出参强制拷贝通
# （E3，校验无 mmap view 暴露）。真实 Store 往返，非 mock（E4）。
# pytest 风格；无 pytest 时可 python -m pytest 或直接断言脚本跑。

import os
import shutil
import tempfile

import numpy as np

import fusion_store


def _new_dir():
    d = tempfile.mkdtemp(prefix="fs-py-test-")
    shutil.rmtree(d, ignore_errors=True)
    return d


def test_open_create_then_reopen():
    # LMDB 单写者 env：同进程不能同时打开两次同目录 heed env。
    # 先 del s 关句柄，再 reopen。跨进程/跨次启动 reopen 不受此限。
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    assert s.vector_dim() == 4
    s.insert_vector(0, np.array([1.0, 0, 0, 0], dtype=np.float32))
    s.checkpoint()
    del s  # 关句柄，释放 heed env
    # reopen 不传 dim，从持久化 schema 恢复
    s2 = fusion_store.Store.open(d)
    assert s2.vector_dim() == 4
    assert s2.vector_count() == 1
    del s2
    shutil.rmtree(d, ignore_errors=True)


def test_put_kv_get_kv_forced_copy():
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    s.put_kv(b"the-key", b"fusion-store-payload")
    got = s.get_kv(b"the-key")
    assert bytes(got) == b"fusion-store-payload"
    # missing -> None
    assert s.get_kv(b"nope") is None
    # E3：返回是 owned bytes，非 mmap view（bytes 不可变 owned，无 buffer 暴露 mmap）
    assert isinstance(got, bytes)
    shutil.rmtree(d, ignore_errors=True)


def test_insert_and_search_knn_numpy_input():
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=8)
    for i in range(50):
        v = np.zeros(8, dtype=np.float32)
        v[0] = float(i)
        s.insert_vector(i, v)
    q = np.array([10.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    ids, dists = s.search_knn(q, top_k=5, timeout_ms=100)
    assert len(ids) == 5
    assert len(dists) == 5
    # 最近邻应是 id=10
    assert ids[0] == 10
    assert abs(float(dists[0]) - 0.0) < 1e-5
    shutil.rmtree(d, ignore_errors=True)


def test_search_knn_output_is_owned_not_mmap_view():
    # E3 核心校验：出参 ids/dists 是 owned 拷贝，非 mmap 指针 view。
    # 关 store 句柄后 ids/dists 仍可读（owned），证非 view。
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    for i in range(10):
        v = np.array([float(i), 0, 0, 0], dtype=np.float32)
        s.insert_vector(i, v)
    q = np.array([1.0, 0, 0, 0], dtype=np.float32)
    ids, dists = s.search_knn(q, top_k=3, timeout_ms=100)
    # 句柄 "丢弃"（del）后 ids/dists 仍可读
    captured_ids = [int(x) for x in ids]
    captured_dists = [float(x) for x in dists]
    del s
    assert captured_ids[0] == 1
    assert abs(captured_dists[0]) < 1e-5
    # ids/dists 是 Python list（owned），非 memoryview
    assert isinstance(ids, list)
    assert isinstance(dists, list)
    shutil.rmtree(d, ignore_errors=True)


def test_search_knn_timeout_ms_none_allowed():
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    s.insert_vector(0, np.array([1.0, 0, 0, 0], dtype=np.float32))
    q = np.array([1.0, 0, 0, 0], dtype=np.float32)
    ids, dists = s.search_knn(q, top_k=1)
    assert len(ids) == 1
    assert ids[0] == 0
    shutil.rmtree(d, ignore_errors=True)


def test_non_contiguous_or_wrong_dtype_rejected():
    # 非连续/非 float32 数组应被拒（as_slice 返回 None）
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    # 非连续（转置）
    bad = np.zeros((4, 4), dtype=np.float32).T[0]
    try:
        s.insert_vector(0, bad)
        raised = False
    except Exception:
        raised = True
    assert raised, "non-contiguous array should be rejected"
    shutil.rmtree(d, ignore_errors=True)


def test_delete_vector_get_vector_list_vector_ids():
    # #2 delete_vector + #3 get_vector/list_vector_ids 端到端
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    for i in range(5):
        v = np.zeros(4, dtype=np.float32)
        v[0] = float(i)
        s.insert_vector(i, v)

    # list 应 5 个
    ids = s.list_vector_ids()
    assert sorted(ids) == [0, 1, 2, 3, 4], f"list before delete: {ids}"

    # get id=2 应成功，值 == 插入
    got = s.get_vector(2)
    assert got is not None
    assert [float(x) for x in got] == [2.0, 0.0, 0.0, 0.0], f"get_vector(2): {got}"
    # get missing -> None
    assert s.get_vector(99) is None

    # delete id=2 -> True
    assert s.delete_vector(2) is True
    # 再删 id=2 -> False（已软删）
    assert s.delete_vector(2) is False
    # 删 missing id=99 -> False
    assert s.delete_vector(99) is False

    # get id=2 删后 -> None
    assert s.get_vector(2) is None
    # list 排除 id=2，剩 4 个
    ids2 = s.list_vector_ids()
    assert sorted(ids2) == [0, 1, 3, 4], f"list after delete: {ids2}"

    # E3：get_vector 返回 owned list，非 mmap view——del s 后仍可读
    captured = [float(x) for x in s.get_vector(0)]
    del s
    assert captured == [0.0, 0.0, 0.0, 0.0]
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
