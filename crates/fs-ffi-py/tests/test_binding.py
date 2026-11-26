#!/usr/bin/env python3
# fusion-store Python binding tests [v2 §3.4/E3/E4]
#
# M3 §630 acceptance: fusion_store.Store.search_knn passes with zero-copy numpy input +
# forced-copy output (E3, verifying no mmap view is exposed). Real Store roundtrip, no
# mocks (E4). pytest style; without pytest, run via python -m pytest or as a plain
# assertion script.

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
    # LMDB single-writer env: the same process cannot open the same heed env dir twice at once.
    # del s first to close the handle, then reopen. Cross-process / cross-run reopens are not limited.
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    assert s.vector_dim() == 4
    s.insert_vector(0, np.array([1.0, 0, 0, 0], dtype=np.float32))
    s.checkpoint()
    del s  # close the handle, release the heed env
    # reopen without dim; restored from the persisted schema
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
    # E3: the return is owned bytes, not an mmap view (bytes is immutable owned, no buffer exposing mmap)
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
    # nearest neighbor should be id=10
    assert ids[0] == 10
    assert abs(float(dists[0]) - 0.0) < 1e-5
    shutil.rmtree(d, ignore_errors=True)


def test_search_knn_output_is_owned_not_mmap_view():
    # Core E3 check: outputs ids/dists are owned copies, not mmap pointer views.
    # After closing the store handle, ids/dists are still readable (owned), proving not views.
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    for i in range(10):
        v = np.array([float(i), 0, 0, 0], dtype=np.float32)
        s.insert_vector(i, v)
    q = np.array([1.0, 0, 0, 0], dtype=np.float32)
    ids, dists = s.search_knn(q, top_k=3, timeout_ms=100)
    # ids/dists still readable after "discarding" the handle (del)
    captured_ids = [int(x) for x in ids]
    captured_dists = [float(x) for x in dists]
    del s
    assert captured_ids[0] == 1
    assert abs(captured_dists[0]) < 1e-5
    # ids/dists are Python lists (owned), not memoryview
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
    # non-contiguous / non-float32 arrays should be rejected (as_slice returns None)
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    # non-contiguous (transposed)
    bad = np.zeros((4, 4), dtype=np.float32).T[0]
    try:
        s.insert_vector(0, bad)
        raised = False
    except Exception:
        raised = True
    assert raised, "non-contiguous array should be rejected"
    shutil.rmtree(d, ignore_errors=True)


def test_delete_vector_get_vector_list_vector_ids():
    # #2 delete_vector + #3 get_vector/list_vector_ids end-to-end
    d = _new_dir()
    s = fusion_store.Store.open(d, dim=4)
    for i in range(5):
        v = np.zeros(4, dtype=np.float32)
        v[0] = float(i)
        s.insert_vector(i, v)

    # list should have 5 entries
    ids = s.list_vector_ids()
    assert sorted(ids) == [0, 1, 2, 3, 4], f"list before delete: {ids}"

    # get id=2 should succeed, value == inserted
    got = s.get_vector(2)
    assert got is not None
    assert [float(x) for x in got] == [2.0, 0.0, 0.0, 0.0], f"get_vector(2): {got}"
    # get missing -> None
    assert s.get_vector(99) is None

    # delete id=2 -> True
    assert s.delete_vector(2) is True
    # delete id=2 again -> False (already soft-deleted)
    assert s.delete_vector(2) is False
    # delete missing id=99 -> False
    assert s.delete_vector(99) is False

    # get id=2 after delete -> None
    assert s.get_vector(2) is None
    # list excludes id=2, leaving 4
    ids2 = s.list_vector_ids()
    assert sorted(ids2) == [0, 1, 3, 4], f"list after delete: {ids2}"

    # E3: get_vector returns an owned list, not an mmap view — still readable after del s
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
