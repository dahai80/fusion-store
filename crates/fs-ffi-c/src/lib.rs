//! fs-ffi-c —— fusion-store C-ABI 绑定 [v2 §3.4/E3]
//!
//! 薄封装：类型转换 + 错误码映射，不含业务逻辑。cbindgen 生成 fs_store.h。
//! 读路径强制拷贝（E3）：out_ids/out_dists 是 caller-owned buffer，Rust 拷贝填入，
//! 不暴露 mmap 指针 view（跨 C-ABI 零拷贝生命周期未解，M0 先拷贝）。
//!
//! 错误码：0=Ok，负数=StoreError 变体（见 err_code）。
//!
//! 目录布局：单 path 下 vec/kv 各占独立子目录（各自 heed env，避免同路径 Env 冲突）。
//!   <path>/vec/{data,meta}   向量索引
//!   <path>/kv/{data,meta}    KV store

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

use fs_core::store::KvStore;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::store::VectorIndex;
use fs_core::StoreError;

/// C 句柄 —— 持向量索引 + KV store
pub struct FsStoreHandle {
    vec: VectorIndex,
    kv: KvStore,
}

// —— 错误码映射 ——
const OK: c_int = 0;
const ERR_IO: c_int = -1;
const ERR_HEED: c_int = -2;
const ERR_NOT_FOUND: c_int = -3;
const ERR_DUP_VECTOR: c_int = -4;
const ERR_DIM_MISMATCH: c_int = -5;
const ERR_QUOTA: c_int = -6;
const ERR_BUSY: c_int = -7;
const ERR_LOCK: c_int = -8;
const ERR_CORRUPT: c_int = -9;
const ERR_TIMEOUT: c_int = -10;
const ERR_SEGMENT_FULL: c_int = -11;
const ERR_ARROW: c_int = -12;
const ERR_SERDE: c_int = -13;
const ERR_OTHER: c_int = -99;

fn err_code(e: &StoreError) -> c_int {
    match e {
        StoreError::Io(_) => ERR_IO,
        StoreError::Heed(_) => ERR_HEED,
        StoreError::NotFound => ERR_NOT_FOUND,
        StoreError::DuplicateVector(_) => ERR_DUP_VECTOR,
        StoreError::DimensionMismatch { .. } => ERR_DIM_MISMATCH,
        StoreError::QuotaExceeded => ERR_QUOTA,
        StoreError::Busy => ERR_BUSY,
        StoreError::LockBusy => ERR_LOCK,
        StoreError::Corrupt(_) => ERR_CORRUPT,
        StoreError::Timeout => ERR_TIMEOUT,
        StoreError::SegmentFull => ERR_SEGMENT_FULL,
        StoreError::Arrow(_) => ERR_ARROW,
        StoreError::SerdeJson(_) => ERR_SERDE,
    }
}

/// 新建 store：建向量索引（锁定 schema dim）+ KV store。
/// dim=向量维度。path 不存在则创建。建后可用 fs_store_open 重开。
///
/// # Safety
/// path 须为合法 C 字符串（NUL 结尾）。out 接收堆分配句柄，caller 用 fs_store_close 释放。
#[no_mangle]
pub unsafe extern "C" fn fs_store_create(
    path: *const c_char,
    dim: usize,
    out: *mut *mut FsStoreHandle,
) -> c_int {
    if path.is_null() || out.is_null() {
        return ERR_OTHER;
    }
    let cstr = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return ERR_OTHER,
    };
    let base = PathBuf::from(cstr);
    let vec_dir = base.join("vec");
    let kv_dir = base.join("kv");
    let schema = VectorSchema::new(dim, MetricKind::L2);
    let vec_idx = match VectorIndex::open(&vec_dir, schema, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_create: vector index open failed");
            return err_code(&e);
        }
    };
    let kv = match KvStore::open(&kv_dir, 0, 0) {
        Ok(k) => k,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_create: kv store open failed");
            return err_code(&e);
        }
    };
    let boxed = Box::new(FsStoreHandle { vec: vec_idx, kv });
    *out = Box::into_raw(boxed);
    OK
}

/// 打开已存在 store：从持久化 schema 读回向量索引 + 重开 KV。
///
/// # Safety
/// path 须为合法 C 字符串；store 须经 fs_store_create 先建。out 接收句柄。
#[no_mangle]
pub unsafe extern "C" fn fs_store_open(path: *const c_char, out: *mut *mut FsStoreHandle) -> c_int {
    if path.is_null() || out.is_null() {
        return ERR_OTHER;
    }
    let cstr = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return ERR_OTHER,
    };
    let base = PathBuf::from(cstr);
    let vec_dir = base.join("vec");
    let kv_dir = base.join("kv");
    let vec_idx = match VectorIndex::reopen(&vec_dir, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_open: vector index reopen failed");
            return err_code(&e);
        }
    };
    let kv = match KvStore::open(&kv_dir, 0, 0) {
        Ok(k) => k,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_open: kv store open failed");
            return err_code(&e);
        }
    };
    let boxed = Box::new(FsStoreHandle { vec: vec_idx, kv });
    *out = Box::into_raw(boxed);
    OK
}

/// 关闭句柄，释放堆内存。
///
/// # Safety
/// h 须为 fs_store_create/open 返回的合法指针，或 NULL（NULL 时 no-op）。
#[no_mangle]
pub unsafe extern "C" fn fs_store_close(h: *mut FsStoreHandle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// 写 KV。
///
/// # Safety
/// key/val 须有效（klen/vlen 字节）；h 非空。
#[no_mangle]
pub unsafe extern "C" fn fs_store_put_kv(
    h: *mut FsStoreHandle,
    key: *const u8,
    klen: usize,
    val: *const u8,
    vlen: usize,
) -> c_int {
    if h.is_null() || key.is_null() || val.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    let k = std::slice::from_raw_parts(key, klen);
    let v = std::slice::from_raw_parts(val, vlen);
    match handle.kv.put_kv(k, v, None) {
        Ok(()) => OK,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_put_kv failed");
            err_code(&e)
        }
    }
}

/// 零拷贝读 KV → 强制拷贝到 caller buffer（E3）。
/// out_val = caller 预分配 buffer（容量 >= vlen_out 返回值）；
/// vlen_out = 实际 value 长度；buffer 不足时返回 ERR_OTHER，vlen_out 写所需长度。
/// key 不存在：返回 ERR_NOT_FOUND。
///
/// # Safety
/// out_val 非空（容量由 caller 保证）；vlen_out 非空。
#[no_mangle]
pub unsafe extern "C" fn fs_store_get_kv(
    h: *mut FsStoreHandle,
    key: *const u8,
    klen: usize,
    out_val: *mut u8,
    out_cap: usize,
    vlen_out: *mut usize,
) -> c_int {
    if h.is_null() || key.is_null() || out_val.is_null() || vlen_out.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    let k = std::slice::from_raw_parts(key, klen);
    match handle.kv.get_kv_zero_copy(k, None) {
        Ok(Some(buf)) => {
            let bytes = buf.as_bytes();
            *vlen_out = bytes.len();
            if bytes.len() > out_cap {
                return ERR_OTHER;
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_val, bytes.len());
            OK
        }
        Ok(None) => ERR_NOT_FOUND,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_get_kv failed");
            err_code(&e)
        }
    }
}

/// 插入向量（建库后灌数据用，非 PRD C-ABI 四函数之一，但端到端必需）。
///
/// # Safety
/// v 须有效 vlen 个 f32；h 非空。
#[no_mangle]
pub unsafe extern "C" fn fs_store_insert_vector(
    h: *mut FsStoreHandle,
    id: u64,
    v: *const f32,
    vlen: usize,
) -> c_int {
    if h.is_null() || v.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    let vec = std::slice::from_raw_parts(v, vlen);
    match handle.vec.insert(id, vec, None) {
        Ok(()) => OK,
        Err(e) => {
            tracing::error!(error = ?e, id, "fs_store_insert_vector failed");
            err_code(&e)
        }
    }
}

/// KNN 检索（读强制拷贝，E3）。
/// q = 查询向量（qlen 个 f32），top_k = 取近邻数。
/// out_ids/out_dists = caller 预分配 buffer（容量 >= top_k），out_n = 实际写入数。
///
/// # Safety
/// q 须有效 qlen 个 f32；out_ids/out_dists/out_n 非空且容量充足。
#[no_mangle]
pub unsafe extern "C" fn fs_store_search_knn(
    h: *mut FsStoreHandle,
    q: *const f32,
    qlen: usize,
    top_k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
    out_n: *mut usize,
) -> c_int {
    if h.is_null() || q.is_null() || out_ids.is_null() || out_dists.is_null() || out_n.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    let query = std::slice::from_raw_parts(q, qlen);
    match handle.vec.search_knn(query, top_k, None) {
        Ok(results) => {
            let n = results.len().min(top_k);
            for (i, (id, dist)) in results.iter().take(n).enumerate() {
                *out_ids.add(i) = *id;
                *out_dists.add(i) = *dist;
            }
            *out_n = n;
            OK
        }
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_search_knn failed");
            err_code(&e)
        }
    }
}

/// 返回句柄的向量维度（供 caller 校验 buffer 大小）。0=出错或空。
///
/// # Safety
/// h 须非空合法句柄。
#[no_mangle]
pub unsafe extern "C" fn fs_store_vector_dim(h: *const FsStoreHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    let handle = &*h;
    handle.vec.schema().dim
}

/// checkpoint：HNSW 图拓扑 snapshot 落盘 + 段 flush [v2 H1]。
/// close 前调此，重开方可从 snapshot 恢复图（否则图为空，需 M4 WAL 重放）。
///
/// # Safety
/// h 须非空合法句柄。
#[no_mangle]
pub unsafe extern "C" fn fs_store_checkpoint(h: *mut FsStoreHandle) -> c_int {
    if h.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    match handle.vec.checkpoint() {
        Ok(()) => OK,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_checkpoint failed");
            err_code(&e)
        }
    }
}

#[cfg(test)]
mod tests {
    //! fs-ffi-c C-ABI 端到端测试 [v2 §3.4/E3/E4]
    //!
    //! 验收（plan M3）：C 接口 fs_store_search_knn 端到端通；读路径强制拷贝（E3）；
    //! put_kv/get_kv 往返；create→close→open 重开持久化。真实 mmap 段，非 mock（E4）。

    use super::*;
    use std::ffi::CString;
    use std::ptr;
    use tempfile::tempdir;

    fn c_path(dir: &std::path::Path) -> CString {
        CString::new(dir.to_string_lossy().as_bytes()).unwrap()
    }

    unsafe fn create_store(path: &CString, dim: usize) -> *mut FsStoreHandle {
        let mut h: *mut FsStoreHandle = ptr::null_mut();
        let rc = fs_store_create(path.as_ptr(), dim, &mut h);
        assert_eq!(rc, 0, "fs_store_create failed: {}", rc);
        assert!(!h.is_null());
        h
    }

    unsafe fn open_store(path: &CString) -> *mut FsStoreHandle {
        let mut h: *mut FsStoreHandle = ptr::null_mut();
        let rc = fs_store_open(path.as_ptr(), &mut h);
        assert_eq!(rc, 0, "fs_store_open failed: {}", rc);
        assert!(!h.is_null());
        h
    }

    #[test]
    fn create_insert_knn_end_to_end() {
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        let dim = 4usize;
        unsafe {
            let h = create_store(&path, dim);
            for i in 0..10u64 {
                let mut v = vec![0.0f32; dim];
                v[0] = i as f32;
                let rc = fs_store_insert_vector(h, i, v.as_ptr(), dim);
                assert_eq!(rc, 0, "insert {} failed: {}", i, rc);
            }
            let q = [3.0f32, 0.0, 0.0, 0.0];
            let mut ids = vec![0u64; 3];
            let mut dists = vec![0.0f32; 3];
            let mut n = 0usize;
            let rc = fs_store_search_knn(
                h,
                q.as_ptr(),
                dim,
                3,
                ids.as_mut_ptr(),
                dists.as_mut_ptr(),
                &mut n,
            );
            assert_eq!(rc, 0, "search_knn failed: {}", rc);
            assert!(n >= 1, "knn returned empty");
            assert_eq!(ids[0], 3, "nearest should be id=3");
            assert_eq!(dists[0], 0.0, "exact match dist=0");
            fs_store_close(h);
        }
    }

    #[test]
    fn read_is_forced_copy_ids_dists_owned() {
        // E3：out_ids/out_dists 是 caller-owned buffer，close 后仍可读（非 mmap view）
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        let dim = 2usize;
        unsafe {
            let h = create_store(&path, dim);
            for i in 0..5u64 {
                let v = [i as f32, 0.0];
                fs_store_insert_vector(h, i, v.as_ptr(), dim);
            }
            let q = [1.0f32, 0.0];
            let mut ids = vec![0u64; 5];
            let mut dists = vec![0.0f32; 5];
            let mut n = 0usize;
            fs_store_search_knn(
                h,
                q.as_ptr(),
                dim,
                5,
                ids.as_mut_ptr(),
                dists.as_mut_ptr(),
                &mut n,
            );
            fs_store_close(h);
            assert!(n >= 1);
            assert_eq!(ids[0], 1, "owned buffer valid after close");
        }
    }

    #[test]
    fn put_kv_get_kv_forced_copy_roundtrip() {
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        unsafe {
            let h = create_store(&path, 4);
            let key = b"the-key";
            let val = b"fusion-store-payload-bytes";
            let rc = fs_store_put_kv(h, key.as_ptr(), key.len(), val.as_ptr(), val.len());
            assert_eq!(rc, 0, "put_kv failed: {}", rc);

            let mut out = vec![0u8; val.len() + 16];
            let mut vlen = 0usize;
            let rc = fs_store_get_kv(
                h,
                key.as_ptr(),
                key.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut vlen,
            );
            assert_eq!(rc, 0, "get_kv failed: {}", rc);
            assert_eq!(vlen, val.len());
            assert_eq!(&out[..vlen], val, "kv value forced-copy roundtrip");
            fs_store_close(h);
        }
    }

    #[test]
    fn get_kv_missing_returns_not_found() {
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        unsafe {
            let h = create_store(&path, 4);
            let key = b"nope";
            let mut out = vec![0u8; 64];
            let mut vlen = 0usize;
            let rc = fs_store_get_kv(
                h,
                key.as_ptr(),
                key.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut vlen,
            );
            assert_eq!(rc, -3, "missing key -> ERR_NOT_FOUND(-3), got {}", rc);
            fs_store_close(h);
        }
    }

    #[test]
    fn create_close_open_reopen_persisted() {
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        let dim = 8usize;
        unsafe {
            {
                let h = create_store(&path, dim);
                let v = vec![5.0f32; dim];
                fs_store_insert_vector(h, 42, v.as_ptr(), dim);
                // 图 checkpoint 落盘，重开方可从 snapshot 恢复（H1）
                let rc = fs_store_checkpoint(h);
                assert_eq!(rc, 0, "checkpoint failed: {}", rc);
                fs_store_close(h);
            }
            let h = open_store(&path);
            let got_dim = fs_store_vector_dim(h);
            assert_eq!(got_dim, dim, "dim restored from persisted schema");
            let q = vec![5.0f32; dim];
            let mut ids = vec![0u64; 1];
            let mut dists = vec![0.0f32; 1];
            let mut n = 0usize;
            fs_store_search_knn(
                h,
                q.as_ptr(),
                dim,
                1,
                ids.as_mut_ptr(),
                dists.as_mut_ptr(),
                &mut n,
            );
            assert_eq!(n, 1, "vector persisted across reopen");
            assert_eq!(ids[0], 42, "id=42 recovered");
            fs_store_close(h);
        }
    }
}
