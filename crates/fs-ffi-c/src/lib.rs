//! fs-ffi-c —— fusion-store C-ABI 绑定 [v2 §3.4/E3]
//!
//! 薄封装：类型转换 + 错误码映射，不含业务逻辑。cbindgen 生成 fs_store.h。
//!
//! A7 诚实定位：C-ABI 读路径**强制拷贝**——out_ids/out_dists/out_val 是 caller-owned
//! buffer，Rust 拷贝填入，不暴露 mmap 指针 view。跨 C-ABI 零拷贝（返回 mmap 指针 +
//! 保活句柄透出）生命周期复杂度高，本阶段不做。故 fusion-store 的「零拷贝」**仅在
//! Rust 进程内有效**（mmap + Arc<MmapHandle> 保活）；C/Swift/Python 经此 C-ABI 读
//! 是普通拷贝存储引擎，unified memory 零拷贝优势在 FFI 消费侧不存在。对外定位按
//! 此修正，不宣称跨语言零拷贝。
//!
//! 错误码：0=Ok，负数=StoreError 变体（见 err_code）。
//!
//! 目录布局：单 path 下 vec/kv 各占独立子目录（各自 heed env，避免同路径 Env 冲突）。
//!   <path>/vec/{data,meta}   向量索引
//!   <path>/kv/{data,meta}    KV store

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine, StoreError};

/// C 句柄 —— 持 Engine（聚合 KV+向量+列式+WAL，F1/A2）
pub struct FsStoreHandle {
    engine: Engine,
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
const ERR_VALUE_TOO_LARGE: c_int = -14;
const ERR_LOCK_POISONED: c_int = -15;
const ERR_MAP_FULL: c_int = -16;
const ERR_INVALID_KEY: c_int = -17;
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
        StoreError::ValueTooLarge(_) => ERR_VALUE_TOO_LARGE,
        StoreError::LockPoisoned => ERR_LOCK_POISONED,
        StoreError::MapFull { .. } => ERR_MAP_FULL,
        StoreError::InvalidKey(_) => ERR_INVALID_KEY,
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
    let schema = VectorSchema::new(dim, MetricKind::L2);
    let engine = match Engine::open(&base, Some(schema), 0) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_create: engine open failed");
            return err_code(&e);
        }
    };
    let boxed = Box::new(FsStoreHandle { engine });
    *out = Box::into_raw(boxed);
    OK
}

/// 打开已存在 store：Engine::open(None) 重开 KV+向量（从持久化 schema 恢复）+ 自动 recover。
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
    let engine = match Engine::open(&base, None, 0) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_open: engine open failed");
            return err_code(&e);
        }
    };
    let boxed = Box::new(FsStoreHandle { engine });
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
        let boxed = Box::from_raw(h);
        // A6/E5：drop 前有序落盘（flush + heed sync + checkpoint），免正常退出丢数据。
        // C-ABI 签名为 void 无法回传错误码，故 engine close 的 Err 在此记 error 日志
        // （非吞没——Rust 内部消费方调 Engine::close 拿到 Result，C 侧仅日志可见）。
        if let Err(e) = boxed.engine.close() {
            tracing::error!(err = ?e, "fs_store_close: engine close FAILED — data may not be fully persisted");
        }
        drop(boxed);
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
    match handle.engine.put_kv(k, v, None) {
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
    match handle.engine.get_kv_zero_copy(k, None) {
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
    match handle.engine.insert_vector(id, vec, None) {
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
    match handle.engine.search_knn(query, top_k, None) {
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

/// 删除向量（软删：图节点标 tombstone，向量字节由 compact COW 回收）。
/// 返回 1=删了存活向量，0=id 不存在或已删（非错误）。
///
/// # Safety
/// h 非空合法句柄。
#[no_mangle]
pub unsafe extern "C" fn fs_store_delete_vector(h: *mut FsStoreHandle, id: u64) -> c_int {
    if h.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    match handle.engine.delete_vector(id, None) {
        Ok(deleted) => {
            if deleted {
                OK
            } else {
                ERR_NOT_FOUND
            }
        }
        Err(e) => {
            tracing::error!(error = ?e, id, "fs_store_delete_vector failed");
            err_code(&e)
        }
    }
}

/// 按 id 取单向量（强制拷贝到 caller buffer，E3）。id 不存在或已软删 → ERR_NOT_FOUND。
/// out_val = caller 预分配 buffer（容量 >= dim*sizeof(f32)），vlen_out = 实际写入字节。
/// buffer 不足时返回 ERR_OTHER，vlen_out 写所需字节。
///
/// # Safety
/// out_val/vlen_out 非空；out_val 容量由 caller 保证（可用 fs_store_vector_dim 取 dim）。
#[no_mangle]
pub unsafe extern "C" fn fs_store_get_vector(
    h: *mut FsStoreHandle,
    id: u64,
    out_val: *mut f32,
    out_cap: usize,
    vlen_out: *mut usize,
) -> c_int {
    if h.is_null() || out_val.is_null() || vlen_out.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    match handle.engine.get_vector(id, None) {
        Ok(Some(v)) => {
            let need_bytes = v.len();
            *vlen_out = need_bytes;
            if v.len() > out_cap {
                return ERR_OTHER;
            }
            std::ptr::copy_nonoverlapping(v.as_ptr(), out_val, v.len());
            OK
        }
        Ok(None) => ERR_NOT_FOUND,
        Err(e) => {
            tracing::error!(error = ?e, id, "fs_store_get_vector failed");
            err_code(&e)
        }
    }
}

/// 枚举存活（非软删）向量 id，拷贝到 caller 预分配 buffer（#3，E3 强制拷贝）。
/// out_ids = caller 预分配 buffer（容量 >= cap），out_n = 实际写入数。
/// 调用方需先查总数：可用一次 capacity=0 调用取 out_n=总需，再分配再调。
/// 容量不足时返回 ERR_OTHER，out_n 写所需长度。
///
/// # Safety
/// out_ids/out_n 非空；out_ids 容量由 caller 保证。
#[no_mangle]
pub unsafe extern "C" fn fs_store_list_vector_ids(
    h: *mut FsStoreHandle,
    out_ids: *mut u64,
    cap: usize,
    out_n: *mut usize,
) -> c_int {
    if h.is_null() || out_ids.is_null() || out_n.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    match handle.engine.list_vector_ids(None) {
        Ok(ids) => {
            *out_n = ids.len();
            if ids.len() > cap {
                return ERR_OTHER;
            }
            std::ptr::copy_nonoverlapping(ids.as_ptr(), out_ids, ids.len());
            OK
        }
        Err(e) => {
            tracing::error!(error = ?e, "fs_store_list_vector_ids failed");
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
    match handle.engine.vec_index() {
        Ok(g) => g.as_ref().unwrap().schema().dim,
        Err(_) => 0,
    }
}

/// checkpoint：图 snapshot + KV 段 flush + WAL marker 推进 applied_seq + 截断 WAL [v2 H1/L6]。
/// close 前调此，重开方可从 snapshot 恢复 + WAL 已截断不重放。
///
/// # Safety
/// h 须非空合法句柄。
#[no_mangle]
pub unsafe extern "C" fn fs_store_checkpoint(h: *mut FsStoreHandle) -> c_int {
    if h.is_null() {
        return ERR_OTHER;
    }
    let handle = &*h;
    match handle.engine.checkpoint() {
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

    #[test]
    fn delete_vector_then_get_and_list() {
        // #2/#3：delete_vector 软删 → get_vector 返回 NOT_FOUND，list_vector_ids 排除之
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        let dim = 4usize;
        unsafe {
            let h = create_store(&path, dim);
            for i in 0..5u64 {
                let mut v = vec![0.0f32; dim];
                v[0] = i as f32;
                fs_store_insert_vector(h, i, v.as_ptr(), dim);
            }
            // list 应有 5 个 id
            let mut n = 0usize;
            let mut ids = vec![0u64; 8];
            let rc = fs_store_list_vector_ids(h, ids.as_mut_ptr(), 8, &mut n);
            assert_eq!(rc, 0, "list_vector_ids failed: {}", rc);
            assert_eq!(n, 5, "5 live ids before delete");

            // get id=2 应成功
            let mut got = vec![0.0f32; dim];
            let mut vlen = 0usize;
            let rc = fs_store_get_vector(h, 2, got.as_mut_ptr(), dim, &mut vlen);
            assert_eq!(rc, 0, "get_vector(2) ok");
            assert_eq!(vlen, dim, "vlen == dim");

            // delete id=2
            let rc = fs_store_delete_vector(h, 2);
            assert_eq!(rc, 0, "delete_vector(2) ok");
            // 再删 id=2 → NOT_FOUND（已软删）
            let rc = fs_store_delete_vector(h, 2);
            assert_eq!(rc, -3, "delete again -> ERR_NOT_FOUND, got {}", rc);
            // 删不存在的 id=99 → NOT_FOUND
            let rc = fs_store_delete_vector(h, 99);
            assert_eq!(rc, -3, "delete missing -> ERR_NOT_FOUND");

            // get id=2 删后 → NOT_FOUND
            let mut got2 = vec![0.0f32; dim];
            let mut vlen2 = 0usize;
            let rc = fs_store_get_vector(h, 2, got2.as_mut_ptr(), dim, &mut vlen2);
            assert_eq!(rc, -3, "get deleted -> ERR_NOT_FOUND, got {}", rc);

            // list 应排除 id=2，剩 4 个
            let mut n2 = 0usize;
            let mut ids2 = vec![0u64; 8];
            let rc = fs_store_list_vector_ids(h, ids2.as_mut_ptr(), 8, &mut n2);
            assert_eq!(rc, 0);
            assert_eq!(n2, 4, "4 live ids after delete");
            assert!(!ids2[..n2].contains(&2), "deleted id excluded from list");

            fs_store_close(h);
        }
    }

    #[test]
    fn list_vector_ids_capacity_probe() {
        // 先 capacity=0 探总数，再分配再调（两段式调用模式）
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        let dim = 2usize;
        unsafe {
            let h = create_store(&path, dim);
            for i in 0..7u64 {
                let v = [i as f32, 0.0];
                fs_store_insert_vector(h, i, v.as_ptr(), dim);
            }
            // 第一段：cap=0 取所需（len=0 的 vec.as_mut_ptr() 是非空对齐悬垂指针，
            // 过 is_null 检查；cap=0 < ids.len() 触 ERR_OTHER 并写 out_n=所需总数）
            let mut need = 0usize;
            let mut zero: Vec<u64> = Vec::new();
            let rc = fs_store_list_vector_ids(h, zero.as_mut_ptr(), 0, &mut need);
            assert_eq!(rc, -99, "cap=0 -> ERR_OTHER, got {}", rc);
            assert_eq!(need, 7, "need == 7");
            // 第二段：足量分配
            let mut ids = vec![0u64; need];
            let mut n = 0usize;
            let rc = fs_store_list_vector_ids(h, ids.as_mut_ptr(), need, &mut n);
            assert_eq!(rc, 0);
            assert_eq!(n, 7);
            fs_store_close(h);
        }
    }
}
