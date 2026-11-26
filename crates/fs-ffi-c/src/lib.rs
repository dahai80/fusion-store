//! fs-ffi-c — fusion-store C-ABI bindings [v2 §3.4/E3]
//!
//! Thin wrapper: type conversion + error code mapping, no business logic.
//! cbindgen generates fs_store.h.
//!
//! A7 honest positioning: the C-ABI read path **forces a copy** — out_ids/out_dists/out_val
//! are caller-owned buffers filled in by Rust; no mmap pointer view is exposed. Zero-copy
//! across the C-ABI (returning an mmap pointer plus a keep-alive handle) has high lifetime
//! complexity and is out of scope for now. So fusion-store's "zero-copy" **only holds within
//! a Rust process** (mmap + Arc<MmapHandle> keep-alive); C/Swift/Python reads via this C-ABI
//! are plain copy-based engine operations — the unified-memory zero-copy advantage does not
//! exist on the FFI consumer side. External positioning follows this correction; no
//! cross-language zero-copy claims.
//!
//! Error codes: 0 = Ok, negative = StoreError variant (see err_code).
//!
//! Directory layout: under a single path, vec/kv each get their own subdirectory
//! (separate heed envs, avoiding same-path Env conflicts).
//!   <path>/vec/{data,meta}   vector index
//!   <path>/kv/{data,meta}    KV store

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine, StoreError};

/// C handle — owns an Engine (aggregated KV + vector + columnar + WAL, F1/A2)
pub struct FsStoreHandle {
    engine: Engine,
}

// ---- error code mapping ----
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

/// Create a new store: builds the vector index (locking the schema dim) + KV store.
/// dim = vector dimension. Creates `path` if it does not exist. Reopen later with fs_store_open.
///
/// # Safety
/// `path` must be a valid C string (NUL-terminated). `out` receives a heap-allocated
/// handle; the caller frees it with fs_store_close.
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

/// Open an existing store: Engine::open(None) reopens KV + vector (restored from the
/// persisted schema) + automatic recover.
///
/// # Safety
/// `path` must be a valid C string; the store must have been created with fs_store_create
/// first. `out` receives the handle.
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

/// Close the handle and free heap memory.
///
/// # Safety
/// `h` must be a valid pointer returned by fs_store_create/open, or NULL (NULL is a no-op).
#[no_mangle]
pub unsafe extern "C" fn fs_store_close(h: *mut FsStoreHandle) {
    if !h.is_null() {
        let boxed = Box::from_raw(h);
        // A6/E5: orderly persistence before drop (flush + heed sync + checkpoint) so a normal
        // exit never loses data. The C-ABI signature is void and cannot return an error code,
        // so engine close Err is logged here as an error (not swallowed — internal Rust
        // consumers call Engine::close and get a Result; on the C side only the log shows it).
        if let Err(e) = boxed.engine.close() {
            tracing::error!(err = ?e, "fs_store_close: engine close FAILED — data may not be fully persisted");
        }
        drop(boxed);
    }
}

/// Write KV.
///
/// # Safety
/// `key`/`val` must be valid for klen/vlen bytes; `h` must not be null.
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

/// Zero-copy KV read -> forced copy into a caller buffer (E3).
/// out_val = caller-allocated buffer (capacity >= the vlen_out result);
/// vlen_out = actual value length; if the buffer is too small, returns ERR_OTHER and
/// writes the required length into vlen_out.
/// Missing key: returns ERR_NOT_FOUND.
///
/// # Safety
/// `out_val` must not be null (capacity guaranteed by the caller); `vlen_out` must not be null.
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

/// Insert a vector (used for bulk loading after store creation; not one of the PRD's four
/// C-ABI functions, but required end-to-end).
///
/// # Safety
/// `v` must be valid for vlen f32 elements; `h` must not be null.
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

/// KNN search (forced-copy read, E3).
/// q = query vector (qlen f32 elements), top_k = number of neighbors to fetch.
/// out_ids/out_dists = caller-allocated buffers (capacity >= top_k); out_n = number written.
///
/// # Safety
/// `q` must be valid for qlen f32 elements; out_ids/out_dists/out_n must not be null and
/// must have sufficient capacity.
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

/// Delete a vector (soft delete: graph node gets a tombstone; vector bytes are reclaimed
/// by compact COW).
/// Returns 1 if a live vector was deleted, 0 if the id is missing or already deleted (not an error).
///
/// # Safety
/// `h` must be a valid non-null handle.
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

/// Fetch a single vector by id (forced copy into a caller buffer, E3).
/// Missing or soft-deleted id -> ERR_NOT_FOUND.
/// out_val = caller-allocated buffer (capacity >= dim * sizeof(f32)); vlen_out = bytes written.
/// If the buffer is too small, returns ERR_OTHER and writes the required bytes into vlen_out.
///
/// # Safety
/// out_val/vlen_out must not be null; out_val capacity is guaranteed by the caller
/// (use fs_store_vector_dim to obtain the dim).
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

/// Enumerate live (non-soft-deleted) vector ids, copied into a caller-allocated buffer
/// (#3, E3 forced copy).
/// out_ids = caller-allocated buffer (capacity >= cap); out_n = number written.
/// To learn the total count first: call once with capacity=0 to get out_n = required total,
/// then allocate and call again.
/// If capacity is insufficient, returns ERR_OTHER and writes the required length into out_n.
///
/// # Safety
/// out_ids/out_n must not be null; out_ids capacity is guaranteed by the caller.
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

/// Return the handle's vector dimension (so callers can size buffers). 0 = error or empty.
///
/// # Safety
/// `h` must be a valid non-null handle.
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

/// Checkpoint: graph snapshot + KV segment flush + WAL marker advancing applied_seq + WAL
/// truncation [v2 H1/L6].
/// Call this before close so a reopen restores from the snapshot and the truncated WAL is not replayed.
///
/// # Safety
/// `h` must be a valid non-null handle.
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
    //! fs-ffi-c C-ABI end-to-end tests [v2 §3.4/E3/E4]
    //!
    //! Acceptance (plan M3): C interface fs_store_search_knn works end-to-end; read path is
    //! forced-copy (E3); put_kv/get_kv roundtrip; create->close->open reopen persistence.
    //! Real mmap segments, no mocks (E4).

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
        // E3: out_ids/out_dists are caller-owned buffers, still readable after close (not mmap views)
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
                // Persist the graph checkpoint so a reopen restores from the snapshot (H1)
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
        // #2/#3: delete_vector soft delete -> get_vector returns NOT_FOUND, list_vector_ids excludes it
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
            // list should return 5 ids
            let mut n = 0usize;
            let mut ids = vec![0u64; 8];
            let rc = fs_store_list_vector_ids(h, ids.as_mut_ptr(), 8, &mut n);
            assert_eq!(rc, 0, "list_vector_ids failed: {}", rc);
            assert_eq!(n, 5, "5 live ids before delete");

            // get id=2 should succeed
            let mut got = vec![0.0f32; dim];
            let mut vlen = 0usize;
            let rc = fs_store_get_vector(h, 2, got.as_mut_ptr(), dim, &mut vlen);
            assert_eq!(rc, 0, "get_vector(2) ok");
            assert_eq!(vlen, dim, "vlen == dim");

            // delete id=2
            let rc = fs_store_delete_vector(h, 2);
            assert_eq!(rc, 0, "delete_vector(2) ok");
            // delete id=2 again -> NOT_FOUND (already soft-deleted)
            let rc = fs_store_delete_vector(h, 2);
            assert_eq!(rc, -3, "delete again -> ERR_NOT_FOUND, got {}", rc);
            // delete nonexistent id=99 -> NOT_FOUND
            let rc = fs_store_delete_vector(h, 99);
            assert_eq!(rc, -3, "delete missing -> ERR_NOT_FOUND");

            // get id=2 after delete -> NOT_FOUND
            let mut got2 = vec![0.0f32; dim];
            let mut vlen2 = 0usize;
            let rc = fs_store_get_vector(h, 2, got2.as_mut_ptr(), dim, &mut vlen2);
            assert_eq!(rc, -3, "get deleted -> ERR_NOT_FOUND, got {}", rc);

            // list should exclude id=2, leaving 4
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
        // Probe the total with capacity=0 first, then allocate and call again (two-phase pattern)
        let dir = tempdir().unwrap();
        let path = c_path(dir.path());
        let dim = 2usize;
        unsafe {
            let h = create_store(&path, dim);
            for i in 0..7u64 {
                let v = [i as f32, 0.0];
                fs_store_insert_vector(h, i, v.as_ptr(), dim);
            }
            // Phase 1: cap=0 to learn the requirement (a len=0 vec.as_mut_ptr() is a non-null,
            // aligned dangling pointer that passes the is_null check; cap=0 < ids.len() triggers
            // ERR_OTHER and writes out_n = required total)
            let mut need = 0usize;
            let mut zero: Vec<u64> = Vec::new();
            let rc = fs_store_list_vector_ids(h, zero.as_mut_ptr(), 0, &mut need);
            assert_eq!(rc, -99, "cap=0 -> ERR_OTHER, got {}", rc);
            assert_eq!(need, 7, "need == 7");
            // Phase 2: allocate enough and call again
            let mut ids = vec![0u64; need];
            let mut n = 0usize;
            let rc = fs_store_list_vector_ids(h, ids.as_mut_ptr(), need, &mut n);
            assert_eq!(rc, 0);
            assert_eq!(n, 7);
            fs_store_close(h);
        }
    }
}
