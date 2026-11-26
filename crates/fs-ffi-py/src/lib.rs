//! fs-ffi-py — fusion-store Python bindings (PyO3) [v2 §3.4/E3]
//!
//! Thin wrapper over fs-core, no business logic.
//! Input vectors: numpy.ndarray f32 contiguous, passed zero-copy (reads the numpy buffer
//! pointer; numpy must stay alive while Rust reads it).
//! Outputs ids/dists/get_kv: forced copy into owned Python objects (E3: no mmap pointer
//! view exposed; cross-FFI zero-copy lifetimes are unsolved — Python GC freeing the wrapper
//! would drop the Arc to zero and dangle -> use-after-free).
//!
//! API:
//!   store = fusion_store.Store.open(path, dim=None)   # dim=Some -> create (locks schema); None -> reopen
//!   store.put_kv(key: bytes, value: bytes)
//!   store.get_kv(key: bytes) -> bytes | None          # forced copy
//!   store.insert_vector(id: int, vec: np.ndarray)     # numpy input zero-copy
//!   store.search_knn(query: np.ndarray, top_k: int, timeout_ms: int|None) -> (ids, dists)
//!                                                      # outputs numpy forced copy (E3)
//!   store.delete_vector(id: int) -> bool              # soft delete; True = live vector deleted (#2)
//!   store.get_vector(id: int) -> list[float] | None   # forced copy; missing/soft-deleted -> None (#3)
//!   store.list_vector_ids() -> list[int]              # live (non-soft-deleted) ids (#3)
//!   store.checkpoint()
//!   store.vector_dim() -> int

use std::path::PathBuf;

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::{Engine, FusionStoreEngine};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList, PyTuple};

/// Python handle — owns an Engine (aggregated KV + vector + columnar + WAL, F1/A2)
#[pyclass(name = "Store")]
pub struct PyStore {
    engine: Engine,
}

#[pymethods]
impl PyStore {
    /// Store.open(path, dim=None)
    /// dim=Some -> create (locks schema dim, builds the vector index); dim=None -> reopen (restored from the persisted schema)
    /// Engine::open runs automatic recover (A1) + WAL is wired into the write path (F2)
    #[staticmethod]
    #[pyo3(signature = (path, dim=None))]
    fn open(path: &str, dim: Option<usize>) -> PyResult<Self> {
        let base = PathBuf::from(expand_tilde(path));
        let schema = dim.map(|d| VectorSchema::new(d, MetricKind::L2));
        if let Some(d) = dim {
            tracing::info!(path = %base.display(), d, "py open(create)");
        } else {
            tracing::info!(path = %base.display(), "py open(reopen)");
        }
        let engine = Engine::open(&base, schema, 0).map_err(map_err_map)?;
        Ok(PyStore { engine })
    }

    /// Write KV. Engine -> WAL (fsync) -> mmap segment (F2).
    fn put_kv(&self, key: &[u8], value: &[u8]) -> PyResult<()> {
        self.engine.put_kv(key, value, None).map_err(map_err_map)?;
        Ok(())
    }

    /// Read KV (forced copy, E3) -> returns owned bytes or None.
    /// No mmap pointer view exposed: the returned PyBytes is a copy, so Python GC freeing
    /// it never dangles.
    fn get_kv<'py>(&self, py: Python<'py>, key: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        match self
            .engine
            .get_kv_zero_copy(key, None)
            .map_err(map_err_map)?
        {
            Some(buf) => {
                let bytes = buf.as_bytes();
                // Forced copy into PyBytes (E3): buf.as_bytes() lives in the mmap region, copy out owned bytes
                let owned = bytes.to_vec();
                tracing::debug!(len = owned.len(), "py get_kv forced-copy");
                Ok(PyBytes::new(py, &owned).into_any())
            }
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// Insert a vector (numpy input read zero-copy from the buffer, E3 input direction).
    /// Engine -> WAL (payload carries the full vector) -> segment write + edge linking (F2).
    fn insert_vector(&self, py: Python<'_>, id: u64, vec: PyBuffer<f32>) -> PyResult<()> {
        let buf = vec.as_slice(py).ok_or_else(|| {
            PyRuntimeError::new_err("vec must be C-contiguous float32 numpy array")
        })?;
        let v: Vec<f32> = buf.iter().map(|c| c.get()).collect();
        self.engine
            .insert_vector(id, &v, None)
            .map_err(map_err_map)?;
        Ok(())
    }

    /// KNN search (forced-copy outputs, E3).
    /// Engine -> zero-copy vector index read. Returns (ids, dists) as an owned Python tuple of lists.
    #[pyo3(signature = (query, top_k, timeout_ms=None))]
    fn search_knn<'py>(
        &self,
        py: Python<'py>,
        query: PyBuffer<f32>,
        top_k: usize,
        timeout_ms: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let qbuf = query.as_slice(py).ok_or_else(|| {
            PyRuntimeError::new_err("query must be C-contiguous float32 numpy array")
        })?;
        let q: Vec<f32> = qbuf.iter().map(|c| c.get()).collect();
        let timeout = timeout_ms.map(std::time::Duration::from_millis);
        let results = self
            .engine
            .search_knn(&q, top_k, timeout)
            .map_err(map_err_map)?;
        // Forced-copy outputs (E3): build owned lists, not mmap views
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        let dists: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
        let ids_list = PyList::new(py, ids)?;
        let dists_list = PyList::new(py, dists)?;
        let tup = PyTuple::new(py, [ids_list.into_any(), dists_list.into_any()])?;
        Ok(tup.into_any())
    }

    /// Checkpoint: graph snapshot + KV flush + WAL marker advancing applied_seq + WAL truncation (L6).
    fn checkpoint(&self) -> PyResult<()> {
        self.engine.checkpoint().map_err(map_err_map)?;
        Ok(())
    }

    /// Return the vector dimension (so callers can size buffers).
    fn vector_dim(&self) -> usize {
        match self.engine.vec_index() {
            Ok(g) => g.as_ref().unwrap().schema().dim,
            Err(_) => 0,
        }
    }

    /// Return the current vector count.
    fn vector_count(&self) -> usize {
        match self.engine.vec_index() {
            Ok(g) => g.as_ref().unwrap().len(),
            Err(_) => 0,
        }
    }

    /// Delete a vector (soft-delete tombstone; vector bytes are reclaimed by compact COW).
    /// Returns True = live vector deleted, False = id missing or already deleted (#2).
    fn delete_vector(&self, id: u64) -> PyResult<bool> {
        let deleted = self.engine.delete_vector(id, None).map_err(map_err_map)?;
        tracing::info!(id, deleted, "py delete_vector");
        Ok(deleted)
    }

    /// Fetch a single vector by id (forced copy, E3) -> returns list[float] or None (#3).
    /// Missing or soft-deleted id -> None. Owned list, not an mmap view.
    fn get_vector<'py>(&self, py: Python<'py>, id: u64) -> PyResult<Bound<'py, PyAny>> {
        match self.engine.get_vector(id, None).map_err(map_err_map)? {
            Some(v) => {
                // Forced copy into an owned Python list (E3): not an mmap view
                tracing::debug!(id, len = v.len(), "py get_vector forced-copy");
                let list = PyList::new(py, v)?;
                Ok(list.into_any())
            }
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// Enumerate live (non-soft-deleted) vector ids -> owned list[int] (#3, E3 forced copy).
    fn list_vector_ids(&self) -> PyResult<Vec<u64>> {
        let ids = self.engine.list_vector_ids(None).map_err(map_err_map)?;
        tracing::debug!(count = ids.len(), "py list_vector_ids");
        Ok(ids)
    }
}

fn map_err_map(e: fs_core::StoreError) -> PyErr {
    tracing::error!(error = ?e, "fs-ffi-py op failed");
    PyRuntimeError::new_err(format!("{e:?}"))
}

/// Expand ~ to $HOME (Python-side paths often contain ~)
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix('~') {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(rest.trim_start_matches('/'))
                .to_string_lossy()
                .into_owned();
        }
    }
    path.to_string()
}

#[pymodule]
fn fusion_store(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // tracing init (idempotent, env-filter)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    m.add_class::<PyStore>()?;
    m.setattr(
        "__doc__",
        "fusion-store Python bindings (PyO3, forced-copy reads E3)",
    )?;
    Ok(())
}
