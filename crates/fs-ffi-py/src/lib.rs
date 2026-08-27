//! fs-ffi-py —— fusion-store Python 绑定（PyO3）[v2 §3.4/E3]
//!
//! 薄封装 fs-core，不含业务逻辑。
//! 入参向量 numpy.ndarray f32 连续零拷贝传参（读 numpy buffer 指针，Rust 读期间 numpy 存活）。
//! 出参 ids/dists/get_kv 强制拷贝为 owned Python 对象（E3：不暴露 mmap 指针 view，
//! 跨 FFI 零拷贝生命周期未解，Python GC 释放包装致 Arc 归零后悬垂 → use-after-free）。
//!
//! API：
//!   store = fusion_store.Store.open(path, dim=None)   # dim=Some→create 锁 schema；None→reopen
//!   store.put_kv(key: bytes, value: bytes)
//!   store.get_kv(key: bytes) -> bytes | None          # 强制拷贝
//!   store.insert_vector(id: int, vec: np.ndarray)     # numpy 入参零拷贝
//!   store.search_knn(query: np.ndarray, top_k: int, timeout_ms: int|None) -> (ids, dists)
//!                                                      # 出参 numpy 强制拷贝（E3）
//!   store.checkpoint()
//!   store.vector_dim() -> int

use std::path::PathBuf;

use fs_core::store::KvStore;
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::store::VectorIndex;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList, PyTuple};

/// Python 句柄 —— 持向量索引 + KV store
#[pyclass(name = "Store")]
pub struct PyStore {
    vec: VectorIndex,
    kv: KvStore,
}

#[pymethods]
impl PyStore {
    /// Store.open(path, dim=None)
    /// dim=Some → create（锁 schema dim，建向量索引）；dim=None → reopen（从持久化 schema 恢复）
    #[staticmethod]
    #[pyo3(signature = (path, dim=None))]
    fn open(path: &str, dim: Option<usize>) -> PyResult<Self> {
        let base = PathBuf::from(expand_tilde(path));
        let vec_dir = base.join("vec");
        let kv_dir = base.join("kv");
        std::fs::create_dir_all(&vec_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        std::fs::create_dir_all(&kv_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let vec_idx = match dim {
            Some(d) => {
                tracing::info!(path = %vec_dir.display(), dim = d, "py open(create)");
                let schema = VectorSchema::new(d, MetricKind::L2);
                VectorIndex::open(&vec_dir, schema, 0)
            }
            None => {
                tracing::info!(path = %vec_dir.display(), "py open(reopen)");
                VectorIndex::reopen(&vec_dir, 0)
            }
        }
        .map_err(map_err_map)?;
        let kv = KvStore::open(&kv_dir, 0, 0).map_err(map_err_map)?;
        Ok(PyStore { vec: vec_idx, kv })
    }

    /// 写 KV。key/value 为 bytes。
    fn put_kv(&self, key: &[u8], value: &[u8]) -> PyResult<()> {
        self.kv.put_kv(key, value, None).map_err(map_err_map)?;
        Ok(())
    }

    /// 读 KV（强制拷贝，E3）→ 返回 owned bytes 或 None。
    /// 不暴露 mmap 指针 view：返回的 PyBytes 是拷贝，Python GC 释放无悬垂。
    fn get_kv<'py>(&self, py: Python<'py>, key: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        match self.kv.get_kv_zero_copy(key, None).map_err(map_err_map)? {
            Some(buf) => {
                let bytes = buf.as_bytes();
                // 强制拷贝到 PyBytes（E3）：buf.as_bytes() 落 mmap 区，拷出 owned
                let owned = bytes.to_vec();
                tracing::debug!(len = owned.len(), "py get_kv forced-copy");
                Ok(PyBytes::new(py, &owned).into_any())
            }
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// 插入向量（numpy 入参零拷贝读 buffer，E3 入参方向）。
    /// vec: numpy.ndarray float32 连续。Rust 经 PyBuffer 直接读 buffer 指针，无 Python 序列化。
    fn insert_vector(&self, py: Python<'_>, id: u64, vec: PyBuffer<f32>) -> PyResult<()> {
        let buf = vec.as_slice(py).ok_or_else(|| {
            PyRuntimeError::new_err("vec must be C-contiguous float32 numpy array")
        })?;
        let v: Vec<f32> = buf.iter().map(|c| c.get()).collect();
        self.vec.insert(id, &v, None).map_err(map_err_map)?;
        Ok(())
    }

    /// KNN 检索（出参强制拷贝，E3）。
    /// query: numpy.ndarray float32 连续；返回 (ids, dists) 为 owned Python tuple of list。
    /// ids/dists 是拷贝后的 owned 数据，非 mmap view —— 校验无 view 暴露。
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
            .vec
            .search_knn(&q, top_k, timeout)
            .map_err(map_err_map)?;
        // 出参强制拷贝（E3）：建 owned list，非 mmap view
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        let dists: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
        let ids_list = PyList::new(py, ids)?;
        let dists_list = PyList::new(py, dists)?;
        let tup = PyTuple::new(py, [ids_list.into_any(), dists_list.into_any()])?;
        Ok(tup.into_any())
    }

    /// checkpoint：HNSW 图 snapshot 落盘（H1）。close 前调，重开方可恢复图。
    fn checkpoint(&self) -> PyResult<()> {
        self.vec.checkpoint().map_err(map_err_map)?;
        Ok(())
    }

    /// 返回向量维度（供 caller 校验 buffer 大小）。
    fn vector_dim(&self) -> usize {
        self.vec.schema().dim
    }

    /// 返回当前向量数。
    fn vector_count(&self) -> usize {
        self.vec.len()
    }
}

fn map_err_map(e: fs_core::StoreError) -> PyErr {
    tracing::error!(error = ?e, "fs-ffi-py op failed");
    PyRuntimeError::new_err(format!("{e:?}"))
}

/// 展开 ~ 为 $HOME（Python 侧路径常含 ~）
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
    // tracing 初始化（幂等，env-filter）
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    m.add_class::<PyStore>()?;
    m.setattr("__doc__", "fusion-store Python 绑定（PyO3，读强制拷贝 E3）")?;
    Ok(())
}
