//! Prometheus metrics layer — standalone metrics, decoupled from tracing [v2 E5]
//!
//! PRD §2.8: op counters + latency histogram + segment usage gauge + write queue depth gauge.
//! Uses `prometheus` crate atomic counters so high QPS does not lose precision (not tracing span counting).
//! Global Registry; the `/metrics` endpoint gathers and outputs the Prometheus text format.

use std::sync::OnceLock;

use prometheus::{
    register_histogram_vec_with_registry, register_int_counter_vec_with_registry,
    register_int_gauge_with_registry, Encoder, HistogramVec, IntCounterVec, IntGauge, Registry,
};

/// Default latency buckets (microsecond scale, covers put/get/knn/insert)
const LATENCY_BUCKETS: &[f64] = &[
    10.0,
    50.0,
    100.0,
    500.0,
    1_000.0,
    5_000.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
];

/// Global Registry + metric handles (OnceLock singleton, unique per daemon process).
/// F-ERR-4: all handles are Option — registration failure degrades to None; observe/inc/set_* record only when Some,
/// and /metrics returns empty. Never panic and take down the daemon on metric registration failure (metrics are non-core; degrade observably).
struct Metrics {
    registry: Registry,
    ops_total: Option<IntCounterVec>,
    ops_latency_us: Option<HistogramVec>,
    seg_used_bytes: Option<IntGauge>,
    seg_capacity_bytes: Option<IntGauge>,
    backpressure_queue_depth: Option<IntGauge>,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

/// F-ERR-4: single metric registration failure → log error + return None (degrade), no panic.
/// Guard: prometheus registration only fails on duplicate name/type conflict (first registration in a process never conflicts), so failure is extremely rare;
/// once failed the metric is permanently missing (degraded); observe paths silently skip, storage correctness is unaffected.
fn register_or_degrade<T, F>(name: &str, reg: F) -> Option<T>
where
    F: FnOnce() -> prometheus::Result<T>,
{
    match reg() {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::error!(
                metric = name,
                error = %e,
                "prometheus metric register failed — degraded (metric disabled, storage unaffected)"
            );
            None
        }
    }
}

/// Initialize global metrics (called once at daemon startup; repeat calls return the existing instance)
pub fn init() {
    METRICS.get_or_init(|| {
        let registry = Registry::new();
        let ops_total = register_or_degrade("fusion_store_ops_total", || {
            register_int_counter_vec_with_registry!(
                "fusion_store_ops_total",
                "storage op count by op kind + result",
                &["op", "result"],
                registry
            )
        });
        let ops_latency_us = register_or_degrade("fusion_store_ops_latency_us", || {
            register_histogram_vec_with_registry!(
                "fusion_store_ops_latency_us",
                "op latency in microseconds",
                &["op"],
                LATENCY_BUCKETS.to_vec(),
                registry
            )
        });
        let seg_used_bytes = register_or_degrade("fusion_store_seg_used_bytes", || {
            register_int_gauge_with_registry!(
                "fusion_store_seg_used_bytes",
                "namespace used bytes (quota)",
                registry
            )
        });
        let seg_capacity_bytes = register_or_degrade("fusion_store_seg_capacity_bytes", || {
            register_int_gauge_with_registry!(
                "fusion_store_seg_capacity_bytes",
                "namespace quota limit bytes",
                registry
            )
        });
        let backpressure_queue_depth = register_or_degrade(
            "fusion_store_backpressure_queue_depth",
            || {
                register_int_gauge_with_registry!(
                    "fusion_store_backpressure_queue_depth",
                    "write queue depth (backpressure gauge, A2)",
                    registry
                )
            },
        );
        let degraded = ops_total.is_none()
            || ops_latency_us.is_none()
            || seg_used_bytes.is_none()
            || seg_capacity_bytes.is_none()
            || backpressure_queue_depth.is_none();
        if degraded {
            tracing::warn!(
                "prometheus metrics partially degraded — some metrics disabled, /metrics may be incomplete"
            );
        } else {
            tracing::info!("prometheus metrics registry initialized");
        }
        Metrics {
            registry,
            ops_total,
            ops_latency_us,
            seg_used_bytes,
            seg_capacity_bytes,
            backpressure_queue_depth,
        }
    });
}

/// Record one op: count + latency (caller passes op name + result label + elapsed us)
pub fn observe(op: &str, result: &str, elapsed_us: u64) {
    if let Some(m) = METRICS.get() {
        if let Some(ops_total) = &m.ops_total {
            ops_total.with_label_values(&[op, result]).inc();
        }
        if let Some(ops_latency_us) = &m.ops_latency_us {
            ops_latency_us
                .with_label_values(&[op])
                .observe(elapsed_us as f64);
        }
    }
}

/// Count only (fast path without latency, e.g. health probes)
pub fn inc(op: &str, result: &str) {
    if let Some(m) = METRICS.get() {
        if let Some(ops_total) = &m.ops_total {
            ops_total.with_label_values(&[op, result]).inc();
        }
    }
}

/// Update segment usage gauge (KV used_bytes / quota_limit)
pub fn set_seg_util(used: u64, capacity: u64) {
    if let Some(m) = METRICS.get() {
        if let Some(seg_used_bytes) = &m.seg_used_bytes {
            seg_used_bytes.set(used as i64);
        }
        if let Some(seg_capacity_bytes) = &m.seg_capacity_bytes {
            seg_capacity_bytes.set(capacity as i64);
        }
    }
}

/// Update the backpressure queue depth gauge
pub fn set_queue_depth(depth: usize) {
    if let Some(m) = METRICS.get() {
        if let Some(backpressure_queue_depth) = &m.backpressure_queue_depth {
            backpressure_queue_depth.set(depth as i64);
        }
    }
}

/// Gather the Prometheus text format (used by the /metrics endpoint)
pub fn gather() -> Vec<u8> {
    match METRICS.get() {
        Some(m) => {
            let encoder = prometheus::TextEncoder::new();
            let mut buf = Vec::new();
            encoder
                .encode(&m.registry.gather(), &mut buf)
                .unwrap_or_else(|e| tracing::warn!(error = %e, "metrics encode failed"));
            buf
        }
        None => Vec::new(),
    }
}

// placeholder — actual endpoint logic lives in main.rs / api.rs
