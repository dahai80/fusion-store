//! Prometheus 指标层 —— 独立 metrics，与 tracing 解耦 [v2 E5]
//!
//! PRD §2.8：op 计数器 + 延迟 histogram + 段使用率 gauge + 写队列深度 gauge。
//! 用 `prometheus` crate 原子计数器，高 QPS 不丢精度（非 tracing span 计数）。
//! 全局 Registry，`/metrics` 端点 gather 输出 Prometheus 文本格式。

use std::sync::OnceLock;

use prometheus::{
    register_histogram_vec_with_registry, register_int_counter_vec_with_registry,
    register_int_gauge_with_registry, Encoder, HistogramVec, IntCounterVec, IntGauge, Registry,
};

/// 默认 latency bucket（us 级，覆盖 put/get/knn/insert）
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

/// 全局 Registry + 指标句柄（OnceLock 单例，daemon 进程唯一）
struct Metrics {
    registry: Registry,
    ops_total: IntCounterVec,
    ops_latency_us: HistogramVec,
    seg_used_bytes: IntGauge,
    seg_capacity_bytes: IntGauge,
    backpressure_queue_depth: IntGauge,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

/// 初始化全局指标（daemon 启动调一次，重复调返回已建实例）
pub fn init() {
    METRICS.get_or_init(|| {
        let registry = Registry::new();
        let ops_total = register_int_counter_vec_with_registry!(
            "fusion_store_ops_total",
            "storage op count by op kind + result",
            &["op", "result"],
            registry
        )
        .expect("register ops_total");
        let ops_latency_us = register_histogram_vec_with_registry!(
            "fusion_store_ops_latency_us",
            "op latency in microseconds",
            &["op"],
            LATENCY_BUCKETS.to_vec(),
            registry
        )
        .expect("register ops_latency_us");
        let seg_used_bytes = register_int_gauge_with_registry!(
            "fusion_store_seg_used_bytes",
            "namespace used bytes (quota)",
            registry
        )
        .expect("register seg_used_bytes");
        let seg_capacity_bytes = register_int_gauge_with_registry!(
            "fusion_store_seg_capacity_bytes",
            "namespace quota limit bytes",
            registry
        )
        .expect("register seg_capacity_bytes");
        let backpressure_queue_depth = register_int_gauge_with_registry!(
            "fusion_store_backpressure_queue_depth",
            "write queue depth (backpressure gauge, A2)",
            registry
        )
        .expect("register backpressure_queue_depth");
        tracing::info!("prometheus metrics registry initialized");
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

/// 记一次 op：计数 + 延迟（caller 传 op 名 + 结果标签 + 耗时 us）
pub fn observe(op: &str, result: &str, elapsed_us: u64) {
    if let Some(m) = METRICS.get() {
        m.ops_total.with_label_values(&[op, result]).inc();
        m.ops_latency_us
            .with_label_values(&[op])
            .observe(elapsed_us as f64);
    }
}

/// 仅计数（无延时的快速路径，如 health 探测）
pub fn inc(op: &str, result: &str) {
    if let Some(m) = METRICS.get() {
        m.ops_total.with_label_values(&[op, result]).inc();
    }
}

/// 更新段使用率 gauge（KV used_bytes / quota_limit）
pub fn set_seg_util(used: u64, capacity: u64) {
    if let Some(m) = METRICS.get() {
        m.seg_used_bytes.set(used as i64);
        m.seg_capacity_bytes.set(capacity as i64);
    }
}

/// 更新背压队列深度 gauge
pub fn set_queue_depth(depth: usize) {
    if let Some(m) = METRICS.get() {
        m.backpressure_queue_depth.set(depth as i64);
    }
}

/// gather Prometheus 文本格式（/metrics 端点用）
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

// placeholder —— 实际端点逻辑见 main.rs / api.rs
