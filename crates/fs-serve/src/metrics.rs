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

/// 全局 Registry + 指标句柄（OnceLock 单例，daemon 进程唯一）。
/// F-ERR-4：所有句柄 Option 化 —— 注册失败降级为 None，observe/inc/set_* 检查 Some 后记，
/// /metrics 返回空。绝不因指标注册失败 panic 拖垮 daemon（指标非核心功能，降级可观测）。
struct Metrics {
    registry: Registry,
    ops_total: Option<IntCounterVec>,
    ops_latency_us: Option<HistogramVec>,
    seg_used_bytes: Option<IntGauge>,
    seg_capacity_bytes: Option<IntGauge>,
    backpressure_queue_depth: Option<IntGauge>,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

/// F-ERR-4：单指标注册失败 → 记 error + 返回 None（降级），不 panic。
/// 守卫：prometheus 注册仅在重名/类型冲突时失败（单进程首次注册不冲突），失败极罕见；
/// 一旦失败该指标永久缺失（降级），observe 路径静默跳过，不影响存储正确性。
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

/// 初始化全局指标（daemon 启动调一次，重复调返回已建实例）
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

/// 记一次 op：计数 + 延迟（caller 传 op 名 + 结果标签 + 耗时 us）
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

/// 仅计数（无延时的快速路径，如 health 探测）
pub fn inc(op: &str, result: &str) {
    if let Some(m) = METRICS.get() {
        if let Some(ops_total) = &m.ops_total {
            ops_total.with_label_values(&[op, result]).inc();
        }
    }
}

/// 更新段使用率 gauge（KV used_bytes / quota_limit）
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

/// 更新背压队列深度 gauge
pub fn set_queue_depth(depth: usize) {
    if let Some(m) = METRICS.get() {
        if let Some(backpressure_queue_depth) = &m.backpressure_queue_depth {
            backpressure_queue_depth.set(depth as i64);
        }
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
