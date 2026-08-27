//! SIMD 距离 —— NEON 与标量位等 [v2 R4]
//!
//! 位等约束：同输入同输出到 bit 级，召回率测试对暴力 KNN 可复现。
//! 实现约束：NEON 用与标量相同求和顺序（成对累加不重排），
//! build profile 强制 opt-level=1 禁激进重排。支持 cosine / L2 / dot。
//! 按 VectorSchema.metric 固定一种（§1.4）。

use crate::vector::schema::MetricKind;

/// 计算两向量距离（按 metric 选）。SIMD 与标量位等。
pub fn distance(metric: MetricKind, a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "distance: dim mismatch");
    match metric {
        MetricKind::Dot => dot(a, b),
        MetricKind::Cosine => cosine(a, b),
        MetricKind::L2 => l2(a, b),
    }
}

/// 点积 —— 4-lane 累加，NEON 与标量同序同结构 [v2 R4 位等]
///
/// NEON 用 vmulq + vaddq（非 fused，每步独立舍入），与标量 4-accumulator
/// (a[i]*b[i] 后 +=) 运算序列逐位等价 → 同输入同输出到 bit 级。
/// NEON 4-wide 处理，标量 1-wide 处理（同 4 累加器结构）→ NEON ≥3x 快。
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // aarch64 全系含 NEON（Apple Silicon 默认），直接走 NEON；他架构标量
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_scalar(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vgetq_lane_f32, vld1q_f32, vmulq_f32};
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a[i..].as_ptr());
        let vb = vld1q_f32(b[i..].as_ptr());
        // 非 fused：先乘（舍入）后加（舍入），与标量 a[i]*b[i] 后 += 等价
        let prod = vmulq_f32(va, vb);
        acc = vaddq_f32(acc, prod);
        i += 4;
    }
    // 横向归约 acc[0..4] —— 与标量 (sum0+sum1)+(sum2+sum3) 同序
    let s0 = vgetq_lane_f32(acc, 0);
    let s1 = vgetq_lane_f32(acc, 1);
    let s2 = vgetq_lane_f32(acc, 2);
    let s3 = vgetq_lane_f32(acc, 3);
    let mut sum = (s0 + s1) + (s2 + s3);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// 标量点积 —— NEON 位等对照基准（bit-equiv test 直接调用）
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = 0;
    // 4 路展开，结构同 NEON：逐乘后累加，顺序等价
    while i + 4 <= n {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = (sum0 + sum1) + (sum2 + sum3);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// cosine 距离 = 1 - dot(a,b)/(|a|*|b|)
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = dot(a, a).sqrt();
    let nb = dot(b, b).sqrt();
    let denom = na * nb;
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - d / denom
}

/// L2 平方距离 = sum((a-b)^2)
///
/// NEON 与标量位等 [v2 R4]：sub→mul→add 非融合（每步独立舍入），
/// 4-lane 累加顺序同标量 4-accumulator（(sum0+sum1)+(sum2+sum3)）。
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    // aarch64 NEON；他架构标量
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { l2_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        l2_scalar(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32, vld1q_f32, vmlaq_f32, vsubq_f32};
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a[i..].as_ptr());
        let vb = vld1q_f32(b[i..].as_ptr());
        // d = a - b（独立舍入）；acc += d*d（非 fused：先乘舍入后加舍入）
        let d = vsubq_f32(va, vb);
        acc = vmlaq_f32(acc, d, d);
        i += 4;
    }
    // 横向归约同标量序 (s0+s1)+(s2+s3)
    let s0 = vgetq_lane_f32(acc, 0);
    let s1 = vgetq_lane_f32(acc, 1);
    let s2 = vgetq_lane_f32(acc, 2);
    let s3 = vgetq_lane_f32(acc, 3);
    let mut sum = (s0 + s1) + (s2 + s3);
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// 标量 L2 —— NEON 位等对照基准（bit-equiv test 直接调用）
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = 0;
    // 4 路展开，结构同 NEON：sub→mul→add 逐舍入，顺序等价
    while i + 4 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        i += 4;
    }
    let mut s = (sum0 + sum1) + (sum2 + sum3);
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_correct_small() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        // 5+12+21+32 = 70
        assert_eq!(dot(&a, &b), 70.0);
    }

    #[test]
    fn dot_handles_non_multiple_of_4() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(dot(&a, &b), 15.0);
    }

    #[test]
    fn cosine_orthogonal_is_max() {
        let a = [1.0f32, 0.0];
        let b = [0.0, 1.0];
        let d = cosine(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "orthogonal cosine dist = 1, got {d}"
        );
    }

    #[test]
    fn cosine_identical_is_zero() {
        let a = [3.0f32, 4.0];
        let d = cosine(&a, &a);
        assert!(d.abs() < 1e-6, "identical cosine dist = 0, got {d}");
    }

    #[test]
    fn l2_correct() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        // (9 + 16 + 0) = 25
        assert_eq!(l2(&a, &b), 25.0);
    }

    #[test]
    fn l2_handles_non_multiple_of_4() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0];
        let b = [0.0; 7];
        // 1+4+9+16+25+25+25 = 105
        assert_eq!(l2(&a, &b), 105.0);
    }

    #[test]
    fn distance_dispatch_by_metric() {
        let a = [1.0f32, 0.0];
        let b = [0.0, 1.0];
        let cos_d = distance(MetricKind::Cosine, &a, &b);
        let l2_d = distance(MetricKind::L2, &a, &b);
        let dot_d = distance(MetricKind::Dot, &a, &b);
        assert!((cos_d - 1.0).abs() < 1e-6);
        assert!((l2_d - 2.0).abs() < 1e-6);
        assert!((dot_d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn neon_and_scalar_bit_equivalent() {
        // R4 位等：NEON 与标量同输入 → 同输出到 bit 级
        // 用非整数值触发舍入差异，验证逐位一致
        let a = [
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 2.2, 3.3, 4.4,
        ];
        let b = [1.5, 2.7, 3.9, 4.1, 5.2, 6.3, 7.4, 8.5, 0.9, 1.8, 2.7, 3.6];
        let neon_d = dot(&a, &b);
        let scalar_d = dot_scalar(&a, &b);
        // bit 级断言：to_bits 完全相等
        assert_eq!(
            neon_d.to_bits(),
            scalar_d.to_bits(),
            "NEON {neon_d} != scalar {scalar_d} at bit level"
        );
    }

    #[test]
    fn l2_neon_and_scalar_bit_equivalent() {
        // R4 位等：L2 NEON 与标量同输入 → 同输出到 bit 级
        let a = [
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9,
        ];
        let b = [
            1.5, 2.7, 3.9, 4.1, 5.2, 6.3, 7.4, 8.5, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1,
        ];
        let neon_d = l2(&a, &b);
        let scalar_d = l2_scalar(&a, &b);
        assert_eq!(
            neon_d.to_bits(),
            scalar_d.to_bits(),
            "L2 NEON {neon_d} != scalar {scalar_d} at bit level"
        );
    }
}
