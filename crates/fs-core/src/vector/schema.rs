//! Vector index schema — locks dim/metric/ef_search as strong invariants at creation [v2 E2]

use serde::{Deserialize, Serialize};

use crate::vector::nsw::DEFAULT_EF_SEARCH;

/// Distance metric kind (fixed to one at index creation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricKind {
    Cosine,
    L2,
    Dot,
}

/// Vector index schema — insert/search validate dimension consistency and
/// report DimensionMismatch on mismatch [v2 E2]
///
/// F-ARCH-2: `ef_search` is locked at index creation; search uses this value
/// (or top_k, whichever is larger) as the candidate-set width.
/// Default is 200 (meets the PRD §2.5 ≥0.95 recall SLA); tunable at creation
/// (raise for high-recall workloads, lower for low-latency ones).
/// serde default: falls back to DEFAULT_EF_SEARCH when older snapshots/schemas
/// did not persist this field (backward compatible).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorSchema {
    pub dim: usize,
    pub metric: MetricKind,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
}

fn default_ef_search() -> usize {
    DEFAULT_EF_SEARCH
}

impl VectorSchema {
    pub fn new(dim: usize, metric: MetricKind) -> Self {
        Self {
            dim,
            metric,
            ef_search: DEFAULT_EF_SEARCH,
        }
    }

    /// Create with an explicit ef_search (F-ARCH-2). Falls back to the default for 0 or missing.
    pub fn with_ef_search(dim: usize, metric: MetricKind, ef_search: usize) -> Self {
        Self {
            dim,
            metric,
            ef_search: if ef_search == 0 {
                DEFAULT_EF_SEARCH
            } else {
                ef_search
            },
        }
    }

    /// Validate vector dimension against the schema; returns DimensionMismatch on mismatch [v2 E2]
    pub fn validate_dim(&self, vector: &[f32]) -> crate::Result<()> {
        if vector.len() == self.dim {
            Ok(())
        } else {
            Err(crate::StoreError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_dim_passes_on_match() {
        let schema = VectorSchema::new(768, MetricKind::Cosine);
        let v = vec![0.0f32; 768];
        assert!(schema.validate_dim(&v).is_ok());
    }

    #[test]
    fn validate_dim_rejects_mismatch() {
        let schema = VectorSchema::new(768, MetricKind::Cosine);
        let v = vec![0.0f32; 1024];
        let err = schema.validate_dim(&v).unwrap_err();
        match err {
            crate::StoreError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 768);
                assert_eq!(got, 1024);
            }
            _ => panic!("expected DimensionMismatch"),
        }
    }
}
