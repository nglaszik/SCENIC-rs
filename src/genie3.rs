//! GENIE3 orchestration: one tree-ensemble regression per target gene, run in
//! parallel over targets with rayon. Each regulator's importance for a target is
//! an edge in the inferred network.

use crate::forest::{forest_importance, MaxFeatures};
use rayon::prelude::*;

#[derive(Clone)]
pub struct Genie3Params {
    pub n_trees: usize,
    pub max_features: MaxFeatures,
    pub min_leaf: usize,
    pub seed: u64,
}

/// `expr` is row-major (n_cells x n_genes). `regulators` are gene indices that are TFs.
/// Returns edges as (regulator_gene_idx, target_gene_idx, importance).
pub fn run_genie3(
    expr: &[f32],
    n_cells: usize,
    n_genes: usize,
    regulators: &[usize],
    params: &Genie3Params,
) -> Vec<(usize, usize, f32)> {
    // extract each regulator as a contiguous column over cells
    let reg_cols: Vec<Vec<f32>> = regulators
        .iter()
        .map(|&g| (0..n_cells).map(|c| expr[c * n_genes + g]).collect())
        .collect();

    (0..n_genes)
        .into_par_iter()
        .flat_map_iter(|target| {
            // candidate regulators excluding the target itself
            let cand: Vec<usize> = (0..regulators.len())
                .filter(|&r| regulators[r] != target)
                .collect();
            if cand.is_empty() {
                return Vec::new().into_iter();
            }
            let cols: Vec<&[f32]> = cand.iter().map(|&r| reg_cols[r].as_slice()).collect();
            let y: Vec<f32> = (0..n_cells).map(|c| expr[c * n_genes + target]).collect();
            let seed = params
                .seed
                ^ (target as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let imp = forest_importance(
                &cols,
                &y,
                params.n_trees,
                params.max_features.clone(),
                params.min_leaf,
                seed,
            );
            let edges: Vec<(usize, usize, f32)> = cand
                .iter()
                .zip(imp.iter())
                .filter(|(_, &w)| w > 0.0)
                .map(|(&r, &w)| (regulators[r], target, w as f32))
                .collect();
            edges.into_iter()
        })
        .collect()
}
