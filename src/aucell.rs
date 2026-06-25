//! AUCell: score each gene signature (regulon) in each cell by the area under
//! the recovery curve of its genes in that cell's expression ranking.
//!
//! Pure-Python AUCell is loop-heavy and slow; this is parallel over cells.

use rayon::prelude::*;
use std::cmp::Ordering;

/// `expr` row-major (n_cells x n_genes). `regulons[r]` = gene indices of regulon r.
/// Returns an (n_cells x n_regulons) matrix of AUC scores, row-major.
pub fn run_aucell(
    expr: &[f32],
    n_cells: usize,
    n_genes: usize,
    regulons: &[Vec<usize>],
    auc_max_rank: usize,
) -> Vec<f32> {
    let n_reg = regulons.len();
    let rows: Vec<Vec<f32>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let row = &expr[c * n_genes..(c + 1) * n_genes];
            // rank genes by descending expression; ties broken by gene index
            let mut order: Vec<usize> = (0..n_genes).collect();
            order.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut rank = vec![0u32; n_genes];
            for (k, &g) in order.iter().enumerate() {
                rank[g] = k as u32;
            }
            regulons
                .iter()
                .map(|genes| {
                    if genes.is_empty() {
                        return 0.0f32;
                    }
                    let mut raw: u64 = 0;
                    for &g in genes {
                        let r = rank[g] as usize;
                        if r < auc_max_rank {
                            raw += (auc_max_rank - r) as u64;
                        }
                    }
                    let max_auc = (auc_max_rank * genes.len()) as f64;
                    (raw as f64 / max_auc) as f32
                })
                .collect()
        })
        .collect();

    let mut flat = vec![0f32; n_cells * n_reg];
    for (c, r) in rows.iter().enumerate() {
        flat[c * n_reg..(c + 1) * n_reg].copy_from_slice(r);
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden values worked out by hand. One cell, four genes:
    //   expr = [0.1, 0.4, 0.3, 0.2]  ->  descending ranks: g1=0, g2=1, g3=2, g0=3
    // For regulon {g0, g1} at auc_max_rank=4:
    //   g1: rank 0 -> +(4-0)=4 ; g0: rank 3 -> +(4-3)=1 ; raw=5
    //   max_auc = 4*2 = 8  ->  score = 5/8 = 0.625
    #[test]
    fn aucell_recovery_score_matches_hand_calc() {
        let expr = vec![0.1f32, 0.4, 0.3, 0.2];
        let regulons = vec![vec![0usize, 1]];
        let out = run_aucell(&expr, 1, 4, &regulons, 4);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.625).abs() < 1e-6, "got {}", out[0]);
    }

    // A gene ranked at or beyond auc_max_rank contributes nothing.
    // {g1} fully inside rank<2: rank 0 -> +2, max_auc=2 -> 1.0
    // {g0} outside rank<2: contributes 0 -> 0.0
    #[test]
    fn aucell_genes_beyond_max_rank_contribute_zero() {
        let expr = vec![0.1f32, 0.4, 0.3, 0.2]; // g0 is rank 3
        let regulons = [vec![1usize], vec![0usize]];
        let out = run_aucell(&expr, 1, 4, &regulons, 2);
        assert!((out[0] - 1.0).abs() < 1e-6, "g1 got {}", out[0]);
        assert!((out[1] - 0.0).abs() < 1e-6, "g0 got {}", out[1]);
    }

    // Empty regulon scores 0; scores stay within [0, 1].
    #[test]
    fn aucell_empty_regulon_is_zero_and_bounds_hold() {
        let expr = vec![0.5f32, 0.2, 0.9, 0.1];
        let regulons = [vec![], vec![0, 1, 2, 3]];
        let out = run_aucell(&expr, 1, 4, &regulons, 4);
        assert_eq!(out[0], 0.0);
        assert!((0.0..=1.0).contains(&out[1]));
    }
}
