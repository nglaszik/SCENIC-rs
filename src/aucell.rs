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
