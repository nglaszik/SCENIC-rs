//! ctx (cisTarget) enrichment math — Rust port of `ctxcore.recovery`.
//!
//! Parity target: ctxcore 0.2.0. These are the numeric building blocks of the
//! `ctx` step: per-motif recovery-curve AUC (`weighted_auc1d`), the NES
//! normalization across all motifs, and the leading-edge critical point. See
//! `docs/ctx_spec.md` for the full algorithm and reference line numbers.

use rayon::prelude::*;

/// `round(auc_threshold * total_genes) - 1`, matching ctxcore `derive_rank_cutoff`.
/// (ctxcore's `rank_threshold` arg only bounds an assertion here; the cutoff is
/// purely a function of `auc_threshold` and `total_genes`.)
pub fn rank_cutoff(auc_threshold: f64, total_genes: usize) -> usize {
    let rc = (auc_threshold * total_genes as f64).round() as i64 - 1;
    rc.max(0) as usize
}

/// Per-motif weighted AUC of the recovery curve (ctxcore `weighted_auc1d`/`aucs`).
///
/// `rankings` is row-major `n_motifs x n_genes`; each entry is the 0-based rank of
/// that module gene for that motif. `weights` has length `n_genes`. Mirrors
/// `maxauc = (rank_cutoff + 1) * weights.sum()` and the trapezoidal sum
/// `sum(diff(x) * cumsum(w)) / maxauc` over genes with rank < rank_cutoff.
pub fn aucs(
    rankings: &[i32],
    n_motifs: usize,
    n_genes: usize,
    weights: &[f64],
    total_genes: usize,
    auc_threshold: f64,
) -> Vec<f64> {
    let rc = rank_cutoff(auc_threshold, total_genes) as i64;
    let wsum: f64 = weights.iter().sum();
    let maxauc = (rc as f64 + 1.0) * wsum;
    (0..n_motifs)
        .into_par_iter()
        .map(|m| {
            let row = &rankings[m * n_genes..(m + 1) * n_genes];
            // genes within the AUC window (rank < rank_cutoff), sorted by rank
            let mut pairs: Vec<(i32, f64)> = (0..n_genes)
                .filter(|&g| (row[g] as i64) < rc)
                .map(|g| (row[g], weights[g]))
                .collect();
            pairs.sort_unstable_by_key(|p| p.0);
            let k = pairs.len();
            let (mut cum, mut auc) = (0.0f64, 0.0f64);
            for i in 0..k {
                cum += pairs[i].1; // cumsum of weights = recovery height
                let next_x = if i + 1 < k { pairs[i + 1].0 } else { rc as i32 };
                auc += (next_x - pairs[i].0) as f64 * cum;
            }
            if maxauc > 0.0 {
                auc / maxauc
            } else {
                0.0
            }
        })
        .collect()
}

/// NES = (auc - mean(aucs)) / std(aucs), population std (ddof=0), over all motifs.
pub fn nes(aucs: &[f64]) -> Vec<f64> {
    let n = aucs.len() as f64;
    let mean = aucs.iter().sum::<f64>() / n;
    let var = aucs.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    aucs.iter().map(|a| (a - mean) / std).collect()
}

/// Recovery curves, row-major `n_motifs x rank_threshold`.
/// `rcc[m][t]` = cumulative weight of module genes with rank <= t (ctxcore
/// `rcc2d`: bincount(weights)[:rank_threshold] then cumsum).
pub fn recovery_curves(
    rankings: &[i32],
    n_motifs: usize,
    n_genes: usize,
    weights: &[f64],
    rank_threshold: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n_motifs * rank_threshold];
    out.par_chunks_mut(rank_threshold)
        .enumerate()
        .for_each(|(m, rcc)| {
            let row = &rankings[m * n_genes..(m + 1) * n_genes];
            for g in 0..n_genes {
                let r = row[g];
                if r >= 0 && (r as usize) < rank_threshold {
                    rcc[r as usize] += weights[g];
                }
            }
            let mut cum = 0.0;
            for v in rcc.iter_mut() {
                cum += *v;
                *v = cum;
            }
        });
    out
}

/// `avgrcc + 2*std(rccs)` per rank position (population std), over all motifs.
pub fn avg2std_rcc(rccs: &[f64], n_motifs: usize, rank_threshold: usize) -> Vec<f64> {
    (0..rank_threshold)
        .into_par_iter()
        .map(|t| {
            let mut s = 0.0;
            for m in 0..n_motifs {
                s += rccs[m * rank_threshold + t];
            }
            let mean = s / n_motifs as f64;
            let mut var = 0.0;
            for m in 0..n_motifs {
                let d = rccs[m * rank_threshold + t] - mean;
                var += d * d;
            }
            mean + 2.0 * (var / n_motifs as f64).sqrt()
        })
        .collect()
}

/// Sequential AUCs (no rayon) — for use inside an outer parallel-over-modules loop.
pub fn aucs_seq(
    rankings: &[i32],
    n_motifs: usize,
    n_genes: usize,
    weights: &[f64],
    total_genes: usize,
    auc_threshold: f64,
) -> Vec<f64> {
    let rc = rank_cutoff(auc_threshold, total_genes) as i64;
    let wsum: f64 = weights.iter().sum();
    let maxauc = (rc as f64 + 1.0) * wsum;
    let mut pairs: Vec<(i32, f64)> = Vec::with_capacity(n_genes);
    (0..n_motifs)
        .map(|m| {
            let row = &rankings[m * n_genes..(m + 1) * n_genes];
            pairs.clear();
            for g in 0..n_genes {
                if (row[g] as i64) < rc {
                    pairs.push((row[g], weights[g]));
                }
            }
            pairs.sort_unstable_by_key(|p| p.0);
            let (mut cum, mut auc) = (0.0f64, 0.0f64);
            for i in 0..pairs.len() {
                cum += pairs[i].1;
                let next_x = if i + 1 < pairs.len() {
                    pairs[i + 1].0
                } else {
                    rc as i32
                };
                auc += (next_x - pairs[i].0) as f64 * cum;
            }
            if maxauc > 0.0 {
                auc / maxauc
            } else {
                0.0
            }
        })
        .collect()
}

/// Fill `buf` (length rank_threshold) with motif m's recovery curve.
fn rcc_into(
    rankings: &[i32],
    m: usize,
    n_present: usize,
    weights: &[f64],
    rank_threshold: usize,
    buf: &mut [f64],
) {
    for v in buf.iter_mut() {
        *v = 0.0;
    }
    let row = &rankings[m * n_present..(m + 1) * n_present];
    for g in 0..n_present {
        let r = row[g];
        if r >= 0 && (r as usize) < rank_threshold {
            buf[r as usize] += weights[g];
        }
    }
    let mut cum = 0.0;
    for v in buf.iter_mut() {
        cum += *v;
        *v = cum;
    }
}

/// `avg + 2*std` of the recovery curves over all motifs, computed WITHOUT
/// materializing the full n_motifs x rank_threshold matrix (two streaming passes,
/// motifs summed in ascending order so the result is bit-identical to
/// `avg2std_rcc`). Memory is O(rank_threshold).
pub fn avg2std_streaming(
    rankings: &[i32],
    n_motifs: usize,
    n_present: usize,
    weights: &[f64],
    rank_threshold: usize,
) -> Vec<f64> {
    let mut buf = vec![0f64; rank_threshold];
    let mut sum = vec![0f64; rank_threshold];
    for m in 0..n_motifs {
        rcc_into(rankings, m, n_present, weights, rank_threshold, &mut buf);
        for t in 0..rank_threshold {
            sum[t] += buf[t];
        }
    }
    let n = n_motifs as f64;
    let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
    let mut sse = vec![0f64; rank_threshold];
    for m in 0..n_motifs {
        rcc_into(rankings, m, n_present, weights, rank_threshold, &mut buf);
        for t in 0..rank_threshold {
            let d = buf[t] - mean[t];
            sse[t] += d * d;
        }
    }
    (0..rank_threshold)
        .map(|t| mean[t] + 2.0 * (sse[t] / n).sqrt())
        .collect()
}

/// Leading-edge critical point for a SINGLE motif: argmax_t (rcc[t] - avg2std[t]).
pub fn rank_at_max_one(
    rankings: &[i32],
    m: usize,
    n_present: usize,
    weights: &[f64],
    rank_threshold: usize,
    avg2std: &[f64],
) -> usize {
    let mut buf = vec![0f64; rank_threshold];
    rcc_into(rankings, m, n_present, weights, rank_threshold, &mut buf);
    let mut best_t = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for t in 0..rank_threshold {
        let v = buf[t] - avg2std[t];
        if v > best_v {
            best_v = v;
            best_t = t;
        }
    }
    best_t
}

/// Leading-edge critical point per motif: argmax_t (rcc[t] - avg2stdrcc[t]).
/// (ctxcore `leading_edge.critical_point`.) Leading-edge genes are those with
/// rank <= the returned rank_at_max.
pub fn rank_at_max(
    rccs: &[f64],
    n_motifs: usize,
    rank_threshold: usize,
    avg2stdrcc: &[f64],
) -> Vec<usize> {
    (0..n_motifs)
        .into_par_iter()
        .map(|m| {
            let rcc = &rccs[m * rank_threshold..(m + 1) * rank_threshold];
            let mut best_t = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for t in 0..rank_threshold {
                let v = rcc[t] - avg2stdrcc[t];
                if v > best_v {
                    best_v = v;
                    best_t = t;
                }
            }
            best_t
        })
        .collect()
}
