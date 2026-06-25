//! Co-expression module generation from GRN adjacencies — Rust port of
//! pyscenic `modules_from_adjacencies` (utils.py). Produces, per TF, several
//! candidate modules (gene sets + importances) via three methods (weight
//! percentile, top-N targets, top-N regulators-per-target), split into
//! activating/repressing by TF↔target expression correlation.
//!
//! Defaults match pyscenic: thresholds=(0.75,0.90) percentiles, top_n_targets=(50,),
//! top_n_regulators=(5,10,50), min_genes=20, rho_threshold=0.03, keep_only_activating.

use std::collections::HashMap;

use rayon::prelude::*;

#[derive(Clone)]
pub struct Module {
    pub tf: String,
    pub activating: bool, // true => (+), false => (-)
    #[allow(dead_code)] // method provenance, surfaced in the regulon Context later
    pub context: String, // e.g. "top50" / "weight>75.0%" / "top5perTarget"
    pub gene2weight: Vec<(String, f64)>, // unique genes incl. the TF (weight 1.0)
}

/// Pearson correlation between expression columns ci and cj. If `mask`, skip cells
/// where either value is 0 (ctxcore masked_rho); else use all cells (np.corrcoef).
fn corr(expr: &[f32], n_cells: usize, n_genes: usize, ci: usize, cj: usize, mask: bool) -> f64 {
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
    for c in 0..n_cells {
        let x = expr[c * n_genes + ci] as f64;
        let y = expr[c * n_genes + cj] as f64;
        if mask && (x == 0.0 || y == 0.0) {
            continue;
        }
        sx += x;
        sy += y;
        n += 1;
    }
    if n == 0 {
        return f64::NAN;
    }
    let (mx, my) = (sx / n as f64, sy / n as f64);
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for c in 0..n_cells {
        let x = expr[c * n_genes + ci] as f64;
        let y = expr[c * n_genes + cj] as f64;
        if mask && (x == 0.0 || y == 0.0) {
            continue;
        }
        let (dx, dy) = (x - mx, y - my);
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Linear-interpolation quantile of a pre-sorted slice (matches pandas/numpy default).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 < n {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    } else {
        sorted[lo]
    }
}

/// Build a module from a list of (edge_index, weight) rows for one TF: dedup
/// targets keeping last weight, then add the TF with weight 1.0 (overriding).
/// `edge_index` indexes into `adj_target` to recover the target gene name.
fn make_module(
    tf: &str,
    activating: bool,
    context: String,
    rows: &[(usize, f64)],
    adj_target: &[String],
) -> Module {
    // preserve first-seen order but last-wins on weight (frozendict semantics)
    let mut order: Vec<String> = Vec::new();
    let mut w: HashMap<String, f64> = HashMap::new();
    for &(edge_i, weight) in rows {
        let g = &adj_target[edge_i];
        if !w.contains_key(g) {
            order.push(g.clone());
        }
        w.insert(g.clone(), weight);
    }
    // add TF with weight 1.0 (overrides if TF was its own target)
    if !w.contains_key(tf) {
        order.push(tf.to_string());
    }
    w.insert(tf.to_string(), 1.0);
    let gene2weight = order
        .into_iter()
        .map(|g| {
            let v = w[&g];
            (g, v)
        })
        .collect();
    Module {
        tf: tf.to_string(),
        activating,
        context,
        gene2weight,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn modules_from_adjacencies(
    adj_tf: &[String],
    adj_target: &[String],
    adj_importance: &[f64],
    expr: &[f32],
    n_cells: usize,
    n_genes: usize,
    gene_names: &[String],
    thresholds: &[f64],
    top_n_targets: &[usize],
    top_n_regulators: &[usize],
    min_genes: usize,
    rho_threshold: f64,
    mask_dropouts: bool,
    keep_only_activating: bool,
) -> Vec<Module> {
    let gene_index: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    let n_edges = adj_tf.len();

    // --- correlation sign per edge (parallel over edges) ---
    let regulation: Vec<i32> = (0..n_edges)
        .into_par_iter()
        .map(|i| {
            let ci = gene_index[adj_tf[i].as_str()];
            let cj = gene_index[adj_target[i].as_str()];
            let rho = corr(expr, n_cells, n_genes, ci, cj, mask_dropouts);
            if rho > rho_threshold {
                1
            } else if rho < -rho_threshold {
                -1
            } else {
                0
            }
        })
        .collect();

    // quantile thresholds computed on ALL edge importances (pyscenic uses the full df)
    let mut sorted_imp = adj_importance.to_vec();
    sorted_imp.sort_by(|a, b| a.total_cmp(b));
    let thr_vals: Vec<f64> = thresholds
        .iter()
        .map(|&q| quantile(&sorted_imp, q))
        .collect();

    let signs: &[i32] = if keep_only_activating { &[1] } else { &[1, -1] };
    let mut modules: Vec<Module> = Vec::new();

    for &sign in signs {
        let activating = sign > 0;
        // edges in this regulation class
        let sub: Vec<usize> = (0..n_edges).filter(|&i| regulation[i] == sign).collect();

        // group helpers over the subset
        let mut by_tf: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut by_target: HashMap<&str, Vec<usize>> = HashMap::new();
        for &i in &sub {
            by_tf.entry(adj_tf[i].as_str()).or_default().push(i);
            by_target.entry(adj_target[i].as_str()).or_default().push(i);
        }

        // --- method 1: weight > percentile threshold ---
        for (k, &thr) in thr_vals.iter().enumerate() {
            let ctx = format!("weight>{}%", thresholds[k] * 100.0);
            let mut grp: HashMap<&str, Vec<(usize, f64)>> = HashMap::new();
            for &i in &sub {
                if adj_importance[i] > thr {
                    grp.entry(adj_tf[i].as_str())
                        .or_default()
                        .push((i, adj_importance[i]));
                }
            }
            for (tf, rows) in &grp {
                modules.push(make_module(tf, activating, ctx.clone(), rows, adj_target));
            }
        }

        // --- method 2: top-N targets per TF ---
        for &n in top_n_targets {
            let ctx = format!("top{}", n);
            for (tf, idxs) in &by_tf {
                let rows: Vec<(usize, f64)> = nlargest(idxs, adj_importance, n)
                    .iter()
                    .map(|&i| (i, adj_importance[i]))
                    .collect();
                if !rows.is_empty() {
                    modules.push(make_module(tf, activating, ctx.clone(), &rows, adj_target));
                }
            }
        }

        // --- method 3: top-N regulators per target, regrouped by TF ---
        for &n in top_n_regulators {
            let ctx = format!("top{}perTarget", n);
            let mut regrp: HashMap<&str, Vec<(usize, f64)>> = HashMap::new();
            for idxs in by_target.values() {
                for &i in nlargest(idxs, adj_importance, n).iter() {
                    regrp
                        .entry(adj_tf[i].as_str())
                        .or_default()
                        .push((i, adj_importance[i]));
                }
            }
            for (tf, rows) in &regrp {
                modules.push(make_module(tf, activating, ctx.clone(), rows, adj_target));
            }
        }
    }

    // filter by minimum gene count (unique genes incl. TF)
    modules
        .into_iter()
        .filter(|m| m.gene2weight.len() >= min_genes)
        .collect()
}

/// Top-n edge indices by importance, ties broken by original (ascending index)
/// order — matches pandas nlargest(keep="first").
fn nlargest(idxs: &[usize], importance: &[f64], n: usize) -> Vec<usize> {
    let mut v = idxs.to_vec();
    v.sort_by(|&a, &b| importance[b].total_cmp(&importance[a]).then(a.cmp(&b)));
    v.truncate(n);
    v
}
