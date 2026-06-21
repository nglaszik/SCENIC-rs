//! GRNBoost2: gradient-boosted regression trees per target gene (the arboreto
//! default). Reference algorithm = sklearn GradientBoostingRegressor. Feature
//! importances (total split gain per regulator) are the network edges.

use crate::forest::MaxFeatures;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[derive(Clone)]
pub struct GbmParams {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub max_features: MaxFeatures,
    pub subsample: f64,
    pub min_leaf: usize,
    /// Rolling-window length for out-of-bag early stopping (GRNBoost2's regularizer).
    /// 0 disables early stopping (then `n_estimators` trees are always built).
    pub early_stop_window: usize,
    pub seed: u64,
}

#[derive(Clone)]
struct Node {
    feature: i32, // -1 = leaf
    threshold: f32,
    left: u32,
    right: u32,
    value: f32,
}

#[allow(clippy::too_many_arguments)]
fn build_tree(
    cols: &[&[f32]],
    resid: &[f32],
    sorted_feat: &[Vec<u32>],
    idx: &[usize],
    depth: usize,
    max_depth: usize,
    mf: usize,
    min_leaf: usize,
    rng: &mut ChaCha8Rng,
    nodes: &mut Vec<Node>,
    gains: &mut [f64],
    feat_buf: &mut Vec<usize>,
    member: &mut [bool],
    order_buf: &mut Vec<u32>,
) -> u32 {
    let n = idx.len();
    let (mut s, mut ss) = (0f64, 0f64);
    for &i in idx {
        let v = resid[i] as f64;
        s += v;
        ss += v * v;
    }
    let mean = (s / n as f64) as f32;
    let make_leaf = |nodes: &mut Vec<Node>| {
        let id = nodes.len() as u32;
        nodes.push(Node { feature: -1, threshold: 0.0, left: 0, right: 0, value: mean });
        id
    };
    if depth >= max_depth || n < 2 * min_leaf {
        return make_leaf(nodes);
    }
    let node_imp = ss - s * s / n as f64;
    if node_imp <= 1e-12 {
        return make_leaf(nodes);
    }

    // Mark this node's samples so we can pull each feature's pre-sorted order in
    // O(n) instead of re-sorting per feature. sorted_feat[f] is the global sample
    // order by feature value (built once per target); filtering it by membership
    // yields exactly the same per-node sorted order the old code re-derived.
    for &i in idx {
        member[i] = true;
    }

    feat_buf.shuffle(rng);
    // sklearn GradientBoostingRegressor selects splits by Friedman's improvement
    // score  (n_l*n_r/(n_l+n_r)) * (mean_l - mean_r)^2 , but accumulates feature
    // importance as the MSE (variance) impurity decrease of the chosen split.
    let mut best: Option<(usize, f32, f64, f64)> = None; // (feat, thr, friedman, mse_dec)
    for &f in feat_buf.iter().take(mf) {
        let col = cols[f];
        order_buf.clear();
        for &smp in &sorted_feat[f] {
            if member[smp as usize] {
                order_buf.push(smp);
            }
        }
        let (mut ls, mut lss, mut ln) = (0f64, 0f64, 0usize);
        for k in 0..n - 1 {
            let i = order_buf[k] as usize;
            let v = resid[i] as f64;
            ls += v;
            lss += v * v;
            ln += 1;
            let (vf, vfn) = (col[order_buf[k] as usize], col[order_buf[k + 1] as usize]);
            if vf == vfn {
                continue;
            }
            let rn = n - ln;
            if ln < min_leaf || rn < min_leaf {
                continue;
            }
            let (lnf, rnf) = (ln as f64, rn as f64);
            let (rs, rss) = (s - ls, ss - lss);
            let diff = ls / lnf - rs / rnf;
            let friedman = lnf * rnf / (lnf + rnf) * diff * diff;
            if best.map_or(true, |(_, _, bf, _)| friedman > bf) {
                let mse_dec = node_imp - (lss - ls * ls / lnf) - (rss - rs * rs / rnf);
                best = Some((f, (vf + vfn) / 2.0, friedman, mse_dec));
            }
        }
    }

    for &i in idx {
        member[i] = false;
    }

    let (f, thr, _friedman, mse_dec) = match best {
        Some(b) if b.2 > 1e-12 => b,
        _ => return make_leaf(nodes),
    };
    gains[f] += mse_dec;
    let col = cols[f];
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for &i in idx {
        if col[i] <= thr {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    if left.is_empty() || right.is_empty() {
        return make_leaf(nodes);
    }
    let id = nodes.len() as u32;
    nodes.push(Node { feature: f as i32, threshold: thr, left: 0, right: 0, value: mean });
    let l = build_tree(cols, resid, sorted_feat, &left, depth + 1, max_depth, mf, min_leaf, rng, nodes, gains, feat_buf, member, order_buf);
    let r = build_tree(cols, resid, sorted_feat, &right, depth + 1, max_depth, mf, min_leaf, rng, nodes, gains, feat_buf, member, order_buf);
    nodes[id as usize].left = l;
    nodes[id as usize].right = r;
    id
}

fn predict_row(nodes: &[Node], cols: &[&[f32]], i: usize) -> f32 {
    let mut n = 0usize;
    loop {
        let node = &nodes[n];
        if node.feature < 0 {
            return node.value;
        }
        n = if cols[node.feature as usize][i] <= node.threshold {
            node.left as usize
        } else {
            node.right as usize
        };
    }
}

fn boost_importance(cols: &[&[f32]], y: &[f32], p: &GbmParams) -> Vec<f64> {
    let n = y.len();
    let n_feat = cols.len();
    let mf = match &p.max_features {
        MaxFeatures::Sqrt => (n_feat as f64).sqrt().round() as usize,
        MaxFeatures::All => n_feat,
        MaxFeatures::Frac(f) => (*f * n_feat as f64).round() as usize,
        MaxFeatures::Count(c) => *c,
    }
    .clamp(1, n_feat);

    let mean_y = (y.iter().map(|&v| v as f64).sum::<f64>() / n as f64) as f32;
    let lr = p.learning_rate as f32;
    let mut pred = vec![mean_y; n];
    let mut gains = vec![0f64; n_feat];
    let mut rng = ChaCha8Rng::seed_from_u64(p.seed);
    let mut feat_buf: Vec<usize> = (0..n_feat).collect();
    let n_sub = ((p.subsample * n as f64).round() as usize).clamp(1, n);

    // Pre-sort the sample order by each feature value ONCE (feature values are
    // constant across boosting rounds). Each tree node then reuses these orders
    // via a membership filter instead of re-sorting — the main GRNBoost2 speedup.
    let sorted_feat: Vec<Vec<u32>> = cols
        .iter()
        .map(|col| {
            let mut v: Vec<u32> = (0..n as u32).collect();
            v.sort_unstable_by(|&a, &b| col[a as usize].total_cmp(&col[b as usize]));
            v
        })
        .collect();
    let mut member = vec![false; n];
    let mut order_buf: Vec<u32> = Vec::with_capacity(n);

    // Out-of-bag early stopping (GRNBoost2): with subsample < 1 each round leaves
    // held-out (OOB) samples. We track each tree's improvement in OOB squared
    // error and stop once the mean improvement over the trailing window goes
    // negative — i.e. added trees no longer help generalization. Mirrors
    // arboreto's EarlyStopMonitor; the stop point is data-driven, per target.
    let can_stop = p.early_stop_window > 0 && n_sub < n;
    let mut oob_imp: Vec<f64> = Vec::new();

    let oob_mse = |pred: &[f32], oob: &[usize]| -> f64 {
        let s: f64 = oob.iter().map(|&i| { let r = (y[i] - pred[i]) as f64; r * r }).sum();
        s / oob.len() as f64
    };

    for _ in 0..p.n_estimators {
        let resid: Vec<f32> = (0..n).map(|i| y[i] - pred[i]).collect();
        let mut all: Vec<usize> = (0..n).collect();
        let (in_bag, out_bag): (&[usize], &[usize]) = if n_sub < n {
            all.shuffle(&mut rng);
            all.split_at(n_sub)
        } else {
            (&all[..], &[][..])
        };
        let loss_before = if can_stop { oob_mse(&pred, out_bag) } else { 0.0 };

        let mut nodes: Vec<Node> = Vec::new();
        build_tree(cols, &resid, &sorted_feat, in_bag, 0, p.max_depth, mf, p.min_leaf,
                   &mut rng, &mut nodes, &mut gains, &mut feat_buf, &mut member, &mut order_buf);
        for i in 0..n {
            pred[i] += lr * predict_row(&nodes, cols, i);
        }

        if can_stop {
            oob_imp.push(loss_before - oob_mse(&pred, out_bag));
            let t = oob_imp.len();
            if t >= p.early_stop_window {
                let mean = oob_imp[t - p.early_stop_window..].iter().sum::<f64>()
                    / p.early_stop_window as f64;
                if mean < 0.0 {
                    break;
                }
            }
        }
    }
    let tot: f64 = gains.iter().sum();
    if tot > 0.0 {
        for g in gains.iter_mut() {
            *g /= tot;
        }
    }
    gains
}

pub fn run_grnboost2(
    expr: &[f32],
    n_cells: usize,
    n_genes: usize,
    regulators: &[usize],
    p: &GbmParams,
) -> Vec<(usize, usize, f32)> {
    let reg_cols: Vec<Vec<f32>> = regulators
        .iter()
        .map(|&g| (0..n_cells).map(|c| expr[c * n_genes + g]).collect())
        .collect();

    (0..n_genes)
        .into_par_iter()
        .flat_map_iter(|target| {
            let cand: Vec<usize> = (0..regulators.len()).filter(|&r| regulators[r] != target).collect();
            if cand.is_empty() {
                return Vec::new().into_iter();
            }
            let cols: Vec<&[f32]> = cand.iter().map(|&r| reg_cols[r].as_slice()).collect();
            let y: Vec<f32> = (0..n_cells).map(|c| expr[c * n_genes + target]).collect();
            let mut pp = p.clone();
            pp.seed = p.seed ^ (target as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let imp = boost_importance(&cols, &y, &pp);
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
