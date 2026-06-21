//! Random-forest variance-reduction feature importances.
//!
//! GENIE3 only needs per-feature importances, not predictions, so we never
//! materialize the trees — we recurse over node sample-index sets and accumulate
//! the weighted impurity (variance) decrease for the feature chosen at each split.

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone)]
pub enum MaxFeatures {
    Sqrt,
    All,
    Frac(f64),
    Count(usize),
}

impl MaxFeatures {
    fn resolve(&self, n: usize) -> usize {
        let k = match self {
            MaxFeatures::Sqrt => (n as f64).sqrt().round() as usize,
            MaxFeatures::All => n,
            MaxFeatures::Frac(f) => (*f * n as f64).round() as usize,
            MaxFeatures::Count(c) => *c,
        };
        k.clamp(1, n.max(1))
    }
}

/// Accumulate variance-reduction importances for one regression tree (no storage).
fn tree_importance(
    cols: &[&[f32]],
    y: &[f32],
    sample: &[usize],
    max_features: usize,
    min_leaf: usize,
    rng: &mut ChaCha8Rng,
    imp: &mut [f64],
) {
    let n_features = cols.len();
    let mut feat_buf: Vec<usize> = (0..n_features).collect();
    let mut stack: Vec<Vec<usize>> = vec![sample.to_vec()];

    while let Some(idx) = stack.pop() {
        let n = idx.len();
        if n < 2 * min_leaf || n < 2 {
            continue;
        }
        // node statistics
        let (mut s, mut ss) = (0f64, 0f64);
        for &i in &idx {
            let v = y[i] as f64;
            s += v;
            ss += v * v;
        }
        let node_imp = ss - s * s / n as f64;
        if node_imp <= 1e-12 {
            continue;
        }
        // pick a random subset of features
        feat_buf.shuffle(rng);
        let mut best: Option<(usize, f32, f64)> = None; // (feature, threshold, decrease)

        for &f in feat_buf.iter().take(max_features) {
            let col = cols[f];
            let mut order = idx.clone();
            order.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap_or(std::cmp::Ordering::Equal));

            let (mut ls, mut lss, mut ln) = (0f64, 0f64, 0usize);
            for k in 0..n - 1 {
                let i = order[k];
                let v = y[i] as f64;
                ls += v;
                lss += v * v;
                ln += 1;
                let vf = col[order[k]];
                let vf_next = col[order[k + 1]];
                if vf == vf_next {
                    continue;
                }
                let rn = n - ln;
                if ln < min_leaf || rn < min_leaf {
                    continue;
                }
                let rs = s - ls;
                let rss = ss - lss;
                let left_imp = lss - ls * ls / ln as f64;
                let right_imp = rss - rs * rs / rn as f64;
                let dec = node_imp - left_imp - right_imp;
                if best.map_or(true, |(_, _, bd)| dec > bd) {
                    best = Some((f, (vf + vf_next) / 2.0, dec));
                }
            }
        }

        if let Some((f, thr, dec)) = best {
            if dec <= 1e-12 {
                continue;
            }
            imp[f] += dec;
            let col = cols[f];
            let (mut left, mut right) = (Vec::new(), Vec::new());
            for &i in &idx {
                if col[i] <= thr {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }
            if !left.is_empty() && !right.is_empty() {
                stack.push(left);
                stack.push(right);
            }
        }
    }
}

/// Random-forest importances over `cols` (each a regulator column of length n_cells),
/// predicting `y`. Returns a per-feature importance vector that sums to ~1.
pub fn forest_importance(
    cols: &[&[f32]],
    y: &[f32],
    n_trees: usize,
    max_features: MaxFeatures,
    min_leaf: usize,
    seed: u64,
) -> Vec<f64> {
    let n_features = cols.len();
    let n_cells = y.len();
    let mf = max_features.resolve(n_features);
    let mut total = vec![0f64; n_features];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..n_trees {
        let sample: Vec<usize> = (0..n_cells).map(|_| rng.gen_range(0..n_cells)).collect();
        let mut imp = vec![0f64; n_features];
        tree_importance(cols, y, &sample, mf, min_leaf, &mut rng, &mut imp);
        let s: f64 = imp.iter().sum();
        if s > 0.0 {
            for (t, v) in total.iter_mut().zip(imp.iter()) {
                *t += v / s;
            }
        }
    }
    for v in total.iter_mut() {
        *v /= n_trees as f64;
    }
    total
}
