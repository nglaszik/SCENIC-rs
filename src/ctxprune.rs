//! cisTarget pruning + regulon assembly — Rust port of pyscenic
//! `module2features_auc1st_impl` + `df2regulons` (transform.py) and
//! `load_motif_annotations` (utils.py).
//!
//! Per (db, module): enrich all motifs (AUC/NES), keep NES >= nes_threshold that
//! are annotated to the module's TF, take the leading-edge target genes, then
//! union per (TF, +/-) across modules & DBs keeping the max weight per gene.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::ctx;
use crate::ctxdb::RankingDb;
use crate::ctxmod::Module;

/// Motif→TF annotations, keyed (TF, motif_id) → best (qvalue, identity, desc).
pub struct MotifAnnotations {
    map: HashMap<(String, String), (f64, f64, String)>,
}

impl MotifAnnotations {
    /// Parse a motif2tf .tbl (columns: #motif_id, gene_name, motif_similarity_qvalue,
    /// orthologous_identity, description). Filters qval<=fdr & identity>=ident_thr,
    /// and keeps, per (TF, motif), the best row (min qval, then max identity) —
    /// matching load_motif_annotations + the keep="last" dedup in transform.py.
    pub fn load(path: &str, fdr: f64, ident_thr: f64) -> Result<MotifAnnotations, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty annotation file")?;
        let cols: Vec<&str> = header.split('\t').collect();
        let idx = |name: &str| cols.iter().position(|c| *c == name);
        let c_motif = idx("#motif_id").ok_or("no #motif_id column")?;
        let c_gene = idx("gene_name").ok_or("no gene_name column")?;
        let c_qval = idx("motif_similarity_qvalue").ok_or("no qvalue column")?;
        let c_ident = idx("orthologous_identity").ok_or("no identity column")?;
        let c_desc = idx("description").ok_or("no description column")?;

        let mut map: HashMap<(String, String), (f64, f64, String)> = HashMap::new();
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let qval: f64 = f[c_qval].parse().unwrap_or(f64::NAN);
            let ident: f64 = f[c_ident].parse().unwrap_or(f64::NAN);
            if !(qval <= fdr && ident >= ident_thr) {
                continue;
            }
            let key = (f[c_gene].to_string(), f[c_motif].to_string());
            let desc = f.get(c_desc).copied().unwrap_or("").to_string();
            // keep best: smallest qval, then largest identity
            match map.get(&key) {
                Some((q0, i0, _)) if !(qval < *q0 || (qval == *q0 && ident > *i0)) => {}
                _ => {
                    map.insert(key, (qval, ident, desc));
                }
            }
        }
        Ok(MotifAnnotations { map })
    }

    fn get(&self, tf: &str, motif: &str) -> Option<&(f64, f64, String)> {
        self.map.get(&(tf.to_string(), motif.to_string()))
    }
}

pub struct CtxRegulon {
    pub name: String, // "{TF}(+)" / "{TF}(-)"
    pub tf: String,
    pub activating: bool,
    pub gene2weight: Vec<(String, f64)>, // union of leading edges, max weight per gene
    pub nes: f64,                        // max NES across the regulon's motifs
}

/// One enriched, annotated motif's contribution to a (TF, type) regulon.
struct Row {
    nes: f64,
    target_genes: Vec<(String, f64)>,
}

#[allow(clippy::too_many_arguments)]
pub fn derive_regulons(
    dbs: &[&RankingDb],
    modules: &[Module],
    ann: &MotifAnnotations,
    rank_threshold: usize,
    auc_threshold: f64,
    nes_threshold: f64,
) -> Vec<CtxRegulon> {
    // Enrich every (db, module) pair in parallel; each produces ((tf, type), Row)
    // contributions. Inner math is sequential (parallelism is over modules) and
    // the leading edge is computed streaming, so peak memory per task is
    // O(rank_threshold), not the full n_motifs x rank_threshold recovery matrix.
    let pairs: Vec<(&&RankingDb, &Module)> = dbs
        .iter()
        .flat_map(|db| modules.iter().map(move |m| (db, m)))
        .collect();

    let per: Vec<Vec<((String, bool), Row)>> = pairs
        .par_iter()
        .map(|(db, m)| {
            let mut out: Vec<((String, bool), Row)> = Vec::new();
            let module_size = m.gene2weight.len();
            let genes: Vec<String> = m.gene2weight.iter().map(|(g, _)| g.clone()).collect();
            let (present, rankings) = db.load_module(&genes);
            let n_present = present.len();
            if n_present == 0 {
                return out;
            }
            // skip if <80% of module genes mapped (frac_missing >= 0.20)
            if (module_size - n_present) as f64 / module_size as f64 >= 0.20 {
                return out;
            }

            let n_motifs = db.n_motifs;
            let weights = vec![1.0; n_present]; // unweighted recovery (ctx default)
            let aucs = ctx::aucs_seq(
                &rankings,
                n_motifs,
                n_present,
                &weights,
                db.total_genes,
                auc_threshold,
            );
            let ness = ctx::nes(&aucs);

            // enriched + annotated-to-this-TF motifs
            let enriched: Vec<usize> = (0..n_motifs)
                .filter(|&i| {
                    ness[i] >= nes_threshold && ann.get(&m.tf, &db.motif_names[i]).is_some()
                })
                .collect();
            if enriched.is_empty() {
                return out;
            }

            // avg+2std over ALL motifs (streaming, O(rank_threshold) memory)
            let a2s =
                ctx::avg2std_streaming(&rankings, n_motifs, n_present, &weights, rank_threshold);
            let w_of: HashMap<&str, f64> = m
                .gene2weight
                .iter()
                .map(|(g, w)| (g.as_str(), *w))
                .collect();

            for &mi in &enriched {
                // rank_at_max computed only for this enriched motif
                let cutoff =
                    ctx::rank_at_max_one(&rankings, mi, n_present, &weights, rank_threshold, &a2s)
                        as i32;
                let target_genes: Vec<(String, f64)> = (0..n_present)
                    .filter(|&j| rankings[mi * n_present + j] <= cutoff)
                    .map(|j| {
                        (
                            present[j].clone(),
                            *w_of.get(present[j].as_str()).unwrap_or(&1.0),
                        )
                    })
                    .collect();
                out.push((
                    (m.tf.clone(), m.activating),
                    Row {
                        nes: ness[mi],
                        target_genes,
                    },
                ));
            }
            out
        })
        .collect();

    // group rows by (TF, type)
    let mut grouped: HashMap<(String, bool), Vec<Row>> = HashMap::new();
    for v in per {
        for (k, r) in v {
            grouped.entry(k).or_default().push(r);
        }
    }

    // df2regulons: union target genes per (TF, type), max weight per gene
    let mut out = Vec::new();
    for ((tf, activating), rows) in grouped {
        let mut g2w: HashMap<String, f64> = HashMap::new();
        let mut max_nes = f64::NEG_INFINITY;
        for r in &rows {
            if r.nes > max_nes {
                max_nes = r.nes;
            }
            for (g, w) in &r.target_genes {
                let e = g2w.entry(g.clone()).or_insert(f64::NEG_INFINITY);
                if *w > *e {
                    *e = *w;
                }
            }
        }
        let suffix = if activating { "(+)" } else { "(-)" };
        out.push(CtxRegulon {
            name: format!("{tf}{suffix}"),
            tf,
            activating,
            gene2weight: g2w.into_iter().collect(),
            nes: max_nes,
        });
    }
    out
}
