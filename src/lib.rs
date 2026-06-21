use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use std::collections::HashMap;

mod aucell;
mod forest;
mod genie3;
mod grnboost2;

use aucell::run_aucell;
use forest::MaxFeatures;
use genie3::{run_genie3, Genie3Params};
use grnboost2::{run_grnboost2, GbmParams};

fn parse_max_features(s: &str) -> MaxFeatures {
    match s {
        "sqrt" => MaxFeatures::Sqrt,
        "all" => MaxFeatures::All,
        other => match other.parse::<f64>() {
            Ok(f) if f > 0.0 && f <= 1.0 => MaxFeatures::Frac(f),
            Ok(f) if f > 1.0 => MaxFeatures::Count(f as usize),
            _ => MaxFeatures::Sqrt,
        },
    }
}

fn map_regulators(gene_names: &[String], tf_names: &[String]) -> Vec<usize> {
    let idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    tf_names
        .iter()
        .filter_map(|t| idx.get(t.as_str()).copied())
        .collect()
}

fn names_for(edges: &[(usize, usize, f32)], gene_names: &[String]) -> (Vec<String>, Vec<String>, Vec<f32>) {
    let tf = edges.iter().map(|&(t, _, _)| gene_names[t].clone()).collect();
    let tg = edges.iter().map(|&(_, g, _)| gene_names[g].clone()).collect();
    let w = edges.iter().map(|&(_, _, w)| w).collect();
    (tf, tg, w)
}

/// GENIE3 (random forest) GRN inference.
#[pyfunction]
#[pyo3(name = "genie3", signature = (expr, gene_names, tf_names, n_estimators=1000, max_features="sqrt", min_samples_leaf=1, seed=42))]
fn genie3_py(
    py: Python<'_>,
    expr: PyReadonlyArray2<f32>,
    gene_names: Vec<String>,
    tf_names: Vec<String>,
    n_estimators: usize,
    max_features: &str,
    min_samples_leaf: usize,
    seed: u64,
) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    let arr = expr.as_array();
    let (n_cells, n_genes) = (arr.shape()[0], arr.shape()[1]);
    let regulators = map_regulators(&gene_names, &tf_names);
    if regulators.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no tf_names matched gene_names"));
    }
    let expr_vec: Vec<f32> = arr.iter().copied().collect();
    let params = Genie3Params {
        n_trees: n_estimators,
        max_features: parse_max_features(max_features),
        min_leaf: min_samples_leaf,
        seed,
    };
    let edges = py.allow_threads(|| run_genie3(&expr_vec, n_cells, n_genes, &regulators, &params));
    Ok(names_for(&edges, &gene_names))
}

/// GRNBoost2 (gradient boosting) GRN inference.
#[pyfunction]
#[pyo3(name = "grnboost2", signature = (expr, gene_names, tf_names, n_estimators=5000, learning_rate=0.01, max_depth=3, max_features="0.1", subsample=0.9, min_samples_leaf=1, early_stop_window=25, seed=42))]
fn grnboost2_py(
    py: Python<'_>,
    expr: PyReadonlyArray2<f32>,
    gene_names: Vec<String>,
    tf_names: Vec<String>,
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    max_features: &str,
    subsample: f64,
    min_samples_leaf: usize,
    early_stop_window: usize,
    seed: u64,
) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    let arr = expr.as_array();
    let (n_cells, n_genes) = (arr.shape()[0], arr.shape()[1]);
    let regulators = map_regulators(&gene_names, &tf_names);
    if regulators.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no tf_names matched gene_names"));
    }
    let expr_vec: Vec<f32> = arr.iter().copied().collect();
    let params = GbmParams {
        n_estimators,
        learning_rate,
        max_depth,
        max_features: parse_max_features(max_features),
        subsample,
        min_leaf: min_samples_leaf,
        early_stop_window,
        seed,
    };
    let edges = py.allow_threads(|| run_grnboost2(&expr_vec, n_cells, n_genes, &regulators, &params));
    Ok(names_for(&edges, &gene_names))
}

/// AUCell: score each regulon (gene signature) in each cell.
/// Returns (n_cells * n_regulons) row-major scores plus the regulon order.
#[pyfunction]
#[pyo3(name = "aucell", signature = (expr, gene_names, regulon_names, regulon_genes, auc_max_rank=None))]
fn aucell_py(
    py: Python<'_>,
    expr: PyReadonlyArray2<f32>,
    gene_names: Vec<String>,
    regulon_names: Vec<String>,
    regulon_genes: Vec<Vec<String>>,
    auc_max_rank: Option<usize>,
) -> PyResult<(Vec<String>, Vec<f32>, usize, usize)> {
    let arr = expr.as_array();
    let (n_cells, n_genes) = (arr.shape()[0], arr.shape()[1]);
    let idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let regulons: Vec<Vec<usize>> = regulon_genes
        .iter()
        .map(|genes| genes.iter().filter_map(|g| idx.get(g.as_str()).copied()).collect())
        .collect();
    let amr = auc_max_rank.unwrap_or_else(|| ((0.05 * n_genes as f64).ceil() as usize).max(1));
    let expr_vec: Vec<f32> = arr.iter().copied().collect();
    let flat = py.allow_threads(|| run_aucell(&expr_vec, n_cells, n_genes, &regulons, amr));
    Ok((regulon_names, flat, n_cells, regulons.len()))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(genie3_py, m)?)?;
    m.add_function(wrap_pyfunction!(grnboost2_py, m)?)?;
    m.add_function(wrap_pyfunction!(aucell_py, m)?)?;
    Ok(())
}
