//! cisTarget ranking-database reader (feather / Arrow IPC).
//!
//! The gene-based v10 DBs are `motifs (rows) x genes (cols)` int16 rank matrices
//! with a trailing `motifs` string column. We load the whole DB into a gene-major
//! contiguous buffer once (≈317 MB for hg38), like ctxcore's MemoryDecorator, so
//! per-module enrichment is a cheap gather + the math in `ctx.rs`.

use std::collections::HashMap;
use std::fs::File;

use arrow::array::{Array, Int16Array, StringArray};
use arrow::ipc::reader::FileReader;

pub struct RankingDb {
    pub name: String,
    gene_index: HashMap<String, usize>, // gene name -> slot in `data`
    pub motif_names: Vec<String>,
    data: Vec<i16>, // gene-major: data[slot * n_motifs + motif]
    pub n_motifs: usize,
    pub n_genes: usize,
    pub total_genes: usize, // == n_genes (ctxcore nbr_total_region_or_gene_ids)
}

impl RankingDb {
    pub fn open(path: &str, name: &str) -> Result<RankingDb, String> {
        let f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let reader = FileReader::try_new(f, None).map_err(|e| e.to_string())?;
        let schema = reader.schema();

        // Separate the trailing "motifs" string column from the gene rank columns.
        let mut gene_cols: Vec<usize> = Vec::new();
        let mut gene_names: Vec<String> = Vec::new();
        let mut motif_col: Option<usize> = None;
        for (i, fld) in schema.fields().iter().enumerate() {
            if fld.name() == "motifs" {
                motif_col = Some(i);
            } else {
                gene_cols.push(i);
                gene_names.push(fld.name().to_string());
            }
        }
        let motif_col = motif_col.ok_or("no 'motifs' column in DB")?;
        let n_genes = gene_cols.len();

        let mut batches = Vec::new();
        for b in reader {
            batches.push(b.map_err(|e| e.to_string())?);
        }
        let n_motifs: usize = batches.iter().map(|b| b.num_rows()).sum();

        // Gene-major fill: each gene's column is contiguous across all motifs.
        let mut data = vec![0i16; n_genes * n_motifs];
        for (slot, &col) in gene_cols.iter().enumerate() {
            let base = slot * n_motifs;
            let mut row_off = 0usize;
            for b in &batches {
                let arr = b
                    .column(col)
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .ok_or_else(|| format!("gene column {col} is not Int16"))?;
                let dst = &mut data[base + row_off..base + row_off + arr.len()];
                for (r, d) in dst.iter_mut().enumerate() {
                    *d = arr.value(r);
                }
                row_off += b.num_rows();
            }
        }

        let mut motif_names = Vec::with_capacity(n_motifs);
        for b in &batches {
            let arr = b
                .column(motif_col)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("'motifs' column is not Utf8")?;
            for r in 0..arr.len() {
                motif_names.push(arr.value(r).to_string());
            }
        }

        let gene_index = gene_names
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        Ok(RankingDb {
            name: name.to_string(),
            gene_index,
            motif_names,
            data,
            n_motifs,
            n_genes,
            total_genes: n_genes,
        })
    }

    /// The DB's gene names, in column-slot order.
    pub fn gene_list(&self) -> Vec<String> {
        let mut v = vec![String::new(); self.n_genes];
        for (name, &slot) in &self.gene_index {
            v[slot] = name.clone();
        }
        v
    }

    /// Gather rankings for the genes present in the DB, in the given input order
    /// (missing genes dropped). Returns (present_genes, rankings_motif_major i32),
    /// where rankings[m * n_present + j] is the rank of present gene j in motif m.
    pub fn load_module(&self, genes: &[String]) -> (Vec<String>, Vec<i32>) {
        let present: Vec<(usize, String)> = genes
            .iter()
            .filter_map(|g| self.gene_index.get(g).map(|&c| (c, g.clone())))
            .collect();
        let ng = present.len();
        let mut out = vec![0i32; self.n_motifs * ng];
        for (j, (col, _)) in present.iter().enumerate() {
            let base = col * self.n_motifs;
            for m in 0..self.n_motifs {
                out[m * ng + j] = self.data[base + m] as i32;
            }
        }
        let present_genes = present.into_iter().map(|(_, g)| g).collect();
        (present_genes, out)
    }
}
