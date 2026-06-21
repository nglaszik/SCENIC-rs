# `ctx` (cisTarget) — Rust port spec

Goal: reimplement pySCENIC's `ctx` step (modules → cisTarget motif enrichment →
pruning → regulons) in Rust, no Dask/Python. Validation bar: per-(module,motif)
**NES numeric parity** vs ctxcore 0.2.0 / pyscenic 0.12.1.

Reference source (read-only, the source of truth for parity):
`~/venvs/pyscenic_clean/lib/python3.10/site-packages/{ctxcore,pyscenic}`

## Pipeline
GRN adjacencies (TF,target,importance) + expr matrix
  → `modules_from_adjacencies` → candidate modules (Regulons)
  → for each (module, db): `module2features_auc1st_impl` → enriched+annotated motifs
  → `df2regulons` → final regulons (TF(+)/(-) → target genes + weights)

## 1. Module generation  (pyscenic/utils.py:267 `modules_from_adjacencies`)
Defaults: thresholds=(0.75,0.90) [PERCENTILES of importance], top_n_targets=(50,),
top_n_regulators=(5,10,50), min_genes=20, rho_dichotomize=True,
keep_only_activating=True, rho_threshold=0.03, rho_mask_dropouts=False.

Correlation (utils.py:108 `add_correlation`): if mask_dropouts=False (default) →
`np.corrcoef` Pearson over ALL cells between TF and target. regulation =
(rho>0.03) - (rho<-0.03) ∈ {-1,0,1}. keep_only_activating → keep rows regulation>0,
add "activating" to context.

Three module methods, each yields Regulon(tf, gene2weight=[(target,importance)], context):
- modules4thr: per percentile p in {0.75,0.90}: thr=importance.quantile(p); per TF,
  targets with importance>thr; context "weight>{p*100}%".
- modules4top_targets: per n in {50}: per TF, nlargest(n, importance); context "top{n}".
- modules4top_factors: per n in {5,10,50}: per TARGET, nlargest(n, importance), regroup
  by TF; context "top{n}perTarget".
Then add TF to its own module with weight 1.0 (genesig.add). Keep modules with
len>=min_genes(20). (len counts target genes + the added TF.)

## 2. cisTarget enrichment  (DEFAULT = pyscenic/transform.py:152 `module2features_auc1st_impl`,
   partial at :267 with rank_threshold=1500, auc_threshold=0.05, nes_threshold=3.0,
   filter_for_annotation=True, weighted_recovery=False)
NOTE: CLI `--rank_threshold` default 5000 but is threaded via prune.py — CONFIRM how
it reaches module2features (the partial default is 1500). rank_threshold only affects
the rccs/leading-edge length, NOT AUC/NES.

For a module's genes present in DB (db.load(gs) = intersection, any column order):
- weights = ones(len(genes))  (weighted_recovery=False).
- AUC fast path (ctxcore/recovery.py:364 `aucs`):
    total_genes = db.total_genes (CisTargetDatabase.nbr_total_region_or_gene_ids).
    rank_cutoff = round(auc_threshold*total_genes) - 1   (derive_rank_cutoff, rank_threshold defaults total_genes-1 here)
    maxauc = (rank_cutoff+1) * weights.sum()
    per motif row: weighted_auc1d → take genes with rank<rank_cutoff, sort by rank,
      y=cumsum(weights[sorted]); x=[sorted ranks, rank_cutoff]; auc = sum(diff(x)*y)/maxauc.
- NES = (auc - auc.mean()) / auc.std()   over ALL motifs in db (numpy std, ddof=0).
- enriched = NES >= nes_threshold(3.0).
- annotate: left-join motif_annotations on motif_id; sort by [similarity_qvalue desc,
  orthologous_identity asc], drop dup motif index keep="last"; filter_for_annotation →
  keep rows with non-null annotation.
- leading edge: recovery(no_auc=True) → rccs (n_motifs x rank_threshold) via
  rcc2d (bincount(ranking,weights)[:rank_threshold] then cumsum, with a sentinel gene
  at rank=total_genes appended). avgrcc=rccs.mean(0); avg2stdrcc=avgrcc+2*rccs.std(0).
  Per enriched+annotated motif: rank_at_max = argmax(rcc - avg2stdrcc); TargetGenes =
  genes with rank <= rank_at_max, paired with their MODULE importance weight (module[gene]),
  sorted. (recovery.py:221 leading_edge.)

Skip module if <80% of its genes map to db (frac_missing>=0.20).

## 3. Regulon assembly  (transform.py:382 `_regulon4group`, :494 `df2regulons`)
Group enriched-annotated rows by (TF, type) where type="(-)" if "repressing" in context
else "(+)". Per group: one Regulon per row with gene2weight=row.TargetGenes; then
Regulon.union across rows = union of target genes keeping MAX weight per gene. Regulon
score = NES * correction (correction = orthologous_identity * min(-log10(qval),10)/10;
direct annotation → 1.0). Name = "{TF}(+)" / "{TF}(-)".

## 4. DB format (verified)
`*.genes_vs_motifs.rankings.feather`: rows=motifs (5876), cols=genes (27015) + final
`motifs` string column (the motif id per row). cell dtype int16 = 0-based rank of that
gene for that motif. total_genes = nbr_total_region_or_gene_ids (CONFIRM: stored value;
likely 27015). Two DBs used (500bp,10kbp); ctx runs each and concatenates enriched dfs
before df2regulons (CONFIRM in prune.py). motif2tf .tbl → motif_annotations via
load_motif_annotations (filters: min_orthologous_identity=0.0, max_similarity_fdr=0.001).

## TODO confirms before coding parity
- prune.py: how rank_threshold/auc/nes flow to module2features; how multiple DBs combine.
- load_motif_annotations exact filtering + the annotation column.
- ctdb.py: nbr_total_region_or_gene_ids value vs n_genes columns.

## pySCENIC 0.12.1 leading-edge bug (we deliberately do NOT replicate)
`pyscenic/transform.py` (`module2features_auc1st_impl` + `module2df`) sorts enriched
motifs by `[motif_similarity_qvalue desc, orthologous_identity asc]` and dedups, which
reorders rows — but `rccs`/`rankings` are boolean-indexed in the ORIGINAL motif order and
`module2df` re-stitches them under the sorted index. Result: each motif's leading-edge
TargetGenes are taken from the WRONG motif's recovery curve + ranks (whenever the sort
reorders, i.e. the common case). Proven via the recovery-curve invariant rcc[t]=0 for
t<min(rank): in a test 8/13 motifs had pyscenic curves nonzero before their own min rank.
ctxcore primitives (recovery/aucs/leading_edge) are correct in isolation; the bug is in
pyscenic's row wiring. scenic-rs aligns rccs/rankings with their motifs correctly, so we
validate vs a corrected ctxcore-primitive reference + exact NES parity, not bug-for-bug.

## Rust stages (each gated on parity vs reference)
1. Feather reader (mmap int16 motifs×genes + gene index + motif names). Dep: arrow2/polars.
2. Enrichment math: weighted_auc1d + NES + leading edge. Unit-test vs ctxcore.recovery on synthetic ranks.
3. Module generation from adjacencies+expr. Test module sets vs modules_from_adjacencies.
4. Annotation + df2regulons union. Test final regulons.
5. pyo3 binding `ctx(...)`, end-to-end + memory benchmark.
