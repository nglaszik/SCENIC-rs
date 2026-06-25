"""Generate the tiny synthetic cisTarget fixtures used by test_ctx_smoke.

Run manually when the fixture needs regenerating (needs pyarrow, which the CI
*test* envs deliberately do NOT install -- the Rust arrow reader loads the
committed .feather at test time with no pyarrow dependency):

    python tests/_make_ctx_fixture.py

It writes two small files into data/:
  - ctx_mini.feather       a ranking DB (motifs x genes int16 ranks + "motifs" col)
  - ctx_mini_motifs.tbl    motif->TF annotations

The DB is constructed so exactly ONE motif (motif_TF1) recovers TF1's module
near the top of the ranking while the rest do not, giving that motif an
NES well above the 3.0 threshold -> ctx returns a TF1(+) regulon.
See test_ctx_smoke.py / make_ctx_inputs() for the matching gene names.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- shared naming convention (must match test_ctx_smoke.make_ctx_inputs) ---
TF = "TF1"
N_TARGETS = 25
N_FILLERS = 594  # -> total_genes = 1 + 25 + 594 = 620, so default auc_threshold
#                  (0.05) gives rank_cutoff = round(0.05*620)-1 = 30, wide enough
#                  to contain all 26 module genes ranked at the top.
N_MOTIFS = 20  # one enriched motif vs 19 flat -> NES ~ sqrt(19) ~ 4.4 >> 3.0

TARGETS = [f"t{i:02d}" for i in range(N_TARGETS)]
FILLERS = [f"f{i:03d}" for i in range(N_FILLERS)]
MODULE = [TF] + TARGETS  # 26 genes recovered by the enriched motif
DB_GENES = MODULE + FILLERS  # column order in the DB
TOTAL = len(DB_GENES)


def build():
    gene_pos = {g: j for j, g in enumerate(DB_GENES)}
    module_cols = [gene_pos[g] for g in MODULE]
    filler_cols = [gene_pos[g] for g in FILLERS]

    ranks = np.zeros((N_MOTIFS, TOTAL), dtype=np.int16)
    for m in range(N_MOTIFS):
        if m == 0:
            # Enriched motif: module genes take the top ranks 0..25, fillers after.
            order = module_cols + filler_cols
        else:
            # Flat motifs: module genes pushed well past the rank cutoff (>=30),
            # so they contribute ~0 to the recovery AUC. Fillers fill the top.
            order = filler_cols[:30] + module_cols + filler_cols[30:]
        for rank, col in enumerate(order):
            ranks[m, col] = rank

    motif_names = ["motif_TF1" if m == 0 else f"motif_flat{m:02d}" for m in range(N_MOTIFS)]
    # Build every column at once (avoids fragmented-frame perf warnings).
    data = {g: ranks[:, j] for j, g in enumerate(DB_GENES)}
    data["motifs"] = motif_names
    df = pd.DataFrame(data)

    out_dir = Path(__file__).resolve().parents[1] / "data"
    # zstd-compressed Arrow IPC -- the Rust reader has ipc_compression enabled.
    df.to_feather(out_dir / "ctx_mini.feather", compression="zstd")

    # Annotate only the enriched motif to TF1 (qval passes default fdr 0.001).
    tbl = out_dir / "ctx_mini_motifs.tbl"
    with open(tbl, "w") as fh:
        fh.write("#motif_id\tgene_name\tmotif_similarity_qvalue\torthologous_identity\tdescription\n")
        fh.write(f"motif_TF1\t{TF}\t0.0\t1.0\tsynthetic direct annotation\n")

    print(f"wrote {out_dir/'ctx_mini.feather'} ({N_MOTIFS} motifs x {TOTAL} genes)")
    print(f"wrote {tbl}")


if __name__ == "__main__":
    build()
