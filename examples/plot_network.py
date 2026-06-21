"""Example: draw a scenic-rs regulatory network.

NOTE: networkx + matplotlib are NOT scenic-rs dependencies. This is standalone
example code — copy `plot_network` into your own analysis and adapt it. The
package itself stays numpy + pandas only.

    pip install networkx matplotlib    # only if you want this example

Run:  python plot_network.py
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_network(adj, top_n=200, focus_tf=None, activity=None, seed=0,
                 figsize=(9, 9), label_targets=None, out=None, ax=None):
    """Draw a TF -> target network from a scenic-rs adjacency DataFrame
    (columns: TF, target, importance).

    top_n      : keep the strongest N edges.
    focus_tf   : str or list of TFs to restrict to (single-regulon view).
    activity   : optional {gene: score} (e.g. mean AUCell) to color nodes.
    """
    df = adj[["TF", "target", "importance"]].copy()
    if focus_tf is not None:
        focus = {focus_tf} if isinstance(focus_tf, str) else set(focus_tf)
        df = df[df["TF"].isin(focus)]
    df = df.sort_values("importance", ascending=False).head(top_n)
    if df.empty:
        raise ValueError("no edges to plot after filtering")

    G = nx.from_pandas_edgelist(df, "TF", "target", edge_attr="importance",
                                create_using=nx.DiGraph)
    tf_set = set(df["TF"])
    out_deg = dict(G.out_degree())
    nodes = list(G.nodes())
    pos = nx.spring_layout(G, seed=seed, weight="importance",
                           k=1.5 / max(np.sqrt(len(nodes)), 1.0))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    w = np.array([d["importance"] for *_, d in G.edges(data=True)], dtype=float)
    wn = w / w.max() if w.size and w.max() > 0 else w
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.3 + 1.7 * wn, alpha=0.25,
                           edge_color="#888888", arrows=False)

    sizes = [(350 + 120 * out_deg.get(n, 0)) if n in tf_set else 70 for n in nodes]
    if activity is not None:
        vals = np.array([float(activity.get(n, np.nan)) for n in nodes])
        nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes,
                                    node_color=vals, cmap="viridis", ax=ax,
                                    linewidths=0.3, edgecolors="white")
        plt.colorbar(nc, ax=ax, fraction=0.03, pad=0.01, label="activity")
    else:
        colors = ["#d62728" if n in tf_set else "#9ecae1" for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes,
                               node_color=colors, ax=ax, linewidths=0.3,
                               edgecolors="white")

    show_targets = label_targets if label_targets is not None else (len(nodes) <= 40)
    labels = {n: n for n in nodes if (n in tf_set) or show_targets}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"{len(tf_set)} regulators · {len(nodes)} nodes · {G.number_of_edges()} edges")
    if out:
        ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    return ax


if __name__ == "__main__":
    import os
    import scenic_rs

    here = os.path.dirname(__file__)
    z = np.load(os.path.join(here, "..", "data", "pbmc3k_prep_500_300_0.npz"), allow_pickle=True)
    X, genes, tfs = z["X"].astype("float32"), list(z["genes"]), list(z["tfs"])

    adj = scenic_rs.genie3(X, genes, tfs, n_estimators=100)
    plot_network(adj, top_n=120, out=os.path.join(here, "network_overview.png"))
    top_tf = adj.iloc[0]["TF"]
    plot_network(adj, focus_tf=top_tf, top_n=30,
                 out=os.path.join(here, f"network_{top_tf}.png"))
    print(f"wrote network_overview.png and network_{top_tf}.png (focus TF = {top_tf})")
