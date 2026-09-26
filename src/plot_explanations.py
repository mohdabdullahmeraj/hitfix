
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("gnnexplainer_results.json") as f:
    results = json.load(f)

def plot_explanation(ax, node_a, node_b, top_edges, score, title, shared_max_imp):
    G = nx.Graph()
    G.add_node(node_a, is_target=True)
    G.add_node(node_b, is_target=True)

    max_imp = shared_max_imp
    for src, dst, imp in top_edges:
        G.add_edge(src, dst, weight=imp)

    pos = nx.spring_layout(G, seed=42, k=0.8)

    # draw target nodes (the actual drug pair being predicted) larger and colored
    target_nodes = [node_a, node_b]
    other_nodes = [n for n in G.nodes() if n not in target_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=200,
                            node_color="lightgray", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_size=500,
                            node_color="tab:red", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in target_nodes},
                             font_size=9, font_weight="bold", ax=ax)

    # edge thickness proportional to importance (normalized within this plot)
    edges = G.edges(data=True)
    widths = [1 + 6 * (d["weight"] / max_imp) for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="tab:blue", alpha=0.6, ax=ax)

    ax.set_title(f"{title}\nprediction score: {score:.3f}", fontsize=10)
    ax.axis("off")

for i, r in enumerate(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    shared_max = max(
        max(e[2] for e in r["baseline_top_edges"]),
        max(e[2] for e in r["fixed_top_edges"]),
    )
    plot_explanation(axes[0], r["node_a"], r["node_b"], r["baseline_top_edges"],
                      r["baseline_score"],
                      f"BASELINE (rank {r['baseline_rank']})", shared_max)
    plot_explanation(axes[1], r["node_a"], r["node_b"], r["fixed_top_edges"],
                      r["fixed_score"],
                      f"FIXED MODEL (rank {r['fixed_rank']})", shared_max)

    fig.suptitle(f"Drug pair ({r['node_a']}, {r['node_b']}) — which edges each model relied on",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"explanation_pair_{i+1}_{r['node_a']}_{r['node_b']}.png", dpi=150, bbox_inches="tight")
    print(f"Saved explanation_pair_{i+1}_{r['node_a']}_{r['node_b']}.png")
    plt.close()
