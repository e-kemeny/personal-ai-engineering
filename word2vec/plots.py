import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

results = [
    [0, 0.008616909384727478, 0.05543749465743722],
    [20.0, 0.2543923556804657, 0.03247252293697082],
    [50.0, 0.5737594962120056, 0.000620743845562995]
]

l1_weights = []
sparsities = []
spearman_scores = []

for result in results:
    l1_weights.append(result[0])
    sparsities.append(result[1])
    spearman_scores.append(result[2])

plt.figure(figsize=(10, 6))

plt.plot(
    sparsities,
    spearman_scores,
    linewidth=2,
    zorder=1
)

scatter = plt.scatter(
    sparsities,
    spearman_scores,
    c=range(len(l1_weights)),
    cmap="viridis",
    s=90,
    zorder=2
)

handles, _ = scatter.legend_elements(num=len(l1_weights))

legend_labels = [f"λ = {weight}" for weight in l1_weights]

plt.legend(
    handles,
    legend_labels,
    title="L1 Weight",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)

plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))

plt.xlabel("Embedding Sparsity")
plt.ylabel("WordSim-353 Spearman Correlation")
plt.title("Sparse Word2Vec: Sparsity vs Semantic Quality")

plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    "sparsity_vs_semantic_quality.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()