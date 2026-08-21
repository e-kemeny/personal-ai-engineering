import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

results = [
    [0, 0.01065340917557478, 1.1226169821939227],
    [0.0001, 0.01278409082442522, 1.1088346402792577],
    [0.0005, 0.03267045319080353, 1.0706996370843564],
    [0.001, 0.09588067978620529, 1.0559896584082362],
    [0.002, 0.3089488744735718, 1.0985557858992754],
    [0.003, 0.515625, 1.1869544624768456],
    [0.005, 0.8217329382896423, 1.3384032108660402],
    [0.01, 0.9978693127632141, 1.3863112513361306]
]

l1_weights = []
sparsities = []
prediction_losses = []

for result in results:
    l1_weights.append(result[0])
    sparsities.append(result[1])
    prediction_losses.append(result[2])

plt.figure(figsize=(10, 6))

plt.plot(
    sparsities,
    prediction_losses,
    linewidth=2,
    zorder=1
)

scatter = plt.scatter(
    sparsities,
    prediction_losses,
    c=range(len(l1_weights)),
    cmap="viridis",
    s=80,
    zorder=2
)

handles, _ = scatter.legend_elements(num=len(l1_weights))

legend_labels = [f"λ = {weight}" for weight in l1_weights]

plt.legend(
    handles,
    legend_labels,
    title="L1 Weight (λ)",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)

plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))

plt.xlabel("Embedding Sparsity")
plt.ylabel("Prediction Loss")
plt.title("Word2Vec: Sparsity vs Prediction Loss")

plt.grid(alpha=0.25)
plt.tight_layout()

plt.show()