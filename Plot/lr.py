import matplotlib.pyplot as plt


def plot_lr_comparison(
    lr_lists, labels, title="Learning Rate Comparison for 10,000 Samples"
):
    """
    Plots learning rate curves for different training methods.
    """
    markers = ["o", "s", "D", "^", "v", "*"]  # Extend as needed
    plt.figure(figsize=(10, 5))

    for i, lr_list in enumerate(lr_lists):
        plt.plot(
            range(1, len(lr_list) + 1),
            lr_list,
            marker=markers[i % len(markers)],
            label=labels[i],
        )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.grid(True)
    plt.xticks(range(1, len(lr_lists[0]) + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()
