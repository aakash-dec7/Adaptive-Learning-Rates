import matplotlib.pyplot as plt


def plot_loss_comparison(loss_lists, labels, title="Training Loss Comparison"):
    """
    Plots loss curves for different training methods.
    """
    markers = ["o", "s", "D", "^", "v", "*"]  # Extend if needed
    plt.figure(figsize=(10, 5))

    for i, loss_list in enumerate(loss_lists):
        plt.plot(
            range(1, len(loss_list) + 1),
            loss_list,
            marker=markers[i % len(markers)],
            label=labels[i],
        )

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(title)
    plt.grid(True)
    plt.xticks(range(1, len(loss_lists[0]) + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()
