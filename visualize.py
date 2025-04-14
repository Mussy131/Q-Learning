import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plotMapWithPath(mapArray, start, goal, path = None, filename=None): 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mapArray, cmap='gray_r')

    # Start and goal
    ax.scatter(start[1], start[0], color='limegreen', s=100, label="Start", edgecolors='black')
    ax.scatter(goal[1], goal[0], color='red', s=100, label="Goal", edgecolors='black')

    # Path
    if path:
        py, px = zip(*path)
        ax.plot(px, py, color='blue', linewidth=2, label="Q-Learning Path")

    ax.legend()
    ax.set_title("QLB8a Path Planning via Q-Learning")
    ax.axis("off")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_q_value_heatmap(qTable, mapArray, title="Q-table Max Value Heatmap"):
    heatmap = np.full(mapArray.shape, np.nan)

    for state, q_values in qTable.items():
        y, x = state
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
            heatmap[y, x] = np.max(q_values)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Max Q-value")
    plt.axis('off')
    plt.tight_layout()
    return fig 

# Function to save the path animation as a GIF
def save_path_animation(mapArray, start, goal, path, save_path="path.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mapArray, cmap='gray_r')
    ax.scatter(goal[1], goal[0], color='red', s=100, label="Goal", edgecolors='black')
    ax.scatter(start[1], start[0], color='green', s=100, label="Start", edgecolors='black')

    path_line, = ax.plot([], [], color='blue', linewidth=2)

    def update(frame):
        if frame == 0:
            return path_line,
        segment = path[:frame]
        py, px = zip(*segment)
        path_line.set_data(px, py)
        return path_line,

    ani = animation.FuncAnimation(fig, update, frames=len(path)+1, interval=200, blit=True)
    ani.save(save_path, writer='pillow')
    plt.close(fig)

# Function to plot the success rate of K-Fold cross-validation
def plot_kfold_success_bar(fold_results):
    fig, ax = plt.subplots()
    ax.bar([f"Fold {i+1}" for i in range(len(fold_results))], fold_results, color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success Rate")
    ax.set_title("K-Fold Success Rates")
    for i, val in enumerate(fold_results):
        ax.text(i, val + 0.01, f"{val:.2%}", ha='center')
    plt.tight_layout()
    plt.savefig("kfold_success.png")
    plt.close()