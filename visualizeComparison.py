import matplotlib.pyplot as plt

def plot_path_comparison(map_array, start, goal, path_q, path_astar, filename="comparison.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(map_array, cmap='gray_r')
    ax.scatter(start[1], start[0], color='limegreen', s=100, label="Start", edgecolors='black')
    ax.scatter(goal[1], goal[0], color='red', s=100, label="Goal", edgecolors='black')
    if path_astar:
        py, px = zip(*path_astar)
        ax.plot(px, py, color='orange', linewidth=2, label="A* Path")
    if path_q:
        py, px = zip(*path_q)
        ax.plot(px, py, color='blue', linewidth=2, label="Q-Learning Path")
    ax.legend()
    ax.set_title("Path Comparison: Q-Learning vs A*")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_path_comparison_3way(map_array, start, goal, path_q, path_astar, path_fusion, filename="compare3.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(map_array, cmap='gray_r')
    ax.scatter(start[1], start[0], color='limegreen', s=100, label="Start", edgecolors='black')
    ax.scatter(goal[1], goal[0], color='red', s=100, label="Goal", edgecolors='black')
    if path_astar:
        py, px = zip(*path_astar)
        ax.plot(px, py, color='orange', linewidth=2, label="A*")
    if path_q:
        py, px = zip(*path_q)
        ax.plot(px, py, color='blue', linewidth=2, label="Q-Learning")
    if path_fusion:
        py, px = zip(*path_fusion)
        ax.plot(px, py, color='green', linewidth=2, label="Fusion")
    ax.legend()
    ax.set_title("Q-Learning vs A* vs A*+Q")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_length_histogram(lengths_q, lengths_astar, filename="length_comparison.png"):
    plt.figure(figsize=(6, 4))
    plt.hist(lengths_q, bins=20, alpha=0.6, label="Q-Learning")
    plt.hist(lengths_astar, bins=20, alpha=0.6, label="A*")
    plt.xlabel("Path Length")
    plt.ylabel("Frequency")
    plt.title("Path Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_3way_length_histogram(lengths_q, lengths_astar, lengths_fusion, filename="length_3way.png"):
    plt.figure(figsize=(6, 4))
    plt.hist(lengths_q, bins=20, alpha=0.5, label="Q-Learning")
    plt.hist(lengths_astar, bins=20, alpha=0.5, label="A*")
    plt.hist(lengths_fusion, bins=20, alpha=0.5, label="A*+Q")
    plt.xlabel("Path Length")
    plt.ylabel("Frequency")
    plt.title("Length Distribution: Q vs A* vs A*+Q")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
