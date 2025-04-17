import matplotlib.pyplot as plt
from utils import loadPTSample
from QlearningPathPlanner import QLearningPlanner, crop_map_around_start_goal
from visualize import plotMapWithPath, plot_q_value_heatmap

# Use a sample map for testing
data_path = "../data/train/0000b83e-6ff4-465d-b019-a244628430b3.pt"  # Change this to your real path
mapArray, start, goal = loadPTSample(data_path)
    
# 
tile_map, start_crop, goal_crop = crop_map_around_start_goal(mapArray, start, goal, margin=20)

# Train the Q-learning planner
planner = QLearningPlanner(tile_map, episodes=3000, maxSteps=500)
planner.train(start_crop, goal_crop)

# Extract the path using the trained Q-table
path = planner.extractPath(start_crop, goal_crop)

# Visualize the results
if path:
    print(f"✅ Successfully find the path, The length is: {len(path)}")
    fig1 = plotMapWithPath(tile_map, start_crop, goal_crop, path)
    plt.show()

    fig2 = plot_q_value_heatmap(planner.q_table, tile_map)
    plt.show()
else:
    print("❌ Cnnnot find the path")
    fig1 = plotMapWithPath(tile_map, start_crop, goal_crop, path)
    plt.show

    fig2 = plot_q_value_heatmap(planner.q_table, tile_map)
    plt.show()

