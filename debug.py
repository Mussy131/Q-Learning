import matplotlib.pyplot as plt
from utils import loadPTSample
from QlearningPathPlanner import QLearningPlanner, crop_map_around_start_goal
from visualize import plotMapWithPath, plot_q_value_heatmap

# 使用一张样本图进行调试
data_path = "data/train/0000b83e-6ff4-465d-b019-a244628430b3.pt"  # 替换为你的实际路径
mapArray, start, goal = loadPTSample(data_path)
    
# 裁剪地图
tile_map, start_crop, goal_crop = crop_map_around_start_goal(mapArray, start, goal, margin=20)

# 训练
planner = QLearningPlanner(tile_map, episodes=3000, maxSteps=500)
planner.train(start_crop, goal_crop)

# 路径提取
path = planner.extractPath(start_crop, goal_crop)

# 可视化
if path:
    print(f"✅ 成功找到路径，长度为 {len(path)}")
    fig1 = plotMapWithPath(tile_map, start_crop, goal_crop, path)
    plt.show()

    fig2 = plot_q_value_heatmap(planner.q_table, tile_map)
    plt.show()
else:
    print("❌ 未能找到有效路径")
    fig1 = plotMapWithPath(tile_map, start_crop, goal_crop, path)
    plt.show

    fig2 = plot_q_value_heatmap(planner.q_table, tile_map)
    plt.show()

