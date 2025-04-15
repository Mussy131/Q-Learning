import os
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from utils import loadPTSample, isValid
from QlearningPathPlanner import QLearningPlanner, crop_map_around_start_goal
from visualize import plotMapWithPath, plot_q_value_heatmap, save_path_animation, plot_kfold_success_bar
from extendedMainPatch import extended_evaluation

# === Parameters ===
TEST_FOLDER = "../data/test"
SAVE_FOLDER = "test2"
COMPARE_FOLDER = "comparison_results_ablation"
NUM_SAMPLES = 10
EPISODES = 5000
MAX_STEPS = 500
MARGIN = 20
K_Fold = 5

# === K-Fold Evaluation ===
def evaluate_with_kfold(pt_files, k=5, test_size=100, episodes=5000, max_steps=300):
    print(f"\nEvaluating success rate with {k}-Fold cross-validation...")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    all_pool = pt_files[:]
    random.shuffle(all_pool)

    def get_valid_file(pool, used_set):
        while pool:
            f = pool.pop()
            if f in used_set:
                continue
            try:
                mapArray, start, goal = loadPTSample(f)
                if isValid(start, mapArray) and isValid(goal, mapArray):
                    return f
            except:
                continue
        return None

    for fold_idx, (_, test_idx) in enumerate(kf.split(pt_files)):
        print(f"\n[Fold {fold_idx+1}/{k}]")
        test_files = [pt_files[i] for i in test_idx]
        random.shuffle(test_files)

        used = set()
        valid_files = []
        pool = all_pool.copy()

        for f in test_files:
            try:
                mapArray, start, goal = loadPTSample(f)
                if isValid(start, mapArray) and isValid(goal, mapArray):
                    valid_files.append(f)
                    used.add(f)
                else:
                    alt = get_valid_file(pool, used)
                    if alt:
                        valid_files.append(alt)
                        used.add(alt)
            except:
                alt = get_valid_file(pool, used)
                if alt:
                    valid_files.append(alt)
                    used.add(alt)

            if len(valid_files) >= test_size:
                break

        print(f"[Fold {fold_idx+1}] Final test set size: {len(valid_files)}")

        success_count = 0
        for path in tqdm(valid_files, desc=f"Fold {fold_idx+1}", ncols=80):
            try:
                mapArray, start, goal = loadPTSample(path)
                map_crop, start_crop, goal_crop = crop_map_around_start_goal(mapArray, start, goal)
                planner = QLearningPlanner(map_crop, episodes=episodes, maxSteps=max_steps)
                planner.train(start_crop, goal_crop)
                result_path = planner.extractPath(start_crop, goal_crop)
                if result_path:
                    success_count += 1
            except:
                continue

        success_rate = success_count / len(valid_files) if valid_files else 0.0
        print(f"[Fold {fold_idx+1}] Success rate: {success_rate:.2%}")
        fold_results.append(success_rate)
        plot_kfold_success_bar(fold_results)

    print("\n=== K-Fold Evaluation Summary ===")
    for i, r in enumerate(fold_results):
        print(f"Fold {i+1}: {r:.2%}")
    print(f"Average success rate: {np.mean(fold_results):.2%}")

# === Main Execution ===

# Create save directory if not exists
# os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(COMPARE_FOLDER, exist_ok=True)

# Get all test files
print(f"Loading test files from {TEST_FOLDER}...")
all_files = glob.glob(os.path.join(TEST_FOLDER, "*.pt"))
original_file_list = all_files.copy()
random.shuffle(all_files)

# plot_id = 1
# valid_samples = 0

# print(f"Testing {NUM_SAMPLES} maps...")

# while valid_samples < NUM_SAMPLES:
#     if not all_files:
#         print("No more files to load!")
#         break

#     file_path = all_files.pop()
#     try:
#         mapArray, start, goal = loadPTSample(file_path)
#     except Exception as e:
#         print(f"[Error] Failed to load file: {file_path} - {str(e)}")
#         continue

#     filename = os.path.splitext(os.path.basename(file_path))[0]

#     if not (isValid(start, mapArray) and isValid(goal, mapArray)):
#         print(f"[Warning] Start or goal is in obstacle: {filename}")
#         fig = plotMapWithPath(mapArray, start, goal, path=None)
#         plt.show()
#         plt.close(fig)
#         continue  # 跳过并自动roll下一张

#     tile_map, start_crop, goal_crop = crop_map_around_start_goal(mapArray, start, goal, margin=MARGIN)

#     planner = QLearningPlanner(tile_map, episodes=EPISODES, maxSteps=MAX_STEPS)
#     planner.train(start_crop, goal_crop)
#     path = planner.extractPath(start_crop, goal_crop)

#     # Path Planning Result
#     if path:
#         print(f"[Info] Path found. Length: {len(path)}")
#     else:
#         print(f"[Info] No path found.")

#     # Save visualizations
#     id_str = f"{plot_id:03d}"
#     fig1 = plotMapWithPath(tile_map, start_crop, goal_crop, path)
#     fig1.savefig(os.path.join(SAVE_FOLDER, f"{id_str}_path.png"))
#     plt.close(fig1)

#     fig2 = plot_q_value_heatmap(planner.q_table, tile_map, title="Q-table Heatmap")
#     fig2.savefig(os.path.join(SAVE_FOLDER, f"{id_str}_heatmap.png"))
#     plt.close(fig2)

#     plt.plot(planner.reward_history)
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.title(f"Reward over Episodes: {filename}")
#     plt.grid(True)
#     plt.savefig(os.path.join(SAVE_FOLDER, f"{id_str}_reward.png"))
#     plt.close()

#     if path:
#         save_path_animation(tile_map, start_crop, goal_crop, path, save_path=os.path.join(SAVE_FOLDER, f"{id_str}_path.gif"))

#     plot_id += 1
#     valid_samples += 1

# === K-Fold Evaluation ===
# evaluate_with_kfold(original_file_list, k=K_Fold)

# === Path Comparison Evaluation ===
print("\nRunning Q-Learning vs A* Comparison...")

# Reward Shaping
extended_evaluation(original_file_list[:30], COMPARE_FOLDER, QLearningPlanner, reward_mode="default")

# Ablation Study(No Reward Shaping)
# extended_evaluation(original_file_list[:30], COMPARE_FOLDER, QLearningPlanner, reward_mode="ablation")