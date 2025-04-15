from classicalPlanners import astar
from visualizeComparison import plot_path_comparison, plot_length_histogram
from utils import loadPTSample, isValid
from QlearningPathPlanner import crop_map_around_start_goal
from ablationStudy import reward_fn_no_shaping as custom_reward

def extended_evaluation(map_list, save_dir, planner_class, reward_mode="default"):
    from classicalPlanners import astar, astar_with_q_heuristic
    from visualizeComparison import plot_path_comparison_3way, plot_3way_length_histogram

    q_lengths, astar_lengths, fusion_lengths = [], [], []

    for i, path in enumerate(map_list):
        mapArray, start, goal = loadPTSample(path)
        if not isValid(start, mapArray) or not isValid(goal, mapArray):
            continue

        tile_map, start_crop, goal_crop = crop_map_around_start_goal(mapArray, start, goal)
        astar_path = astar(tile_map, start_crop, goal_crop)
        if astar_path:
            astar_lengths.append(len(astar_path))

        if reward_mode == "ablation":
            from ablationStudy import reward_fn_no_shaping as custom_reward
            planner = planner_class(tile_map, episodes=5000, maxSteps=300)
            planner.train(start_crop, goal_crop, custom_reward)
        else:
            planner = planner_class(tile_map, episodes=5000, maxSteps=300)
            planner.train(start_crop, goal_crop)

        q_path = planner.extractPath(start_crop, goal_crop)
        if q_path:
            q_lengths.append(len(q_path))

        fusion_path = astar_with_q_heuristic(tile_map, start_crop, goal_crop, planner.q_table)
        if fusion_path:
            fusion_lengths.append(len(fusion_path))

        plot_path_comparison_3way(tile_map, start_crop, goal_crop, q_path, astar_path, fusion_path,
            filename=f"{save_dir}/compare3_{i:03d}.png")

    plot_3way_length_histogram(q_lengths, astar_lengths, fusion_lengths,
        filename=f"{save_dir}/length_3way.png")