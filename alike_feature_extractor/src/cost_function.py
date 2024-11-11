# import numpy as np

# def select_optimal_features(keypoints, scores, num_features=80, min_distance=100 ):
#     """
#     根据分数和距离优化选择的 cost_function。
#     - keypoints: 提取的所有特征点坐标，形状为 (N, 2)。
#     - scores: 每个特征点的分数，形状为 (N,)。
#     - num_features: 选择的特征点数目默认为80。
#     - min_distance: 特征点之间的最小距离约束。

#     返回：
#     - selected_keypoints: 优化后的特征点坐标。
#     - selected_scores: 优化后的特征点分数。
#     """
#     # 1. 按分数从高到低排序
#     sorted_indices = np.argsort(-scores)
#     sorted_keypoints = keypoints[sorted_indices]
#     sorted_scores = scores[sorted_indices]

#     # 2. 初始化存储选择的特征点
#     selected_keypoints = []
#     selected_scores = []

#     # 3. 通过距离约束逐步添加特征点
#     for i, pt in enumerate(sorted_keypoints):
#         # 检查 selected_keypoints 是否为空
#         if not selected_keypoints:
#             # 第一个点直接添加，不用考虑距离
#             selected_keypoints.append(pt)
#             selected_scores.append(sorted_scores[i])
#             continue
        
#         # 检查新加入的特征点是否满足距离约束
#         if all(np.linalg.norm(pt - np.array(selected_keypoints), axis=1) > min_distance):
#             selected_keypoints.append(pt)
#             selected_scores.append(sorted_scores[i])
#             if len(selected_keypoints) >= num_features:  # 达到目标数量
#                 break

#     return np.array(selected_keypoints), np.array(selected_scores)


import numpy as np

def select_optimal_features(keypoints, scores, num_features=150, min_distance=10):
    """
    基于分数和距离优化选择的 cost_function。
    - keypoints: 提取的所有特征点坐标，形状为 (N, 2)。
    - scores: 每个特征点的分数，形状为 (N,)。
    - num_features: 选择的特征点数目，默认为80。
    - min_distance: 特征点之间的最小距离约束。

    返回：
    - selected_keypoints: 优化后的特征点坐标。
    - selected_scores: 优化后的特征点分数。
    """
    # 1. 按分数从高到低排序
    sorted_indices = np.argsort(-scores)
    sorted_keypoints = keypoints[sorted_indices]
    sorted_scores = scores[sorted_indices]

    # 2. 初始化存储选择的特征点
    selected_keypoints = []
    selected_scores = []

    # 3. 根据距离和分数的综合约束添加特征点
    for i, pt in enumerate(sorted_keypoints):
        # 第一个点直接添加，不用考虑距离
        if not selected_keypoints:
            selected_keypoints.append(pt)
            selected_scores.append(sorted_scores[i])
            continue
        
        # 计算当前特征点与已选特征点的距离，确保最小距离约束
        distances = np.linalg.norm(pt - np.array(selected_keypoints), axis=1)
        
        # 只选择满足最小距离约束的特征点
        if all(distances > min_distance):
            selected_keypoints.append(pt)
            selected_scores.append(sorted_scores[i])
            if len(selected_keypoints) >= num_features:  # 达到目标数量则停止
                break

    return np.array(selected_keypoints), np.array(selected_scores)
