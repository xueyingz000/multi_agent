def extract_curve_endpoints(points):
    """找出一组点中最可能的端点（距离最远的两点）"""
    max_dist = 0
    endpoints = (0, 0)
    
    # 找出距离最远的两个点
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                if dist > max_dist:
                    max_dist = dist
                    endpoints = (i, j)
    
    return endpoints[0], endpoints[1]

def sort_points_by_path(points):
    """根据路径长度对点进行排序"""
    if len(points) <= 2:
        return points
        
    # 找出可能的端点
    start_idx, end_idx = extract_curve_endpoints(points)
    
    # 使用Dijkstra算法找出从起点到终点的最短路径
    # 简化版本：使用贪心算法，但强制第一个点为识别的起点
    ordered = [points[start_idx]]
    remaining = points.copy()
    remaining.pop(start_idx)
    
    # 贪心排序，但增加方向约束
    while remaining and len(remaining) > 1:
        last = ordered[-1]
        if len(ordered) >= 2:
            prev = ordered[-2]
            # 计算当前方向向量
            dir_x = last[0] - prev[0]
            dir_y = last[1] - prev[1]
            dir_len = (dir_x**2 + dir_y**2)**0.5
            if dir_len > 0:
                dir_x /= dir_len
                dir_y /= dir_len
                
            # 寻找最接近当前方向的点
            best_score = -float('inf')
            best_idx = 0
            
            for i, point in enumerate(remaining):
                # 计算距离分量
                dist_x = point[0] - last[0]
                dist_y = point[1] - last[1]
                dist = (dist_x**2 + dist_y**2)**0.5
                
                # 如果距离太小，可能是重复点
                if dist < 0.001:
                    best_idx = i
                    break
                    
                # 计算方向余弦（点乘）
                cos_angle = (dir_x*dist_x + dir_y*dist_y) / dist
                
                # 结合距离和方向的评分
                score = cos_angle - dist * 0.1  # 优先考虑方向，其次考虑距离
                
                if score > best_score:
                    best_score = score
                    best_idx = i
        else:
            # 第一个点，仅考虑距离
            closest_idx = 0
            closest_dist = float('inf')
            
            for i, point in enumerate(remaining):
                dist = (point[0] - last[0])**2 + (point[1] - last[1])**2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
                    
            best_idx = closest_idx
            
        ordered.append(remaining[best_idx])
        remaining.pop(best_idx)
    
    # 添加最后一个点
    if remaining:
        ordered.append(remaining[0])
        
    return ordered 