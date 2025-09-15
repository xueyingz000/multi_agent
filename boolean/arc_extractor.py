import numpy as np
from scipy import optimize
import math

def fit_circle(points):
    """尝试将点集拟合为圆弧
    
    Args:
        points: 点集[(x1,y1), (x2,y2), ...]
        
    Returns:
        (xc, yc, r): 圆心和半径，如果拟合失败则返回None
    """
    def calc_R(xc, yc):
        """ 计算距离函数 """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ 计算距离平方和 """
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    # 将点集转换为数组
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # 初始猜测圆心位置
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    # 使用莱文伯格-马夸特算法拟合圆
    center_estimate = x_m, y_m
    try:
        center, _ = optimize.leastsq(f_2, center_estimate)
        xc, yc = center
        r = calc_R(xc, yc).mean()
        
        # 计算残差评估拟合质量
        Ri = calc_R(xc, yc)
        residuals = np.sum((Ri - r)**2) / len(points)
        
        # 如果残差太大，认为拟合失败
        if residuals > 0.1:  # 阈值根据实际情况调整
            return None
            
        return xc, yc, r
    except:
        return None

def extract_arc_points(points):
    """智能提取弧形点序列
    
    首先尝试拟合圆弧，如果拟合成功则按角度排序
    如果拟合失败，则应用其他策略
    
    Args:
        points: 原始点集
        
    Returns:
        有序点集
    """
    if len(points) < 3:
        return points
        
    # 尝试拟合圆弧
    result = fit_circle(points)
    if result:
        xc, yc, r = result
        # 计算每个点相对于圆心的角度
        angles = []
        for i, p in enumerate(points):
            angle = math.atan2(p[1] - yc, p[0] - xc)
            angles.append((i, angle))
        
        # 按角度排序
        sorted_indices = [i for i, _ in sorted(angles, key=lambda x: x[1])]
        
        # 确保角度连续（可能需要旋转排序结果）
        # 如果角度跨越了-pi到pi的边界，需要特殊处理
        angle_diffs = [
            abs(angles[sorted_indices[i]][1] - angles[sorted_indices[i-1]][1])
            for i in range(1, len(sorted_indices))
        ]
        max_jump_idx = angle_diffs.index(max(angle_diffs)) + 1
        
        # 如果存在大的角度跳变，从这里切开重新排序
        if max(angle_diffs) > math.pi:
            sorted_indices = sorted_indices[max_jump_idx:] + sorted_indices[:max_jump_idx]
            
        return [points[i] for i in sorted_indices]
    
    # 圆弧拟合失败，使用方向约束算法
    ordered = [points[0]]
    remaining = points.copy()
    remaining.pop(0)
    
    # 维护一个方向向量，并逐步更新
    dir_x, dir_y = 0, 0  # 初始方向未定义
    
    while remaining:
        last = ordered[-1]
        
        # 如果已经有至少2个点，计算当前方向向量
        if len(ordered) >= 2:
            prev = ordered[-2]
            # 更新方向向量 (使用滑动平均以减少噪声影响)
            cur_dir_x = last[0] - prev[0]
            cur_dir_y = last[1] - prev[1]
            cur_dir_len = math.sqrt(cur_dir_x**2 + cur_dir_y**2)
            
            if cur_dir_len > 0:
                cur_dir_x /= cur_dir_len
                cur_dir_y /= cur_dir_len
                
                # 平滑更新方向
                if dir_x == 0 and dir_y == 0:  # 首次设置方向
                    dir_x, dir_y = cur_dir_x, cur_dir_y
                else:
                    # 计算加权平均 (给当前方向较高权重)
                    dir_x = 0.7 * dir_x + 0.3 * cur_dir_x
                    dir_y = 0.7 * dir_y + 0.3 * cur_dir_y
                    
                    # 重新归一化
                    dir_len = math.sqrt(dir_x**2 + dir_y**2)
                    if dir_len > 0:
                        dir_x /= dir_len
                        dir_y /= dir_len
        
        # 选择下一个点
        best_score = -float('inf')
        best_idx = 0
        
        for i, point in enumerate(remaining):
            # 计算距离和方向
            dx = point[0] - last[0]
            dy = point[1] - last[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            # 如果距离太小，可能是重复点
            if dist < 0.001:
                best_idx = i
                break
                
            # 第一个点或方向未定义时，只考虑距离
            if dir_x == 0 and dir_y == 0:
                score = -dist  # 负距离作为分数
            else:
                # 计算与当前方向的一致性 (点积)
                if dist > 0:
                    cos_angle = (dir_x*dx + dir_y*dy) / dist
                    
                    # 考虑弧线曲率变化 - 优先选择轻微转向的点
                    # 这允许曲线自然弯曲，但避免急转或回折
                    if cos_angle > 0.7:  # 大致30度以内
                        turning_bonus = 0
                    elif cos_angle > 0:  # 0-90度
                        turning_bonus = 0.2 * (1 - cos_angle)  # 小幅转向奖励
                    else:  # 大于90度
                        turning_bonus = -0.5  # 惩罚急转
                        
                    score = cos_angle + turning_bonus - 0.1 * dist
                else:
                    score = -float('inf')
            
            if score > best_score:
                best_score = score
                best_idx = i
                
        ordered.append(remaining[best_idx])
        remaining.pop(best_idx)
    
    return ordered

def extract_wall_centerline(wall_points):
    """提取墙体中心线，支持多种几何形状"""
    # 先处理简单情况
    if len(wall_points) <= 2:
        return wall_points
    
    # 判断点集的分布特征
    xs = [p[0] for p in wall_points]
    ys = [p[1] for p in wall_points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    # 计算点之间的距离矩阵
    distances = []
    for i, p1 in enumerate(wall_points):
        for j, p2 in enumerate(wall_points[i+1:], i+1):
            dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            distances.append((i, j, dist))
    
    # 找出最远的两点
    i, j, _ = max(distances, key=lambda x: x[2])
    
    # 如果最远两点之间的距离接近矩形对角线，可能是U形或复杂形状
    max_dist = math.sqrt(width**2 + height**2)
    diag_ratio = max(distances, key=lambda x: x[2])[2] / max_dist
    
    if diag_ratio > 0.9:  # 接近矩形对角线
        # 可能是复杂形状，尝试按坐标分区
        if width > height:  # 水平延伸的墙
            # 按x坐标排序
            sorted_points = sorted(wall_points, key=lambda p: p[0])
        else:  # 垂直延伸的墙
            # 按y坐标排序
            sorted_points = sorted(wall_points, key=lambda p: p[1])
    else:
        # 尝试圆弧拟合
        sorted_points = extract_arc_points(wall_points)
    
    return sorted_points

# 添加测试函数以便手动验证该模块是否正常工作
if __name__ == "__main__":
    # 简单测试
    test_points = [(1,1), (2,2), (3,2), (4,1)]
    sorted_points = extract_wall_centerline(test_points)
    print("测试点排序结果:", sorted_points) 