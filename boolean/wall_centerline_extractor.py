import numpy as np
import math
from sklearn.cluster import DBSCAN
from scipy import optimize
from sklearn.linear_model import LinearRegression

def fit_circle(points):
    """尝试将点集拟合为圆弧"""
    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    center_estimate = x_m, y_m
    try:
        center, _ = optimize.leastsq(f_2, center_estimate)
        xc, yc = center
        r = calc_R(xc, yc).mean()
        
        # 评估拟合质量
        Ri = calc_R(xc, yc)
        residuals = np.sum((Ri - r)**2) / len(points)
        
        if residuals > 0.8:  # 放宽残差要求
            return None
            
        return float(xc), float(yc), float(r)  # 转换为标量避免格式化错误
    except:
        return None

def identify_wall_edges(points, eps=0.5):
    """识别墙体的内外边缘"""
    if len(points) < 4:
        return [points], []
    
    # 将点集转换为numpy数组以便聚类
    points_array = np.array(points)
    
    # 使用DBSCAN聚类
    db = DBSCAN(eps=eps, min_samples=1).fit(points_array)
    labels = db.labels_
    
    # 收集不同聚类的点
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[i])
    
    # 排除噪声点 (label=-1)
    if -1 in clusters:
        del clusters[-1]
    
    # 如果只有一个聚类或没有聚类，返回原始点集
    if len(clusters) <= 1:
        # 尝试使用原始点集进行圆弧拟合
        if fit_circle(points):
            # 如果能拟合圆弧，则人为分为两组进行处理
            return create_artificial_edges(points)
        return [points], []
    
    # 只保留两个最大的聚类（内外边缘）
    largest_clusters = sorted(clusters.values(), key=len, reverse=True)[:2]
    
    return largest_clusters

def create_artificial_edges(points):
    """当聚类失败但形状像弧时，人为创建两条边缘"""
    # 尝试拟合圆弧
    result = fit_circle(points)
    if not result:
        return [points], []
        
    xc, yc, r = result
    
    # 计算每个点到圆心的距离
    distances = []
    for i, p in enumerate(points):
        dist = math.sqrt((p[0] - xc)**2 + (p[1] - yc)**2)
        distances.append((i, dist))
    
    # 按距离排序
    sorted_dist = sorted(distances, key=lambda x: x[1])
    
    # 中点作为分界
    mid_idx = len(sorted_dist) // 2
    
    # 创建内外两组点
    inner_points = [points[idx] for idx, _ in sorted_dist[:mid_idx]]
    outer_points = [points[idx] for idx, _ in sorted_dist[mid_idx:]]
    
    return [inner_points, outer_points]

def compute_centerline_from_edges(edge1, edge2):
    """根据墙体内外边缘计算中心线
    
    Args:
        edge1, edge2: 墙体两侧的边缘点
        
    Returns:
        中心线点集
    """
    # 如果一边为空，直接返回另一边
    if not edge1:
        return edge2
    if not edge2:
        return edge1
    
    # 合并点集尝试拟合圆弧
    all_points = edge1 + edge2
    result = fit_circle(all_points)
    
    if result:
        xc, yc, r = result
        # 计算边缘点的平均半径
        r1 = np.mean([math.sqrt((p[0]-xc)**2 + (p[1]-yc)**2) for p in edge1]) if edge1 else r
        r2 = np.mean([math.sqrt((p[0]-xc)**2 + (p[1]-yc)**2) for p in edge2]) if edge2 else r
        
        # 计算中心线的半径
        center_r = (r1 + r2) / 2
        
        # 计算角度范围
        all_angles = []
        for p in all_points:
            angle = math.atan2(p[1]-yc, p[0]-xc)
            all_angles.append(angle)
        
        min_angle = min(all_angles)
        max_angle = max(all_angles)
        
        # 创建中心线 - 增加分辨率
        num_points = min(25, max(20, len(all_points)))
        centerline = []
        
        # 处理角度范围跨越-π到π的情况
        if max_angle - min_angle > math.pi:
            # 调整角度范围
            adjusted_angles = []
            for angle in all_angles:
                if angle < 0:
                    adjusted_angles.append(angle + 2*math.pi)
                else:
                    adjusted_angles.append(angle)
            min_angle = min(adjusted_angles)
            max_angle = max(adjusted_angles)
            
            # 生成中心线点
            for i in range(num_points):
                angle = min_angle + (max_angle - min_angle) * i / (num_points - 1)
                if angle > math.pi:
                    angle -= 2*math.pi
                x = xc + center_r * math.cos(angle)
                y = yc + center_r * math.sin(angle)
                centerline.append((x, y))
        else:
            # 生成中心线点
            for i in range(num_points):
                angle = min_angle + (max_angle - min_angle) * i / (num_points - 1)
                x = xc + center_r * math.cos(angle)
                y = yc + center_r * math.sin(angle)
                centerline.append((x, y))
        
        print(f"成功拟合圆弧 - 圆心: ({xc:.2f}, {yc:.2f}), 半径: {center_r:.2f}")
        return centerline
    
    print("圆弧拟合失败，使用点对点插值")
    # 圆弧拟合失败，使用点对点插值
    # 对点进行主方向排序
    xs1 = [p[0] for p in edge1]
    ys1 = [p[1] for p in edge1]
    xs2 = [p[0] for p in edge2]
    ys2 = [p[1] for p in edge2]
    
    # 判断墙体主方向
    x_range = max(max(xs1 + xs2)) - min(min(xs1 + xs2))
    y_range = max(max(ys1 + ys2)) - min(min(ys1 + ys2))
    
    if x_range > y_range:
        # 水平方向为主
        edge1.sort(key=lambda p: p[0])
        edge2.sort(key=lambda p: p[0])
    else:
        # 垂直方向为主
        edge1.sort(key=lambda p: p[1])
        edge2.sort(key=lambda p: p[1])
    
    # 重采样较长的边缘使两边点数相等
    if len(edge1) > len(edge2):
        long_edge = edge1
        short_edge = edge2
    else:
        long_edge = edge2
        short_edge = edge1
    
    # 重采样长边为短边的长度
    resampled_long = []
    if len(short_edge) > 1:
        for i in range(len(short_edge)):
            idx = int(i * (len(long_edge) - 1) / (len(short_edge) - 1))
            resampled_long.append(long_edge[idx])
    else:
        resampled_long = long_edge
    
    # 计算中心线
    centerline = []
    for i in range(len(short_edge)):
        x = (short_edge[i][0] + resampled_long[i][0]) / 2
        y = (short_edge[i][1] + resampled_long[i][1]) / 2
        centerline.append((x, y))
    
    return centerline

def is_straight_wall(points, threshold=0.2):  # 提高阈值，更宽松地识别弧形墙
    """判断墙体是否为直线墙"""
    if len(points) < 5:
        return True
        
    # 提取x和y坐标
    x = np.array([p[0] for p in points]).reshape(-1, 1)
    y = np.array([p[1] for p in points])
    
    # 计算主方向
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    
    # 拟合线性模型
    model = LinearRegression()
    
    # 根据主方向选择自变量和因变量
    if x_range > y_range:
        model.fit(x, y)
        y_pred = model.predict(x)
        errors = y - y_pred
    else:
        model.fit(y.reshape(-1, 1), x)
        x_pred = model.predict(y.reshape(-1, 1))
        errors = x - x_pred.flatten()
    
    # 计算平均误差
    mean_error = float(np.mean(np.abs(errors)))
    max_error = float(np.max(np.abs(errors)))
    
    # 计算长度
    length = float(max(x_range, y_range))
    
    # 误差与长度的比值
    error_ratio = max_error / length if length > 0 else 0
    
    print(f"直线拟合 - 平均误差: {mean_error:.4f}, 最大误差: {max_error:.4f}, 误差比: {error_ratio:.4f}")
    
    # 如果误差比小于阈值，则认为是直线墙
    return error_ratio < threshold

def evaluate_arc_fit(points, fit_result):
    """评估圆弧拟合的质量"""
    if not fit_result:
        return float('inf')
        
    xc, yc, r = fit_result
    
    # 计算每个点与拟合圆的距离
    distances = []
    for p in points:
        dist = abs(math.sqrt((p[0]-xc)**2 + (p[1]-yc)**2) - r)
        distances.append(dist)
    
    # 计算与圆弧的拟合误差
    mean_error = np.mean(distances)
    max_error = np.max(distances)
    
    # 评估弧形性 - 计算点在圆周上分布的角度范围
    angles = []
    for p in points:
        angle = math.atan2(p[1]-yc, p[0]-xc)
        angles.append(angle)
    
    min_angle = min(angles)
    max_angle = max(angles)
    angle_range = max_angle - min_angle
    
    # 如果角度范围跨越-π到π边界
    if angle_range > math.pi:
        # 重新计算角度
        adjusted = []
        for a in angles:
            if a < 0:
                adjusted.append(a + 2*math.pi)
            else:
                adjusted.append(a)
        min_angle = min(adjusted)
        max_angle = max(adjusted)
        angle_range = max_angle - min_angle
    
    # 计算弧长与半径的比值
    arc_ratio = angle_range / (2 * math.pi)
    
    # 如果弧长比例太小，拟合分数应该更高(更差)
    quality_score = float(mean_error) + (0.05 / arc_ratio if arc_ratio > 0 else float('inf'))
    
    print(f"圆弧拟合 - 平均误差: {float(mean_error):.4f}, 最大误差: {float(max_error):.4f}, 弧长比例: {arc_ratio:.4f}, 质量分数: {quality_score:.4f}")
    
    return quality_score

def generate_smooth_arc(xc, yc, r, start_angle, end_angle, num_points=40):
    """生成平滑的圆弧点集"""
    centerline = []
    
    # 确保角度差为正值
    if end_angle < start_angle:
        if end_angle < 0 and start_angle > 0:
            end_angle += 2*math.pi
        else:
            start_angle, end_angle = end_angle, start_angle
            
    angle_range = end_angle - start_angle
    
    # 生成均匀分布的点
    for i in range(num_points):
        angle = start_angle + (angle_range * i / (num_points - 1))
        if angle > math.pi:
            angle -= 2*math.pi
        x = xc + r * math.cos(angle)
        y = yc + r * math.sin(angle)
        centerline.append((x, y))
        
    return centerline

def extract_wall_true_centerline(wall_points, tolerance=0.5):
    """提取考虑墙体厚度的真实中心线"""
    # 处理简单情况
    if len(wall_points) <= 3:
        return wall_points
    
    # 尝试拟合圆弧
    circle_fit = fit_circle(wall_points)
    
    # 评估拟合质量
    fit_quality = float('inf')
    if circle_fit:
        fit_quality = evaluate_arc_fit(wall_points, circle_fit)
    
    # 检查是否为直线墙
    is_straight = is_straight_wall(wall_points, threshold=0.2)
    
    # 弧形墙判定更为宽松
    if not is_straight and fit_quality < 1.5:  # 大幅提高阈值
        print("检测到弧形墙")
        xc, yc, r = circle_fit
        
        # 计算角度范围
        angles = []
        for p in wall_points:
            angle = math.atan2(p[1]-yc, p[0]-xc)
            angles.append(angle)
        
        # 处理角度范围
        adjusted = []
        for a in angles:
            if a < 0:
                adjusted.append(a + 2*math.pi)
            else:
                adjusted.append(a)
        
        start_angle = min(adjusted)
        end_angle = max(adjusted)
        
        # 使用平滑arc生成器
        return generate_smooth_arc(xc, yc, r, start_angle, end_angle)
    else:
        print("检测到直线墙")
        return sort_along_principal_direction(wall_points)

def sort_along_principal_direction(points):
    """沿主方向对点进行排序"""
    # 计算主方向
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    
    # 根据主方向排序
    if x_range > y_range:
        return sorted(points, key=lambda p: p[0])
    else:
        return sorted(points, key=lambda p: p[1])

if __name__ == "__main__":
    # 测试点集模拟墙体内外边缘
    inner_edge = [(0,0), (1,0.1), (2,0.3), (3,0.6), (4,1.1)]
    outer_edge = [(0,1), (1,1.1), (2,1.3), (3,1.6), (4,2.1)]
    all_points = inner_edge + outer_edge
    
    centerline = extract_wall_true_centerline(all_points)
    print("测试中心线:", centerline) 