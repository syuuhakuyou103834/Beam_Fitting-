import numpy as np
from scipy.interpolate import RectBivariateSpline
import pandas as pd
import math

def interpolate_grid(data, factor=10):
    """将网格分辨率从1mm提升到0.1mm"""
    n = data.shape[0]
    x = np.linspace(-15, 15, n)
    y = np.linspace(-15, 15, n)
    interp_fn = RectBivariateSpline(x, y, data)
    x_fine = np.linspace(-15, 15, n * factor)
    y_fine = np.linspace(-15, 15, n * factor)
    return interp_fn(x_fine, y_fine)

def fwhm_1d(profile):
    """计算一维分布的真实FWHM"""
    max_val = np.max(profile)
    if max_val <= 0:
        return 0
    
    half_max = max_val / 2
    above_half = np.where(profile > half_max)[0]
    
    if len(above_half) < 2:
        return 0
    
    left_idx = above_half[0]
    right_idx = above_half[-1]
    
    # 线性插值提高精度
    if left_idx > 0:
        left_interp = left_idx - (profile[left_idx] - half_max) / (profile[left_idx] - profile[left_idx-1])
    else:
        left_interp = left_idx
        
    if right_idx < len(profile)-1:
        right_interp = right_idx + (profile[right_idx] - half_max) / (profile[right_idx] - profile[right_idx+1])
    else:
        right_interp = right_idx
        
    return (right_interp - left_interp) * 0.1  # 转换为mm单位

def directional_fwhm(grid, angle_deg):
    """计算指定方向上的FWHM"""
    size = grid.shape[0]
    center = size // 2
    angle_rad = math.radians(angle_deg)
    
    # 沿给定方向生成单位向量
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    
    # 最大投影距离（网格覆盖[-15,15]mm）
    max_len = int(size * math.sqrt(2))
    line_points = []
    
    # 沿直线采样点（步长0.1mm）
    for d in np.linspace(-max_len, max_len, 2*max_len):
        x = center + d * dx
        y = center + d * dy
        
        # 确保在网格范围内
        if (0 <= x < size) and (0 <= y < size):
            i, j = int(x), int(y)
            # 线性插值
            di, dj = x - i, y - j
            val = (1-di)*(1-dj)*grid[i,j]
            if i < size-1:
                val += di*(1-dj)*grid[i+1,j]
            if j < size-1:
                val += (1-di)*dj*grid[i,j+1]
            if i < size-1 and j < size-1:
                val += di*dj*grid[i+1,j+1]
            line_points.append(val)
    
    return fwhm_1d(np.array(line_points))

def fwhm_map(grid):
    """计算所有指定方向上的FWHM"""
    angles = [0, 15, 45, 75, 90]
    return [directional_fwhm(grid, angle) for angle in angles]

def ensure_monotonicity(grid):
    """确保网格满足物理规律：中心强，边缘弱，单峰"""
    size = grid.shape[0]
    center = size // 2
    center_val = grid[center, center]
    
    for i in range(size):
        for j in range(size):
            dist = math.sqrt((i-center)**2 + (j-center)**2)
            # 中心点确保为最大值
            if (i, j) != (center, center) and grid[i, j] > center_val:
                grid[i, j] = min(grid[i, j], center_val)
            
            # 近似圆形边界条件
            max_radius = 0.8 * center  # 比实际半径稍小，避免边缘影响
            if dist > max_radius:
                grid[i, j] = max(0, min(grid[i, j], grid[center, center] * 0.05))
    return grid

def equivalence_transformation(data):
    """进行等价变换：减少FWHM，同时保持行列和不变"""
    size = data.shape[0]
    center = size // 2
    
    # 计算原始行列和
    row_sums = np.sum(data, axis=1)
    col_sums = np.sum(data, axis=0)
    total_sum = np.sum(row_sums)
    
    # 初始变换参数
    transform_matrix = np.zeros_like(data)
    iterations = 0
    fwhms = [10]  # 初始化一个大于3.5的值
    
    # 迭代变换直到所有FWHM<3.5
    while max(fwhms[0:2]) >= 3.5 and iterations < 50:  # 重点关注x/y轴方向
        iterations += 1
        # 中心点增值幅度（随迭代减少）
        center_increase = 0.025 * (1 + np.exp(-iterations/10)) * total_sum / 1000
        transform_matrix = np.zeros_like(data)
        
        # 核心变换：中心点增值，其他点按比例减值
        transform_matrix[center, center] = center_increase
        others_sum = center_increase
        for i in range(size):
            for j in range(size):
                if (i, j) != (center, center):
                    proportional_weight = abs(i-center) + abs(j-center) + 1
                    reduction = center_increase * proportional_weight / (2*size**2)
                    transform_matrix[i, j] = -reduction
                    others_sum -= reduction
        
        # 应用变换
        data += transform_matrix
        
        # 强制非负且保持总和不变
        data = np.maximum(data, 0)
        data[center, center] = max(0, data[center, center])
        
        # 重新平衡行列和
        for r in range(size):
            row_diff = np.sum(data[r, :]) - row_sums[r]
            avg_diff = row_diff / size
            for c in range(size):
                if data[r, c] > 0: 
                    data[r, c] = max(0, data[r, c] - avg_diff)
        
        for c in range(size):
            col_diff = np.sum(data[:, c]) - col_sums[c]
            avg_diff = col_diff / size
            for r in range(size):
                if data[r, c] > 0:
                    data[r, c] = max(0, data[r, c] - avg_diff)
        
        # 确保物理规律
        data = ensure_monotonicity(data)
        
        # FWHM计算（使用插值后的高分辨率）
        high_res = interpolate_grid(data)
        fwhms = fwhm_map(high_res)
        print(f"Iteration {iterations}: FWHM={np.array(fwhms).round(2)}")
    
    return data

def main():
    # 1. 从CSV读取数据
    data_str = """
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,1.03864531,1.3496959,1.74063544,2.37242795,3.49764811,4.69864067,6.04029359,7.81432366,5.83078221,3.88345243,2.65721636,1.58433916,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,1.17063155,1.49132069,1.93793724,2.49926094,3.40640916,5.02203684,6.74646099,8.67284987,11.2200599,8.37202661,5.57598722,3.81531761,2.27484566,1.15713718,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,1.16732987,1.54093292,1.96306441,2.55095745,3.28984251,4.48394544,6.61063839,8.88054299,11.41629904,14.7692582,11.02031752,7.33981777,5.02220233,2.99443882,1.52316993,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,1.01527489,1.42769264,1.88462461,2.40090886,3.11992633,4.02361328,5.48405049,8.08508381,10.86127089,13.96260529,18.06341283,13.47830353,8.9769003,6.14236089,3.66232237,1.86289974,1.18165484,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,1.28339059,1.80472039,2.3823198,3.03494537,3.94384232,5.08617661,6.9322888,10.22020787,13.72953563,17.64987621,22.83363269,17.03767914,11.34753693,7.76444705,4.62947531,2.35485779,1.49370846,1.05070759,0,0,0,0,0,0,0,0
    0,0,0,0,1.17656548,1.56324308,2.19825257,2.90180167,3.6967369,4.80382533,6.19525378,8.44392394,12.44879727,16.72335904,21.49855791,27.81266955,20.75286689,13.82194849,9.45754024,5.6389655,2.86835136,1.81942227,1.27982189,0,0,0,0,0,0,0,0
    0,0,0,0,1.40735704,1.86988415,2.6294552,3.47101037,4.42187771,5.74612928,7.41049616,10.1002587,14.89071595,20.00376292,25.71565046,33.26831927,24.82368693,16.53322039,11.31270293,6.74508804,3.43099855,2.17631467,1.5308679,1.15810747,0,0,0,0,0,0,0
    0,0,0,0,1.66080584,2.20662876,3.10298981,4.09609939,5.218207,6.78094103,8.74504123,11.91919904,17.57236251,23.60621039,30.34674314,39.25956068,29.29414725,19.5106631,13.34999053,7.95980076,4.0488819,2.5682439,1.8065596,1.3666693,0,0,0,0,0,0,0
    0,0,0,0,2.16179863,2.87227255,4.03902669,5.33171417,6.79231277,8.82645561,11.38303928,15.51470227,22.87317894,30.72717593,39.50103384,51.10246024,38.1309156,25.3961804,17.37710123,10.36092597,5.27025323,3.34297122,2.35151994,1.7789339,1.20452274,0,0,0,0,0,0
    0,0,0,2.03203155,3.15990235,4.19840251,5.90384774,7.79336981,9.92832765,12.90163545,16.63859535,22.67784961,33.43373937,44.91393148,57.73868484,74.69649677,55.7359822,37.12161211,25.40011928,15.14457171,7.7035323,4.88642304,3.43721811,2.60026875,1.7606516,1.19848361,0,0,0,0,0
    0,0,0,2.81559201,4.37837484,5.81732531,8.18039785,10.79852808,13.75673523,17.87656382,23.05451217,31.42252991,46.3259389,62.23294446,80.0029801,103.4998002,77.22802636,51.43587187,35.19451895,20.98438633,10.67404884,6.77064964,4.76262480,3.60294401,2.43956674,1.66062425,1.18294556,0,0,0,0
    0,0,0,3.58909213,5.58120303,7.41546233,10.42771872,13.76510232,17.53598885,22.78761775,29.38805333,40.05493491,59.05261202,79.32959399,101.9814181,131.9332903,98.44412842,65.56634702,44.8631657,26.74922201,13.60642611,8.63068414,6.07101423,4.59274566,3.10976511,2.11683135,1.50792464,1.02859762,0,0,0
    0,0,0,4.84707511,7.53742432,10.01459466,14.08265212,18.58979443,23.68238319,30.77471707,39.6886167,54.09425868,79.75065439,107.1347535,137.7260811,178.1761361,132.9489652,88.54746473,60.58778265,36.12487046,18.37550193,11.65575382,8.19891519,6.20251094,4.19974314,2.85878439,2.03645483,1.38912286,0,0,0
    0,0,0,3.72829489,5.79766972,7.70307066,10.83215728,14.29898113,18.21612131,23.67143437,30.52786752,41.60846357,61.34296465,82.40638836,105.9367624,137.0503164,102.2622791,68.10933453,46.60318135,27.78668926,14.13415067,8.96542480,6.30647822,4.77087508,3.23037719,2.19893255,1.56640943,1.06849173,0,0,0
    0,0,0,2.29203214,3.56421520,4.73559256,6.65925130,8.79053971,11.19866768,14.55241335,18.76752122,25.57950445,37.71162169,50.66071651,65.12641063,84.25399253,62.86746007,41.87136164,28.65009141,17.08233567,8.68920741,5.51164604,3.87701380,2.93297588,1.98592884,1.35183085,0,0,0,0,0
    0,0,0,1.53228405,2.38277204,3.16586875,4.45188545,5.87670807,7.48660524,9.72867284,12.54658382,17.10057460,25.21121554,33.86802758,43.53872631,56.32602022,42.02855818,27.99211161,19.15334312,11.41999276,5.80896475,3.68468101,2.59188617,1.96077188,1.32764590,0,0,0,0,0,0
    0,0,0,0,1.28676560,1.70966041,2.40414650,3.17359180,4.04298269,5.25376384,6.77551701,9.23480334,13.61478330,18.28971140,23.51216754,30.41767494,22.69663321,15.11654735,10.34335752,6.16712536,3.13700845,1.98983398,1.39969326,1.05887335,0,0,0,0,0,0,0
    0,0,0,0,0,0,1.09293390,1.44272658,1.83795490,2.38838049,3.08017512,4.19817579,6.18933090,8.31457053,10.68871843,13.82798766,10.31797350,6.87203840,4.70212863,2.80359802,1.42609566,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,1.09335026,1.42078374,1.83231388,2.49738260,3.68186757,4.94611584,6.35843299,8.22590038,6.13788674,4.08799202,2.79717068,1.66778555,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,1.21753808,1.65946373,2.44653170,3.28660087,4.22505903,5.46595596,4.07851020,2.71639375,1.85866726,1.10821211,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,1.07606733,1.46664383,2.16225913,2.90471722,3.73413208,4.83084409,3.60461135,2.40076481,1.64270108,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,1.05266096,1.55192810,2.08481593,2.68011564,3.46726375,2.58715414,1.72311187,1.17902334,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,1.34114355,1.80165398,2.31609944,2.99633625,2.23576407,1.48907695,1.01888711,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,1.15889804,1.55683056,2.00136897,2.58916966,1.93195023,1.28672904,0,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,1.11007251,1.49123973,1.91704930,2.48008536,1.85055524,1.23251786,0,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    """
    lines = [line.strip() for line in data_str.strip().split('\n')]
    grid_1mm = np.zeros((31, 31))
    for i, line in enumerate(lines):
        values = line.split(',')
        grid_1mm[i] = [float(v) for v in values]
    
    # 2. 插值到0.1mm分辨率
    high_res_grid = interpolate_grid(grid_1mm)
    
    # 3. 计算变换前FWHM
    angles = [0, 15, 45, 75, 90]
    original_fwhm = {}
    for angle in angles:
        fwhm = directional_fwhm(high_res_grid, angle)
        original_fwhm[angle] = round(fwhm, 4)
    
    # 4. 检查是否需要等价变换
    if max(original_fwhm.values()) >= 3.5:
        print("Starting equivalence transformation...")
        transformed_1mm = equivalence_transformation(grid_1mm.copy())
    else:
        transformed_1mm = grid_1mm
    
    # 5. 生成变换后的高分辨率网格
    transformed_high_res = interpolate_grid(transformed_1mm)
    
    # 6. 计算变换后FWHM
    transformed_fwhm = {}
    for angle in angles:
        fwhm = directional_fwhm(transformed_high_res, angle)
        transformed_fwhm[angle] = round(fwhm, 4)
    
    # 7. 输出结果
    # 高分辨率CSV (-15mm到15mm, 0.1mm步长)
    fine_points = np.linspace(-15, 15, 301)
    high_res_df = pd.DataFrame(transformed_high_res, index=fine_points, columns=fine_points)
    high_res_df.to_csv("transformed_high_res.csv", float_format='%.4f')
    
    # 1mm分辨率CSV
    coarse_points = np.linspace(-15, 15, 31)
    reshaped_1mm = transformed_1mm.reshape(31, 31)
    coarse_df = pd.DataFrame(reshaped_1mm, index=coarse_points, columns=coarse_points)
    coarse_df.to_csv("transformed_low_res.csv", float_format='%.4f')
    
    # 生成报告
    with open("fwhm_report.txt", "w") as f:
        f.write("Original FWHM (mm):\n")
        for angle in sorted(original_fwhm.keys()):
            f.write(f"{angle}°: {original_fwhm[angle]} | ")
        
        f.write("\n\nTransformed FWHM (mm):\n")
        for angle in sorted(transformed_fwhm.keys()):
            f.write(f"{angle}°: {transformed_fwhm[angle]} | ")

if __name__ == "__main__":
    main()
