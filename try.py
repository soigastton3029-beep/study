import pandas as pd
import numpy as np
import warnings
import os
from mgtwr.sel import SearchMGTWRParameter
from mgtwr.model import MGTWR
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def run_mgtwr_final():
    # =============================================
    # 1. 路径设置
    # =============================================
    input_file = r"D:\城乡融合2\xgboost\merged_with_coords.xlsx"
    output_dir = r"D:\城乡融合2\交互和依赖\面板MGTWR结果"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # =============================================
    # 2. 数据读取与列名规范化
    # =============================================
    print("正在读取原始数据...")
    data = pd.read_excel(input_file)

    column_mapping = {
        'FID': 'ID',
        'X22': 'x1', 'X23': 'x2', 'X24': 'x3', 'X25': 'x4', 
        '坐标_WGS1984 X': 'lon', '坐标_WGS1984 Y': 'lat',
        '经度': 'lon', '纬度': 'lat'
    }
    data = data.rename(columns=column_mapping)

    # =============================================
    # 3. 缺失值清理
    # =============================================
    # 【强制写死4个变量，杜绝灵异事件】
    x_columns = ['x1', 'x2', 'x3', 'x4']
    y_column = 'Y'

    cols_to_check = ['city', '年份', 'lon', 'lat', y_column] + x_columns
    
    # 强制清理
    data = data.dropna(subset=cols_to_check).reset_index(drop=True)

    # =============================================
    # 4. 数据归一化
    # =============================================
    scaler = MinMaxScaler()
    normalize_cols = [y_column] + x_columns
    data[normalize_cols] = scaler.fit_transform(data[normalize_cols])

    # =============================================
    # 5. 提取矩阵 (加入防崩溃抗体)
    # =============================================
    coords = data[['lon', 'lat']].values
    Y = data[[y_column]].values
    
    # 【核心：强制只取这4列】
    X = data[x_columns].values

    # 【终极抢救：加入千万分之一的微小扰动，打破局部数据"死水"导致的矩阵奇异】
    np.random.seed(42) 
    X = X + np.random.normal(0, 1e-7, X.shape)

    min_year = data['年份'].min()
    t = (data['年份'] - min_year).values.reshape(-1, 1)

    print("\n" + "=" * 50)
    print("【核对矩阵维度】")
    print(f"坐标矩阵形状: {coords.shape}")
    print(f"时间矩阵形状: {t.shape}")
    print(f"自变量矩阵形状: {X.shape}  <-- 必须是4！如果还是8，请检查是否跑错了文件！")
    print(f"因变量矩阵形状: {Y.shape}")
    print("=" * 50)

    # =============================================
    # 6. MGTWR 模型训练 (大带宽自适应)
    # =============================================
    print("\n开始搜索 MGTWR 最优时空带宽...")
    # 【修改：使用自适应带宽 fixed=False，且最小邻居数加大到 50，避免局部共线性】
    sel_multi = SearchMGTWRParameter(coords, t, X, Y, kernel='gaussian', fixed=False)
    bws = sel_multi.search(multi_bw_min=[50], verbose=True, tol_multi=1.0e-4, time_cost=True)

    print("\n带宽搜索完成，开始拟合模型...")
    mgtwr = MGTWR(coords, t, X, Y, sel_multi, kernel='gaussian', fixed=False).fit()

    # =============================================
    # 7. 结果保存与导出
    # =============================================
    variable_names = ['常数项'] + x_columns

    model_eval = pd.DataFrame({
        'R2': [mgtwr.R2],
        'Adj_R2': [mgtwr.adj_R2],
        'AICc': [mgtwr.aic_c],
        'RSS': [mgtwr.RSS],
        '样本量': [mgtwr.n],
        '变量数': [len(variable_names)]
    })
    model_eval.to_csv(os.path.join(output_dir, '模型精度评价_真实面板.csv'), index=False, encoding='utf-8-sig')

    coefficients = pd.DataFrame(mgtwr.betas, columns=variable_names)
    coefficients['city'] = data['city'].values
    coefficients['年份'] = data['年份'].values
    coefficients['lon'] = coords[:, 0]
    coefficients['lat'] = coords[:, 1]

    cols_order = ['city', '年份', 'lon', 'lat'] + variable_names
    coefficients = coefficients[cols_order]

    coefficients.to_csv(os.path.join(output_dir, 'MGTWR动态系数_真实面板.csv'), index=False, encoding='utf-8-sig')
    
    print("\n✅ 分析大功告成，所有报错已被肃清！")

if __name__ == "__main__":
    run_mgtwr_final()