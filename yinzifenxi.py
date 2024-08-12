import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis

def yinzi(X):
    # 假设数据包含在名为"Sheet1"的工作表中，你可以根据实际情况调整工作表的名称
    # 假设数据矩阵是DataFrame中的所有列（除去标签列等不需要的列）

    # 创建因子分析模型，指定你想要保留的因子数量
    n_factors = 16  # 指定保留的因子数量
    fa = FactorAnalysis(n_components=n_factors)

    # 拟合模型并获取因子载荷矩阵
    fa.fit(X)
    factor_loadings = fa.components_.T  # 转置以匹配属性和因子的维度

    # 计算每个属性的贡献度百分比
    total_variance = np.sum(np.var(X, axis=0))  # 总方差
    contribution_percentages = np.var(factor_loadings, axis=1) / total_variance * 100

    # 打印每个属性的贡献度百分比
    for i, percentage in enumerate(contribution_percentages):
        print(f"属性{i+1}的贡献度百分比: {percentage:.2f}%")
