import datetime
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import scipy.stats as st
from yinzifenxi import yinzi
# 1. 读取新数据文件
data = pd.read_excel("2021年6~8月.xlsx")


# 处理时间列
def convert_time(row):
    row = row.replace("上午", "AM").replace("下午", "PM")
    format_str = "%m/%d/%y %p%I时%M分%S秒"
    row = datetime.strptime(row, format_str)
    return row


def add_time_features(data):
    data['is_weekend'] = data['日子'].isin([5, 6]).astype(int)
    return data


def process_data(data):
    data["记录时间"] = [convert_time(dt_str) for dt_str in data["记录时间"]]
    # 获取第一个时间点的日期（假设第一个时间点是 datetime.datetime 类型）
    date = data['记录时间'].iloc[0].date()

    first_time = data['记录时间'].iloc[0]
    data['记录时间'] = pd.to_datetime(data['记录时间'])
    data['日子'] = (data['记录时间'] - first_time).astype('timedelta64[h]') // 24
    data['时间差（小时）'] = (data['记录时间'] - first_time).astype('timedelta64[h]') % 24
    data = add_time_features(data)
    # 使用KNN插值填充缺失值
    imputer = KNNImputer(n_neighbors=5)
    data_filled = imputer.fit_transform(data.select_dtypes(include=np.number))
    data.loc[:, data.select_dtypes(include=np.number).columns] = data_filled
    return data


# threshold = 80000  # 50w，单位千瓦时（kwh）
# 计算Q1, Q3, 和 IQR
def IQR(data):
    Q1 = data['逐时负荷/kWh'].quantile(0.25)
    Q3 = data['逐时负荷/kWh'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    for i in range(len(data)):
        if data.loc[i, '逐时负荷/kWh'] < lower_bound or data.loc[i, '逐时负荷/kWh'] > upper_bound:
            data_ = [index for index in data['日子'] if
                     index == i and data.loc[index, '逐时负荷/kWh'] > lower_bound or data.loc[
                         index, '逐时负荷/kWh'] < upper_bound]
            average = np.mean(data_)
            data.loc[i, '逐时负荷/kWh'] = average


IQR(data)
data = process_data(data)
# fill_missing_values('L(h-24)')
# fill_missing_values('L(h-1)')
# fill_missing_values('T（h-1）')


# 2. 数据预处理
# X = data[
#     ['is_weekend', '日子', '日期因子', '星期因子映射', '辐射强度', '时间差（小时）', '温度 °C ', '相对湿度%', 'T(h-1)',
#      'L(h-1)',
#      'L(h-24)']]
Y = data['逐时负荷/kWh']
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(data[['is_weekend', '星期因子映射']])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    data[['日子', '日期因子', '辐射强度', '时间差（小时）', '温度 °C ', '相对湿度%', 'T(h-1)', 'L(h-1)', 'L(h-24)']])
X = np.hstack((scaled_features, encoded_features))
# yinzi(X)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# 定义测试集的大小比例，例如20%
test_size = 0.2

# 计算分割点
split_idx = int(len(X_poly) * (1 - test_size))
# 3. 拆分数据集
X_train, X_test = X_poly[:split_idx], X_poly[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# 4. 设置超参数搜索空间
params = {
    'n_estimators': st.randint(100, 1000),  # 增加最大值
    'learning_rate': st.uniform(0.001, 0.3),  # 增加最大值，减小最小值
    'subsample': st.beta(10, 1),  # 这个参数的范围通常在0.5到1之间，可以保持不变
    'max_depth': st.randint(3, 50),  # 增加最大值
    'colsample_bytree': st.beta(10, 1),  # 这个参数的范围通常在0.5到1之间，可以保持不变
    'min_child_weight': st.expon(0, 100),  # 增加尾部范围
    'gamma': st.uniform(0, 10),  # 增加最大值
    'reg_alpha': st.uniform(0, 1),  # 增加最大值
    'reg_lambda': st.uniform(0, 1)  # 增加最大值
}

xgb_model = XGBRegressor(objective='reg:squarederror', booster='gbtree', seed=42)
random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=200,
                                   scoring='neg_mean_absolute_error', n_jobs=-1, cv=5, random_state=42)

# # 进行超参数搜索
# random_search.fit(X_train, Y_train)
#
# # 输出最佳参数
# print("最佳参数:", random_search.best_params_)
#
# # 5. 使用最佳参数重新训练模型
# best_model = random_search.best_estimator_
params = {'colsample_bytree': 0.9723294044494839, 'gamma': 8.306425177506245, 'learning_rate': 0.03710162861049878,
          'max_depth': 45, 'min_child_weight': 19.422577002336727, 'n_estimators': 317,
          'reg_alpha': 0.036731937426613626, 'reg_lambda': 0.9572702629833256, 'subsample': 0.8278471530596209}

best_model = xgb.XGBRegressor(**params)
best_model.fit(X_train, Y_train)

# 预测并计算评估指标
predictions = best_model.predict(X_test)
mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)
non_zero_mask = Y_test != 0
mape = (np.sum(np.abs((Y_test[non_zero_mask] - predictions[non_zero_mask]) / Y_test[non_zero_mask])) / len(
    Y_test))
R_square = r2_score(Y_test, predictions)
# 打印评估指标
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('平均百分比误差:', mape)
print("R square :", R_square)
# 绘制折线图
predictions_full = best_model.predict(X_test)

plt.figure(figsize=(12, 6))
x = range(len(Y_test))
plt.plot(x, Y_test, label='Actual', color='blue')
x = range(len(Y_test))
plt.plot(x, predictions_full, label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()

# 绘制折线图
predictions_full = best_model.predict(X_test)
x = range(len(Y))
plt.figure(figsize=(12, 6))
plt.plot(x, Y, label='Actual', color='blue')
x = range(len(predictions_full))
plt.plot([z + split_idx for z in x], predictions_full, label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()