import os
import pickle
from datetime import datetime
import statsmodels.api as sm
import numpy as np
import pandas as pd
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
# 保存参数的文件名
parameters_file = 'best_sarima_parameters.pkl'
def convert_time(row):
    row = row.replace("上午", "AM").replace("下午", "PM")
    format_str = "%m/%d/%y %p%I时%M分%S秒"
    row = datetime.strptime(row, format_str)
    return row


def process_data(data):
    data["记录时间"] = [convert_time(dt_str) for dt_str in data["记录时间"]]

    data['记录时间'] = pd.to_datetime(data['记录时间'])
    data['月'] = data['记录时间'].dt.month
    data['日'] = data['记录时间'].dt.day
    data['星期几'] = data['记录时间'].dt.weekday
    data['是否周末'] = data['星期几'].isin([5, 6]).astype(int)

    window_size = 7  # 例如，使用一周的数据
    data['滚动平均负荷'] = data['逐时负荷/kWh'].rolling(window=window_size).mean().shift(1)
    data['滚动最大负荷'] = data['逐时负荷/kWh'].rolling(window=window_size).max().shift(1)
    data['滚动最小负荷'] = data['逐时负荷/kWh'].rolling(window=window_size).min().shift(1)
    start_time = data['记录时间'].min()

    # 计算小时差
    data['小时差'] = (data['记录时间'] - start_time).astype('timedelta64[h]')

    # 使用KNN插值填充缺失值
    imputer = KNNImputer(n_neighbors=5)
    data_filled = imputer.fit_transform(data.select_dtypes(include=np.number))
    data.loc[:, data.select_dtypes(include=np.number).columns] = data_filled

    return data


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


def process(data1):

    week_factor_mapping_categories = [0.1, 0.2, 0.3, 0.7, 1]
    encoder = OneHotEncoder(sparse_output=False, categories=[week_factor_mapping_categories])
    scaler = StandardScaler()
    encoded_features = encoder.fit_transform(data1[['星期因子映射']])
    scaled_features = scaler.fit_transform(
        data1[['辐射强度', '温度 °C ', 'T(h-1)', '相对湿度%',
               # 'L(h-1)', 'L(h-24)', '滚动平均负荷', '滚动最大负荷', '滚动最小负荷'
               ]])
    # print(scaled_features.shape)
    # poly = PolynomialFeatures(degree=3)
    # X1 = poly.fit_transform(scaled_features)
    X1 = data1[['辐射强度', '温度 °C ', 'T(h-1)', '相对湿度%',
                'L(h-1)', 'L(h-24)',
                '滚动平均负荷', '滚动最大负荷', '滚动最小负荷'
                ]]
    encoded_features = data1[['日期因子', '星期因子映射']]
    X1 = np.hstack((X1, encoded_features))


    return X1



def save_parameters(parameters):
    with open(parameters_file, 'wb') as f:
        pickle.dump(parameters, f)

def load_parameters():
    with open(parameters_file, 'rb') as f:
        return pickle.load(f)

def sarima(data, data_test, test_size, train=True):
    # 选择SARIMA模型的参数
    p, d, q = 1, 1, 1  # 非季节性ARIMA参数
    P, D, Q, s = 1, 1, 1, 24  # 季节性参数，这里假设为一天的小时数
    if train:
        # 定义p, d, q的范围
        p = d = q = range(0, 3)  # 根据需要调整这些范围
        pdq = list(itertools.product(p, d, q))

        # 季节性参数的范围
        seasonal_pdq = [(x[0], x[1], x[2], 24) for x in pdq]  # 季节周期设置为24小时

        # 分割数据集为训练集和测试集，这里暂且认为最后24个观测为测试集
        train = data['逐时负荷/kWh'][:-24]
        test = data['逐时负荷/kWh'][-24:]

        best_aic = float("inf")
        best_pdq = None
        best_seasonal_pdq = None
        best_model = None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit(disp=False)

                    # 如果当前模型的AIC更低，那么就保存该模型
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_pdq = param
                        best_seasonal_pdq = param_seasonal
                        best_model = results
                except Exception as e:
                    continue  # 如果模型无法拟合则继续尝试其他参数组合

        print('最佳SARIMA模型的参数组合是:')
        print('ARIMA参数:', best_pdq)
        print('季节性参数:', best_seasonal_pdq)
        print('该模型的AIC是:', best_aic)

        # 使用最佳模型进行预测
        predictions = best_model.get_prediction(start=pd.to_datetime(test.index[0]), end=pd.to_datetime(test.index[-1]),
                                                dynamic=False)
        pred_values = predictions.predicted_mean

        # 计算预测的MSE
        mse = mean_squared_error(test, pred_values)
        print('预测的均方误差是:', mse)
        # 保存最佳参数
        best_parameters = {
            'pdq': best_pdq,
            'seasonal_pdq': best_seasonal_pdq,
            'aic': best_aic
        }
        save_parameters(best_parameters)

    else:
        if os.path.exists(parameters_file):  # 检查参数文件是否存在
            # 加载已保存的最佳参数
            best_parameters = load_parameters()
        else:
            print("未找到参数文件，请运行网格搜索以找到最佳参数。")
            return

    best_pdq = best_parameters['pdq']
    best_seasonal_pdq = best_parameters['seasonal_pdq']
    # 使用最佳参数拟合模型
    best_model = sm.tsa.statespace.SARIMAX(data['逐时负荷/kWh'],
                                           order=best_pdq,
                                           seasonal_order=best_seasonal_pdq,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False).fit(disp=False)
    # 拟合模型
    sarima_results = best_model.fit(disp=False)

    # 进行预测，这里预测最后几天
    # 假设您想预测最后72小时的数据
    num_hours_to_predict = test_size
    sarima_forecast = sarima_results.get_forecast(steps=num_hours_to_predict)

    # 获取预测的均值
    forecast = sarima_forecast.predicted_mean

    # 获取预测的置信区间
    conf_int = sarima_forecast.conf_int()

    # 将预测结果与实际值进行比较，这假设您有最后几天的实际值可用于比较
    # 如果您没有实际值，则可以跳过这一步
    actual_values = data_test['逐时负荷/kWh']  # 您需要提供这部分数据
    plt.figure(figsize=(10, 5))
    plt.plot(actual_values.index, actual_values, label='Actual')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.legend()
    plt.show()
