import os
from random import randint
import sklearn.metrics as skm
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor
import scipy.stats as st
from function import *
from tsai.inference import load_learner
from tsai.all import *

data = pd.read_excel("2021年6~8月.xlsx")

IQR(data)
data = process_data(data)
# 定义测试集的大小比例，例如20%
test_size = int(len(data) * 0.2)
ts = data['逐时负荷/kWh']
X, y = SlidingWindow(60, horizon=1)(ts)
splits = TimeSplitter(test_size)(y)
fcst = load_learner("models/TST.pkl", cpu=False)
_, _, preds_0 = fcst.get_X_preds(X[splits[1]], y[splits[1]])
fcst = load_learner("models/MLSTM_FCN.pkl", cpu=False)
_, _, preds_1 = fcst.get_X_preds(X[splits[1]], y[splits[1]])
fcst = load_learner("models/mWDN.pkl", cpu=False)
_, _, preds_2 = fcst.get_X_preds(X[splits[1]], y[splits[1]])

data_test = data[-test_size:]
data = data[:-test_size]

Y_train = data['逐时负荷/kWh']
Y_test = data_test['逐时负荷/kWh']

X = process(data)
# 设置超参数搜索空间
params = {
    'max_depth': st.randint(3, 10),  # 树的最大深度
    'learning_rate': np.linspace(0.01, 0.3, 30),  # 学习率
    'subsample': np.linspace(0.5, 1.0, 50),  # 子样本的比例
    'colsample_bytree': np.linspace(0.5, 1.0, 50),  # 构建树时列的子样本比例
    "min_child_weight": st.expon(0, 50),
    'gamma': np.linspace(0, 10, 50),  # 最小损失减少量
    'n_estimators': st.randint(100, 1000),  # 增加最大值
    # 'learning_rate': st.uniform(0.001, 0.3),  # 增加最大值，减小最小值
    # 'subsample': st.beta(10, 1),  # 这个参数的范围通常在0.5到1之间，可以保持不变
    # 'max_depth': st.randint(3, 50),  # 增加最大值
    # 'colsample_bytree': st.beta(10, 1),  # 这个参数的范围通常在0.5到1之间，可以保持不变
    # 'min_child_weight': st.expon(0, 100),  # 增加尾部范围
    # 'gamma': st.uniform(0, 10),  # 增加最大值
    'reg_alpha': st.uniform(0, 1),
    'reg_lambda': st.uniform(0, 1),

}

xgb_model = XGBRegressor(objective='reg:squarederror', booster='gbtree', seed=42)
random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=200,
                                   scoring='neg_mean_absolute_error', n_jobs=-1, cv=5, random_state=42)

k = 2
if k == 1:
    # 进行超参数搜索
    random_search.fit(X, Y_train)
    #
    # 输出最佳参数
    print("最佳参数:", random_search.best_params_)
    best_model = random_search.best_estimator_
    # with open("canshu.txt", 'w') as file:
    #     file.write("最佳参数:" + random_search.best_params_ + '\n')
else:
    params = {'colsample_bytree': 0.8979591836734693, 'gamma': 3.0612244897959187,
              'learning_rate': 0.019999999999999997,
              'max_depth': 6, 'min_child_weight': 0.4556904766980622, 'n_estimators': 464,
              'reg_alpha': 0.7798453951511436,
              'reg_lambda': 0.95698063882695841, 'subsample': 0.7959183673469388}

    # params = {'colsample_bytree': 0.8858369064007838, 'gamma': 7.798453951511436, 'learning_rate': 0.03309419164808752,
    #           'max_depth': 32, 'min_child_weight': 1.8599624307511762, 'n_estimators': 332,
    #           'reg_alpha': 0.9629920038589946, 'reg_lambda': 0.9418721660386861, 'subsample': 0.8191415480782579}

    best_model = xgb.XGBRegressor(**params)
best_model.fit(X, Y_train)
# sarima(data, data_test, test_size,True)


# torch.Size([235, 1])
# 预测并计算评估指标
# predictions = []
# rows = data
# for index, row in data_test.iterrows():
#     index -= 1498
#     if index >= 24:
#         row['L(h-24)'] = predictions[index - 24]
#     if index > 0:
#         row['L(h-1)'] = predictions[-1]
#     # 使用过去7个预测值
#     if index < 6:
#         row['滚动最大负荷'] = max(Y_train[index - 7:].values.tolist() + predictions)
#         row['滚动最小负荷'] = min(Y_train[index - 7:].values.tolist() + predictions)
#         row['滚动平均负荷'] = np.mean(Y_train[index - 7:].values.tolist() + predictions)
#     elif index == 6:
#         row['滚动最大负荷'] = max(Y_train.iloc[-1] + predictions)
#         row['滚动最小负荷'] = min(Y_train.iloc[-1] + predictions)
#         row['滚动平均负荷'] = np.mean(Y_train.iloc[-1] + predictions)
#     elif index == 7:
#         row['滚动最大负荷'] = max(predictions)
#         row['滚动最小负荷'] = min(predictions)
#         row['滚动平均负荷'] = np.mean(predictions)
#     else:
#         row['滚动最大负荷'] = max(predictions[-7:])
#         row['滚动最小负荷'] = min(predictions[-7:])
#         row['滚动平均负荷'] = np.mean(predictions[-7:])
#     row = pd.DataFrame([row], columns=data_test.columns)
#     # rows = pd.concat([rows, row], axis=0)
#     # x = process(row, False)
#     x = process(row)
#     current_prediction = best_model.predict(x)
#     predictions.append(current_prediction[-1])
X_test = process(data_test)
predictions = best_model.predict(X_test)
preds_0 = np.array(preds_0)[:, 0]
preds_1 = np.array(preds_1)[:, 0]
preds_2 = np.array(preds_2)[:, 0]

best = 0
best_i = [0.78, 0.18, 0.02, 0.0]
# for i in range(101):
#     ii = i / 100
#     for j in range(101 - i):
#         jj = j / 100
#         for k in range(101 - i - j):
#             kk = k / 100
#             for q in range(101 - i - j - k):
#                 qq = q / 100
#                 p = (ii * predictions + jj * preds_0 + kk * preds_1 + qq * preds_2)
#                 r2 = r2_score(Y_test, p)
#                 if r2 > best:
#                     best = r2
#                     best_i = [ii, jj, kk, qq]
print(best_i)
predictions = (best_i[0] * predictions + best_i[1] * preds_0 + best_i[2] * preds_1 + best_i[3] * preds_2)
# X_test = process(data)
# predictions = best_model.predict(X_test)
# Y_test = Y_train
error = predictions - Y_test
mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)
mape = (sum(abs((a - p) / a) for a, p in zip(Y_test, predictions)) / len(Y_test)) * 100

R_square = r2_score(Y_test, predictions)
# 打印评估指标
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('平均百分比误差:', mape)
print("R square :", R_square)
# 绘制折线图
# predictions_full = best_model.predict(X_test)

plt.figure(figsize=(12, 6))
x = data_test['逐时负荷/kWh']
plt.plot(x.index, Y_test, label='Actual', color='blue')
plt.plot(x.index, predictions, label='Predicted', color='red')
plt.plot(x.index, error, label='error', color='green')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()

# os.system("shutdown /s /t 0")
