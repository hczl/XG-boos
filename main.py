import math
import operator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import scipy.stats as st
from util import preprocess, date_transform, get_unseen_data, bucket_avg, config_plot

config_plot()

def xgb_data_split(df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols):
    # generate unseen data
    unseen = get_unseen_data(unseen_start_date, steps,
                             encode_cols, bucket_size)
    df = pd.concat([df, unseen], axis=0)
    df = date_transform(df, encode_cols)

    # data for forecast ,skip the connecting point
    df_unseen = df[unseen_start_date:].iloc[:, 1:]
    test_start = '2010-11-26 00:00:00'
    # skip the connecting point
    df_test = df[test_start_date: unseen_start_date].iloc[:-1, :]
    df_train = df[:test_start_date]
    return df_unseen, df_test, df_train


def xgb_importance(df, test_ratio, xgb_params, ntree, early_stop, plot_title):
    df = pd.DataFrame(df)
    # split the data into train/test set
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_ratio,
                                                        random_state=42)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'validate')]

    xgb_model = xgb.train(xgb_params, dtrain, ntree, evals=watchlist,
                          early_stopping_rounds=early_stop, verbose_eval=True)

    importance = xgb_model.get_fscore()
    importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
    feature_importance_plot(importance_sorted, plot_title)


def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    # df.plot()
    df.plot(kind='barh', x='feature', y='fscore',
            legend=False, figsize=(12, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.show()


def xgb_forecasts_plot(plot_start, Y, Y_test, Y_hat, forecasts, title):
    Y = pd.concat([Y, Y_test])
    ax = Y[plot_start:].plot(label='observed', figsize=(15, 10))
    # Y_test.plot(label='test_observed', ax=ax)
    # 由于 Y_hat 是 numpy 数组，需要将其转换为 pandas Series
    # 绘制测试集上的预测值
    print(Y_hat)
    Y_hat.plot(label='Predicted', ax=ax)
    forecasts.plot(label="forecast", ax=ax)

    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(Y_test.index[0]), Y_test.index[-1],
                     alpha=.1, zorder=-1)
    print(ax.get_ylim())
    ax.set_xlabel('Time')
    ax.set_ylabel('Global Active Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.show()


# 加载逐时负荷数据集
def load_hourly_load_data(filename):
    return pd.read_excel(filename)


# 计算评估指标
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mse, rmse, mape

N_rows = 18000
parse_dates = [['Date', 'Time']]
filename = "household_power_consumption.txt"
encode_cols = ['Month', 'DayofWeek', 'Hour']

df = preprocess(N_rows, parse_dates, filename)
# keep all features
df = date_transform(df, encode_cols)
val_ratio = 0.3
ntree = 300
early_stop = 50
# base parameters
xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',  # regression task
    'subsample': 0.80,  # 80% of data to grow trees and prevent overfitting
    'colsample_bytree': 0.85,  # 85% of features used
    'eta': 0.1,
    'max_depth': 10,
    'seed': 42}  # for reproducible results


fig_allFeatures = xgb_importance(
    df, val_ratio, xgb_params, ntree, early_stop, 'All Features')
plt.show()



test_start_date = '2010-11-25 20:00:00'
unseen_start_date = '2010-11-26 21:10:00'
steps = 200
bucket_size = "5T"

df = preprocess(N_rows, parse_dates, filename)
G_power = df["Global_active_power"]

df = pd.DataFrame(bucket_avg(G_power, bucket_size))
df.dropna(inplace=True)

df_unseen, df_test, df = xgb_data_split(
    df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols)
xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,  # 初始估计器的数量
    seed=42  # 随机种子
)
# 假设我们对家庭总有功功率进行预测
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# 定义超参数的搜索空间
param_distributions = {
    'max_depth': randint(3, 10),  # 树的最大深度
    'learning_rate': np.linspace(0.01, 0.3, 30),  # 学习率
    'subsample': np.linspace(0.5, 1.0, 50),  # 子样本的比例
    'colsample_bytree': np.linspace(0.5, 1.0, 50),  # 构建树时列的子样本比例
    "min_child_weight": st.expon(0, 50),
    'gamma': np.linspace(0, 5, 50)  # 最小损失减少量
}

# 划分训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=val_ratio,
                                                  random_state=42)
X_test = xgb.DMatrix(df_test.iloc[:, 1:])
y_test = df_test.iloc[:, 0]
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
watchlist = [(dtrain, 'train'), (dval, 'validate')]

# xg_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
# 进行预测
random_search = RandomizedSearchCV(
    estimator=xg_reg,
    param_distributions=param_distributions,
    n_iter=20,  # 随机搜索迭代次数
    cv=5,  # 交叉验证的折数
    scoring='neg_mean_squared_error',  # 使用负均方误差作为评分标准
    random_state=42,
    verbose=1
)
# 执行搜索
random_search.fit(X_train, y_train)
print(random_search.best_params_)
# 打印最佳参数和分数
print("Best parameters found: ", random_search.best_params_)
print("Best score found: ", -random_search.best_score_)
params_new = {**xgb_params, **random_search.best_params_}
xg_reg_best = random_search.best_estimator_
xg_reg=xgb.train(params_new,dtrain, evals=watchlist,
                        early_stopping_rounds=early_stop, verbose_eval=True)
y_pred = xg_reg.predict(X_test)

# # 使用最佳参数重新训练模型
# xg_reg_best = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     colsample_bytree=0.8367346938775511,
#     gamma=.9183673469387755,
#     learning_rate=0.3,
#     max_depth=6,
#     subsample=0.6326530612244898,
#     n_estimators=100,  # 初始估计器的数量
#     seed=42  # 随机种子
# )
# xg_reg_best.fit(X_train, y_train, eval_set=[(X_val, y_val)])

xg_reg_best=xgb.train(params_new, dtrain, evals=watchlist,
                        early_stopping_rounds=early_stop, verbose_eval=True)
# 用最佳参数模型进行预测
y_pred_best = xg_reg_best.predict(X_test)
# 计算评估指标
mae, mse, rmse, mape = calculate_metrics(y_test, y_pred_best)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}")

# 获取特征重要性
importance = xg_reg_best.get_fscore()
importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
# 分离特征名和重要性值
sorted_features = [item[0] for item in importance_sorted]
sorted_importances = [item[1] for item in importance_sorted]

# # ________________
# booster = xg_reg_best.get_booster()
# importance_dict = booster.get_score(importance_type='weight')
#
# # 转换为条形图所需的格式
# importance_df = pd.DataFrame(
#     {'Feature': list(importance_dict.keys()),
#      'Importance': list(importance_dict.values())}
# )
#
# # 对特征重要性进行排序
# importance_df.sort_values(by='Importance', ascending=True, inplace=True)
#
# # 绘制条形图
# plt.figure(figsize=(10, 8))
# plt.barh(importance_df['Feature'], importance_df['Importance'])
# plt.xlabel('Relative Importance')
# plt.title('XGBoost Feature Importance')
# # plt.show()

# 可视化预测结果

X_unseen = xgb.DMatrix(df_unseen)
unseen_y = xg_reg_best.predict(X_unseen)
forecasts = pd.DataFrame(
    unseen_y, index=df_unseen.index, columns=["forecasts"])
plot_start = '2010-11-24 00:00:00'
print(y_pred_best.shape)
y_pred_best = pd.DataFrame(y_pred_best, index=y_test.index, columns=["test_predicted"])
xgb_forecasts_plot(
    plot_start, y, y_test, y_pred_best, forecasts, 'Initial Model')
