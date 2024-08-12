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

def main():
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





if __name__ == '__main__':
    main()