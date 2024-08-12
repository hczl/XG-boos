from tsai.all import *

# 获取和处理时间序列数据
ts = get_forecasting_time_series("Sunspots").values
X, y = SlidingWindow(60, horizon=1)(ts)

# 划分数据集
splits = TimeSplitter(235)(y)

# 设置转换
tfms = [None, TSForecasting()]
batch_tfms = TSStandardize()

# 创建和训练模型
fcst = TSForecaster(X, y, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=512, arch="TSTPlus", metrics=mae, cbs=ShowGraph())
fcst.fit_one_cycle(50, 1e-3)

# 保存模型
fcst.export("fcst.pkl")