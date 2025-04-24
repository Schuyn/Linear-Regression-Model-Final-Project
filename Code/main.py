import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
data = pd.read_csv('Data/nvidia_stock_2015_to_2024.csv')

# 检查数据集大小和基本结构
print("Data shape:", data.shape)
print("Data head:\n", data.head())

# 2. 日期时间戳处理
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')  # 确保数据按时间排序
data.set_index('Date', inplace=True)



# 4. 特征选择（如有需要，可自行修改）
# 假设数据包含开盘价(Open)、收盘价(Close)、最高价(High)、最低价(Low)、成交量(Volume)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[features]

# 5. 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 保存scaler以供未来还原真实值
import joblib
joblib.dump(scaler, 'scaler.save')

# 转换回DataFrame以便后续操作
data_scaled_df = pd.DataFrame(data_scaled, columns=features, index=data.index)

# 6. 划分数据为训练集、验证集、测试集（Informer推荐划分）
total_length = len(data_scaled_df)
train_size = int(total_length * 0.7)
val_size = int(total_length * 0.1)
test_size = total_length - train_size - val_size

train_data = data_scaled_df.iloc[:train_size]
val_data = data_scaled_df.iloc[train_size:train_size + val_size]
test_data = data_scaled_df.iloc[train_size + val_size:]

# 保存预处理后的数据到csv，以备Informer模型使用
train_data.to_csv('train_data.csv')
val_data.to_csv('val_data.csv')
test_data.to_csv('test_data.csv')

# 输出划分结果
print(f'Train set length: {len(train_data)}')
print(f'Validation set length: {len(val_data)}')
print(f'Test set length: {len(test_data)}')