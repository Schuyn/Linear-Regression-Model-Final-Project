# import sys
# print("当前 Python 可执行文件：", sys.executable)

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np


def time_features(dates, timeenc=1, freq='d'):
    """
    Generate simple time features for daily frequency: month, day, weekday.
    `dates` is a DataFrame with a 'date' column of datetime-like values.
    Returns an array of shape (len(dates), 3).
    """
    dates = dates.copy()
    dates['date'] = pd.to_datetime(dates['date'])
    dates['month'] = dates['date'].dt.month
    dates['day'] = dates['date'].dt.day
    dates['weekday'] = dates['date'].dt.weekday
    # Return features in order: month, day, weekday
    return dates[['month', 'day', 'weekday']].values

class Dataset_train(Dataset):
    """
    Dataset for training: sliding-window monthly forecasting with seq_len=84, label_len=21, pred_len=21
    """
    def __init__(
        self,
        root_path,
        data_path,
        size=[84, 21, 21],
        features='MS',
        target='close',
        scale=True,
        timeenc=1,
        freq='d',
        cols=None
    ):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self._read_data()

    def _read_data(self):
        # Load and sort
        file_full = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(file_full, parse_dates=['date'])
        # Ensure 'date' is in datetime format
        df.sort_values('date', inplace=True)

        # Select features and target columns
        if self.cols:
            selected = self.cols.copy()
            if self.target in selected:
                selected.remove(self.target)
        else:
            selected = [c for c in df.columns if c not in ['date', self.target]]
        df = df[['date'] + selected + [self.target]]

        # Build data array
        if self.features in ('M', 'MS'):
            data_cols = selected + [self.target]
            data = df[data_cols].values
        else:  # 'S'
            data = df[[self.target]].values

        # Scale
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data = self.scaler.transform(data)
        self.data = data

        # Time features
        df_stamp = df[['date']]
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        self.data_stamp = data_stamp

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
class Dataset_prediction(Dataset):
    """
    Dataset for prediction: use the last seq_len points to forecast the next pred_len.
    Returns (seq_x, seq_y, seq_x_mark, seq_y_mark) where
      - seq_x:  [seq_len, D]  encoder 输入
      - seq_y:  [label_len+pred_len, D]  decoder 输入（后半段为 0）
      - seq_x_mark: [seq_len, 3]  encoder 时间特征
      - seq_y_mark: [label_len+pred_len, 3] decoder 时间特征
    """
    def __init__(
        self,
        root_path,
        data_path,
        size=[84, 21, 21],
        features='MS',
        target='close',
        scale=True,
        timeenc=1,
        freq='d',
        cols=None
    ):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self._read_data()

    def _read_data(self):
        import pandas as pd

        file_full = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(file_full, parse_dates=['date'])
        df.sort_values('date', inplace=True)

        # —— 列选择逻辑同 DatasetTrain
        if self.cols:
            selected = self.cols.copy()
            if self.target in selected:
                selected.remove(self.target)
        else:
            selected = [c for c in df.columns if c not in ['date', self.target]]
        if self.features in ('M', 'MS'):
            df = df[['date'] + selected + [self.target]]
            data = df[selected + [self.target]].values
        else:
            df = df[['date', self.target]]
            data = df[[self.target]].values

        # —— 标准化
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data = self.scaler.transform(data)
        self.data = data

        # —— 原始日期序列备份 & 时间特征
        self.df_stamp = df['date'].reset_index(drop=True)
        self.data_stamp = time_features(df[['date']], timeenc=self.timeenc, freq=self.freq)

    def __len__(self):
        # 预测时只返回一条样本
        return 1

    def __getitem__(self, idx):
        import pandas as pd

        # —— encoder 输入：最后 seq_len 条
        s_begin = len(self.data) - self.seq_len
        s_end = len(self.data)
        seq_x = self.data[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        # —— decoder 输入：先拿最后 label_len 条，再 pad pred_len 条 0
        dec_in_hist = self.data[s_end - self.label_len:s_end]
        zeros = np.zeros((self.pred_len, self.data.shape[1]), dtype=self.data.dtype)
        seq_y = np.vstack([dec_in_hist, zeros])

        # —— decoder 时间特征：历史后 label_len + 未来 pred_len
        #    先取历史时间戳
        hist_dates = list(self.df_stamp.iloc[-self.label_len:].values)
        #    再生成未来时间戳
        freq_pd = self.freq.upper()  # 'd' -> 'D', 'h'-> 'H'
        last_date = self.df_stamp.iloc[-1]
        future_dates = pd.date_range(last_date, periods=self.pred_len + 1, freq=freq_pd)[1:]
        all_dates = pd.DataFrame({'date': hist_dates + list(future_dates)})

        seq_y_mark = time_features(all_dates, timeenc=self.timeenc, freq=self.freq)

        return seq_x, seq_y, seq_x_mark, seq_y_mark