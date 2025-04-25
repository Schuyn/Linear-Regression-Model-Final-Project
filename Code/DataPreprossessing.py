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

class DatasetTrain(Dataset):
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
        self.data_stamp = data_stamp.values

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