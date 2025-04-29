import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def time_features(dates: pd.DataFrame, timeenc=1, freq='d') -> np.ndarray:
    """
    Generate simple time features for daily frequency: month, day, weekday.
    `dates` is a DataFrame with a 'date' column of datetime-like values.
    Returns an array of shape (len(dates), 3).
    """
    dates = dates.copy()
    dates['date'] = pd.to_datetime(dates['date'])
    return dates[['date']].assign(
        month=lambda x: x['date'].dt.month,
        day=lambda x: x['date'].dt.day,
        weekday=lambda x: x['date'].dt.weekday
    )[['month', 'day', 'weekday']].values


class Dataset_train(Dataset):
    """
    Sliding-window dataset for training/validation/testing with time-based split.
    Splits the full series into train/val/test by date, then produces
    (seq_len, label_len+pred_len) windows.
    """
    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: list = [84, 21, 21],
        features: str = 'MS',
        target: str = 'close',
        scale: bool = True,
        timeenc: int = 1,
        freq: str = 'd',
        cols: list = None,
        split: str = 'train',        # 'train' | 'val' | 'test'
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
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
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self._read_data()

    def _read_data(self):
        # 1) Load and sort
        file_full = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(file_full, parse_dates=['date'])
        df.sort_values('date', inplace=True)
        n = len(df)

        # 2) Determine split indices
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # 3) Determine feature columns (consistent across splits)
        if self.cols:
            selected = list(self.cols)
            if self.target in selected:
                selected.remove(self.target)
        else:
            selected = [c for c in df.columns if c not in ['date', self.target]]
        if self.features in ('M', 'MS'):
            data_cols = selected + [self.target]
        else:
            data_cols = [self.target]

        # 4) Fit scaler on training portion only
        if self.scale:
            train_df = df.iloc[:train_end]
            self.scaler = StandardScaler()
            self.scaler.fit(train_df[data_cols].values)
        else:
            self.scaler = None

        # 5) Slice df according to split
        if self.split == 'train':
            df_split = df.iloc[:train_end]
        elif self.split == 'val':
            df_split = df.iloc[train_end:val_end]
        else:
            df_split = df.iloc[val_end:]

        # 6) Extract and scale data
        values = df_split[data_cols].values
        if self.scale:
            values = self.scaler.transform(values)
        self.data = values

        # 7) Time features
        df_stamp = df_split[['date']]
        self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

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
    Dataset for one-shot future prediction: uses full series to forecast next pred_len points.
    Provides inverse_transform to map back to original scale.
    """
    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: list = [84, 21, 21],
        features: str = 'MS',
        target: str = 'close',
        scale: bool = True,
        timeenc: int = 1,
        freq: str = 'd',
        cols: list = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
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
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self._read_data()

    def _read_data(self):
        # 1) Load and sort
        file_full = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(file_full, parse_dates=['date'])
        df.sort_values('date', inplace=True)
        n = len(df)

        # 2) Determine split index for scaler fit
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # 3) Determine feature columns
        if self.cols:
            selected = list(self.cols)
            if self.target in selected:
                selected.remove(self.target)
        else:
            selected = [c for c in df.columns if c not in ['date', self.target]]
        if self.features in ('M', 'MS'):
            data_cols = selected + [self.target]
        else:
            data_cols = [self.target]

        # 4) Fit scaler on training portion only
        if self.scale:
            train_df = df.iloc[:train_end]
            self.scaler = StandardScaler()
            self.scaler.fit(train_df[data_cols].values)
        else:
            self.scaler = None

        # 5) Transform full series
        values = df[data_cols].values
        if self.scale:
            values = self.scaler.transform(values)
        self.data = values

        # 6) Time features
        self.df_stamp = df['date'].reset_index(drop=True)
        self.data_stamp = time_features(df[['date']], timeenc=self.timeenc, freq=self.freq)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Last seq_len history
        s_begin = len(self.data) - self.seq_len
        seq_x = self.data[s_begin:]
        seq_x_mark = self.data_stamp[s_begin:]
        # Decoder input: last label_len history + zeros for pred_len
        hist = self.data[-self.label_len:]
        zeros = np.zeros((self.pred_len, self.data.shape[1]), dtype=self.data.dtype)
        seq_y = np.vstack([hist, zeros])
        hist_dates = list(self.df_stamp.iloc[-self.label_len:].values)
        last_date = self.df_stamp.iloc[-1]
        future_dates = pd.date_range(last_date, periods=self.pred_len+1, freq=self.freq.upper())[1:]
        all_dates = pd.DataFrame({'date': hist_dates + list(future_dates)})
        seq_y_mark = time_features(all_dates, timeenc=self.timeenc, freq=self.freq)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions or true values back to original scale.
        `data` shape should be (N, D) or (N,)
        """
        if self.scale and self.scaler is not None:
            # if data is 1D, reshape
            arr = data if data.ndim>1 else data.reshape(-1,1)
            full = np.zeros((arr.shape[0], len(self.scaler.scale_)))
            # assume target is last column
            full[:, -1] = arr[:, -1] if arr.shape[1]>1 else arr.flatten()
            inv = self.scaler.inverse_transform(full)
            return inv[:, -1]
        return data
