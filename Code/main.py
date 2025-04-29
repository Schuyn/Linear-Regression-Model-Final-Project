# main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DataPreprossessing import Dataset_train
from Encoder import Encoder  # 简化版多尺度 Encoder

def main():
    # 环境与路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    code_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(code_dir, os.pardir, 'Data'))
    csv_file = 'nvidia_stock_2015_to_2024.csv'

    dataset = Dataset_train(
        root_path=data_dir,        # 目录到 Data 文件夹
        data_path=csv_file,        # 只写文件名
        size=[84, 21, 21],         # seq_len, label_len, pred_len
        features='MS',             # 多元时序
        target='close',            # 以哪一列作为预测目标
        scale=True,                # 是否做 StandardScaler
        timeenc=1,                 # 时间编码方式
        freq='d'                   # 日频
    )

    # 用 DataLoader 打包
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )

    # 5. 示范取一个 batch
    for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
        print("x shape: ", seq_x.shape)         # [B, seq_len, D]
        print("y shape:", seq_y.shape)         # [B, label_len+pred_len, D]
        print("X time features:", seq_x_mark.shape)   # [B, seq_len, T]
        print("Y time features:", seq_y_mark.shape)   # [B, label_len+pred_len, T]
        break

if __name__ == '__main__':
    main()
    


