# main.py
import os
from torch.utils.data import DataLoader
# 如果类名是 DatasetTrain，请用下面这一行；如果是 Dataset_train，请把名字改成 Dataset_train
from DataPreprossessing import Dataset_train  

def main():
    # 1. 定位到 Data 文件夹
    code_dir = os.path.dirname(__file__)                                # .../Code
    data_dir = os.path.abspath(os.path.join(code_dir, os.pardir, 'Data'))  # .../Data

    # 2. 指定 CSV 文件名
    csv_file = 'nvidia_stock_2015_to_2024.csv'

    # 3. 创建 Dataset
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

    # 4. 用 DataLoader 打包
    loader = DataLoader(
        dataset,
        batch_size=32,
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