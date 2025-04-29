# main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataPreprossessing import Dataset_train
from Encoder import EncoderStack
from embed import SimpleEmbedding

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
    
    # 定义嵌入层：将原始特征投射到 d_model
    c_in = dataset.data.shape[1]
    d_model = 256
    embed = SimpleEmbedding(c_in, d_model).to(device)  
    
    n_heads = 8
    d_ff = 4 * d_model
    encoder = EncoderStack(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=3,
        distill_levels=[0, 1, 2],
        dropout=0.1,
        factor=5
    ).to(device)
    
    for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
        # seq_x: (B, 84, c_in), seq_x_mark: (B, 84, 3)
        x_val  = seq_x.float().to(device)
        x_time = seq_x_mark.float().to(device)
        print(f"Input shape - x_val: {x_val.shape}, x_time: {x_time.shape}")
        
        # 嵌入
        x_emb = embed(x_val, x_time)           # → (B, 84, d_model)
        print(f"Embedding shape: {x_emb.shape}")
        
        # EncoderStack 前向
        enc_out = encoder(x_emb)               # → (B, 84 + 42 + 21, d_model)

        print("Embedding output shape: ", x_emb.shape)
        print("EncoderStack output shape:", enc_out.shape)
        break

if __name__ == '__main__':
    main()
    


