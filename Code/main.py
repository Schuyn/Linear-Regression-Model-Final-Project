# main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataPreprossessing import Dataset_train
from Encoder import EncoderStack
from embed import SimpleEmbedding
from Decoder import SimpleDecoder

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
    
    pred_len = dataset.pred_len
    decoder = SimpleDecoder(
        pred_len=pred_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=2,
        factor=5,
        dropout=0.1
    ).to(device)
    

    # 优化器和损失函数
    params = list(embed.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    criterion = torch.nn.MSELoss()
    epochs = 20

    # 训练循环
    for epoch in range(1, epochs + 1):
        embed.train(); encoder.train(); decoder.train()
        total_loss = 0.0
        for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
            x_val = seq_x.float().to(device)                    # (B,84,c_in)
            x_time = seq_x_mark.float().to(device)               # (B,84,3)
            # 嵌入 + 编码
            x_emb = embed(x_val, x_time)                         # (B,84,d_model)
            enc_out = encoder(x_emb)                             # (B,147,d_model)

            # 解码：取未来时间特征
            x_time_dec = seq_y_mark[:, -pred_len:, :].float().to(device)  # (B,21,3)
            pred = decoder(enc_out, x_time_dec)                  # (B,21)

            # 真实标签：收盘价为最后一列
            true_y = seq_y[:, -pred_len:, -1].float().to(device) # (B,21)

            loss = criterion(pred, true_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

    # 保存模型
    torch.save({
        'embed': embed.state_dict(),
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
    }, 'model_epoch.pth')
    print("train completed! see more in: model_epoch.pth")


if __name__ == '__main__':
    main()
    


