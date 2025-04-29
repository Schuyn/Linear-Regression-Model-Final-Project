import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from DataPreprossessing import Dataset_train, Dataset_prediction
from embed import SimpleEmbedding
from Encoder import EncoderStack
from Decoder import SimpleDecoder

def main():
    # 环境配置及路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, os.pardir, 'Data'))
    csv_file = 'nvidia_stock_2015_to_2024.csv'

    # 1) 划分训练/验证/测试数据集
    train_ds = Dataset_train(root_path=data_dir, data_path=csv_file,
                             size=[84,21,21], features='MS', target='close',
                             scale=True, timeenc=1, freq='d', split='train')
    val_ds   = Dataset_train(root_path=data_dir, data_path=csv_file,
                             size=[84,21,21], features='MS', target='close',
                             scale=True, timeenc=1, freq='d', split='val')
    test_ds  = Dataset_train(root_path=data_dir, data_path=csv_file,
                             size=[84,21,21], features='MS', target='close',
                             scale=True, timeenc=1, freq='d', split='test')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, drop_last=False)

    # 2) 模型构建
    c_in = train_ds.data.shape[1]
    d_model = 256
    embed = SimpleEmbedding(c_in, d_model).to(device)
    encoder = EncoderStack(d_model=d_model, n_heads=4, d_ff=4*d_model,
                           num_layers=3, distill_levels=[0,1,2], dropout=0.1,
                           factor=5).to(device)
    pred_len = train_ds.pred_len
    decoder = SimpleDecoder(pred_len=pred_len, d_model=d_model,
                             n_heads=4, d_ff=4*d_model,
                             num_layers=2, factor=5, dropout=0.1).to(device)

    # 3) 优化器与学习率调度
    params = list(embed.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    # 每个 epoch lr 缩半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    criterion = nn.MSELoss()
    epochs = 80
    early_stop_patience = 3

    # 日志
    train_losses = []
    best_val = float('inf')
    early_count = 0

    # 4) 训练-验证循环
    for epoch in range(1, epochs+1):
        # 训练
        embed.train(); encoder.train(); decoder.train()
        total_train = 0.0
        for x, y, x_mark, y_mark in train_loader:
            x = x.float().to(device); xm = x_mark.float().to(device)
            y_target = y[:, -pred_len:, -1].float().to(device)
            # 前向
            x_emb = embed(x, xm)
            enc_out = encoder(x_emb)
            dec_time = y_mark[:, -pred_len:, :].float().to(device)
            pred = decoder(enc_out, dec_time)
            loss = criterion(pred, y_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        # 验证
        embed.eval(); encoder.eval(); decoder.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, y, x_mark, y_mark in val_loader:
                x = x.float().to(device); xm = x_mark.float().to(device)
                y_target = y[:, -pred_len:, -1].float().to(device)
                x_emb = embed(x, xm)
                enc_out = encoder(x_emb)
                dec_time = y_mark[:, -pred_len:, :].float().to(device)
                pred = decoder(enc_out, dec_time)
                total_val += criterion(pred, y_target).item()
        avg_val = total_val / len(val_loader)

        # 学习率更新 & 早停检查
        scheduler.step()
        if avg_val < best_val:
            best_val = avg_val
            early_count = 0
            # 保存最佳模型
            torch.save({'embed': embed.state_dict(),
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict()}, 'best_model.pth')
        else:
            early_count += 1
            if early_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}/{epochs} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 5) 测试集滑窗评估
    embed.eval(); encoder.eval(); decoder.eval()
    total_test = 0.0
    with torch.no_grad():
        for x, y, x_mark, y_mark in test_loader:
            x = x.float().to(device); xm = x_mark.float().to(device)
            y_target = y[:, -pred_len:, -1].float().to(device)
            x_emb = embed(x, xm)
            enc_out = encoder(x_emb)
            dec_time = y_mark[:, -pred_len:, :].float().to(device)
            pred = decoder(enc_out, dec_time)
            total_test += criterion(pred, y_target).item()
    test_mse = total_test / len(test_loader)
    print(f"Final Test MSE: {test_mse:.6f}")

    # 6) 画出 Train 错误曲线
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train MSE')
    plt.title('Train MSE over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid(True); plt.legend(); plt.show()

    # 7) 未来 21 天预测与画图
    pred_ds = Dataset_prediction(root_path=data_dir, data_path=csv_file,
                                 size=[84,21,21], features='MS', target='close',
                                 scale=True, timeenc=1, freq='d')
    embed.load_state_dict(torch.load('best_model.pth')['embed'])
    encoder.load_state_dict(torch.load('best_model.pth')['encoder'])
    decoder.load_state_dict(torch.load('best_model.pth')['decoder'])
    embed.eval(); encoder.eval(); decoder.eval()
    with torch.no_grad():
        x, _, x_mark, y_mark = next(iter(DataLoader(pred_ds, batch_size=1)))
        x = x.float().to(device); xm = x_mark.float().to(device)
        x_emb = embed(x, xm)
        enc_out = encoder(x_emb)
        dec_time = y_mark[:, -pred_len:, :].float().to(device)
        future_pred = decoder(enc_out, dec_time).cpu().numpy().reshape(-1,1)
        # 反归一化
        future_price = pred_ds.inverse_transform(future_pred)
    # 图示
    plt.figure(figsize=(8,5))
    plt.plot(range(1, pred_len+1), future_price, marker='o')
    plt.title('Future 21-day Stock Price Prediction')
    plt.xlabel('Day'); plt.ylabel('Price (USD)'); plt.grid(True); plt.show()

if __name__ == '__main__':
    main()
