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
    # 1) 环境与路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    code_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(code_dir, os.pardir, 'Data'))
    csv_file = 'nvidia_stock_2015_to_2024.csv'

    # 2) 时间切分数据集：train/val/test
    train_ds = Dataset_train(
        root_path=data_dir,
        data_path=csv_file,
        size=[84, 21, 21], features='MS', target='close',
        scale=True, timeenc=1, freq='d', split='train'
    )
    val_ds = Dataset_train(
        root_path=data_dir,
        data_path=csv_file,
        size=[84, 21, 21], features='MS', target='close',
        scale=True, timeenc=1, freq='d', split='val'
    )
    test_ds = Dataset_train(
        root_path=data_dir,
        data_path=csv_file,
        size=[84, 21, 21], features='MS', target='close',
        scale=True, timeenc=1, freq='d', split='test'
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, drop_last=False)

    # 3) 模型组件
    c_in = train_ds.data.shape[1]
    d_model = 256
    embed = SimpleEmbedding(c_in, d_model).to(device)
    n_heads = 4
    d_ff = 4 * d_model
    encoder = EncoderStack(d_model, n_heads, d_ff, num_layers=3,
                           distill_levels=[0,1,2], dropout=0.1, factor=5).to(device)
    pred_len = train_ds.pred_len
    decoder = SimpleDecoder(pred_len, d_model, n_heads, d_ff, num_layers=2,
                             factor=5, dropout=0.1).to(device)

    # 4) 优化器、调度器、损失
    params = list(embed.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.MSELoss()
    epochs = 20

    # 5) 日志存储
    train_losses = []
    val_losses = []

    # 6) 训练+验证循环
    for epoch in range(1, epochs+1):
        # 训练
        embed.train(); encoder.train(); decoder.train()
        total_train = 0.0
        for sx, sy, sxm, sym in train_loader:
            x_val = sx.float().to(device); x_time = sxm.float().to(device)
            x_emb = embed(x_val, x_time)
            enc_out = encoder(x_emb)
            x_time_dec = sym[:, -pred_len:, :].float().to(device)
            pred = decoder(enc_out, x_time_dec)
            true_y = sy[:, -pred_len:, -1].float().to(device)
            loss = criterion(pred, true_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        # 验证
        embed.eval(); encoder.eval(); decoder.eval()
        total_val = 0.0
        with torch.no_grad():
            for sx, sy, sxm, sym in val_loader:
                x_val = sx.float().to(device); x_time = sxm.float().to(device)
                x_emb = embed(x_val, x_time)
                enc_out = encoder(x_emb)
                x_time_dec = sym[:, -pred_len:, :].float().to(device)
                pred = decoder(enc_out, x_time_dec)
                true_y = sy[:, -pred_len:, -1].float().to(device)
                total_val += criterion(pred, true_y).item()
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)

        # 调度
        scheduler.step(avg_val)

        print(f"Epoch {epoch}/{epochs} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

    # 7) 测试集滑动窗口上的 Test MSE
    embed.eval(); encoder.eval(); decoder.eval()
    total_test = 0.0
    with torch.no_grad():
        for sx, sy, sxm, sym in test_loader:
            x_val = sx.float().to(device); x_time = sxm.float().to(device)
            x_emb = embed(x_val, x_time)
            enc_out = encoder(x_emb)
            x_time_dec = sym[:, -pred_len:, :].float().to(device)
            pred = decoder(enc_out, x_time_dec)
            true_y = sy[:, -pred_len:, -1].float().to(device)
            total_test += criterion(pred, true_y).item()
    test_mse = total_test / len(test_loader)
    print(f"Final Test MSE: {test_mse:.6f}")

    # 8) 绘制 Train/Val/Test Loss
    ep = list(range(1, epochs+1))
    plt.figure(figsize=(8,5))
    plt.plot(ep, train_losses, marker='o', label='Train MSE')
    plt.plot(ep, val_losses,   marker='s', label='Val MSE')
    plt.hlines(test_mse, 1, epochs, colors='r', linestyles='--', label='Test MSE')
    plt.title('MSE over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.xticks(ep); plt.grid(True); plt.legend(); plt.show()

    # 9) 最后一次预测（未来 pred_len）
    pred_ds = Dataset_prediction(
        root_path=data_dir, data_path=csv_file,
        size=[84,21,21], features='MS', target='close',
        scale=True, timeenc=1, freq='d'
    )
    px, py, pxm, pym = next(iter(DataLoader(pred_ds, batch_size=1)))
    px_val = px.float().to(device); px_time = pxm.float().to(device)
    x_emb = embed(px_val, px_time); enc_out = encoder(x_emb)
    x_time_dec = pym[:, -pred_len:, :].float().to(device)
    future_pred = decoder(enc_out, x_time_dec)
    print("Future 21-step predictions:", future_pred.cpu().numpy())

if __name__ == '__main__':
    main()
