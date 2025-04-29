import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from DataPreprossessing import Dataset_train, Dataset_prediction
from embed import SimpleEmbedding
from Encoder import EncoderStack
from Decoder import SimpleDecoder
from Prediction import predict_and_plot

def main():
    # 环境配置及路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, os.pardir, 'Data'))
    csv_file = 'nvidia_stock_1999_to_2025.csv'

    # 划分训练/验证/测试数据集
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

    # 模型构建
    c_in = train_ds.data.shape[1]
    d_model = 256
    embed = SimpleEmbedding(c_in, d_model).to(device)
    encoder = EncoderStack(d_model=d_model, n_heads=4, d_ff=4*d_model,
                           num_layers=3, distill_levels=[0,1,2], dropout=0.1,
                           factor=5).to(device)
    pred_len = train_ds.pred_len
    decoder = SimpleDecoder(pred_len=pred_len, d_model=d_model,
                             n_heads=4, d_ff=8*d_model,
                             num_layers=4, factor=5, dropout=0.1).to(device)

    # 优化器与学习率调度
    params = list(embed.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    # 每个 epoch lr 缩半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()
    epochs = 8
    early_stop_patience = 3

    # 日志
    train_losses = []
    best_val = float('inf')
    early_count = 0

    # 训练-验证循环
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
        # print("Example target:", y_target[0].cpu().numpy())

    # 测试集滑窗评估
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
    # print(f"Final Test MSE: {test_mse:.6f}")

    save_dir = os.path.join(base_dir, os.pardir, 'Report', 'Latex', 'Image')
    os.makedirs(save_dir, exist_ok=True)

    # 画出 Train 错误曲线
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train MSE')
    plt.title('Training MSE over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    save_path = os.path.join(save_dir, 'TrainingMSE.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved: {save_path}")
    plt.show()

    # 未来 21 天预测与画图
    pred_ds = Dataset_prediction(
        root_path=data_dir, 
        data_path=csv_file,
        size=[84,21,21], 
        features='MS', 
        target='close',
        scale=True, 
        timeenc=1, 
        freq='d'
    )
    
    # 调用预测函数
    future_price = predict_and_plot(
        data_dir=data_dir,
        csv_file=csv_file,
        save_dir=save_dir,
        pred_len=pred_len,
        device=device,
        pred_ds=pred_ds,
        embed=embed,
        encoder=encoder,
        decoder=decoder
    )
    
if __name__ == '__main__':
    main()
