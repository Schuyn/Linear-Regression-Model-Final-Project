import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

def predict_and_plot(data_dir, csv_file, save_dir, pred_len, device, 
                        pred_ds, embed, encoder, decoder):
    """
    执行预测并绘制结果图表
    """
    # 加载最佳模型
    embed.load_state_dict(torch.load('best_model.pth')['embed'])
    encoder.load_state_dict(torch.load('best_model.pth')['encoder'])
    decoder.load_state_dict(torch.load('best_model.pth')['decoder'])
    
    # 设置为评估模式
    embed.eval(); encoder.eval(); decoder.eval()
    
    # 预测
    with torch.no_grad():
        x, _, x_mark, y_mark = next(iter(DataLoader(pred_ds, batch_size=1)))
        x = x.float().to(device); xm = x_mark.float().to(device)
        x_emb = embed(x, xm)
        enc_out = encoder(x_emb)
        dec_time = y_mark[:, -pred_len:, :].float().to(device)
        future_pred = decoder(enc_out, dec_time).cpu().numpy().reshape(-1,1)
        # print("预测值 (归一化后):", future_pred[:5]) 
        
        # 反归一化
        future_price = pred_ds.inverse_transform(future_pred).reshape(-1)
    
    # 绘图
    plt.figure(figsize=(12,6))
    days = range(1, pred_len+1)
    plt.plot(days, future_price, marker='o', linewidth=2, markersize=8)
    
    # 添加数值标注
    for i, price in enumerate(future_price):
        plt.annotate(f'{price:.2f}', 
                    xy=(days[i], price),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8)
    
    plt.title('Future 21 Days of Stock Price Prediction of Nvidia', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(days)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'Pred21DayPrice.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved: {save_path}")
    plt.show()
    
    return future_price