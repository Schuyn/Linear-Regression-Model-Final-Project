import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

def predict_and_plot(data_dir, csv_file, save_dir, pred_len, device, 
                        pred_ds, embed, encoder, decoder):
    """
   Perform prediction and plot the result chart
    """
    # Modify the model loading path
    base_dir = os.path.dirname(__file__)
    result_dir = os.path.abspath(os.path.join(base_dir, os.pardir, 'Result'))
    model_path = os.path.join(result_dir, 'best_model.pth')

    # Load the Model
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        embed.load_state_dict(checkpoint['embed'])
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        print(f"Load successfully：{model_path}")
    else:
        print(f"Unanble to laod：{model_path}")
    
    # Set to evaluation mode
    embed.eval(); encoder.eval(); decoder.eval()
    
    # Prediction
    with torch.no_grad():
        x, _, x_mark, y_mark = next(iter(DataLoader(pred_ds, batch_size=1)))
        x = x.float().to(device); xm = x_mark.float().to(device)
        x_emb = embed(x, xm)
        enc_out = encoder(x_emb)
        dec_time = y_mark[:, -pred_len:, :].float().to(device)
        future_pred = decoder(enc_out, dec_time).cpu().numpy().reshape(-1,1)
        # print("Predicted values (after normalization):", future_pred[:5]) 
        
        # Denormalization 
        future_price = pred_ds.inverse_transform(future_pred).reshape(-1)
    
    # Plot
    plt.figure(figsize=(12,6))
    days = range(1, pred_len+1)
    plt.plot(days, future_price, marker='o', linewidth=2, markersize=8)
    
    # Add value annotations
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
