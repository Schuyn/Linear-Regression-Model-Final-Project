import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# === 读取数据 ===
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, os.pardir, 'Data'))
csv_file = 'nvidia_stock_1999_to_2025.csv'
file_path = os.path.join(data_dir, csv_file)

df = pd.read_csv(file_path, parse_dates=['Date'])
df.sort_values('Date', inplace=True)

assert all(col in df.columns for col in ['Date', 'Close', 'Volume']), "数据中缺少必要字段"

# === 保存路径 ===
save_dir = r"C:\Users\Schuyn\Desktop\文件\GitHub\Linear-Regression-Model-Final-Project\Report\Latex\Image"
os.makedirs(save_dir, exist_ok=True)

# === 平滑volume（7日移动平均）===
df['volume_smooth'] = df['Volume'].rolling(window=7, min_periods=1).mean()

# === 新版绘图函数：区分长周期/短周期 ===
def plot_close_volume(df_plot, title, save_name):
    fig, ax1 = plt.subplots(figsize=(14,7))
    
    # --- 收盘价 ---
    ax1.plot(df_plot['Date'], df_plot['Close'], color='black', linewidth=2.0, label='Close Price')
    ax1.set_ylabel('Price (USD)', fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', linewidth=0.8, alpha=0.4)

    # --- 自动时间轴格式 ---
    date_span = (df_plot['Date'].max() - df_plot['Date'].min()).days
    if date_span <= 60:
        # 数据少于60天，每天一个刻度
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    else:
        # 数据跨度大，3个月一个刻度
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # --- 成交量缩放处理 ---
    ax2 = ax1.twinx()
    volume_scale = df_plot['Close'].max() * 0.2
    scaled_volume = df_plot['volume_smooth'] / df_plot['volume_smooth'].max() * volume_scale
    ax2.plot(df_plot['Date'], scaled_volume, color='blue', linewidth=1.8, alpha=0.6, label='Volume (7d MA)')
    ax2.set_ylabel('Volume (scaled)', fontsize=14, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # --- 图例合并 ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

    plt.title(title, fontsize=16, pad=15)
    plt.tight_layout()

    # 保存
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved: {save_path}")
    plt.close()

# === 画全数据集 ===
plot_close_volume(
    df,
    'NVIDIA 1999-2025 Close Price and Volume (Smoothed)',
    'nvidia_close_volume_full.png'
)

# === 画最后一个月（日频刻度）===
last_month = df['Date'].max().to_period('M')
df_last_month = df[df['Date'].dt.to_period('M') == last_month]

plot_close_volume(
    df_last_month,
    f'NVIDIA {last_month} Close Price and Volume (Smoothed)',
    'nvidia_close_volume_last_month.png'
)
