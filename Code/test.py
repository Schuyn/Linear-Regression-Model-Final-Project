'''
author:        Schuyn <98257102+Schuyn@users.noreply.github.com>
date:          2025-04-28 22:41:38
'''
import torch
from Decoder import SimpleDecoder

# 假设
B        = 4
L_enc    = 147    # 你 EncoderStack 的输出长度：84+42+21=147
pred_len = 21
d_model  = 256
n_heads  = 4
d_ff     = 4 * d_model

# 1) 随机伪造 Encoder 输出和时间特征
enc_out     = torch.randn(B, L_enc, d_model)
# 未来 21 天的时间特征 —— 3 维: month/day/weekday
# 这里取 0~30、1~31、0~6 的随机数来模拟
x_time_dec  = torch.randint(0, 32,   (B, pred_len, 1)).float()
x_time_dec2 = torch.randint(1, 32,   (B, pred_len, 1)).float()
x_time_dec3 = torch.randint(0, 7,    (B, pred_len, 1)).float()
# 拼成 (B,21,3)
x_time_dec  = torch.cat([x_time_dec, x_time_dec2, x_time_dec3], dim=-1)

# 2) 实例化 Decoder
decoder = SimpleDecoder(
    pred_len=pred_len,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    num_layers=2,
    factor=5,
    dropout=0.1
)

# 3) 前向
pred = decoder(enc_out, x_time_dec)

# 4) 验证输出
print("Decoder output shape:", pred.shape)
# 预期 torch.Size([4, 21])
