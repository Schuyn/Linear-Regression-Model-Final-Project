# 使用 Informer 对 Nvidia 月度收盘价进行预测

author : **Chuyang Su**, **Liyuan Zheng**, **Jiantong Tian**



### 1. 数据预处理

#### 1.简介

本文完全借鉴 zhouhaoyi/Informer2020 的架构，使用 Nvidia 2015-2024 的日频数据————约9.4年（∼2369 个交易日）的数据量————进行训练、验证与测试。

我们将进行月度预测，按照传统的transformer架构4:1:1的比例，我们决定选择 size = [seq_len, label_len, pred_len] 分别为4个月、1个月和1个月的数据量，即 size=[84, 21, 21]，可以保证数据量充足。

我们在data preprocessing的模块文件中共封装了两个模块：dataset_train 和dataset_prediction

#### 2. 读取与清洗



#### 3. 数据集划分：训练、验证、测试

#### 4.