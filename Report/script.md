# 使用 Informer 对 Nvidia 月度收盘价进行预测

author : **Chuyang Su**, **Liyuan Zheng**, **Jiantong Tian**

## 1. 数据预处理

### 1.简介

本文完全借鉴 zhouhaoyi/Informer2020 的架构，使用 Nvidia 2015-2024 的日频数据————约9.4年（∼2369 个交易日）的数据量————进行训练、验证与测试。

我们将进行月度预测，按照传统的transformer架构4:1:1的比例，我们决定选择 size = [seq_len, label_len, pred_len] 分别为4个月、1个月和1个月的数据量，即 size=[84, 21, 21]，可以保证数据量充足。

我们在Data preprocessing的模块文件中共封装了两个模块：Dataset_train 和 Dataset_prediction，分别封装在训练模型时运用的数据预处理部分和在进行真实场景预测时所使用的数据预处理部分。需要注意的是，在本研究中，我们只使用 train 进行训练、调参和测试，并没有涉及真正的部署使用，因此并不会真的使用 prediction 进行任何学术性的、严谨的讨论，但是我们的确会让模型使用2024年全年的数据使用prediction进行未来一个月的收盘价预测，并将结果与实际上的数据进行比对，但是这仅仅只是一个演示内容，并没有进行任何严谨的论证，我们也无法预期能得出良好的推测。

本部分接下来的描述如无特殊说明，均是Dataset_train的内容。

### 2. 读取与清洗

我们通过调用 Dataset_train 进行读取本次训练用的数据集，首先在 __init__ 中进行初始化，包括初始化参数，以及根据传递的参数或初始化的参数进行选取 features 列和 target 列、构建数据数组以及归一化等操作。

### 3. 数据集划分：训练、验证、测试

我们使用 __len__ 进行数据长度的计算，使用 __getitem__ 进行划分。我们选择的方式是将时间分段，即取前4个月的数据作为训练集，之后的1个月的数据作为验证集，再之后的1个月的数据作为测试集。

### 4. 特征选取

在这里我选择 close 也就是收盘价作为目标列，但在 encoder 阶段输入的是所有特征，decoder 阶段只输出目标列。

### 5. 时间特征编码

我们使用 time_features 进行时间特征编码，具体来讲就是通过这个函数将数据集中的原始日期数据转换拆分成几个有用的日历属性，并打包成一个数据矩阵，以供模型后续学习周期性和日历效应。

## 2. 模型部分

### 1. Prob Attention

### 2. Encoder


### 3. Decoder