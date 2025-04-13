# 从零开始构建三层神经网络分类器，实现图像分类

## 仓库介绍

本项目为课程DATA620004——神经网络和深度学习作业的代码仓库

* 作业：从零开始构建三层神经网络分类器，实现图像分类

* 任务描述：
  手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类。

* 基本要求：
  （1）本次作业要求自主实现反向传播，不允许使用 pytorch，tensorflow 等现成的支持自动微分的深度学习框架，可以使用 numpy；
  （2）最终提交的代码中应至少包含**模型**、**训练**、**测试**和**参数查找**四个部分，鼓励进行模块化设计；
  （3）其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。
## Requirements

```bash
pip install numpy
pip install argparse
pip install pickle

# for visualization
pip install seaborn
pip install matplotlib
```

## 文件说明
```bash
- cifar-10-batches-py  # cifar-10 数据集
- data/  # 数据处理代码
  - data_load.py  # 读取cifar-10 数据集
  - metrics.py  # 定义softmax函数和accuracy函数
- model/  # 模型代码
  - model_mlp.py  # 神经网络
  - activations.py    # 激活函数
- train.py # 模型训练主程序
- test.py # 模型测试主程序
- para_search.py # 模型参数探索主程序
- visualize.py # 可视化主程序
```

## 一、 模型的训练与测试

### 数据下载

从[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)网页下载CIFAR-10数据集cifar-10-batches-py，存在在仓库根目录下即可

### 模型训练

* 进入仓库根目录，在命令行中设置需要的模型参数以及超参数，运行：
```bash
python train.py --input_dim 3072 \
                --hidden_dim 256 \
                --hidden_dim_2 128 \
                --activation_type "relu" \
                --epochs 30 \
                --learning_rate 0.1 \
                --batch_size 64 \
                --store
```

生成的模型权重会以`npy`的形式自动保存在`save_model_path`文件夹中；训练中产生的loss和Accuracy信息会以`json`文件的形式保存在`save_loss_path`文件夹中
### 模型测试

* 模型权重地址：[https://pan.baidu.com/s/1r3OuJWtuX7Np-Zfwd90wYg?pwd=2ttu](https://pan.baidu.com/s/14VxRoux2bBZOMeWqDkNRwQ?pwd=5pmb)]
* 将模型权重文件放至目录`save_model_path`中；
* 运行：
```bash
python test.py --model_dir save_model_path/3072_256_128_10_relu/
```
推荐使用的模型参数为`3072_256_128_10_relu`，在测试集上准确率可达53%。

## 二、模型参数搜索与可视化

### 1. 模型参数搜索
* 在命令行中增加想要探索的参数，运行：
```bash
python para_search.py --learning_rates 0.1 0.01 \
                      --batch_sizes 16 32 64 \
                      --activation_types "relu" "tanh" \
                      --regularization_strengths 0.1 0.01
```
最后会返回给定候选参数中，最优的参数组合以及在验证集上的最高准确率。

### 2. 训练信息以及模型参数可视化

[`visualize.py`](visualize.py)提供了
- 训练过程信息的可视化（包括loss和Accuracy）（函数`plot_training_history`可以通过修改`json_files`的值选取不同参数下训练模型的信息进行可视化）。
- 对模型网络初始化和训练后各层参数的可视化代码（包括小提琴图和热力图） （函数`plot_network_para`）
