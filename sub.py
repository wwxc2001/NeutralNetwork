import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 假设 CIFAR - 10 数据集存储在以下路径
path = './cifar-10-batches-py'

# 初始化训练数据和标签列表
X_train = []
y_train = []

# 加载 5 个训练数据批次
for i in range(1, 6):
    file_path = os.path.join(path, f'data_batch_{i}')
    with open(file_path, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        X_train.append(data['data'])
        y_train.append(data['labels'])

# 将列表转换为 numpy 数组
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# 定义类别名称
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 选择要可视化的样本数量
num_samples = 10

# 创建一个新的 PDF 文件
with PdfPages('cifar10_visualization.pdf') as pdf:
    # 创建一个子图布局
    fig, axes = plt.subplots(2, 5, figsize=(4, 2))
    axes = axes.flatten()

    for i in range(num_samples):
        # 重塑图像数据
        img = X_train[i].reshape(3, 32, 32).transpose(1, 2, 0)
        label = y_train[i]

        # 显示图像
        axes[i].imshow(img)
        axes[i].set_title(label_names[label])
        axes[i].axis('off')

    # 调整子图布局
    plt.tight_layout()

    # 将当前图形保存到 PDF 文件中
    pdf.savefig()

    # 关闭图形
    plt.close()

print("可视化结果已保存为 cifar10_visualization.pdf")
    
    