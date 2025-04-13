import os
import pickle
import numpy as np


def load_cifar10(path):
    """
    加载 CIFAR - 10 数据集
    :param path: 数据集所在路径
    :return: 训练集数据、训练集标签、测试集数据、测试集标签
    """
    X_train = []
    y_train = []

    # 加载 5 个训练数据批次
    for i in range(1, 6):
        file_path = os.path.join(path, f'data_batch_{i}')
        with open(file_path, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
            X_train.append(data['data'])
            y_train.append(data['labels'])

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # 加载测试数据批次
    test_file_path = os.path.join(path, 'test_batch')
    with open(test_file_path, 'rb') as fo:
        test_data = pickle.load(fo, encoding='latin1')
        X_test = test_data['data']
        y_test = test_data['labels']

    # 数据预处理：将图像数据展平并归一化，将标签转换为独热编码
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    data_path = './cifar-10-batches-py'  # 请替换为你本地 CIFAR - 10 数据集的实际路径
    X_train, y_train, X_test, y_test = load_cifar10(data_path)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_train[:1])