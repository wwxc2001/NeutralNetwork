import os
import pickle
import numpy as np
from PIL import Image


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

    # 数据预处理：将图像数据转换为合适的形状并归一化，将标签转换为独热编码
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # 数据增强
    X_train = augment_data(X_train)

    # 将图像数据展平为一维向量
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test


def augment_data(images):
    augmented_images = []
    for img in images:
        # 转换为 PIL 图像
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        # 随机左右翻转
        if np.random.random() > 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转换回 numpy 数组
        augmented_img = np.array(pil_img).astype(np.float32) / 255.0
        augmented_images.append(augmented_img)

    return np.array(augmented_images)


if __name__ == "__main__":
    data_path = './cifar-10-batches-py'  # 请替换为你本地 CIFAR - 10 数据集的实际路径
    X_train, y_train, X_test, y_test = load_cifar10(data_path)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_train[:1])