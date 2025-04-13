import numpy as np
from data.data_load import *

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(pred_classes == true_classes)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cifar10('./cifar-10-batches-py')
    # 测试 softmax 函数
    test_softmax_input = np.random.randn(10, 10) 
    softmax_output = softmax(test_softmax_input)
    print("Softmax 函数输出示例:")
    print(softmax_output[:5])  

    # 测试 accuracy 函数
    random_prediction = np.random.randn(*y_test.shape)
    random_prediction_softmax = softmax(random_prediction)
    acc = accuracy(y_test, random_prediction_softmax)
    print(f"随机预测的准确率: {acc}")