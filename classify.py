import numpy as np
import pickle
import os


# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


class ThreeLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.input_size = input_size+1
        self.hidden_size = hidden_size
        self.output_size = output_size
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function")

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, reg_lambda):
        m = X.shape[0]
        dZ2 = output - y
        dW2 = np.dot(self.a1.T, dZ2) / m + reg_lambda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / m + reg_lambda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2


def cross_entropy_loss(y_true, y_pred, reg_lambda, W1, W2):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    l2_reg = (np.sum(np.square(W1)) + np.sum(np.square(W2))) * reg_lambda / 2
    return loss + l2_reg


def train(X_train, y_train, X_val, y_val, hidden_size, activation, learning_rate, reg_lambda, num_epochs, lr_decay):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = ThreeLayerNeuralNetwork(input_size, hidden_size, output_size, activation)
    best_val_acc = 0
    best_params = None

    for epoch in range(num_epochs):
        output = model.forward(X_train)
        loss = cross_entropy_loss(y_train, output, reg_lambda, model.W1, model.W2)
        dW1, db1, dW2, db2 = model.backward(X_train, y_train, output, reg_lambda)

        model.W1 -= learning_rate * dW1
        model.b1 -= learning_rate * db1
        model.W2 -= learning_rate * dW2
        model.b2 -= learning_rate * db2

        if epoch % 10 == 0:
            val_output = model.forward(X_val)
            val_acc = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (model.W1, model.b1, model.W2, model.b2)
            print(f'Epoch {epoch}, Loss: {loss}, Val Acc: {val_acc}')

        learning_rate *= lr_decay

    return best_params


def test(X_test, y_test, params):
    input_size = X_test.shape[1]
    hidden_size = params[0].shape[1]
    output_size = params[2].shape[1]
    model = ThreeLayerNeuralNetwork(input_size, hidden_size, output_size)
    model.W1, model.b1, model.W2, model.b2 = params
    output = model.forward(X_test)
    acc = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))
    return acc


def hyperparameter_search(X_train, y_train, X_val, y_val):
    learning_rates = [1e-2, 1e-3, 1e-4]
    hidden_sizes = [50, 100, 200]
    reg_lambdas = [0.001, 0.01, 0.1]
    activation = 'relu'
    num_epochs = 100
    lr_decay = 0.95

    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in reg_lambdas:
                print(f'Training with lr={lr}, hs={hs}, reg={reg}')
                best_params = train(X_train, y_train, X_val, y_val, hs, activation, lr, reg, num_epochs, lr_decay)
                val_output = test(X_val, y_val, best_params)
                print(f'Validation accuracy: {val_output}')


def load_cifar10_data(data_dir):
    X_train = []
    y_train = []

    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            X_train.append(data[b'data'])
            y_train.append(data[b'labels'])

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    test_file_path = os.path.join(data_dir, 'test_batch')
    with open(test_file_path, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
        X_test = test_data[b'data']
        y_test = test_data[b'labels']

    X_train = X_train.reshape(X_train.shape[0], -1).astype(float) / 255
    X_test = X_test.reshape(X_test.shape[0], -1).astype(float) / 255
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    data_dir = 'cifar-10-batches-py'  # 替换为你本地 CIFAR - 10 数据的路径
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)

    # 划分训练集和验证集
    num_train = int(0.8 * X_train.shape[0])
    X_val = X_train[num_train:]
    y_val = y_train[num_train:]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    # 超参数搜索
    hyperparameter_search(X_train, y_train, X_val, y_val)

    # 训练最终模型
    best_params = train(X_train, y_train, X_val, y_val, hidden_size=128, activation='relu', learning_rate=0.01,
                        reg_lambda=0.01, num_epochs=20, lr_decay=0.95)

    # 保存模型
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    # 测试模型
    test_acc = test(X_test, y_test, best_params)
    print(f'Test accuracy: {test_acc}')