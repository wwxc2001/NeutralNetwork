import numpy as np
from model.activations import ActivationFunction


class ThreeLayerNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = activation.function
        self.activation_derivative = activation.derivative
        self.weights = []
        self.biases = []
        
        # 初始化权重
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.normal(0, pow(self.layer_sizes[i], -0.5), (self.layer_sizes[i], self.layer_sizes[i + 1]))
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        activation = X
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.activation(z)
            self.activations.append(activation)
        
        return self.activations[-1]

    def backward(self, output, delta, lambda_reg = 0.001):
        m = output.shape[0]
        delta *= self.activation_derivative(self.activations[-1])

        deltas = [delta]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(self.activations[i])
            deltas.insert(0, delta)

        gradients = []
        for i in range(len(self.weights)):
            # print(f"activations[{i}].T shape: {self.activations[i].T.shape}")
            # print(f"deltas[{i}] shape: {deltas[i].shape}")
            grad_w = np.dot(self.activations[i].T, deltas[i])  + lambda_reg * self.weights[i]
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)
            gradients.append((grad_w, grad_b))

        return gradients

if __name__ == "__main__":
    # 测试
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 3], [1, 1, 4]])
    y = np.array([[0], [1], [1], [0]])

    # 定义网络结构
    input_size = 3
    hidden_size = 4
    output_size = 1

    # Example of usage:
    activation = ActivationFunction('sigmoid')
    network = ThreeLayerNeuralNetwork(input_size, [hidden_size, hidden_size], output_size, activation)

    print(network.forward(X))