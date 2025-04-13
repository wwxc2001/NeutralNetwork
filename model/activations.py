import numpy as np

class ActivationFunction:
    def __init__(self, function_type='sigmoid'):
        self.function_type = function_type

    def function(self, x):
        """
        根据 fuction_type 实现三类激活函数
        :param x: 输入值
        :return: 激活函数的输出
        """
        if self.function_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.function_type == 'relu':
            return np.maximum(0, x)
        elif self.function_type == 'tanh':
            return np.tanh(x)
    
    def derivative(self, x):
        """
        根据 fuction_type 实现三类激活函数的导数
        :param x: 输入值
        :return: 激活函数的导数输出
        """
        if self.function_type == 'sigmoid':
            sigmoid_x = 1 / (1 + np.exp(-x))
            return sigmoid_x * (1 - sigmoid_x)
        elif self.function_type == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.function_type == 'tanh':
            return 1 - np.tanh(x) ** 2

if __name__ == "__main__":
    # 测试代码
    test_input = np.array([-2, -1, 0, 1, 2])

    # 测试 sigmoid 函数及其导数
    activation = ActivationFunction('sigmoid')
    sigmoid_result = activation.function(test_input)
    sigmoid_derivative_result = activation.derivative(test_input)
    print("Sigmoid 函数结果:", sigmoid_result)
    print("Sigmoid 导数结果:", sigmoid_derivative_result)

    # 测试 ReLU 函数及其导数
    activation = ActivationFunction('relu')
    relu_result = activation.function(test_input)
    relu_derivative_result = activation.derivative(test_input)
    print("ReLU 函数结果:", relu_result)
    print("ReLU 导数结果:", relu_derivative_result)

    # 测试 tanh 函数及其导数
    activation = ActivationFunction('tanh')
    tanh_result = activation.function(test_input)
    tanh_derivative_result = activation.derivative(test_input)
    print("Tanh 函数结果:", tanh_result)
    print("Tanh 导数结果:", tanh_derivative_result)
    