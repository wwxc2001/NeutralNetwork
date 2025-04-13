import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model.model_mlp import *

'''
绘制loss下降曲线以及acc曲线
'''
# 根据路径读取模型
def load_model_parameters(file_path):
    if os.path.exists(file_path):
        config = file_path.split("/")[-1]
        # 解析参数配置
        layer_sizes = [size for size in config.split('_')]
        # 加载对应参数的模型
        file_name = "{}_{}_{}_{}_{}".format(layer_sizes[0], layer_sizes[1], layer_sizes[2], layer_sizes[3], layer_sizes[4])
        # file_path = os.path.join(directory, file_name)
        activation = ActivationFunction(layer_sizes[-1])  # 这里假设激活函数为ReLU，你可以根据实际情况进行修改
        model = ThreeLayerNeuralNetwork(int(layer_sizes[0]), [int(layer_sizes[1]), int(layer_sizes[2])], 10, activation)
        # 加载模型参数
        for i, weights in enumerate(model.weights):
            model.weights[i] = np.load(os.path.join(file_path, f'weights_{i}.npy'))
        for i, biases in enumerate(model.biases):
            model.biases[i] = np.load(os.path.join(file_path, f'biases_{i}.npy'))
        return model
    else:
        raise FileNotFoundError("Configuration file does not exist.")

def plot_training_history(json_files, labels):
    plt.figure(figsize=(10, 8), dpi=200)
    
    for json_file, label in zip(json_files, labels):
        with open(json_file, "r") as f:
            history = json.load(f)
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        plt.plot(epochs, history["train_loss"], label=json_file.split("/")[-1][:-5] + ' Train Loss')
        plt.plot(epochs, history["val_loss"], label=json_file.split("/")[-1][:-5] + ' Validation Loss')
        # 打印标签名字
        plt.text(epochs[-1], history["train_loss"][-1], json_file.split("/")[-1][:-5], verticalalignment='bottom', horizontalalignment='left')
        # plt.text(epochs[-1], history["val_loss"][-1], json_file.split("/")[-1][:-5], verticalalignment='bottom', horizontalalignment='right')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    num1 = 1.0
    num2 = 1.0
    num3 = 0
    num4 = 0.1
    plt.legend(bbox_to_anchor = (num1, num2), loc = num3, borderaxespad = num4)
    plt.grid(True)
    plt.savefig("./save_image_path/loss.pdf")
    plt.show()

    plt.figure(figsize=(10, 8), dpi=200)
    
    for json_file, label in zip(json_files, labels):
        with open(json_file, "r") as f:
            history = json.load(f)
        
        epochs = range(1, len(history["train_acc"]) + 1)
        
        # plt.plot(epochs, history["train_acc"], label=json_file.split("/")[-1][:-5] + ' Train Accuracy')
        plt.plot(epochs, history["val_acc"], label=json_file.split("/")[-1][:-5] + ' Validation Accuracy')
        plt.text(epochs[-1], history["val_acc"][-1], json_file.split("/")[-1][:-5], verticalalignment='bottom', horizontalalignment='left')
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("./save_image_path/acc.pdf")
    plt.show()

'''
神经网络参数可视化
'''
def plot_network_para(model_path):
    # 加载目标模型
    model = load_model_parameters(model_path)

    for i, param in enumerate(model.weights):
        # 绘制小提琴图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.violinplot(data=param.ravel())
        plt.title(f'Layer {i+1} Parameters Violin Plot')
        plt.ylabel('Value')
        
        # 绘制热力图
        plt.subplot(1, 2, 2)
        sns.heatmap(param, cmap='viridis')
        plt.title(f'Layer {i+1} Parameters Heatmap')
        plt.boxplot(param)

        plt.tight_layout()
        plt.savefig(f"./save_image_path/para_visualize/layer_{i+1}.pdf")
        plt.show()
def plot_network_para_1(model_path):
    model = load_model_parameters(model_path)
    for i, param in enumerate(model.weights):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.violinplot(data=param.ravel())
        plt.title(f'Layer {i + 1} Parameters Violin Plot')
        plt.ylabel('Value')

        plt.subplot(1, 2, 2)
        # 使用奇异值分解进行降维，保留 90% 的能量
        u, s, vh = np.linalg.svd(param)
        explained_variance_ratio = s ** 2 / np.sum(s ** 2)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        num_components = np.argmax(cumulative_explained_variance >= 0.5) + 1
        reduced_param = u[:, :num_components] @ np.diag(s[:num_components]) @ vh[:num_components, :]
        sns.heatmap(reduced_param, cmap='viridis')
        plt.title(f'Layer {i + 1} Parameters Heatmap')
        plt.boxplot(param)

        plt.tight_layout()
        plt.savefig(f"./save_image_path/para_visualize/layer_{i + 1}.pdf")
        plt.show()



if __name__ == "__main__":
    json_files = ['./save_loss_path/3072_128_128_10_relu.json','./save_loss_path/3072_256_128_10_relu.json', './save_loss_path/3072_256_256_10_relu.json', './save_loss_path/3072_512_256_10_relu.json', './save_loss_path/3072_512_512_10_relu.json']
    labels = ['Model 1', 'Model 2', 'Model 3', "Model 4"]
    plot_training_history(json_files, labels)
    model_file = "./save_model_path/3072_128_128_10_relu"
    plot_network_para_1(model_file)