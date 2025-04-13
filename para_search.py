import numpy as np
import argparse
from data.data_load import *
from model.model_mlp import *
from data.metrics import *
from train import *

def main(args):
    # 定义超参数搜索范围和默认值
    default_params = {
        "learning_rate":  0.1,
        "hidden_layer_sizes": [128, 128],
        "regularization_strength": 0.01,
        "batch_size": 16,
        "activation_type": "relu"
    }

    # 替换默认参数
    for param, value in vars(args).items():
        if value is not None:
            default_params[param] = value

    # 导入训练数据
    X_train, y_train, X_test, y_test = load_cifar10('cifar-10-batches-py')

    # 存储结果的字典
    results = {}

    # 循环遍历搜索超参数
    for learning_rate in args.learning_rates:
        for hidden_layer_sizes in args.hidden_layer_sizes:
            for regularization_strength in args.regularization_strengths:
                for batch_size in args.batch_sizes:
                    for activation_type in args.activation_types:
                        print(f"learning_rate: {learning_rate} | hidden_size: {hidden_layer_sizes} | regularization_strength: {regularization_strength} | batch_size: {batch_size} | activation_type: {activation_type}")
                        # 创建神经网络
                        activation = ActivationFunction(activation_type)
                        network = ThreeLayerNeuralNetwork(input_size=3072,
                                                          hidden_sizes=[hidden_layer_sizes, hidden_layer_sizes],
                                                          output_size=10,
                                                          activation = activation)

                        # 训练神经网络
                        model, best_eval_acc = train(network, X_train, y_train, epochs=args.epochs,
                                      learning_rate=learning_rate, lambda_reg=regularization_strength,
                                      batch_size=batch_size, activation_type=activation_type, 
                                      save_dir= args.save_dir, store = False)

                        # 更新结果字典
                        results[(learning_rate, hidden_layer_sizes, regularization_strength, batch_size, activation_type)] = best_eval_acc

    # 找到最佳结果
    best_params = max(results, key=results.get)
    best_accuracy = results[best_params]

    print("Best hyperparameters:", best_params)
    print("Best validation accuracy:", best_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters Search")
    parser.add_argument('--learning_rates', default = [0.1],nargs='+', type=float, help='Learning rates to explore')
    parser.add_argument('--hidden_layer_sizes', default = [128], nargs='+', type=int, help='Hidden layer sizes to explore')
    parser.add_argument('--regularization_strengths', default = [0.01], nargs='+', type=float, help='Regularization strengths to explore')
    parser.add_argument('--batch_sizes', default = [16], nargs='+', type=int, help='Batch sizes to explore')
    parser.add_argument('--activation_types', default = ['relu'], nargs='+', help='Activation types to explore')
    parser.add_argument('--epochs', default = 20, type=int, help='Number of epochs for training')
    parser.add_argument('--save_dir', default= "./save_model_path", help= "save path")
    args = parser.parse_args()
    main(args)