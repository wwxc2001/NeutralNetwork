'''
训练部分应实现SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重
'''
import argparse
import copy 
import json
from model.model_mlp import *
from data.metrics import *
from data.data_load import *

# SGD优化器
def SGDOptimizer(model, gradients, learning_rate): 
    for i in range(len(model.weights)):
        model.weights[i] -= learning_rate * gradients[i][0]
        model.biases[i] -= learning_rate * gradients[i][1]
    return model

# 实现学习率下降
def exponential_decay(epoch, initial_lr, decay_rate=0.4):
    lr = initial_lr * (1 / (1 + decay_rate * epoch))
    return lr

# 交叉熵损失和L2正则化
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def compute_loss(model, y, output, lambda_reg=0.01):
    test_y = copy.deepcopy(y)
    if test_y.ndim == 1 or test_y.shape[1] == 1:
        num_classes = output.shape[1]
        test_y = np.eye(num_classes)[test_y.reshape(-1)]  # Convert to one-hot encoding
    # Ensure that the output shape matches test_y
    if test_y.shape[0] != output.shape[0]:
        raise ValueError(f"Shape mismatch: labels have shape {test_y.shape} but output has shape {output.shape}")
    # Ensure output is passed through softmax for multiclass classification
    p = softmax(output)
    # Clip probabilities to prevent log(0) error
    p = np.clip(p, 1e-12, 1 - 1e-12)
    # Calculate cross-entropy loss
    cross_entropy = -np.sum(test_y * np.log(p)) / test_y.shape[0]
    # Calculate L2 regularization cost
    l2_cost = 0.5 * lambda_reg * sum(np.sum(w**2) for w in model.weights)
    # Total loss
    total_loss = cross_entropy + l2_cost
    # Derivative of loss w.r.t. the output of the network
    dL_dout = (p - test_y) / test_y.shape[0]
    return total_loss, dL_dout

# 定义保存模型参数的函数
def save_model_parameters(model, directory, validation_accuracy, activation_type):
    if not os.path.exists(directory):
        os.makedirs(directory)
    layer_sizes = model.layer_sizes
    file_name = "_".join([str(size) for size in layer_sizes] + [activation_type])
    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # 保存模型参数
    for i, weights in enumerate(model.weights):
        np.save(os.path.join(file_path, f'weights_{i}.npy'), weights)
    for i, biases in enumerate(model.biases):
        np.save(os.path.join(file_path, f'biases_{i}.npy'), biases)

    # 保存模型配置和验证准确率
    config_file = os.path.join(directory, 'config.json')
    if os.path.exists(config_file):
        # 如果配置文件已存在，则尝试加载配置
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    # 检查是否存在相同参数层的结果，如果存在则保留较大的准确率
    key = str(layer_sizes)
    if key in config:
        config[key] = max(validation_accuracy, config[key])
    else:
        config[key] = validation_accuracy
    # 保存配置文件
    with open(config_file, 'w') as f:
        json.dump(config, f)

def create_batches(X, y, batch_size):
    data = list(zip(X, y))
    np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield [np.array([d[0] for d in data[i:i + batch_size]]),
               np.array([d[1] for d in data[i:i + batch_size]])]

# 在测试集上进行测试
def test(model, X_test, y_test):
    # 进行测试
    output = model.forward(X_test)
    test_accuracy = accuracy(y_test, output)
    print(f"Test Accuracy: {test_accuracy:.4f}")

# 训练过程
def train(model, X, y, epochs, learning_rate, activation_type, save_dir, batch_size=32, decay_rate=0.001, validation_split=0.1, save_loss_path = "./save_loss_path", lambda_reg=0.01, store = True):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_accuracy = 0
    initial_lr = copy.copy(learning_rate)

    # 打乱数据并划分训练集和验证集
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    split_index = int(len(X) * (1 - validation_split))
    X_train, y_train = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]

    for epoch in range(epochs):
        for batch_X, batch_y in create_batches(X_train, y_train, batch_size):
            # 前向传播
            output = model.forward(batch_X)
            # 计算损失
            loss, dl_loss = compute_loss(model, batch_y, output, lambda_reg=lambda_reg)
            # 计算梯度
            gradients = model.backward(output, dl_loss)
            # 优化模型
            model = SGDOptimizer(model, gradients, learning_rate)

        # 记录训练集和验证集的损失和准确率
        train_output = model.forward(X_train)
        train_loss = compute_loss(model, y_train, train_output, lambda_reg=lambda_reg)[0]
        train_acc = accuracy(y_train, train_output)

        val_output = model.forward(X_val)
        val_loss = compute_loss(model, y_val, val_output, lambda_reg=lambda_reg)[0]
        val_acc = accuracy(y_val, val_output)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # 学习率下降
        learning_rate = exponential_decay(epoch, initial_lr)

        # Save the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = copy.deepcopy(model)
            # 保存模型参数
            if store:
                save_model_parameters(model, save_dir, best_accuracy, activation_type)
                print("update")

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Learning Rate: {learning_rate:.6f}")

    if store:
        # 保存训练过程记录为JSON文件
        history = {
            "train_loss": train_loss_history,
            "train_acc": train_acc_history,
            "val_loss": val_loss_history,
            "val_acc": val_acc_history
        }
        layer_sizes = model.layer_sizes
        loss_file_name = "_".join([str(size) for size in layer_sizes] + [activation_type])
        with open(os.path.join(save_loss_path, f"{loss_file_name}.json"), "w") as f:
            json.dump(history, f)
    return best_model, best_accuracy

# 主程序
# 主程序
def main(args):
    # 导入训练数据        
    X_train, y_train, X_test, y_test = load_cifar10('./cifar-10-batches-py')
    # X_train = X_train.reshape(-1, 3072) / 255.0
    # X_test = X_test.reshape(-1, 3072) / 255.0

    # 定义网络结构
    input_size = args.input_dim
    hidden_size = args.hidden_dim
    hidden_size_2 = args.hidden_dim_2
    output_size = args.num_classes

    # 创建神经网络
    activation = ActivationFunction(args.activation_type)
    network = ThreeLayerNeuralNetwork(input_size, [hidden_size, hidden_size_2], output_size, activation)

    # 训练细节
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    # 储存验证集上最优模型的路径
    savedir = args.save_dir
    # 训练神经网络
    model, best_eval_acc = train(network, X_train, y_train, epochs=epochs, learning_rate=learning_rate, activation_type = args.activation_type, save_dir=savedir, batch_size = batch_size, store = args.store)
    print(best_eval_acc)
    # 在测试集上进行测试
    test(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Network model for classification.")
    parser.add_argument('--input_dim', type=int, default=3072, help='Input feature dimension (M)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--hidden_dim_2', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--activation_type', type=str, default="relu", choices= ["sigmoid", "relu", "tanh"], help='Activation Function Type')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--save_dir', type=str, default="save_model_path", help='Directory to save the model')
    parser.add_argument('--store', action='store_true', help="choose to save model or not")
    parser.add_argument('--lambda_reg', type=float, default=0.01, help="L2 regularization parameter")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    args = parser.parse_args()
    main(args)