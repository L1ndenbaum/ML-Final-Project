import pickle
import numpy as np
import matplotlib.pyplot as plt
import os, random
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def unpickle(file):
    with open(file, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict

def load_cifar10(data_dir, normalize=False, unsupervised=False):
    """加载CIFAR10数据集为展平后的(num_samples, num_features) Numpy数组"""
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    train_data = np.vstack(train_data)  # (50000, 3072)
    train_labels = np.array(train_labels)  # (50000,)

    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    test_data = test_batch[b'data']  # (10000, 3072)
    test_labels = np.array(test_batch[b'labels'])  # (10000,)

    if normalize:
        train_data = train_data / 255.0
        test_data = test_data / 255.0

    if unsupervised:
        data = np.concatenate((train_data, test_data), axis=0)

    if unsupervised:
        return data
    else:
        return train_data, train_labels, test_data, test_labels

def get_cifar10_str_label(idx_lable):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return class_names[idx_lable]

def show_predict_res(X_test, y_test, y_pred):
    """随机抽取16个结果进行展示"""
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 变形为三通道图片(10000, 32, 32, 3)
    random_idxs = random.sample(list(range(X_test.shape[0])), 16)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    for i in range(16):
        axes[i].imshow(X_test[random_idxs[i]])
        axes[i].set_title(f"True:{get_cifar10_str_label(y_test[random_idxs[i]])}\nPred:{get_cifar10_str_label(y_pred[random_idxs[i]])}")
        axes[i].axis('off')
    plt.tight_layout()

def get_test_scores(y_test, y_pred):
    """打印在测试集上的准确率、精确率、召回率"""
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    print(f"在测试集上的准确率(Accuracy): {test_accuracy*100:.2f}%")
    print(f"在测试集上的精确率(Precision): {test_precision*100:.2f}%")
    print(f"在测试集上的召回率(Recall): {test_recall*100:.2f}%")

def get_report_and_cm(y_test, y_pred):
    """打印分类报告并绘制混淆矩阵"""
    # 导出分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()