import os

import matplotlib.pyplot as plt
import numpy as np

from cnn_model import build_model
from cnn_train import train, test
from data_utils import download_cifar10, load_cifar10, normalize_and_encode
from mode_utils import load_model, save_model


def show_feature_maps(feature_maps, title="Feature maps"):
    if feature_maps.shape[0] < feature_maps.shape[-1]:
        maps = feature_maps
    else:
        maps = np.transpose(feature_maps, (2, 0, 1))

    num_filters = maps.shape[0]
    cols = 4
    rows = (num_filters + cols - 1) // cols
    plt.figure(figsize=(12, 3 * rows))
    for i in range(num_filters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(maps[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def get_or_train_model(X_train, y_train, model_path='model.pkl'):
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}...")
        model = load_model(model_path)
    else:
        print("Saved model not found, training new model...")
        model = build_model()
        train(model, X_train, y_train)
        save_model(model, model_path)
    return model


def main():
    download_dir = './data'
    download_cifar10(download_dir)

    data_path = os.path.join(download_dir, 'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_cifar10(data_path)
    X_train, y_train, X_test, y_test = normalize_and_encode(X_train, y_train, X_test, y_test)

    # Загружаем модель или тренируем, если нет сохранённой
    model = get_or_train_model(X_train[:5000], y_train[:5000], model_path='model.pkl')

    # Тестируем модель
    test(model, X_test[:5000], y_test[:5000])

    classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    img = X_test[37]
    label = y_test[37]

    print("True label:", classes[np.argmax(label)])

    # Пропускаем через первый сверточный блок
    out_conv1 = model[0].forward(img)
    show_feature_maps(out_conv1, "Conv1 output")

    out_relu1 = model[1].forward(out_conv1)
    show_feature_maps(out_relu1, "ReLU1 output")

    out_pool1 = model[2].forward(out_relu1)
    show_feature_maps(out_pool1, "MaxPool1 output")

    out_conv2 = model[3].forward(out_pool1)
    show_feature_maps(out_conv2, "Conv2 output")

    out_relu2 = model[4].forward(out_conv2)
    show_feature_maps(out_relu2, "ReLU2 output")

    out_pool2 = model[5].forward(out_relu2)
    show_feature_maps(out_pool2, "MaxPool2 output")

    out_dense = model[6].forward(out_pool2)
    probs, loss = model[7].forward(out_dense, label)

    plt.bar(classes, probs)
    plt.xticks(rotation=45)
    plt.title(f"Predicted probabilities\nTrue class: {classes[np.argmax(label)]}")
    plt.show()


if __name__ == "__main__":
    main()
