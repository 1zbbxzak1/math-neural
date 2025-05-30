import os
import pickle
import tarfile
import urllib.request

import numpy as np


def download_cifar10(download_dir='data'):
    """
    Скачивает архив с датасетом CIFAR-10 и распаковывает его.

    Args:
        download_dir (str): Путь к директории для сохранения данных.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"  # Ссылка на архив с CIFAR-10
    filename = url.split('/')[-1]  # Имя файла архива
    filepath = os.path.join(download_dir, filename)  # Полный путь к файлу архива

    # Создаем директорию, если она не существует
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Если архив еще не скачан — скачиваем
    if not os.path.exists(filepath):
        print("Скачиваю CIFAR-10...")
        urllib.request.urlretrieve(url, filepath)
        print("Скачано.")
    else:
        print("Архив уже скачан.")

    extracted_dir = os.path.join(download_dir, 'cifar-10-batches-py')  # Путь к распакованной папке
    # Если архив еще не распакован — распаковываем
    if not os.path.exists(extracted_dir):
        print("Распаковываю архив...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=download_dir)
        print("Распаковано.")
    else:
        print("Архив уже распакован.")


def load_batch(batch_filename):
    """
    Загружает один batch данных из CIFAR-10.

    Args:
        batch_filename (str): Путь к файлу batch.

    Returns:
        data (np.ndarray): Массив изображений размером (N, 3, 32, 32).
        labels (np.ndarray): Массив меток классов размером (N,).
    """
    with open(batch_filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')  # Загружаем содержимое файла
        data = batch[b'data']  # Извлекаем данные изображений
        labels = batch[b'labels']  # Извлекаем метки классов
        # Преобразуем данные в формат (N, 3, 32, 32) и тип float32
        data = data.reshape(len(data), 3, 32, 32).astype("float32")
        labels = np.array(labels)
        return data, labels


def load_cifar10(data_dir='data/cifar-10-batches-py'):
    """
    Загружает все тренировочные и тестовые данные из CIFAR-10.

    Args:
        data_dir (str): Путь к папке с распакованными данными CIFAR-10.

    Returns:
        X_train, y_train, X_test, y_test (np.ndarray): Массивы данных и меток для обучения и тестирования.
    """
    X_train = []
    y_train = []

    # Загружаем 5 тренировочных batch-ов
    for i in range(1, 6):
        data_batch, labels_batch = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        X_train.append(data_batch)
        y_train.append(labels_batch)

    # Объединяем все батчи в один массив
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # Загружаем тестовый набор
    X_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))

    return X_train, y_train, X_test, y_test


def normalize_and_encode(X_train, y_train, X_test, y_test):
    """
    Нормализует изображения по каналам (RGB) и преобразует метки в one-hot encoding.

    Args:
        X_train, y_train, X_test, y_test (np.ndarray): Исходные данные и метки.

    Returns:
        Нормализованные данные и one-hot метки.
    """

    # Вычисляем среднее и стандартное отклонение по каждому каналу (RGB),
    # учитывая все изображения и пиксели
    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)

    # Нормализуем тренировочные и тестовые изображения
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Преобразуем метки в формат one-hot encoding (10 классов)
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test
