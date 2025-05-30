import numpy as np


# --- Оптимизатор Adam ---
class AdamOptimizer:
    def __init__(self, shape, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        # Инициализация параметров Adam
        self.lr = lr  # Скорость обучения
        self.beta1 = beta1  # Коэффициент для экспоненциального сглаживания градиентов (момент первого порядка)
        self.beta2 = beta2  # Коэффициент для экспоненциального сглаживания квадратов градиентов (момент второго порядка)
        self.eps = eps  # Малое значение для предотвращения деления на ноль
        self.m = np.zeros(shape)  # Инициализация первого момента (градиенты)
        self.v = np.zeros(shape)  # Инициализация второго момента (квадраты градиентов)
        self.t = 0  # Счетчик итераций (необходим для смещения моментов)

    def update(self, grad):
        self.t += 1  # Увеличение счётчика шагов

        # Обновление экспоненциально взвешенного среднего градиентов
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        # Обновление экспоненциально взвешенного среднего квадратов градиентов
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # Коррекция смещения для первого и второго моментов
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Вычисление обновления параметров
        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_state(self):
        # Сохранение текущего состояния оптимизатора
        return {
            'm': self.m,
            'v': self.v,
            't': self.t
        }

    def set_state(self, state):
        # Загрузка состояния оптимизатора (например, при продолжении обучения)
        self.m = state['m']
        self.v = state['v']
        self.t = state['t']


# --- Сверточный слой ---
class Conv2D:
    def __init__(self, num_filters, filter_size, input_depth):
        # Инициализация сверточного слоя
        self.num_filters = num_filters  # Количество фильтров
        self.filter_size = filter_size  # Размер стороны фильтра (предполагается квадратный)

        # Инициализация фильтров случайными значениями, нормализованными по размеру фильтра
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) / (filter_size ** 2)

        # Инициализация оптимизатора Adam для фильтров
        self.opt = AdamOptimizer(self.filters.shape)

    def iterate_regions(self, image):
        # Генерация всех возможных регионов изображения, к которым применяются фильтры
        h, w = image.shape[1], image.shape[2]  # Высота и ширина входного изображения

        # Перебор всех возможных позиций для фильтра
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                # Вырезаем регион из входного изображения соответствующего размера
                region = image[:, i:i + self.filter_size, j:j + self.filter_size]
                yield i, j, region  # Возвращаем координаты и регион

    def forward(self, input):
        # Прямой проход (прямое распространение)
        self.last_input = input  # Сохраняем вход для последующего обратного прохода
        depth, height, width = input.shape  # Размерности входа
        output = np.zeros((self.num_filters, height - self.filter_size + 1, width - self.filter_size + 1))

        # Применение фильтров к каждому региону входа
        for i, j, region in self.iterate_regions(input):
            for f in range(self.num_filters):
                # Свертка региона с фильтром f
                output[f, i, j] = np.sum(region * self.filters[f])
        return output

    def backward(self, d_L_d_out):
        # Обратное распространение ошибки
        d_L_d_filters = np.zeros_like(self.filters)  # Градиенты по фильтрам
        d_L_d_input = np.zeros_like(self.last_input)  # Градиенты по входу

        # Перебираем регионы входного изображения
        for i, j, region in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # Градиент по фильтрам: ошибка * соответствующий регион
                d_L_d_filters[f] += d_L_d_out[f, i, j] * region
                # Градиент по входу: ошибка * фильтр
                d_L_d_input[:, i:i + self.filter_size, j:j + self.filter_size] += d_L_d_out[f, i, j] * self.filters[f]

        # Обновляем фильтры с помощью оптимизатора Adam
        self.filters += self.opt.update(d_L_d_filters)
        return d_L_d_input  # Возвращаем градиент по входу для дальнейшего распространения

    def get_state(self):
        # Сохраняем состояние фильтров и оптимизатора (например, для сохранения модели)
        return {
            'filters': self.filters,
            'opt_state': self.opt.get_state()
        }

    def set_state(self, state):
        # Восстанавливаем состояние фильтров и оптимизатора (например, при загрузке модели)
        self.filters = state['filters']
        self.opt.set_state(state['opt_state'])


class ReLU:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, d_L_d_out):
        return d_L_d_out * (self.last_input > 0)


# --- Слой подвыборки (макспулинг 2x2) ---
class MaxPool2:
    def iterate_regions(self, image):
        # Функция генерации всех неперекрывающихся 2x2 регионов входного изображения
        c, h, w = image.shape  # Получаем количество каналов, высоту и ширину
        new_h = h // 2  # Новая высота после пуллинга
        new_w = w // 2  # Новая ширина после пуллинга

        # Проходим по каждому 2x2 региону изображения
        for i in range(new_h):
            for j in range(new_w):
                region = image[:, i * 2:i * 2 + 2, j * 2:j * 2 + 2]  # Вырезаем 2x2 область для каждого канала
                yield i, j, region  # Возвращаем координаты и сам регион

    def forward(self, input):
        # Прямой проход слоя MaxPooling
        self.last_input = input  # Сохраняем вход для обратного прохода
        c, h, w = input.shape  # Получаем размерности входа
        output = np.zeros((c, h // 2, w // 2))  # Создаем выходной массив с уменьшенными размерами

        # Для каждого региона берем максимум по каждому каналу
        for i, j, region in self.iterate_regions(input):
            output[:, i, j] = np.max(region, axis=(1, 2))  # Максимум по осям 1 и 2 (пространственные оси)
        return output  # Возвращаем результат подвыборки

    def backward(self, d_L_d_out):
        # Обратное распространение ошибки через слой MaxPooling
        d_L_d_input = np.zeros_like(self.last_input)  # Градиент по входу, такой же формы как и вход

        # Перебираем те же регионы, что и в прямом проходе
        for i, j, region in self.iterate_regions(self.last_input):
            max_vals = np.max(region, axis=(1, 2))  # Получаем максимальные значения в каждом канале

            # Распределяем градиент только на те элементы, которые были максимумами
            for ch in range(region.shape[0]):  # Для каждого канала
                for ii in range(2):
                    for jj in range(2):
                        if region[ch, ii, jj] == max_vals[ch]:
                            # Передаем градиент по тому элементу, который был максимумом
                            d_L_d_input[ch, i * 2 + ii, j * 2 + jj] = d_L_d_out[ch, i, j]
        return d_L_d_input  # Возвращаем градиент для следующего слоя


# --- Полносвязный (Dense) слой ---
class Dense:
    def __init__(self, input_len, nodes):
        # Инициализация весов: случайные значения с нормализацией по числу входов
        self.weights = np.random.randn(input_len, nodes) / np.sqrt(input_len)
        self.biases = np.zeros(nodes)  # Инициализация нулевых смещений (bias)

        # Создание оптимизаторов для весов и смещений (используется Adam)
        self.opt_w = AdamOptimizer(self.weights.shape)
        self.opt_b = AdamOptimizer(self.biases.shape)

    def forward(self, input):
        # Прямой проход: линейное преобразование входа
        self.last_input_shape = input.shape  # Сохраняем оригинальную форму входа
        self.last_input = input.flatten()  # Преобразуем вход в вектор
        self.last_output = np.dot(self.last_input, self.weights) + self.biases  # Линейное преобразование
        return self.last_output  # Возвращаем выход слоя

    def backward(self, d_L_d_out):
        # Обратное распространение ошибки

        # Вычисляем градиенты по весам и смещениям
        d_L_d_weights = np.outer(self.last_input, d_L_d_out)  # Градиент весов: внешнее произведение входа и ошибки
        d_L_d_biases = d_L_d_out  # Градиент смещений — это просто ошибка
        d_L_d_input = np.dot(self.weights, d_L_d_out)  # Градиент по входу (для предыдущего слоя)

        # Обновляем параметры с помощью оптимизаторов
        self.weights += self.opt_w.update(d_L_d_weights)
        self.biases += self.opt_b.update(d_L_d_biases)

        # Возвращаем градиент по входу в исходной форме
        return d_L_d_input.reshape(self.last_input_shape)

    def get_state(self):
        # Получение текущего состояния слоя (для сохранения модели)
        return {
            'weights': self.weights,
            'biases': self.biases,
            'opt_w_state': self.opt_w.get_state(),
            'opt_b_state': self.opt_b.get_state()
        }

    def set_state(self, state):
        # Установка состояния слоя (для загрузки модели)
        self.weights = state['weights']
        self.biases = state['biases']
        self.opt_w.set_state(state['opt_w_state'])
        self.opt_b.set_state(state['opt_b_state'])


class SoftmaxCrossEntropy:
    def forward(self, input, label):
        self.last_input = input
        self.last_label = label

        exp = np.exp(input - np.max(input))
        self.probs = exp / np.sum(exp)

        loss = -np.log(self.probs[np.argmax(label)])
        return self.probs, loss

    def backward(self):
        return self.probs - self.last_label


# --- Model builder ---
def build_model():
    conv1 = Conv2D(num_filters=8, filter_size=3, input_depth=3)
    relu1 = ReLU()
    pool1 = MaxPool2()

    conv2 = Conv2D(num_filters=16, filter_size=3, input_depth=8)
    relu2 = ReLU()
    pool2 = MaxPool2()

    dense = Dense(input_len=6 * 6 * 16, nodes=10)
    softmax = SoftmaxCrossEntropy()

    return [conv1, relu1, pool1, conv2, relu2, pool2, dense, softmax]
