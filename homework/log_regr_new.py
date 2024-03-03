from scipy import sparse
from scipy.sparse import coo_array, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# A = coo_array([[1, 2], [3, 4]])
# B = coo_array([[5], [6]])
# new_arr = hstack([A, B]).toarray()
# pass
#
#
# l1 = ['the best pizza flour', 'pie making with sweet rice flour', 'good and cheap', 'great treats for the treat ball.', 'the babies love it']
# l2 = [word.split() for word in l1]
# l2
#
# words = []
# for word_list in l2:
#     for word in word_list:
#         words.append(word)
#
# words_uniq = set(words)
# pass





import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sys import getsizeof

from dmia.classifiers.logistic_regression import LogisticRegression


class LogReg():

    # в методе .__init__() объявим переменные для весов и уровня ошибки
    def __init__(self):
        self.thetas = None
        self.loss_history = []

    # метод .fit() необходим для обучения модели
    # этому методу мы передадим признаки и целевую переменную
    # кроме того, мы зададим значения по умолчанию
    # для количества итераций и скорости обучения
    def fit(self, x, y, iter=1, learning_rate=1e-1, batch_size=200, reg=1e-5):

        # метод создаст "правильные" копии датафрейма
        # x, y = x.copy(), y.copy()

        num_train, dim = x.shape

        # добавит столбец из единиц
        self.add_ones(x)

        # инициализирует веса и запишет в переменную n количество наблюдений
        # thetas, n = np.zeros(x.shape[1]), x.shape[0]
        thetas, n = np.random.randn(dim) * 0.01, x.shape[0]

        np.random.randn(dim) * 0.01

        # создадим список для записи уровня ошибки
        loss_history = []

        # в цикле равном количеству итераций
        for i in range(iter):
            print(i)
            for batch_start in range(0, x.shape[0], batch_size):
                # print(batch_start)
                random_indices = np.random.choice(num_train, batch_size)
                X_batch = x[random_indices]
                y_batch = y[random_indices]

                # метод сделает прогноз с текущими весами
                y_pred = self.h(X_batch, thetas)


                # найдет и запишет уровень ошибки
                loss = self.objective(y_batch, y_pred)
                regularization_loss = (reg / (2 * n)) * np.sum(np.square(thetas[1:]))
                loss += regularization_loss
                loss_history.append(loss)


                # рассчитает градиент
                grad = self.gradient(X_batch, y_batch, y_pred, n)
                regularization_grad = (reg / n) * thetas
                regularization_grad[0] = 0
                grad += regularization_grad

                # и обновит веса
                thetas -= learning_rate * grad

        # метод выдаст веса и список с историей ошибок
        self.thetas = thetas
        self.loss_history = loss_history

    # метод .predict() делает прогноз с помощью обученной модели
    def predict(self, x):

        # метод создаст "правильную" копию модели
        # x = x.copy()
        # добавит столбец из единиц
        self.add_ones(x)
        # рассчитает значения линейной функции
        z = np.dot(x.toarray(), self.thetas)
        # передаст эти значения в сигмоиду
        probs = np.array([self.stable_sigmoid(value) for value in z])
        # выдаст принадлежность к определенному классу и соответствующую вероятность
        return np.where(probs >= 0.5, 1, 0), probs

    # ниже приводятся служебные методы, смысл которых был разобран ранее
    def add_ones(self, X):
        # return x.insert(0, 'x0', np.ones(x.shape[0]))
        # return np.c_[np.ones(x.shape[0]), x]
        return sparse.hstack((np.ones(X.shape[0])[:, np.newaxis], X)).tocsr()

    def h(self, x, thetas):
        z = np.dot(x.toarray(), thetas)
        return np.array([self.stable_sigmoid(value) for value in z])

    def objective(self, y, y_pred):
        y_one_loss = y * np.log(y_pred + 1e-9)
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def gradient(self, x, y, y_pred, n):
        return np.dot(x.toarray().T, (y_pred - y)) / n

    def stable_sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (np.exp(z) + 1)


# train_df = pd.read_csv('./data/train.csv')
train_df = pd.read_csv('/home/soren/PycharmProjects/OTUS/10_numpy/homework/data/train.csv')
# train_df_pos = train_df[:40000]
# train_df_neg = train_df[-40000:]
# train_df = pd.concat([train_df_pos, train_df_neg])

review_summaries = train_df['Reviews_Summary'].values.tolist()
review_summaries = [l.lower() for l in review_summaries]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(review_summaries)
y = train_df.Prediction.values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
del X


# создадим объект класса LogReg
model = LogisticRegression()

# и обучим модель
model.train(X_train, y_train, num_iters=10)


# сделаем прогноз on test
y_pred, probs = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
pass

