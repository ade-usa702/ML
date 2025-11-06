import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split


ALPHA = 1


class LinearModel:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.coef_ = None

    def predict(self, x):
        # парная линейная регрессия y = a + b*x
        if len(x.shape) == 1:
            return self.a + self.b * x
        # множественная линейная (мультилинейная) регрессия y = a + b1*x1 + b2*x2 + ... + bn*xn
        X = np.asarray(x)
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
        return self.a + np.dot(X, self.coef_)

    def fit(self, X, y, alpha) -> None:
        if len(X.shape) == 1:
            dJ0 = sum(self.predict(X) - y) / len(X)
            dJ1 = sum((self.predict(X) - y) * X) / len(X)
            self.a -= alpha * dJ0
            self.b -= alpha * dJ1
        else:
            X = np.asarray(X)
            y = np.asarray(y)
            dJ0 = np.sum(self.predict(X) - y) / len(y)
            dJ_coef_ = np.dot(X.T, (self.predict(X) - y)) / len(y)
            self.coef_ -= alpha * dJ_coef_
            self.a -= alpha * dJ0

    # среднеквадратич.ошибка(MSE) для функции потерь
    def error(self, X, Y):  
        """J = y_pred - y_true
        """    
        if len(X.shape) == 1:    
            return sum((self.predict(X) - Y)**2) / (2 * len(Y))
        return np.sum((self.predict(X) - Y)**2) / (len(Y))
    
    def optimal_alpha(self, error_modific, alpha) -> None:
        if error_modific < -1:
            alpha /= 2
        return alpha

    def score(self, X, y):
        y_pred = self.predict(X)
        res_s = np.sum((y - y_pred) ** 2)
        total_s = np.sum((y - np.mean(y)) ** 2)
        coef_determination = 1 - (res_s / total_s)
        return coef_determination

    def gradient_descent(self, x, y):
        alpha = ALPHA
        step = 0
        err = self.error(x, y)
        while True:
            step += 1
            self.fit(x, y, alpha)
            last_err = self.error(x, y)
            if err < pow(10, -6) or step > pow(10, 6):
                break
            alpha = self.optimal_alpha((err - last_err), alpha)
            err = last_err
        print(f'Кол-во итераций: {step}; Остановлен при сумме ошибок: {last_err:.8f}')
        return step
    
    def show_plot(self, x, y):
        if len(x.shape) == 1: 
            x_min, x_max = x.min(), x.max()
            margin = 0.1 * (x_max - x_min)
            X0 = np.linspace(x_min - margin, x_max + margin, 100)
            Y0 = self.predict(X0)
            plt.figure()
            plt.scatter(x, y)
            plt.xlabel("Признаки")
            plt.ylabel("Целевая переменная")
            plt.plot(X0, Y0, "r")
            plt.show()
        else:
            y_pred = self.predict(x)
            plt.figure()
            plt.scatter(y_pred, y)
            plt.plot(y, y, "r")
            plt.show()

