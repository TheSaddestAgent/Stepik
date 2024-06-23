import numpy as np
import pandas as pd
import random

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0,
                 sgd_sample=None, random_state=42):
        self.X = None
        self.y = None
        self.verbose = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights if weights is not None else np.array([])
        self.metric = metric
        self.loss_fun = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.new_X = None
        self.new_y = None

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X, y, verbose=False):
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.X.reset_index(inplace=True, drop=True)
        self.y.reset_index(inplace=True, drop=True)
        self.verbose = verbose
        ones_series = pd.Series([1] * len(X))
        self.X = pd.concat([ones_series, self.X], axis=1)
        num_features = self.X.shape[1]
        self.weights = np.ones(num_features)
        random.seed(self.random_state)

        y_pred_start = np.dot(self.X, self.weights)
        mse_start = np.mean((y - y_pred_start) ** 2)
        # if verbose:
        #   print(f"start | loss: {mse_start:.2f} | {self.metric}: {self.calculate_metric(y, y_pred_start):.2f}")

        for i in range(self.n_iter):
            i += 1

            if self.reg == 'l1':
                reg_term = self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                reg_term = 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                l1_term = self.l1_coef * np.sign(self.weights)
                l2_term = 2 * self.l2_coef * self.weights
                reg_term = l1_term + l2_term
            else:
                reg_term = 0

            if self.sgd_sample is None:
                y_pred = np.dot(self.X, self.weights)
                gradient = -2 * np.dot(self.X.T, (y - y_pred)) / len(y) + reg_term
                self.new_y = y
            else:
                if self.sgd_sample > 1:
                    sample_rows_idx = random.sample(range(self.X.shape[0]), self.sgd_sample)
                else:
                    cnt_datas = int(len(self.X) * self.sgd_sample)
                    sample_rows_idx = random.sample(range(self.X.shape[0]), cnt_datas)

                self.new_X = self.X.loc[sample_rows_idx]
                self.new_y = self.y.loc[sample_rows_idx]
                y_pred = np.dot(self.new_X, self.weights)

                gradient = -2 * np.dot(self.new_X.T, self.new_y - y_pred) / len(self.new_y) + reg_term

            mse = np.mean((self.new_y - y_pred) ** 2)
            if callable(self.learning_rate):
                self.weights -= self.learning_rate(i) * gradient
            else:
                self.weights -= self.learning_rate * gradient

            # if verbose and (i % verbose == 0):
            #  print(f"{i} | loss: {(mse + reg_term):.2f} | {self.metric}: {self.calculate_metric(y, y_pred):.2f}")

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        self.X = pd.DataFrame(X)
        ones_series = pd.Series([1] * len(X))
        self.X = pd.concat([ones_series, self.X], axis=1)
        y = self.X.dot(self.weights)
        return y

    def get_best_score(self):
        y_pred = np.dot(self.X, self.weights)
        return self.calculate_metric(self.y, y_pred)

    def calculate_metric(self, y, y_pred):
        regul = self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        if self.metric == 'mae':
            return np.mean(np.abs(y - y_pred)) + regul
        elif self.metric == 'mse':
            return np.mean((y - y_pred) ** 2) + regul
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y - y_pred) ** 2)) + regul
        elif self.metric == 'mape':
            return np.mean(np.abs((y - y_pred) / y)) * 100 + regul
        elif self.metric == 'r2':
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) + regul
