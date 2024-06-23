import numpy as np
import pandas as pd
import random

class MyLogReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.X = None
        self.y = None
        self.verbose = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights if weights is not None else np.array([])
        self.eps = 1e-15
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.new_X = None
        self.new_y = None

    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

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

        z = np.dot(self.X, self.weights)
        y_pred_start = 1 / (1 + np.exp(-z))
        loss_start = -np.mean(self.y * np.log(y_pred_start + self.eps) - (1 - self.y) * np.log(1 - y_pred_start + self.eps))

        if verbose:
            print(f"start | loss: {loss_start:.2f}")

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
                z = np.dot(self.X, self.weights)
                y_pred = 1 / (1 + np.exp(-z))
                gradient = (np.dot(y_pred - y, self.X)) / len(y) + reg_term
                loss = -np.mean(
                    self.y * np.log(y_pred + self.eps) - (1 - self.y) * np.log(1 - y_pred + self.eps)) + reg_term

            else:
                if self.sgd_sample > 1:
                    sample_rows_idx = random.sample(range(self.X.shape[0]), self.sgd_sample)
                else:
                    cnt_datas = int(len(self.X) * self.sgd_sample)
                    sample_rows_idx = random.sample(range(self.X.shape[0]), cnt_datas)

                self.new_X = self.X.loc[sample_rows_idx]
                self.new_y = self.y.loc[sample_rows_idx]
                z = np.dot(self.new_X, self.weights)
                y_pred = 1 / (1 + np.exp(-z))
                gradient = (np.dot(y_pred - self.new_y, self.new_X)) / len(self.new_y) + reg_term

                loss = -np.mean(self.new_y * np.log(y_pred + self.eps) - (1 - self.new_y) * np.log(1 - y_pred + self.eps)) + reg_term

            if callable(self.learning_rate):
                self.weights -= self.learning_rate(i) * gradient
            else:
                self.weights -= self.learning_rate * gradient

            if verbose and (i % verbose == 0):
                print(f"{i} | loss: {loss:.2f}")

            self.best_score = self.calculate_accuracy(self.metric)
        self.best_score = self.calculate_accuracy(self.metric)

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        ones_series = pd.Series([1] * len(X))
        X = pd.concat([ones_series, X], axis=1)
        z = np.dot(X, self.weights)
        y_pred = 1 / (1 + np.exp(-z))
        return y_pred

    def predict(self, X):
        X = pd.DataFrame(X)
        ones_series = pd.Series([1] * len(X))
        X = pd.concat([ones_series, X], axis=1)
        z = np.dot(X, self.weights)
        y_pred = 1 / (1 + np.exp(-z))
        binary_y = np.array(y_pred > 0.5)
        return binary_y

    def calculate_accuracy(self, metric='accuracy'):
        new_y = 1 / (1 + np.exp(-self.y))
        new_y = np.array(new_y > 0.5)

        if metric == 'accuracy':
            y_pred = self.predict(self.X.iloc[:, 1:])
            accuracy = np.mean(y_pred == new_y)
            return accuracy

        if metric == 'precision':
            tp = 0
            fp = 0
            y_pred = self.predict(self.X.iloc[:, 1:])
            for i in range(len(new_y)):
                if new_y[i] == True and y_pred[i] == True:
                    tp += 1
                if new_y[i] == False and y_pred[i] == True:
                    fp += 1
            precision = tp / (tp + fp)
            return precision

        if metric == 'recall':
            tp = 0
            fn = 0
            y_pred = self.predict(self.X.iloc[:, 1:])
            for i in range(len(new_y)):
                if new_y[i] == True and y_pred[i] == True:
                    tp += 1
                if new_y[i] == True and y_pred[i] == False:
                    fn += 1
            recall = tp / (tp + fn)
            return recall

        if metric == 'f1':
            p = self.calculate_accuracy('precision')
            r = self.calculate_accuracy('recall')
            f1 = 2 * (p * r) / (p + r)
            return f1

        if metric == 'roc_auc':
            new_y = 1 / (1 + np.exp(-self.y))
            new_y = np.array(new_y > 0.5)
            y_pred = self.predict_proba(self.X.iloc[:, 1:])
            y_pred_rounded = [round(pred, 10) for pred in y_pred]

            sorted_data = sorted(zip(y_pred_rounded, new_y), key=lambda x: x[0], reverse=True)
            sorted_y_pred_rounded, sorted_new_y = zip(*sorted_data)
            _sum = 0
            pos_above_cnt = 0  # колво 1 сверху
            pos_same_cnt = 0  # колво 1 с одной и той же вероятностью как у 0
            n = len(new_y)
            pos_count = sum(new_y)
            neg_count = n - pos_count

            for i in range(n):
                if i == 0:
                    if sorted_new_y[i] == True:
                        pos_above_cnt += 1
                        pos_same_cnt += 1
                else:
                    if sorted_y_pred_rounded[i - 1] != sorted_y_pred_rounded[i]:  # поменялась вероятность
                        pos_same_cnt = 0
                    if sorted_new_y[i] == True:
                        pos_above_cnt += 1
                        pos_same_cnt += 1
                    if sorted_new_y[i] == False:
                        _sum += (pos_above_cnt - pos_same_cnt) + 0.5 * pos_same_cnt
            auc = _sum * (1 / (neg_count * pos_count))
            print(auc)
            return auc

    def get_best_score(self):
        return self.best_score
