import numpy as np
import pandas as pd
import random

class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
        return f"MyKNNReg class: {', '.join(params)}"

    def fit(self, X_train, y_train, verbose=False):
        self.X_train = pd.DataFrame(X_train)
        self.y_train = pd.Series(y_train)
        self.train_size = X.shape

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def chebyshev_distance(self, x1, x2):
        return np.max((np.abs(x1 - x2)))

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def cosine_distance(self, x1, x2):
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - dot_product / (norm_x1 * norm_x2)

    def predict(self, X_test):
        X_test = pd.DataFrame(X_test)

        predictions = []
        for i in range(len(X_test)):

            if self.metric == 'euclidean':
                distances = np.array(
                    [self.euclidean_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            elif self.metric == 'chebyshev':
                distances = np.array(
                    [self.chebyshev_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            elif self.metric == 'manhattan':
                distances = np.array(
                    [self.manhattan_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            elif self.metric == 'cosine':
                distances = np.array([self.cosine_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            else:
                distances = None

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_classes = self.y_train.iloc[nearest_indices]

            if self.weight == 'uniform':
                q = nearest_classes.sum()
                q /= len(nearest_classes)
                predictions.append(q)

            if self.weight == 'rank':

                tot_weight = 0
                for j in range(len(nearest_classes)):
                    tot_weight += 1 / (j + 1)
                res = 0
                for j in range(len(nearest_classes)):
                    w = (1 / (j + 1)) / tot_weight
                    res += w * nearest_classes.iloc[j]
                predictions.append(res)

            if self.weight == 'distance':
                k_distances = distances[nearest_indices]
                tot_weight = 0
                for j in range(len(nearest_classes)):
                    tot_weight += 1 / (k_distances[j])

                res = 0
                for j in range(len(nearest_classes)):
                    w = (1 / k_distances[j]) / tot_weight
                    res += w * nearest_classes.iloc[j]
                predictions.append(res)

        return np.array(predictions)
