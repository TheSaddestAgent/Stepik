import numpy as np
import pandas as pd
import random

class MyKNNClf:
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
        return f"MyKNNClf class: {', '.join(params)}"

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
                mode_class = nearest_classes.mode().iloc[0]
                count_0 = (nearest_classes == 0).sum()
                count_1 = (nearest_classes == 1).sum()
                if count_1 == count_0:
                    predictions.append(1)
                else:
                    predictions.append(mode_class)

            if self.weight == 'rank':
                q_0 = 0
                q_1 = 0
                for j in range(len(nearest_classes)):

                    if nearest_classes.iloc[j] == 0:
                        q_0 += 1 / (j + 1)
                    if nearest_classes.iloc[j] == 1:
                        q_1 += 1 / (j + 1)
                tot_weight = 0
                for j in range(len(nearest_classes)):
                    tot_weight += 1 / (j + 1)
                q_0 /= tot_weight  # на самом деле можно не делить в predict, потому что нам нужно только сравнить q_0 и q_1
                q_1 /= tot_weight  #
                if q_1 >= q_0:
                    predictions.append(1)
                else:
                    predictions.append(0)

            if self.weight == 'distance':
                k_distances = distances[nearest_indices]
                q_0 = 0
                q_1 = 0
                for j in range(len(nearest_classes)):

                    if nearest_classes.iloc[j] == 0:
                        q_0 += 1 / (k_distances[j])
                    if nearest_classes.iloc[j] == 1:
                        q_1 += 1 / (k_distances[j])

                tot_weight = 0
                for j in range(len(nearest_classes)):
                    tot_weight += 1 / (k_distances[j])
                q_0 /= tot_weight
                q_1 /= tot_weight
                if q_1 >= q_0:
                    predictions.append(1)
                else:
                    predictions.append(0)

        return np.array(predictions)

    def predict_proba(self, X_test):
        probabilities = []
        for i in range(len(X_test)):

            if self.metric == 'euclidean':
                distances = np.array([self.euclidean_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            elif self.metric == 'chebyshev':
                distances = np.array([self.chebyshev_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            elif self.metric == 'manhattan':
                distances = np.array([self.manhattan_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            elif self.metric == 'cosine':
                distances = np.array([self.cosine_distance(X_test.iloc[i].values, x) for x in self.X_train.values])
            else:
                distances = None

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_classes = self.y_train.iloc[nearest_indices]
            prob_class1 = -1

            if self.weight == 'uniform':
                prob_class1 = nearest_classes.sum() / self.k

            if self.weight == 'rank':
                q_1 = 0
                for j in range(len(nearest_classes)):
                    if nearest_classes.iloc[j] == 1:
                        q_1 += 1 / (j + 1)
                tot_weight = 0
                for j in range(len(nearest_classes)):
                    tot_weight += 1 / (j + 1)
                q_1 /= tot_weight
                prob_class1 = q_1

            if self.weight == 'distance':
                k_distances = distances[nearest_indices]
                q_1 = 0
                for j in range(len(nearest_classes)):
                    if nearest_classes.iloc[j] == 1:
                        q_1 += 1 / (k_distances[j])
                tot_weight = 0
                for j in range(len(nearest_classes)):
                    tot_weight += 1 / (k_distances[j])
                q_1 /= tot_weight
                prob_class1 = q_1

            probabilities.append(prob_class1)
        return np.array(probabilities)
