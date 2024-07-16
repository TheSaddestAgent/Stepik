import numpy as np
import pandas as pd

class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.left = None
        self.right = None
        self.side = None
        self.depth = 0

    def __str__(self):
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
        return f"Node class: {', '.join(params)}"


class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.tree = None
        self.bins = bins
        self.histo_split = None
        self.criterion = criterion
        self.fi = {}
        self._sum_leaf = 0

    def __str__(self):
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
        return f"MyTreeReg class: {', '.join(params)}"

    def information_gain(self, left_y, right_y, parent_impurity):
        total_samples = len(left_y) + len(right_y)
        p_left = len(left_y) / total_samples
        p_right = len(right_y) / total_samples

        ig = parent_impurity - (p_left * self.mse(left_y) + p_right * self.mse(right_y))

        return ig

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def get_best_split(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        best_ig = 0
        best_col = None
        best_split = None

        for col in X.columns:
            unique_values = X[col].unique()
            unique_values.sort()

            for i in range(1, len(unique_values)):
                split_value = (unique_values[i - 1] + unique_values[i]) / 2
                left_indices = X[col] <= split_value
                right_indices = X[col] > split_value

                left_y = y[left_indices]
                right_y = y[right_indices]

                ig = self.information_gain(left_y, right_y, self.mse(y))

                if ig > best_ig:
                    best_ig = ig
                    best_col = col
                    best_split = split_value

        return best_col, best_split, best_ig

    def make_histo(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)

        histo_split = {}
        for col in X.columns:
            unique_values = X[col].unique()
            if len(unique_values) > self.bins - 1:
                hist, bin_edges = np.histogram(X[col], bins=self.bins)
                unique_values = bin_edges[1:-1]
            else:
                unique_values = X[col].unique()
            unique_values.sort()
            histo_split[col] = unique_values
        return histo_split

    def get_best_histo_split(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        best_ig = 0
        best_col = None
        best_split = None
        for col in X.columns:
            for split_value in self.histo_split[col]:
                left_indices = X[col] <= split_value
                right_indices = X[col] > split_value

                left_y = y[left_indices]
                right_y = y[right_indices]

                ig = self.information_gain(left_y, right_y, self.mse(y))

                if ig > best_ig:
                    best_ig = ig
                    best_col = col
                    best_split = split_value

        return best_col, best_split, best_ig

    def fit(self, X, y):
        self.tree = None
        if self.bins is not None:
            self.histo_split = self.make_histo(X, y)

        for col in X.columns:
            self.fi[col] = 0

        def build(root, X_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()
            best_col = None
            best_split = None
            if self.bins is None:
                best_col, best_split, best_ig = self.get_best_split(X_root, y_root)
            elif self.histo_split is not None:
                best_col, best_split, best_ig = self.get_best_histo_split(X_root, y_root)
            proportion_ones = np.mean(y_root) if len(y_root) else 0

            if best_col is None:
                root.value_leaf = proportion_ones
                root.side = side
                root.feature = None
                root.value_split = None
                self._sum_leaf += proportion_ones
                return root

            if proportion_ones == 0 or proportion_ones == 1 or depth >= self.max_depth or len(y_root) < self.min_samples_split or (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                root.value_leaf = proportion_ones
                self._sum_leaf += proportion_ones

                return root

            root.feature = best_col
            root.value_split = best_split
            self.leafs_cnt += 1

            X_1 = X_root.loc[X_root[best_col] <= best_split]
            y_1 = y_root.loc[X_root[best_col] <= best_split]

            X_2 = X_root.loc[X_root[best_col] > best_split]
            y_2 = y_root.loc[X_root[best_col] > best_split]

            if y_1 is None or y_2 is None or len(y_1) == 0 or len(y_2) == 0:
                root.value_leaf = proportion_ones
                self._sum_leaf += proportion_ones

                root.side = side
                root.feature = None
                root.value_split = None
            else:
                root.left = build(root.left, X_1, y_1, 'left', depth + 1)
                root.right = build(root.right, X_2, y_2, 'right', depth + 1)
                self.fi[best_col] += len(y_root) / len(y) * (self.mse(y_root) - len(y_1) / len(y_root) * self.mse(y_1) - len(y_2) / len(y_root) * self.mse(y_2))
            return root

        self.tree = build(self.tree, X, y)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{'  ' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{'  ' * depth}{node.side} = {node.value_leaf}")

    def predict(self, X):
        for _, row in X.iterrows():
            node = self.tree
            while node.feature is not None:
                if row[node.feature] <= node.value_split:
                    node = node.left
                else:
                    node = node.right
            yield node.value_leaf
