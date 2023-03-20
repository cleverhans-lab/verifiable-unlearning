
from pathlib import Path

import numpy as np
from pmlb import fetch_data
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import copy

class Dataset: 

    def __init__(self, train, test=None):
        if len(train) == 0:
            self.X, self.Y = [], []
        else:
            self.X, self.Y = zip(*train)

        if test is None or len(test) == 0:
            self.X_test, self.Y_test = self.X, self.Y
        else:
            self.X_test, self.Y_test = zip(*test)

    # def __init__(self, X, Y, X_test=None, Y_test=None, normalize=True):

    #     if len(X) == 0 and len(Y) == 0:
    #         self.X, self.Y, self.X_test, self.Y_test = [], [], [], []
    #         return

    #     if X_test is None and Y_test is None:
    #         if normalize:
    #             # normalize X
    #             X = preprocessing.StandardScaler().fit(X).transform(X)
    #             # normalize Y
    #             assert len(set(Y)) == 2, Y
    #             if min(Y) == 0 and max(Y) == 2:
    #                 Y = Y // 2
    #             assert abs(list(set(Y))[0] - list(set(Y))[1]) == 1, set(Y)
    #             if max(Y) == 2:
    #                 Y = Y - 1
    #             assert Y.min() == 0 and Y.max() == 1, Y
    #         self.X, self.X_test, self.Y, self.Y_test = train_test_split(X, Y, test_size=0.2)
    #     else:
    #         self.X, self.Y, self.X_test, self.Y_test = X, Y, X_test, Y_test

    def __str__(self):
        return f'Dataset(no_samples={len(self.X)}+{len(self.X_test)}, no_features={len(self.X[0])})'

    def __len__(self):
        return len(self.X)

    def __getitem__(self, key):
        return (self.X[key], self.Y[key])

    def __add__(self, other):
        X = [ x for x in self.X ] + [ x for x in other.X ]
        Y = [ y for y in self.Y ] + [ y for y in other.Y ]
        return Dataset(list(zip(X, Y)))

    @property
    def data(self):
        return  [ [ y ] + x for x, y in zip(self.X, self.Y) ]

    @property
    def size(self):
        return len(self)

    @property
    def max_size(self):
        return max(self.size, 1)

    @property
    def no_features(self):
        return len(self.X[0])

    @property
    def no_coefs(self):
        return len(self.X[0]) + 1

    def add_shift(self, input, shift):
        if shift != 1:
            return int(input * shift)
        else:
            return input

    def shift(self, precision):
        X_sh, Y_sh = [], []
        for x, y in zip(self.X, self.Y):
            X_sh += [ [ self.add_shift(x_i, precision) for x_i in x]  ]
            Y_sh += [ self.add_shift(y, precision) ]
        X_test_sh, Y_test_sh = [], []
        for x, y in zip(self.X_test, self.Y_test):
            X_test_sh += [ [ self.add_shift(x_i, precision) for x_i in x]  ]
            Y_test_sh += [ self.add_shift(y, precision) ]
        return Dataset(train=list(zip(X_sh, Y_sh)), test=list(zip(X_test_sh, Y_test_sh))) 

    @staticmethod
    def from_pmlb(dataset_name):
        X, Y = fetch_data(dataset_name, return_X_y=True, local_cache_dir=Path.home().joinpath('verifiable-unlearning/data').as_posix())
        # normalize X
        X = preprocessing.StandardScaler().fit(X).transform(X)
        # normalize Y
        assert len(set(Y)) == 2, Y
        if min(Y) == 0 and max(Y) == 2:
            Y = Y // 2
        assert abs(list(set(Y))[0] - list(set(Y))[1]) == 1, set(Y)
        if max(Y) == 2:
            Y = Y - 1
        assert Y.min() == 0 and Y.max() == 1, Y
        # split
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)
        return Dataset(train=list(zip(X, Y)), test=list(zip(X_test, Y_test))) 

    @staticmethod
    def make_classification(no_features):
        X, Y = make_classification(n_samples=10000, random_state=1)
        # X = [ [ x[0] ] for x in X ] # keep only one feature
        X = [ list(x[:no_features]) for x in X ] # keep only one feature
        # normalize X
        X = preprocessing.StandardScaler().fit(X).transform(X)
        for x, y in zip(X, Y):
            yield (list(x), y)        

    def remove(self, idxes):
        data_points = [ (x, y) for idx, (x, y) in enumerate(zip(self.X, self.Y)) if idx not in idxes]
        return Dataset(data_points)