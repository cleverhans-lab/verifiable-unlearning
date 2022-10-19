
from pathlib import Path

import numpy as np
from pmlb import fetch_data
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class Dataset: 

    def __init__(self, X, Y, X_test=None, Y_test=None, normalize=True):

        if X_test is None and Y_test is None:
            if normalize:
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
            self.X, self.X_test, self.Y, self.Y_test = train_test_split(X, Y, test_size=0.2)
        else:
            self.X, self.Y, self.X_test, self.Y_test = X, Y, X_test, Y_test

    def __str__(self):
        return f'Dataset(no_samples={len(self.X)}, no_features={len(self.X[0])}) {self.Y}'

    def __len__(self):
        return len(self.X)

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
        return Dataset(X_sh, Y_sh, X_test_sh, Y_test_sh)    

    @staticmethod
    def from_pmlb(dataset_name):
        X, Y = fetch_data(dataset_name, return_X_y=True, local_cache_dir=Path.home().joinpath('verifiable-unlearning/data').as_posix())
        return Dataset(X, Y)

    @staticmethod
    def make_classification(no_samples, no_features):
        assert no_features == 1
        X, Y = make_classification(n_samples=int(np.ceil(no_samples/0.8)), random_state=1)
        X = [ [ x[0] ] for x in X ] # keep only one feature
        dataset = Dataset(X, Y)
        assert len(dataset) == no_samples
        return dataset
