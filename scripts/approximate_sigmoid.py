import math

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

X = np.random.uniform(low=-8, high=8, size=(10000000,))
Y = [ sigmoid(x) for x in tqdm(X, ncols=80) ]

poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(X.reshape(-1, 1) )

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, Y)

print(f'{poly_reg_model.intercept_:.4f}', " ".join([ f'{c:.4f}' for c in poly_reg_model.coef_]))

X = np.arange(-8, 8.01, 0.01)
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(X.reshape(-1, 1) )
poly_reg_model.predict(X)
