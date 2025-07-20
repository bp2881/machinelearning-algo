import pandas as pd
import numpy as np
from plot import ov_plot as plot


def train(lr, n_iters, x_train, y_train):
    w, b = 0, 0
    for i in range(n_iters):
        if i % 1000 == 0 or i == n_iters - 1:
            print("Iteration: ", i)
        dw = 0
        db = 0
        for j in range(x_train.size):
            dw += (w*x_train[j] + b - y_train[j])*x_train[j]
            db += w*x_train[j] + b - y_train[j]
    
        w -= lr*(1/x_train.size)*dw
        b -= lr*(1/x_train.size)*db
    
    return w, b

# Sqaured Error Cost Function
def sec(X, Y, w, b):
    suma = 0
    for i in range(X.size):
        suma += (w*X[i] + b - Y[i]) ** 2
    
    cost = (1/(2*X.size)) * suma
    return cost

def predict(lr, n_iters, x_train, y_train, x_test, y_test):
    w, b = train(lr, n_iters, x_train, y_train)
    print("Cost = ", sec(x_test, y_test, w, b))
    print(f"(w, b) = ({w:.4f}, {b:.4f})")
    plot(w, b)


if __name__ == "__main__":
    # Data Processing
    data = pd.read_csv("RealEstate.csv")

    x_total = data['X2 house age']
    y_total = data['Y house price of unit area']

    x_train = x_total.head(400).to_numpy()
    y_train = y_total.head(400).to_numpy()

    x_test = x_total.tail(15).to_numpy()
    y_test = y_total.tail(15).to_numpy()

    lr = 0.003
    n_iters = 10000

    predict(lr, n_iters, x_train, y_train, x_test, y_test)