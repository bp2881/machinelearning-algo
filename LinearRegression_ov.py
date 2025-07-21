import pandas as pd
import numpy as np
from plot import ov_plot as plot
from numba import njit

@njit
def train_step(lr, n_iters, x_train, y_train, x_test, y_test):
    w, b = 0.0, 0.0
    history = np.zeros(n_iters // 1000 + 1)
    idx = 0
    n = x_train.size

    for i in range(n_iters):
        if i % 1000 == 0 or i == n_iters - 1:
            cost = sec(x_test, y_test, w, b)
            history[idx] = cost
            idx += 1
        
        y_pred = w * x_train + b
        error = y_pred - y_train

        dw = np.dot(error, x_train)
        db = np.sum(error)

        w -= lr * dw / n
        b -= lr * db / n
    
    return w, b, history


def train(lr, n_iters, x_train, y_train, x_test, y_test):
    w, b, history = train_step(lr, n_iters, x_train, y_train, x_test, y_test)
    for i, cost in enumerate(history):
        step = i * 1000
        if step > n_iters:
            step = n_iters
        print(f"Iteration: {step}, Cost: {cost}")
    
    return w, b

# Sqaured Error Cost Function
@njit
def sec(X, Y, w, b):
    y_pred = w * X + b
    error = y_pred - Y
    cost = (1 / (2 * X.size)) * np.sum(error ** 2)
    return cost


def predict(lr, n_iters, x_train, y_train, x_test, y_test):
    w, b = train(lr, n_iters, x_train, y_train, x_test, y_test)
    cost = sec(x_test, y_test, w, b)
    print("Cost = ", cost)
    print(f"(w, b) = ({w:.4f}, {b:.4f})")
    plot(w, b, cost)


if __name__ == "__main__":
    # Data Processing
    data = pd.read_csv("RealEstate.csv")

    x_total = data['X2 house age']
    y_total = data['Y house price of unit area']

    x_train = x_total.head(400).to_numpy()
    y_train = y_total.head(400).to_numpy()

    x_test = x_total.tail(15).to_numpy()
    y_test = y_total.tail(15).to_numpy()

    # Feature Scaling
    scaling = lambda x: (x - x.min()) / (x.max() - x.min())
    x_train = scaling(x_train).astype(np.float64)
    x_test = scaling(x_test).astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    lr = 0.01
    n_iters = 10_000

    predict(lr, n_iters, x_train, y_train, x_test, y_test)