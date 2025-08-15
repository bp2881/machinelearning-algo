import numpy as np
import pandas as pd
from plot import plr_plot as plot
from numba import njit

@njit
def sec(y_pred, y_true):
    m = y_true.size
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

@njit
def train_step(lr, n_iters, x, y):
    m = x.size
    w4, w3, w2, w1, b = 0.0, 0.0, 0.0, 0.0, 0.0
    history = np.zeros(n_iters // 1000 + 1)
    idx = 0
    for i in range(n_iters):
        y_pred = w4 * (x ** 4) + w3 * (x ** 3) + w2 * (x ** 2) + w1 * x + b
        error  = y_pred - y

        dw4 = (1 / m) * np.dot(error, x**4)       
        dw3 = (1 / m) * np.dot(error, x**3)
        dw2 = (1 / m) * np.dot(error, x**2)
        dw1 = (1 / m) * np.dot(error, x)
        db  = (1 / m) * np.sum(error)

        w4 -= lr * dw4
        w3 -= lr * dw3
        w2 -= lr * dw2
        w1 -= lr * dw1
        b  -= lr * db

        if i % 1000 == 0 or i == n_iters - 1:
            cost = sec(y_pred, y)
            history[idx] = cost
            idx += 1
    return w4, w3, w2, w1, b, history

def train(lr, n_iters, x, y):
    w4, w3, w2, w1, b, history = train_step(lr, n_iters, x, y)
    for i, cost in enumerate(history):
        print(f"Iteration: {i*1000}, Cost: {cost}")
    return w4, w3, w2, w1, b

def predict(lr, n_iters, x_train, y_train, x_test, y_test):
    w4, w3, w2, w1, b = train(lr, n_iters, x_train, y_train)

    y_train_pred = w4**2 + x_train * (w3 + w2 + w1) + b
    y_test_pred  = w4**2 + x_test  * (w3 + w2 + w1) + b

    train_cost = sec(y_train_pred, y_train)
    test_cost  = sec(y_test_pred,  y_test)
    cost = max(train_cost, test_cost)
    print(f"(w4, w3, w2, w1, b) = ({w4:.4f}, {w3:.4f}, {w2:.4f}, {w1:.4f}, {b:.4f})")
    print("Cost = ", cost)
    plot(w4, w3, w2, w1, b, cost)

if __name__ == "__main__":
    df = pd.read_csv("RealEstate.csv")
    x_raw = df["X2 house age"].to_numpy(dtype=float)
    y_raw = df["Y house price of unit area"].to_numpy(dtype=float)

    x_scaled = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())

    x_train = x_scaled[:400]
    y_train = y_raw[:400]
    x_test  = x_scaled[400:]
    y_test  = y_raw[400:]

    learning_rate = 0.01
    iterations = 10_000_000

    predict(learning_rate, iterations, x_train, y_train, x_test, y_test)
