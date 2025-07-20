import pandas as pd
import numpy as np
from plot import ov_plot as plot


def train(lr, n_iters, x_train, y_train):
    n = x_train.size
    w, b = 0, 0
    for i in range(n_iters):
        if i % 1000 == 0 or i == n_iters - 1:
            print("Iteration: ", i)
        
        y_pred = w*x_train + b
        error = y_pred - y_train

        dw = np.dot(error, x_train)
        db = np.sum(error)

        w -= lr * dw / n
        b -= lr * db / n
    
    return w, b

# Sqaured Error Cost Function
def sec(X, Y, w, b):
    suma = 0
    n = X.size
    for i in range(n):
        suma += (w*X[i] + b - Y[i]) ** 2
    
    cost = (1/(2*X.size)) * suma
    return cost

def predict(lr, n_iters, x_train, y_train, x_test, y_test):
    w, b = train(lr, n_iters, x_train, y_train)
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

    lr = 0.001
    n_iters = 10000

    predict(lr, n_iters, x_train, y_train, x_test, y_test)