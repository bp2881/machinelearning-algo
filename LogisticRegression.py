import numpy as np
import pandas as pd
from plot import log_plot as plot

def sigmoid(w, b, X):
    z = np.dot(w, X) + b
    g_x = 1/(1 + np.exp(-z))
    return g_x

def train(lr, n_iters, x_train, y_train):
    w, b = 0, 0
    m = x_train.size
    for i in range(n_iters):
        y_pred = sigmoid(w, b, x_train)
        error = y_pred - y_train
        dw = np.dot(error, x_train)
        db = np.sum(error)

        w -= lr * dw / m
        b -= lr * db / m
    
    return w, b

def predict(lr, n_iters, x_train, y_train, x_test, y_test):
    w, b = train(lr, n_iters, x_train, y_train)
    cost = cost_function(x_test, y_test, w, b)
    print(f"w, b = ({w:.4f}, {b:.4f})")
    print(f"Cost: {cost}")
    print(f"Accuracy: {accuracy(w, b, x_test, y_test)*100:.2f}%")
    ## PLOT
    plot(w, b, cost, x_test, y_test)


def cost_function(x_test, y_test, w, b):
    f_x = sigmoid(w, b, x_test)
    loss = -(y_test * np.log(f_x + 1e-15)) - (1 - y_test) * np.log(1 - f_x + 1e-15)
    cost = np.mean(loss)
    return cost

def accuracy(w, b, x, y):
    preds = sigmoid(w, b, x) >= 0.5
    return np.mean(preds == y)

if __name__ == "__main__":
    data = pd.read_csv("cancer.csv")

    x_total = data["perimeter_mean"]
    y_total = data["diagnosis"]
    

    x_train = x_total.head(500).to_numpy()
    y_train = y_total.head(500).to_numpy()
    y_train = np.where(y_train == 'M', 1, 0)

    x_test = x_total.tail(50).to_numpy()
    y_test = y_total.tail(50).to_numpy()
    y_test = np.where(y_test == 'M', 1, 0)

    lr = 0.001
    n_iters = 10_000
    predict(lr, n_iters, x_train, y_train, x_test, y_test)