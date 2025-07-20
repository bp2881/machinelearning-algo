import numpy as np
import pandas as pd
from plot import mv_plot as plot

def train(lr, n_iters, x1_train, x2_train, x3_train, y_train):
    n = x1_train.size
    w1, w2, w3, b = 0, 0, 0, 0
    for i in range(n_iters):
        if i % 1000 == 0 or i == n_iters - 1:
            print("Iteration: ", i)
        dw1, dw2, dw3 = 0, 0, 0
        db = 0
        for j in range(n):
            y_pred = w1 * x1_train[j] + w2 * x2_train[j] + w3 * x3_train[j] + b
            error = y_pred - y_train[j]
            dw1 += (w1*x1_train[j] + b - y_train[j])*x1_train[j]
            dw2 += (w2*x2_train[j] + b - y_train[j])*x2_train[j]
            dw3 += (w3*x3_train[j] + b - y_train[j])*x3_train[j]
            db += error
        
        w1 -= lr * dw1 / n
        w2 -= lr * dw2 / n
        w3 -= lr * dw3 / n 
        b -= lr * db / n
    
    return w1, w2, w3, b

# Squared Error Cost Function
def sec(x1_test, x2_test, x3_test, y_test, w1, w2, w3, b):
    suma = 0
    for i in range(x1_test.size):
        suma += (w1*x1_test[i] + w2*x2_test[i] + w3*x3_test[i] + b - y_test[i]) ** 2
    
    cost = (1/(2*x1_test.size)) * suma
    return cost

def predict(lr, n_iters, x1_train, x2_train, x3_train, y_train, x1_test, x2_test, x3_test, y_test):
    w1, w2, w3, b = train(lr, n_iters, x1_train, x2_train, x3_train, y_train)
    print("Cost = ", sec(x1_test, x2_test, x3_test, y_test, w1, w2, w3, b))
    plot(w1, w2, w3, b)



if __name__ == "__main__":
    # Data Processing
    data = pd.read_csv("RealEstate.csv")

    x1_total = data['X2 house age']
    x2_total = data['X5 latitude']
    x3_total = data['X6 longitude']
    y_total = data['Y house price of unit area']

    x1_train = x1_total.head(400).to_numpy()
    x2_train = x2_total.head(400).to_numpy()
    x3_train = x3_total.head(400).to_numpy()
    y_train = y_total.head(400).to_numpy()

    x1_test = x1_total.tail(15).to_numpy()
    x2_test = x2_total.tail(15).to_numpy()
    x3_test = x3_total.tail(15).to_numpy()
    y_test = y_total.tail(15).to_numpy()

    # Need to apply feature scaling
    # Standardization method
    x1_mean = x1_train.mean()  
    x1_std = x1_train.std()

    x2_mean = x2_train.mean()  
    x2_std = x2_train.std()

    x3_mean = x3_train.mean()  
    x3_std = x3_train.std()

    x1_train = (x1_train - x1_mean) / x1_std
    x2_train = (x2_train - x2_mean) / x2_std
    x3_train = (x3_train - x3_mean) / x3_std

    x1_test = (x1_test - x1_mean) / x1_std
    x2_test = (x2_test - x2_mean) / x2_std
    x3_test = (x3_test - x3_mean) / x3_std

    lr = 0.001
    n_iters = 100000
    
    predict(lr, n_iters, x1_train, x2_train, x3_train, y_train, x1_test, x2_test, x3_test, y_test)