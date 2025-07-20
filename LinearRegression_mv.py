import numpy as np
import pandas as pd
from plot import mv_plot as plot

def train(lr, n_iters, x1_train, x2_train, x3_train, y_train):
    n = x1_train.size
    w1, w2, w3, b = 0, 0, 0, 0
    for i in range(n_iters):
        if i % 10000 == 0 or i == n_iters - 1:
            print("Iteration: ", i)
        
        y_pred = w1 * x1_train + w2 * x2_train + w3 * x3_train + b
        error = y_pred - y_train
        
        dw1 = np.dot(error, x1_train)
        dw2 = np.dot(error, x2_train)
        dw3 = np.dot(error, x3_train)
        db  = np.sum(error)

        w1 -= lr * dw1 / n
        w2 -= lr * dw2 / n
        w3 -= lr * dw3 / n
        b  -= lr * db  / n
           
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
    cost = sec(x1_test, x2_test, x3_test, y_test, w1, w2, w3, b)
    print("Cost = ", cost)
    plot(w1, w2, w3, b, cost)



if __name__ == "__main__":
    # Data Processing
    data = pd.read_csv("RealEstate.csv")

    x1_total = data['X2 house age']
    x2_total = data['X3 distance to the nearest MRT station']
    x3_total = data['X4 number of convenience stores']
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
    scaling = lambda x: (x - x.min()) / (x.max() - x.min())

    x1_test = scaling(x1_test)
    x2_test = scaling(x2_test)
    x3_test = scaling(x3_test)

    x1_train = scaling(x1_train)
    x2_train = scaling(x2_train)
    x3_train = scaling(x3_train)


    lr = 0.001
    n_iters = 1000000
    
    predict(lr, n_iters, x1_train, x2_train, x3_train, y_train, x1_test, x2_test, x3_test, y_test)