import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

area = df['area'][:10_000]
latitude = df['geo_lat'][:10_000]
longitude = df['geo_lon'][:10_000]
house_price = df['price'][:10_000]
rooms = df['rooms'][:10_000]
kitchen_area = df['kitchen_area'][:10_000]

#df2 = pd.read_csv("cancer.csv")
#perimeter_mean = df2["perimeter_mean"]
#diagnosis = df2["diagnosis"]


def ov_plot(w, b, cost):
    x_min = area.min()
    x_max = area.max()

    x_line_scaled = np.linspace(0, 1, 100)
    y_line = w * x_line_scaled + b

    x_line_original = x_line_scaled * (x_max - x_min) + x_min

    plt.figure(figsize=(10, 6))
    plt.scatter(area, house_price, alpha=0.7, edgecolors='k', label='Data')
    plt.plot(x_line_original, y_line, color='blue', linewidth=2,
             label=f'y = {w:.4f}x + {b:.4f}')
    plt.xlabel('House Area')
    plt.ylabel('House Price')
    plt.title(f'Prediction with Cost: {cost:.4f}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig("./assets/ov_plot.png")
    plt.show()


def mv_plot(w1, w2, w3, b, cost):
    scale = lambda x: (x - x.min()) / (x.max() - x.min())
    kitchen_scaled = scale(kitchen_area)
    rooms_scaled = scale(rooms)

    x1_min, x1_max = area.min(), area.max()
    x1_line_raw = np.linspace(x1_min, x1_max, 100)
    x1_line_scaled = (x1_line_raw - x1_min) / (x1_max - x1_min)

    x2_fixed = kitchen_scaled.mean()
    x3_fixed = rooms_scaled.mean()

    y_line = w1 * x1_line_scaled + w2 * x2_fixed + w3 * x3_fixed + b

    plt.figure(figsize=(10, 6))
    plt.scatter(area, house_price, alpha=0.7, edgecolors='k', label='Actual Data')
    plt.plot(x1_line_raw, y_line, color='blue', linewidth=2,
             label=f'y = {w1:.4f}*x1 + {w2:.4f}*mean(x2) + {w3:.4f}*mean(x3) + {b:.4f}')
    plt.xlabel('House Area')
    plt.ylabel('House Price of Unit Area')
    plt.title(f'Prediction with Cost: {cost:.4f}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig("./assets/mv_plot.png")
    plt.show()


def log_plot(w, b, x_test, y_test, cost, accuracy):
    x_vals = np.linspace(min(x_test), max(x_test), 100)
    y_vals = 1 / (1 + np.exp(-(w * x_vals + b)))  

    plt.plot(x_vals, y_vals, label="Sigmoid Curve")
    plt.scatter(x_test, y_test, color='red', label="Test Data")
    plt.xlabel("Bolld Glucose level")
    plt.ylabel("Diabetes")
    plt.title(f"Logistic Regression\nCost: {cost:.4f} Accuracy: {accuracy:.2f}%")
    plt.legend()
    plt.grid(True)
    #plt.savefig("./assets/log_plot.png")
    plt.show()


def plr_plot(w4, w3, w2, w1, b, cost, x_train, y_train):
    area = x_train  
    house_price = y_train

    x_min, x_max = area.min(), area.max()
    x_line = np.linspace(x_min, x_max, 200)

    y_line = (
        w4 * (x_line ** 4) +
        w3 * (x_line ** 3) +
        w2 * (x_line ** 2) +
        w1 * x_line +
        b
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(area, house_price, alpha=0.7, edgecolors='k', label='Training Data')
    plt.plot(x_line, y_line, linewidth=2, label='Polynomial Fit')

    plt.xlabel("Area")
    plt.ylabel("House Price")
    plt.title(f"Polynomial Regression (Cost = {cost:.4f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
