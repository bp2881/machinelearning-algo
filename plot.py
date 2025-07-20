import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("RealEstate.csv")
house_age = df['X2 house age']
latitude = df['X5 latitude']
longitude = df['X6 longitude']
house_price = df['Y house price of unit area']
convenience_stores = df['X4 number of convenience stores']
nearest_mrt_station = df['X3 distance to the nearest MRT station']

def ov_plot(w, b, cost):
    x_line = np.linspace(house_age.min(), house_age.max(), 100)
    y_line = w * x_line + b

    plt.figure(figsize=(10, 6))
    plt.scatter(house_age, house_price, alpha=0.7, edgecolors='k', label='Data')
    plt.plot(x_line, y_line, label=f'y = {w}x + {b}', color='blue', linewidth=2)
    plt.xlabel('House Age (years)')
    plt.ylabel('House Price of Unit Area')
    plt.title(f'Prediction with Cost: {cost:.4f}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ov_plot.png")
    plt.show()


def mv_plot(w1, w2, w3, b, cost):
    scale = lambda x: (x - x.min()) / (x.max() - x.min())

    x1_min, x1_max = house_age.min(), house_age.max()
    x1_line_raw = np.linspace(x1_min, x1_max, 100)
    x1_line_scaled = (x1_line_raw - x1_min) / (x1_max - x1_min)

    x2_fixed = (nearest_mrt_station.mean() - nearest_mrt_station.min()) / (nearest_mrt_station.max() - nearest_mrt_station.min())
    x3_fixed = (convenience_stores.mean() - convenience_stores.min()) / (convenience_stores.max() - convenience_stores.min())

    y_line = w1 * x1_line_scaled + w2 * x2_fixed + w3 * x3_fixed + b

    c = w2 * x2_fixed + w3 * x3_fixed + b

    plt.figure(figsize=(10, 6))
    plt.scatter(house_age, house_price, alpha=0.7, edgecolors='k', label='Actual Data')
    plt.plot(x1_line_raw, y_line, color='blue', linewidth=2, label=f'y = {w1}x + {c}')
    plt.xlabel('House Age')
    plt.ylabel('House Price of Unit Area')
    plt.title(f'Prediction with Cost: {cost:.4f}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mv_plot.png")
    plt.show()
