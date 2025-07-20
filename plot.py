import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("RealEstate.csv")
house_age = df['X2 house age']
latitude = df['X5 latitude']
longitude = df['X6 longitude']
house_price = df['Y house price of unit area']

def ov_plot(w, b):
    x_line = np.linspace(house_age.min(), house_age.max(), 100)
    y_line = w * x_line + b

    plt.figure(figsize=(10, 6))
    plt.scatter(house_age, house_price, alpha=0.7, edgecolors='k', label='Data')
    plt.plot(x_line, y_line, label=f'y = {w}x + {b}', color='blue', linewidth=2)
    plt.xlabel('House Age (years)')
    plt.ylabel('House Price of Unit Area')
    plt.title('House Age vs. House Price of Unit Area with Line y = wx + b')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ov_plot.png")
    plt.show()


def mv_plot(w1, w2, w3, b):
    x1_line = np.linspace(house_age.min(), house_age.max(), 100)
    x2_fixed = latitude.mean()
    x3_fixed = longitude.mean()
    
    y_line = w1 * x1_line + w2 * x2_fixed + w3 * x3_fixed + b

    plt.figure(figsize=(10, 6))
    plt.scatter(house_age, house_price, alpha=0.7, edgecolors='k', label='Data')
    plt.plot(x1_line, y_line, color='blue', linewidth=2, label='Model Prediction')
    plt.xlabel('House Age, Latitutude and Longitude')
    plt.ylabel('House Price of Unit Area')
    plt.title('House Age vs. House Price (Model Output)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mv_plot.png")
    plt.show()