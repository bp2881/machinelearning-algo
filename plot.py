import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("RealEstate.csv")
house_age = df['X2 house age']
house_price = df['Y house price of unit area']

def plot(w, b):
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
    plt.show()
