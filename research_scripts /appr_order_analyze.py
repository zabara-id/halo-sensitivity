import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Папка с CSV ---
dir_path = "data/output/L1-192-results"

# --- генерация имени файла по порядку ---
def fname(order: int) -> str:
    if order == 1:
        return "1st_order.csv"
    elif order == 2:
        return "2nd_order.csv"
    elif order == 3:
        return "3rd_order.csv"
    else:
        return f"{order}th_order.csv"

orders_all = range(1, 7)
orders, med_errs = [], []

for n in orders_all:
    path = os.path.join(dir_path, fname(n))
    if not os.path.exists(path):
        print(f"⚠️  '{path}' not found, skip")
        continue

    df = pd.read_csv(path)
    err = np.sqrt((df["x_poly"]-df["x_int"])**2 +
                  (df["y_poly"]-df["y_int"])**2 +
                  (df["z_poly"]-df["z_int"])**2)
    med = err.max()
    orders.append(n)
    med_errs.append(med)
    print(f"order {n}: max  = {med:.2e}")

if not med_errs:
    print("Нет данных для графика.")
else:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(orders, med_errs, width=0.6)

    # логарифмическая ось
    plt.yscale('log')

    # сетка
    plt.grid(which='major', axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.4, alpha=0.4)

    # подписи
    plt.xlabel("Порядок аппроксимации конченого состояния [-]", fontsize=12)
    plt.ylabel("Максимальная ошибка по положению [км]", fontsize=12)

    # аннотации над столбцами
    for bar, val in zip(bars, med_errs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height*1.1,  # чуть выше
                 f"{val:.1e}",
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(orders)
    plt.tight_layout()
    plt.show()