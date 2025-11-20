import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from utils.libration_sense import du2km, tu2days

file_path = "../data/output/coefficients_wide/L2.csv"

df = pd.read_csv(file_path)

df.sort_values('T', inplace=True)
# Извлекаем столбцы в виде NumPy-массивов
T_array = tu2days(df['T'].values)
alpha1 = df['Alpha1'].values
alpha2 = df['Alpha2'].values
dev_max = df['Deviation Max'].values

# --- Функция для сглаживания данных сплайном ---
def spline_smooth(x, y, num_points_spline=300, k=5):
    """
    x, y - исходные массивы,
    num_points_spline - на сколько точек интерполировать,
    k - степень сплайна (3 = кубический).
    Возвращает (x_s, y_s) - сглаженные массивы.
    """
    x_smooth = np.linspace(x.min(), x.max(), num_points_spline)
    spline = make_interp_spline(x, y, k=k)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Сглаживаем Alpha1 и Alpha2
z0_alpha1_smooth, alpha1_smooth = spline_smooth(T_array, alpha1)
z0_alpha2_smooth, alpha2_smooth = spline_smooth(T_array, alpha2)

# Создаем фигуру и два объекта осей
fig, ax1 = plt.subplots(figsize=(12, 8))

# --- Левая ось: Alpha1 и Alpha2 ---
ax1.set_xlabel(r'T [дни]', fontsize=16)
ax1.set_ylabel(r'$\alpha_1$, $\alpha_2$ [безразм. ед.]', fontsize=16)

# 1) Точки (scatter) для Alpha1/Alpha2
ax1.scatter(T_array, alpha1, color='blue', s=20, label='Alpha1', alpha=0.6)
ax1.scatter(T_array, alpha2, color='green', s=20, label='Alpha2', alpha=0.6)

# 2) Сглаженные линии
line1, = ax1.plot(z0_alpha1_smooth, alpha1_smooth, color='blue', linewidth=2, label=r'$\alpha_1$')
line2, = ax1.plot(z0_alpha2_smooth, alpha2_smooth, color='green', linewidth=2, label=r'$\alpha_2$')

# Сетка для наглядности
ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)

# --- Правая ось: Deviation Max ---
ax2 = ax1.twinx()
ax2.set_ylabel(r'$d_{max}$ [тыс. км]', fontsize=16, color='red')
ax2.tick_params(axis='y', labelsize=14, colors='red')

# Окрасим правую ось в красный, чтобы выделить
ax2.spines['right'].set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.yaxis.label.set_color('red')

line3 = ax2.plot(T_array, np.array([du2km(x) for x in dev_max]) / 1000, color='red', linestyle='-', marker='D', markersize=4, 
                 label=r'$d_{max}$')[0]

# Собираем легенду
lines_1 = [line1, line2]
lines_2 = [line3]
labels_1 = [l.get_label() for l in lines_1]
labels_2 = [l.get_label() for l in lines_2]

# У ax1 будет своя легенда (Alpha1, Alpha2)
legend1 = ax1.legend(lines_1, labels_1, loc='center left', fontsize=14)
# Добавим ещё одну легенду на тот же Axes для Deviation
legend2 = ax1.legend(lines_2, labels_2, loc='lower right', fontsize=14)
# Чтобы легенды не перекрывали друг друга
ax1.add_artist(legend1)

# plt.title('Гало-орбиты L2', fontsize=14)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
plt.grid(True)
plt.show()