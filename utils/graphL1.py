import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.libration_sense import du2km


file_path = "data/output/L1_DA_opt.csv"
print(file_path)
df = pd.read_csv(file_path)

df.sort_values('z0', inplace=True)
# Извлекаем столбцы в виде NumPy-массивов
z0 = np.array([du2km(x) for x in df['z0'].values]) / 1000
alpha1 = df['Alpha1'].values
alpha2 = df['Alpha2'].values
dev_max = df['Deviation Max'].values

# Создаем фигуру и два объекта осей
fig, ax1 = plt.subplots(figsize=(12, 8))

# --- Левая ось: Alpha1 и Alpha2 ---
ax1.set_xlabel(r'$z_0$ [тыс. км]', fontsize=16)
ax1.set_ylabel(r'$\alpha_1$, $\alpha_2$ [-]', fontsize=16)

# 1) Точки (scatter) для Alpha1/Alpha2
scatter1 = ax1.scatter(z0, alpha1, color='blue', s=20, label=r'$\alpha_1$', alpha=0.6)
scatter2 = ax1.scatter(z0, alpha2, color='green', s=20, label=r'$\alpha_2$', alpha=0.6)

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

scatter3 = ax2.scatter(z0, np.array([du2km(x) for x in dev_max]) / 1000, color='red', s=16, marker='D', alpha=0.7,
                       label=r'$d_{max}$')

# Собираем легенду
lines_1 = [scatter1, scatter2]
lines_2 = [scatter3]
labels_1 = [h.get_label() for h in lines_1]
labels_2 = [h.get_label() for h in lines_2]

# У ax1 будет своя легенда (Alpha1, Alpha2)
legend1 = ax1.legend(lines_1, labels_1, loc='center left', fontsize=14)
# Добавим ещё одну легенду на тот же Axes для Deviation
legend2 = ax1.legend(lines_2, labels_2, loc='right', fontsize=14)
# Чтобы легенды не перекрывали друг друга
ax1.add_artist(legend1)

# plt.title('Гало-орбиты L1', fontsize=14)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
plt.grid(True)
plt.show()
