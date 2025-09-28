import numpy as np
from daceypy import DA

from utils.libration_sense import (
    du2km,
    km2du,
    get_xf,
    vu2kms,
    kmS2vu,
    initial_state_parser,
    get_maxdeviation_wo_integrate,
)


def alpha_xfinder(n: float, orbit_type: str,
                 number_of_orbit: int,
                 xf: DA,
                 grid_density: int = 5) -> tuple:
    # Создаем сетку значений
    std_pos_values = np.linspace(0, km2du(8), grid_density)  # от 0 до 8 км
    std_vel_values = np.linspace(0, kmS2vu(0.05e-3), grid_density)  # от 0 до 0.05 м / с 

    # Данные для нормировки
    pos_max = np.max(std_pos_values)
    vel_max = np.max(std_vel_values)

    # Генерируем матрицу A и вектор y
    N = grid_density**2
    A = np.zeros((N, 2))
    y = np.zeros(N)

    # Заполняем нормированную матрицу A и вектор y
    index = 0
    for std_pos in std_pos_values:
        for std_vel in std_vel_values:
            A[index] = [std_pos / pos_max, std_vel / vel_max]
            y[index] = get_maxdeviation_wo_integrate(orbit_type, number_of_orbit, xf, std_pos, std_vel)
            index += 1

    deviation_max = np.max(y)
    
    y_normed = y / deviation_max
    y_powered = np.power(y_normed, n)
    A_powered = np.power(A, n)
    alpha_star = np.linalg.inv(A_powered.T @ A_powered) @ A_powered.T @ y_powered
    
    return alpha_star, deviation_max

def alpha_finder_of_n(A_normed, y, n):
    rmin_max = np.max(y)
    y_normed = y / rmin_max
    y_powered = np.power(y_normed, n)
    A_powered = np.power(A_normed, n)
    return np.linalg.inv(A_powered.T @ A_powered) @ A_powered.T @ y_powered, rmin_max