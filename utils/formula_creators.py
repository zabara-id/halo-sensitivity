import numpy as np
from tqdm import tqdm
from itertools import product
from tqdm import tqdm
from daceypy import DA
from scipy.optimize import minimize

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
    std_pos_values = np.linspace(0, km2du(1), grid_density)  # от 0 до 8 км
    std_vel_values = np.linspace(0, kmS2vu(0.01e-3), grid_density)  # от 0 до 0.05 м / с 

    # Данные для нормировки
    pos_max = np.max(std_pos_values)
    vel_max = np.max(std_vel_values)

    # Генерируем матрицу A и вектор y
    N = grid_density**2
    A = np.zeros((N, 2))
    y = np.zeros(N)

    # Заполняем нормированную матрицу A и вектор y
    index = 0
    for std_pos in tqdm(std_pos_values, desc="std_pos loop"):
        for std_vel in std_vel_values:
            A[index] = [std_pos / pos_max, std_vel / vel_max]
            y[index] = get_maxdeviation_wo_integrate(
                orbit_type, number_of_orbit, xf, std_pos, std_vel
            )
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


def n_finder(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,
    grid_density: int = 5,
    amount_of_points: int = 10000,
    seed: int | None = None,
    reuse_noise: bool = True,
) -> float:
    
    std_pos_values = np.linspace(0, km2du(1), grid_density)  # от 0 до 1 км
    std_vel_values = np.linspace(0, kmS2vu(0.01e-3), grid_density)  # от 0 до 0.01 м / с 

    # Генерируем матрицу A и вектор y
    N = grid_density**2
    A = np.zeros((N, 2))
    y_du = np.zeros(N)

    # pos_max = np.max(std_pos_values)
    # vel_max = np.max(std_vel_values)

    pos_max = std_pos_values[-1]
    vel_max = std_vel_values[-1]

    # Подготовим переиспользуемый шум (единичные нормальные отклонения)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    unit_deltas = rng.normal(0.0, 1.0, (amount_of_points, 6)) if reuse_noise else None

    # Заполняем A и y
    index = 0
    for std_pos, std_vel in tqdm(
        product(std_pos_values, std_vel_values),
        total=len(std_pos_values) * len(std_vel_values),
        desc="progress"
    ):
        A[index] = [std_pos / pos_max, std_vel / vel_max]
        y_du[index] = get_maxdeviation_wo_integrate(
            orbit_type,
            number_of_orbit,
            xf,
            std_pos,
            std_vel,
            amount_of_points=amount_of_points,
            unit_deltas=unit_deltas,
            seed=None if reuse_noise else seed,
        )
        index += 1

    y_normed = y_du / np.max(y_du)
    A_normed_du_and_vu = A

    def loss(n: float) -> float:
        """
        Целевая функция для поиска степени n оптимизационной процедурой.

        Args:
            n (float): Степень нормы функционала для аппроксимирующей формулы.

        Returns:
            float: Значение среднеквадратичного отклонения в точке n.
        """
        y_powered = np.power(y_normed, n)
        A_powered = np.power(A, n)
        alpha_star = np.linalg.inv(A_powered.T @ A_powered) @ A_powered.T @ y_powered
        core = np.power(y_powered - np.dot(A_powered, alpha_star), 2)

        return np.sum(core) / np.shape(y_powered)[0]

    # Ограничения на n
    bounds = [(0.000001, None)]

    # Начальное приближение
    n_initial = 1.991

    # Метод сопряжённых направлений
    n_opt = minimize(loss, n_initial, method='Powell', bounds=bounds)

    return n_opt.x[0], A_normed_du_and_vu, y_du
