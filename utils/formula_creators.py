from itertools import product

import numpy as np
from tqdm import tqdm
from tqdm.auto import tqdm as atqdm
from itertools import product
from tqdm import tqdm
from daceypy import DA
from scipy.optimize import minimize, minimize_scalar

from utils.libration_sense import (
    km2du,
    kmS2vu,
    get_maxdev_sampling_no_integrate,
    get_maxdev_optimization_ellipsoid,
    get_maxdev_sampling_ellipsoid,
    get_maxdev_linear_ellipsoid
)


def alpha_xfinder(
    n: float,
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,
    grid_density: int = 5,
    amount_of_points: int = 10_000,
    seed: int | None = 42,
    reuse_noise: bool = True,
) -> tuple:
    """Оценка коэффициентов alpha для фиксированного n.

    Делает то же сглаживание, что и на графиках: один и тот же
    набор единичных шумов используется во всех точках сетки,
    чтобы убрать «рваность» из-за независимых выборок.
    """
    # Создаем сетку значений
    std_pos_values = np.linspace(0, km2du(1), grid_density)
    std_vel_values = np.linspace(0, kmS2vu(0.01e-3), grid_density)

    # Данные для нормировки
    pos_max = std_pos_values[-1]
    vel_max = std_vel_values[-1]

    # Общий набор шумов для всей сетки
    rng = np.random.default_rng(seed)
    unit_deltas = (
        rng.normal(0.0, 1.0, (amount_of_points, 6)) if reuse_noise else None
    )

    # Генерируем матрицу A и вектор y
    N = grid_density**2
    A = np.zeros((N, 2))
    y = np.zeros(N)

    # Заполняем нормированную матрицу A и вектор y
    index = 0
    for std_pos in tqdm(std_pos_values, desc="std_pos loop"):
        for std_vel in std_vel_values:
            A[index] = [std_pos / pos_max, std_vel / vel_max]
            y[index] = get_maxdev_sampling_no_integrate(
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
    
    std_pos_values = np.linspace(0, km2du(10), grid_density)  # от 0 до 10 км
    std_vel_values = np.linspace(0, kmS2vu(0.10e-3), grid_density)  # от 0 до 0.10 см / с

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
    for std_pos, std_vel in product(std_pos_values, std_vel_values):
        A[index] = [std_pos / pos_max, std_vel / vel_max]
        # y_du[index] = get_maxdev_sampling_no_integrate(
        #     orbit_type,
        #     number_of_orbit,
        #     xf,
        #     std_pos,
        #     std_vel,
        #     amount_of_points=amount_of_points,
        #     unit_deltas=unit_deltas,
        #     seed=None if reuse_noise else seed,
        # )

        # y_du[index] = get_maxdev_sampling_ellipsoid(
        #     orbit_type,
        #     number_of_orbit,
        #     xf,
        #     std_pos,
        #     std_vel,
        #     amount_of_points=amount_of_points,
        #     unit_deltas=unit_deltas,
        #     seed=None if reuse_noise else seed,
        #     radius=3.0
        # )

        y_du[index] = get_maxdev_optimization_ellipsoid(
            orbit_type,
            number_of_orbit,
            xf,
            std_pos,
            std_vel,
            radius=3.0
        )

        # y_du[index] = get_maxdev_linear_ellipsoid(
        #     xf,
        #     std_pos,
        #     std_vel,
        #     radius=3.0
        # )

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
    n_initial = 1.995

    # Метод сопряжённых направлений
    n_opt = minimize(loss, n_initial, method='Powell', bounds=bounds)

    return n_opt.x[0], A_normed_du_and_vu, y_du


def n_finder_performed(
    orbit_type: str,
    number_of_orbit: int,
    xf,
    grid_density: int = 5,
    amount_of_points: int = 10_000,
    seed: int | None = None,
    reuse_noise: bool = True,
    n_bounds=(0.5, 4.0),
    ridge: float = 0.0,
):
    # сетка
    std_pos_values = np.linspace(0, km2du(8), grid_density)
    std_vel_values = np.linspace(0, kmS2vu(0.05e-3), grid_density)
    POS, VEL = np.meshgrid(std_pos_values, std_vel_values, indexing="ij")
    pos_max, vel_max = std_pos_values[-1], std_vel_values[-1]

    # нормированные признаки (N x 2)
    A = np.column_stack([(POS/pos_max).ravel(), (VEL/vel_max).ravel()])

    # общий набор шумов
    rng = np.random.default_rng(seed)
    unit_deltas = rng.normal(0.0, 1.0, (amount_of_points, 6)) if reuse_noise else None

    # вычисление цели для каждой точки сетки (узкое место → желательно распараллелить/кэшировать)
    def one_val(sp, sv):
        return get_maxdev_sampling_no_integrate(
            orbit_type, number_of_orbit, xf, sp, sv,
            amount_of_points=amount_of_points,
            unit_deltas=unit_deltas,
            seed=None if reuse_noise else seed,
        )

    # векторизованный расчёт y (можно заменить на joblib.Parallel)
    y_du = np.array([one_val(sp, sv) for sp, sv in zip(POS.ravel(), VEL.ravel())])

    # нормируем на максимум
    y_normed = y_du / np.max(y_du)
    A_base = A.copy()

    # небольшое eps для нулей (только в потере, чтобы n→0 не портил кондиционирование)
    eps = 1e-12
    A_pos = np.maximum(A_base, eps)
    y_pos = np.maximum(y_normed, eps)

    def loss(n: float) -> float:
        y_p = y_pos**n
        A_p = A_pos**n
        if ridge > 0.0:
            # (A^T A + λI)^{-1} A^T y  — через solve
            AtA = A_p.T @ A_p
            AtA.flat[0::AtA.shape[0]+1] += ridge  # добавим λ на диагональ
            alpha = np.linalg.solve(AtA, A_p.T @ y_p)
        else:
            # устойчивее и обычно быстрее, чем explicit inv норм. уравнений
            alpha, *_ = np.linalg.lstsq(A_p, y_p, rcond=None)
        resid = y_p - A_p @ alpha
        return float(np.mean(resid*resid))

    res = minimize_scalar(loss, bounds=n_bounds, method="bounded", options={"xatol": 1e-3})
    n_opt = float(res.x)

    return n_opt, A_base, y_du
