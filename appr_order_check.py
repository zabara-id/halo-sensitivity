import csv
from multiprocessing import Pool, cpu_count

import numpy as np
from daceypy import DA, array
from scipy.integrate import solve_ivp


from utils.libration_sense import (
    CR3BP,
    RK78,
    cr3bp, 
    initial_state_parser,
    km2du,
    du2km,
    vu2kms,
    kmS2vu
    )


def du_vu2km_kms(du_vu_vec: np.ndarray):

    km = du2km(du_vu_vec[:3])
    kms = vu2kms(du_vu_vec[3:])

    km_kms_vec = np.concatenate((km, kms), axis=None)

    return km_kms_vec
    


def get_haloorbit_dots(
        haloorbit_type: str,
        haloorbit_num: int,
        haloorbit_dots_num: int
    ) -> np.ndarray:
    # НУ
    x0, z0, vy0, T, _, __ = initial_state_parser(haloorbit_type, haloorbit_num)
    initial_state = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
    t_eval = np.linspace(0.0, T, haloorbit_dots_num, endpoint=True)

    # Интегрирование
    sol = solve_ivp(
        fun=cr3bp,
        t_span=(0.0, T),
        y0=initial_state,
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-13,
        atol=1e-13
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return sol.y.T


def process_halodot(args):

    (ord, halodot, T, dots_around_halodots_amount, seed) = args

    # Инициализация DA-среды
    DA.init(ord, 6)

    # Начальное положение как DA-массив для подхода полиномиальной аппрокимации
    initial_state_da = array.identity(6)
    initial_state_da += halodot
    initial_state_cons = initial_state_da.cons()

    # Отклонения от начального положения
    np.random.seed(seed)
    std_pos = km2du(1)                                      # 1 км
    std_vel = kmS2vu(0.01e-3)                               # 0.01 м/c
    std_devs = np.array([std_pos] * 3 + [std_vel] * 3)
    initial_deviations = np.random.normal(0,
                                          std_devs,
                                          (dots_around_halodots_amount, 6))
    limits = np.array([3 * std_pos] * 3 + [3 * std_vel] * 3)
    initial_deviations = np.clip(initial_deviations, -limits, limits)

    new_initial_states = initial_state_cons + initial_deviations   # новые НУ вокруг точки гало-орбиты

    xf = RK78(initial_state_da, 0.0, T, CR3BP)

    results = []
    for (initial_deviation, new_initial_state)  in zip(initial_deviations, new_initial_states):
        xf_of_T_polyappr = xf.eval(initial_deviation)

        xf_of_T_integr = solve_ivp(
            fun=cr3bp,
            t_span=(0.0, T),
            y0=new_initial_state,
            method='DOP853',
            rtol=1e-13,
            atol=1e-13
        ).y.T[-1]

        xf_of_T_polyappr = du_vu2km_kms(xf_of_T_polyappr)
        xf_of_T_integr = du_vu2km_kms(xf_of_T_integr)

        results.append(
            tuple(xf_of_T_polyappr) +
            tuple(xf_of_T_integr)
        )

    print(f'Окрестность точки {halodot} гало-орбиты проанализирована!')

    return results


if __name__ == "__main__":
    # Порядок полиномиальной аппроксимации, который проверяется
    ord = 7

    # Файл для записи результатов
    filepath="L1-192-results/7th_order.csv"
    
    # Гало-орбита из каталога
    haloorbit_type = 'L1'
    haloorbit_num = 192

    haloorbit_dots_num = 100       # кол-во получаемых точек гало-орбиты
    dots_around_num = 10_000       # кол-во генерируемых отклонений вокруг каждой точки гало-орбиты

    # Период      
    _, _, _, T, _, _ = initial_state_parser(haloorbit_type, haloorbit_num)

    # Получение самих точек гало-орбиты
    halodots = get_haloorbit_dots(haloorbit_type,                   
                                  haloorbit_num,
                                  haloorbit_dots_num)

    args_list = [
        (ord, halodot, T, dots_around_num, 42 + i)
        for i, halodot in enumerate(halodots)
    ]

    # Параллельное вычисление
    with Pool(cpu_count()) as pool:
        results_list = pool.map(process_halodot, args_list)

        # results_list — это список списков
    header = [
        "x_poly", "y_poly", "z_poly", "vx_poly", "vy_poly", "vz_poly",
        "x_int",  "y_int",  "z_int",  "vx_int",  "vy_int",  "vz_int"
    ]

    # Собираем всё в один большой список
    all_rows = [row for results in results_list for row in results]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
