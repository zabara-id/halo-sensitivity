import os
import csv
from multiprocessing import Pool, cpu_count
from typing import Iterable, Tuple

from libration_sense import (
    ORBIT_TYPES_NUMS,
    initial_state_parser,
    get_xf,
    get_maxdev_sampling_ellipsoid_with_vector,
    get_maxdev_linear_ellipsoid_with_vector,
    get_maxdev_optimization_ellipsoid_with_vector,
    get_maxdev_optimization_ellipsoid_integrate_with_vector,
    get_maxdev_floquet_ellipsoid_with_vector,
    km2du,
    kmS2vu,
    du2km,
)


ORBIT_TYPE = "L1"
NUM_ORBITS = ORBIT_TYPES_NUMS[ORBIT_TYPE]

# Параметры эксперимента — такие же, как в main() в libration_sense.py
STD_POS_DU = km2du(2.0)          # 2 км по положению
STD_VEL_VU = kmS2vu(0.02e-3)     # 0.02 м/с по скорости
RADIUS = 4.0
DERORDER = 3
N_SAMPLES = 100_000
N_RANDOM_STARTS = 10
INIT_STRATEGY = "mixed_multistart"


def _compute_for_orbit(orbit_num: int) -> Tuple[int, float, float, float, float, float, float]:
    """
    Считает значения пяти методов для одной орбиты семейства L1.

    Возвращает:
      (номер орбиты,
       sampling_km,
       linear_ellipsoid_km,
       da_optimization_km,
       ivp_optimization_km,
       floquet_km)
    """
    orbit_type = ORBIT_TYPE
    std_pos = STD_POS_DU
    std_vel = STD_VEL_VU

    # Берём MAX_MUL из таблицы исходных НУ
    _, _, _, _, _, max_mul = initial_state_parser(orbit_type, orbit_num)

    # DA‑карта через период (для методов 1–3)
    xf = get_xf(orbit_type, orbit_num, derorder=DERORDER)

    # 1) sampling (ellipsoid)
    dev_sampling_du, _ = get_maxdev_sampling_ellipsoid_with_vector(
        orbit_type,
        orbit_num,
        xf,
        std_pos,
        std_vel,
        derorder=DERORDER,
        amount_of_points=N_SAMPLES,
        radius=RADIUS,
        seed=42
    )

    # 2) linear ellipsoid (semi-analytic formula)
    dev_linear_du, _ = get_maxdev_linear_ellipsoid_with_vector(
        xf,
        std_pos,
        std_vel,
        radius=RADIUS,
    )

    # 3) DA optimization (multistart)
    dev_da_du, _ = get_maxdev_optimization_ellipsoid_with_vector(
        orbit_type,
        orbit_num,
        xf,
        std_pos,
        std_vel,
        radius=RADIUS,
        verbose=False,
        n_random_starts=N_RANDOM_STARTS,
        init_strategy=INIT_STRATEGY,
    )

    # 4) IVP optimization (multistart)
    dev_ivp_du, _ = get_maxdev_optimization_ellipsoid_integrate_with_vector(
        orbit_type,
        orbit_num,
        xf,
        std_pos,
        std_vel,
        radius=RADIUS,
        verbose=False,
        n_random_starts=N_RANDOM_STARTS,
        init_strategy=INIT_STRATEGY,
    )

    # 5) Floquet (monodromy eigen)
    dev_floq_du, _ = get_maxdev_floquet_ellipsoid_with_vector(
        orbit_type,
        orbit_num,
        std_pos,
        std_vel,
        radius=RADIUS,
    )

    # Переводим все значения из DU в км
    return (
        orbit_num,
        float(max_mul),
        du2km(dev_sampling_du),
        du2km(dev_linear_du),
        du2km(dev_da_du),
        du2km(dev_ivp_du),
        du2km(dev_floq_du),
    )


def _orbit_numbers() -> Iterable[int]:
    return range(1, NUM_ORBITS + 1)


def main() -> None:
    """
    Запускает батч‑расчёт для всех орбит семейства L1 (1..251),
    сохраняет результаты в CSV и печатает прогресс в консоль.
    """
    # Путь к CSV: data/output/ellipsoid_deviation_L1.csv (от корня проекта)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ellipsoid_deviation_L1__.csv")

    print(f"Saving results to: {output_path}")
    print(f"Total orbits for {ORBIT_TYPE}: {NUM_ORBITS}")
    print(f"Using up to {cpu_count()} processes\n")

    rows: list[Tuple[int, float, float, float, float, float, float]] = []

    with Pool(processes=cpu_count()) as pool:
        for (
            orbit_num,
            max_mul,
            s_km,
            lin_km,
            da_km,
            ivp_km,
            floq_km,
        ) in pool.imap_unordered(_compute_for_orbit, _orbit_numbers()):
            rows.append((orbit_num, max_mul, s_km, lin_km, da_km, ivp_km, floq_km))
            print(
                f"Orbit {ORBIT_TYPE} #{orbit_num} done "
                f"(MAX_MUL={max_mul:.6f}): "
                f"sampling={s_km:.3f} km, linear={lin_km:.3f} km, "
                f"DA={da_km:.3f} km, IVP={ivp_km:.3f} km, Floquet={floq_km:.3f} km",
                flush=True,
            )

    # Сортируем по номеру орбиты перед записью
    rows.sort(key=lambda r: r[0])

    # Записываем CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "orbit_num",
                "max_mul",
                "sampling_ellipsoid_km",
                "linear_ellipsoid_km",
                "da_optimization_km",
                "ivp_optimization_km",
                "floquet_km",
            ]
        )
        for row in rows:
            writer.writerow(row)

    print("\nAll orbits processed. CSV saved.")


if __name__ == "__main__":
    main()
