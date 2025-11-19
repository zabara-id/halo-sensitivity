import argparse
import csv
import os
from multiprocessing import Pool, cpu_count
from typing import Iterable, Tuple

from libration_sense import (
    ORBIT_TYPES_NUMS,
    du2km,
    get_maxdev_floquet_ellipsoid_with_vector,
    get_maxdev_linear_ellipsoid_with_vector,
    get_maxdev_optimization_ellipsoid_integrate_with_vector,
    get_maxdev_optimization_ellipsoid_with_vector,
    get_maxdev_sampling_ellipsoid_with_vector,
    get_xf,
    initial_state_parser,
    km2du,
    kmS2vu,
)


# Параметры эксперимента — такие же, как в main() в libration_sense.py
STD_POS_DU = km2du(3.0)          # 3 км по положению
STD_VEL_VU = kmS2vu(0.03e-3)     # 0.03 м/с по скорости
RADIUS = 3.0
DERORDER = 3
N_SAMPLES = 100_000
N_RANDOM_STARTS = 10
INIT_STRATEGY = "mixed_multistart"


def _csv_output_path(project_root: str, orbit_type: str) -> str:
    return os.path.join(project_root, "data", "output", f"ellipsoid_deviation_{orbit_type}__.csv")


def _compute_for_orbit(args: Tuple[str, int]) -> Tuple[int, float, float, float, float, float, float]:
    """
    Считает значения пяти методов для одной орбиты семейства L1 или L2.

    Возвращает:
      (номер орбиты,
       sampling_km,
       linear_ellipsoid_km,
       da_optimization_km,
       ivp_optimization_km,
       floquet_km)
    """
    orbit_type, orbit_num = args
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


def _orbit_numbers(orbit_type: str) -> Iterable[int]:
    return range(1, ORBIT_TYPES_NUMS[orbit_type] + 1)


def run_batch(orbit_type: str) -> str:
    """
    Запускает батч‑расчёт для всех орбит семейства, сохраняет CSV и печатает прогресс.

    Возвращает путь к сохранённому CSV.
    """
    orbit_type = orbit_type.upper()
    if orbit_type not in ORBIT_TYPES_NUMS:
        raise ValueError(f"Unsupported orbit type '{orbit_type}'. Use one of {list(ORBIT_TYPES_NUMS)}.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = _csv_output_path(project_root, orbit_type)

    print(f"Saving results to: {output_path}")
    print(f"Total orbits for {orbit_type}: {ORBIT_TYPES_NUMS[orbit_type]}")
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
        ) in pool.imap_unordered(_compute_for_orbit, ((orbit_type, n) for n in _orbit_numbers(orbit_type))):
            rows.append((orbit_num, max_mul, s_km, lin_km, da_km, ivp_km, floq_km))
            print(
                f"Orbit {orbit_type} #{orbit_num} done "
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
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch deviation calculation for L1/L2 halo orbits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--orbit-type",
        choices=sorted(ORBIT_TYPES_NUMS),
        default="L1",
        help="Halo family to process",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_batch(args.orbit_type)


if __name__ == "__main__":
    main()
