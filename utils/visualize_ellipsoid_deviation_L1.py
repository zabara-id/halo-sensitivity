import os
import csv
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def _load_ellipsoid_data() -> Tuple[np.ndarray, ...]:
    """
    Загружает данные из data/output/ellipsoid_deviation_L1.csv.

    Возвращает:
        (orbit_nums, max_mul, sampling_km, linear_km, da_km, ivp_km, floquet_km)
        все в виде numpy-массивов, отсортированных по номеру орбиты.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "output", "ellipsoid_deviation_L1.csv")

    orbit_nums: List[int] = []
    max_mul: List[float] = []
    sampling: List[float] = []
    linear: List[float] = []
    da_opt: List[float] = []
    ivp_opt: List[float] = []
    floquet: List[float] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orbit_nums.append(int(row["orbit_num"]))
            max_mul.append(float(row["max_mul"]))
            sampling.append(float(row["sampling_ellipsoid_km"]))
            linear.append(float(row["linear_ellipsoid_km"]))
            da_opt.append(float(row["da_optimization_km"]))
            ivp_opt.append(float(row["ivp_optimization_km"]))
            floquet.append(float(row["floquet_km"]))

    # Сортируем по номеру орбиты на случай, если CSV перемешан
    order = np.argsort(orbit_nums)
    orbit_nums_arr = np.array(orbit_nums)[order]
    max_mul_arr = np.array(max_mul)[order]
    sampling_arr = np.array(sampling)[order]
    linear_arr = np.array(linear)[order]
    da_opt_arr = np.array(da_opt)[order]
    ivp_opt_arr = np.array(ivp_opt)[order]
    floquet_arr = np.array(floquet)[order]

    return (
        orbit_nums_arr,
        max_mul_arr,
        sampling_arr,
        linear_arr,
        da_opt_arr,
        ivp_opt_arr,
        floquet_arr,
    )


def plot_ellipsoid_deviation_L1() -> None:
    """
    Строит составной график:
      - верхняя панель: абсолютные отклонения (км) 5 методов vs номер орбиты;
      - средняя панель: отношение отклонения каждого метода к IVP optimization;
      - нижняя панель: MAX_MUL (мультипликатор Флоке) vs номер орбиты.

    Так видно и абсолютные значения, и относительные ошибки методов
    относительно «эталонного» IVP, и поведение MAX_MUL без общей шкалы.
    """
    (
        orbit_nums,
        max_mul,
        sampling,
        linear,
        da_opt,
        ivp_opt,
        floquet,
    ) = _load_ellipsoid_data()

    fig, (ax_abs, ax_ratio, ax_mul) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [2.0, 1.6, 1.2]},
    )

    # 1) Абсолютные отклонения
    ax_abs.plot(
        orbit_nums,
        sampling,
        label="sampling (ellipsoid)",
        color="tab:blue",
        linewidth=1.3,
    )
    ax_abs.plot(
        orbit_nums,
        linear,
        label="linear ellipsoid (semi-analytic)",
        color="tab:orange",
        linewidth=1.3,
    )
    ax_abs.plot(
        orbit_nums,
        da_opt,
        label="DA optimization (multistart)",
        color="tab:green",
        linewidth=1.3,
    )
    ax_abs.plot(
        orbit_nums,
        ivp_opt,
        label="IVP optimization (multistart)",
        color="tab:red",
        linewidth=1.8,
    )
    ax_abs.plot(
        orbit_nums,
        floquet,
        label="Floquet (monodromy eigen)",
        color="tab:purple",
        linewidth=1.4,
        linestyle="--",
    )

    ax_abs.set_ylabel("Максимальное отклонение, км", fontsize=11)
    ax_abs.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_abs.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # 2) Отношения к IVP
    eps = 1e-9
    denom = np.maximum(ivp_opt, eps)
    sampling_rel = sampling / denom
    linear_rel = linear / denom
    da_rel = da_opt / denom
    floquet_rel = floquet / denom

    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="IVP baseline")

    ax_ratio.plot(
        orbit_nums,
        sampling_rel,
        label="sampling / IVP",
        color="tab:blue",
        linewidth=1.1,
    )
    ax_ratio.plot(
        orbit_nums,
        linear_rel,
        label="linear / IVP",
        color="tab:orange",
        linewidth=1.1,
    )
    ax_ratio.plot(
        orbit_nums,
        da_rel,
        label="DA / IVP",
        color="tab:green",
        linewidth=1.1,
    )
    ax_ratio.plot(
        orbit_nums,
        floquet_rel,
        label="Floquet / IVP",
        color="tab:purple",
        linewidth=1.2,
        linestyle="--",
    )

    ax_ratio.set_ylabel("Отношение к IVP\n(безразмерно)", fontsize=11)
    ax_ratio.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_ratio.legend(loc="upper right", fontsize=9, framealpha=0.9)
    # ax_ratio.set_yscale("log")

    # 3) MAX_MUL отдельно
    ax_mul.plot(
        orbit_nums,
        max_mul,
        label="MAX_MUL (Флоке)",
        color="black",
        linewidth=1.3,
        linestyle="-.",
    )
    ax_mul.set_xlabel("Номер гало-орбиты L1", fontsize=11)
    ax_mul.set_ylabel("MAX_MUL\n(мультипликатор Флоке)", fontsize=11)
    ax_mul.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_mul.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_mul.set_yscale('log')

    ax_mul.set_xlim(orbit_nums.min(), orbit_nums.max())

    fig.suptitle(
        "Отклонения на эллипсоиде неопределённости для семейства L1\n"
        "Абсолютные значения, относительные ошибки и MAX_MUL",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()


def main() -> None:
    plot_ellipsoid_deviation_L1()


if __name__ == "__main__":
    main()
