import argparse
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _csv_path(project_root: str, orbit_type: str) -> str:
    return os.path.join(project_root, "data", "output", f"ellipsoid_deviation_{orbit_type}__.csv")


def _load_ellipsoid_data(orbit_type: str) -> Tuple[np.ndarray, ...]:
    """
    Загружает данные из data/output/ellipsoid_deviation_<orbit>.csv.

    Возвращает:
        (orbit_nums, max_mul, sampling_km, linear_km, da_km, ivp_km, floquet_km)
        все в виде numpy-массивов, отсортированных по номеру орбиты.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = _csv_path(project_root, orbit_type)

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


def plot_ellipsoid_deviation(orbit_type: str = "L1") -> None:
    """
    Строит составной график:
      - верхняя панель: абсолютные отклонения (км) 5 методов vs номер орбиты;
      - средняя панель: отношение отклонения каждого метода к IVP optimization;
      - нижняя панель: MAX_MUL (мультипликатор Флоке) vs номер орбиты.

    Так видно и абсолютные значения, и относительные ошибки методов
    относительно «эталонного» IVP, и поведение MAX_MUL без общей шкалы.
    """
    orbit_type = orbit_type.upper()
    (
        orbit_nums,
        max_mul,
        sampling,
        linear,
        da_opt,
        ivp_opt,
        floquet,
    ) = _load_ellipsoid_data(orbit_type)

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
        label=r'$d_{\max}^{\text{samp}}$',
        color="tab:blue",
        linewidth=1.3,
    )
    ax_abs.plot(
        orbit_nums,
        linear,
        label=r'$d_{\max}^{\text{lin}}$',
        color="tab:orange",
        linewidth=1.3,
    )
    ax_abs.plot(
        orbit_nums,
        da_opt,
        label=r'$d_{\max}^{\text{DA-opt}}$',
        color="tab:green",
        linewidth=1.3,
    )
    ax_abs.plot(
        orbit_nums,
        ivp_opt,
        label=r'$d_{\max}^{\text{IVP-opt}}$',
        color="tab:red",
        linewidth=1.8,
    )
    ax_abs.plot(
        orbit_nums,
        floquet,
        label=r'$d_{\max}^{\text{Fl}}$',
        color="tab:purple",
        linewidth=1.4,
        linestyle="--",
    )

    ax_abs.set_ylabel(r'$d_{\max}$ [км]', fontsize=11)
    ax_abs.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_abs.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # 2) Отношения к IVP
    eps = 1e-9
    denom = np.maximum(ivp_opt, eps)
    sampling_rel = sampling / denom
    linear_rel = linear / denom
    da_rel = da_opt / denom
    floquet_rel = floquet / denom

    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)

    ax_ratio.plot(
        orbit_nums,
        sampling_rel,
        label=r'$d_{\max}^{samp} / d_{\max}^{\text{IVP-opt}}$',
        color="tab:blue",
        linewidth=1.1,
    )
    ax_ratio.plot(
        orbit_nums,
        linear_rel,
        label=r'$d_{\max}^{lin} / d_{\max}^{\text{IVP-opt}}$',
        color="tab:orange",
        linewidth=1.1,
    )
    ax_ratio.plot(
        orbit_nums,
        da_rel,
        label=r'$d_{\max}^{DA-opt} / d_{\max}^{\text{IVP-opt}}$',
        color="tab:green",
        linewidth=1.1,
    )
    ax_ratio.plot(
        orbit_nums,
        floquet_rel,
        label=r'$d_{\max}^{Fl} / d_{\max}^{\text{IVP-opt}}$',
        color="tab:purple",
        linewidth=1.2,
        linestyle="--",
    )

    ax_ratio.set_ylabel(r'$d_{\max}^{method} / d_{\max}^{\text{IVP-opt}}$ [-]', fontsize=11)
    ax_ratio.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_ratio.legend(loc="upper right", fontsize=9, framealpha=0.9)
    # ax_ratio.set_yscale("log")

    # 3) MAX_MUL отдельно
    ax_mul.plot(
        orbit_nums,
        max_mul,
        color="black",
        linewidth=1.3,
        linestyle="-.",
    )
    ax_mul.set_xlabel(r"Порядковый номер гало-орбиты семейства $L_2$ [-]", fontsize=11)

    ax_mul.set_ylabel(r'$\mu_{\max}$ [-]', fontsize=11)
    ax_mul.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    # ax_mul.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_mul.set_yscale('log')

    ax_mul.set_xlim(orbit_nums.min(), orbit_nums.max())

    # fig.suptitle(
    #     f"Отклонения на эллипсоиде неопределённости для семейства {orbit_type}\n"
    #     "Абсолютные значения, относительные ошибки и MAX_MUL",
    #     fontsize=13,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ellipsoid deviation metrics for L1/L2 halo orbits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--orbit-type",
        choices=["L1", "L2"],
        default="L1",
        help="Halo family to visualize",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_ellipsoid_deviation(args.orbit_type)


if __name__ == "__main__":
    main()
