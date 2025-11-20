"""
Аппроксимация Alpha1/Alpha2 для L2 с гибридом
    baseline (квадратик) + общая логистика,
чтобы дать лёгкий загиб слева, который виден в данных.
График без сплайнов: точки + наши кривые + d_max справа.

f(T) = p0 + p1*T + p2*T^2 + A / (1 + exp(k * (T - T_mid)))^(1/nu)

* T — период в днях.
* A, k, T_mid, nu управляют хвостом справа; p0..p2 задают мягкий наклон/кривизну слева.
Запуск: python3 utils/fit_alpha_param_L2.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utils.libration_sense import du2km, tu2days


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "output" / "coefficients" / "koefficients_data_L2.csv"


def quad_logistic(
    x: np.ndarray,
    p0: float,
    p1: float,
    p2: float,
    A: float,
    k: float,
    x_mid: float,
    nu: float,
) -> np.ndarray:
    """Базовый квадратик + асимметричная логистика для сурового хвоста справа."""
    baseline = p0 + p1 * x + p2 * x * x
    logistic = A / (1.0 + np.exp(k * (x - x_mid))) ** (1.0 / nu)
    return baseline + logistic


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def load_alpha_data(path: Path = DEFAULT_DATA) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает T (дни) и Alpha1/Alpha2/Deviation Max из CSV."""
    df = pd.read_csv(path).sort_values("T")
    t_days = tu2days(df["T"].values)
    return (
        t_days,
        df["Alpha1"].to_numpy(),
        df["Alpha2"].to_numpy(),
        df["Deviation Max"].to_numpy(),
    )


@dataclass
class FitResult:
    label: str
    p0: float
    p1: float
    p2: float
    A: float
    k: float
    x_mid: float
    nu: float
    r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return quad_logistic(x, self.p0, self.p1, self.p2, self.A, self.k, self.x_mid, self.nu)


def fit_alpha_curve(x: np.ndarray, y: np.ndarray, label: str) -> FitResult:
    """Подбирает квадратический baseline + логистику; bounds мягкие для устойчивости."""
    y_min = float(np.quantile(y, 0.05))
    y_max = float(np.quantile(y, 0.95))
    span = float(x.max() - x.min())
    p0_guess = float(np.median(y))
    p1_guess = 0.0
    p2_guess = 0.0
    A_guess = float(y_max - y_min)
    p0_tuple = (p0_guess, p1_guess, p2_guess, A_guess, 0.5, float(np.median(x)), 1.0)
    bounds = (
        # p0, p1, p2, A, k, x_mid, nu
        (
            y_min - 0.2,
            -1.0,
            -0.1,
            -2.0,  # A может быть отриц. для растущей/убывающей сигмоиды
            -5.0,
            x.min() - 0.2 * span,
            0.2,
        ),
        (
            y_max + 0.2,
            1.0,
            0.1,
            2.0,
            5.0,
            x.max() + 0.2 * span,
            5.0,
        ),
    )
    try:
        params, _ = curve_fit(
            quad_logistic,
            x,
            y,
            p0=p0_tuple,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        params, _ = curve_fit(quad_logistic, x, y, p0=p0_tuple, maxfev=20000)

    y_hat = quad_logistic(x, *params)
    return FitResult(
        label=label,
        p0=float(params[0]),
        p1=float(params[1]),
        p2=float(params[2]),
        A=float(params[3]),
        k=float(params[4]),
        x_mid=float(params[5]),
        nu=float(params[6]),
        r2=float(r2_score(y, y_hat)),
    )


def print_result(result: FitResult) -> None:
    direction = "убывающая" if result.k > 0 else "возрастающая"
    formula = (
        f"{result.label}(T) = ({result.p0:.5f}) + ({result.p1:.5f})*T + ({result.p2:.5f})*T^2 "
        f"+ ({result.A:.5f}) / (1 + exp({result.k:.5f} * (T - {result.x_mid:.3f})))^(1/{result.nu:.3f})"
    )
    print(f"{result.label}: {direction} сигмоида, R^2 = {result.r2:.4f}")
    print(f"    {formula}")


def plot_with_dev(
    t: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    dev_max: np.ndarray,
    fit1: FitResult,
    fit2: FitResult,
) -> None:
    """Построение в стиле graphL2: точки, наши кривые и d_max справа."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Точки данных
    ax1.scatter(t, alpha1, color="blue", s=20, alpha=0.6, label="Alpha1")
    ax1.scatter(t, alpha2, color="green", s=20, alpha=0.6, label="Alpha2")

    # Сигмоидальные аппроксимации
    t_dense = np.linspace(t.min(), t.max(), 1000)
    line1, = ax1.plot(
        t_dense,
        fit1.predict(t_dense),
        color="blue",
        linewidth=2.5,
        label=f"alpha1 fit (R2={fit1.r2:.3f})",
    )
    line2, = ax1.plot(
        t_dense,
        fit2.predict(t_dense),
        color="green",
        linewidth=2.5,
        label=f"alpha2 fit (R2={fit2.r2:.3f})",
    )

    ax1.set_xlabel(r"T [дни]", fontsize=16)
    ax1.set_ylabel(r"$\alpha_1$, $\alpha_2$ [безразм. ед.]", fontsize=16)
    ax1.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)

    # Правая ось для d_max
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$d_{max}$ [тыс. км]", fontsize=16, color="red")
    ax2.tick_params(axis="y", labelsize=14, colors="red")
    ax2.spines["right"].set_color("red")
    ax2.tick_params(axis="y", colors="red")
    ax2.yaxis.label.set_color("red")
    line3 = ax2.plot(
        t,
        du2km(dev_max) / 1000.0,
        color="red",
        linestyle="-",
        marker="D",
        markersize=4,
        label="d_max",
    )[0]

    legend1 = ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="center left", fontsize=14)
    legend2 = ax1.legend([line3], [line3.get_label()], loc="lower right", fontsize=14)
    ax1.add_artist(legend1)

    plt.tight_layout()
    plt.minorticks_on()
    ax1.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    plt.show()


def run_fit_and_plot() -> None:
    """Запуск без аргументов: фит, печать параметров и отрисовка графика."""
    t, alpha1, alpha2, dev_max = load_alpha_data(DEFAULT_DATA)
    fit1 = fit_alpha_curve(t, alpha1, "Alpha1")
    fit2 = fit_alpha_curve(t, alpha2, "Alpha2")

    print("Параметризация: f(T) = p0 + p1*T + p2*T^2 + A / (1 + exp(k*(T - T_mid)))^(1/nu)")
    print_result(fit1)
    print_result(fit2)

    plot_with_dev(t, alpha1, alpha2, dev_max, fit1, fit2)


if __name__ == "__main__":
    run_fit_and_plot()
