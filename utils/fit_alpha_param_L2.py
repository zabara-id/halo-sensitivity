"""
Аппроксимация Alpha1/Alpha2 для L2 теми же сигмоидами, что использованы для L1,
и построение графика без сплайнов (точки + наши кривые + d_max справа).

f(x) = lower + (upper - lower) / (1 + exp(k * (x - x_mid)))

* x — период T в днях.
* k > 0 даёт убывание (Alpha2), k < 0 — возрастание (Alpha1).
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


def logistic4(x: np.ndarray, lower: float, upper: float, k: float, x_mid: float) -> np.ndarray:
    """Четырехпараметрическая сигмоида с независимыми асимптотами."""
    return lower + (upper - lower) / (1.0 + np.exp(k * (x - x_mid)))


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
    lower: float
    upper: float
    k: float
    x_mid: float
    r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return logistic4(x, self.lower, self.upper, self.k, self.x_mid)


def fit_alpha_curve(x: np.ndarray, y: np.ndarray, label: str) -> FitResult:
    """Подбирает сигмоиду; bounds мягкие для устойчивости."""
    lower_guess = float(np.quantile(y, 0.05))
    upper_guess = float(np.quantile(y, 0.95))
    span = float(x.max() - x.min())
    p0 = (lower_guess, upper_guess, 0.5, float(np.median(x)))
    bounds = (
        (lower_guess - 0.2, lower_guess, -5.0, x.min() - 0.2 * span),
        (upper_guess, upper_guess + 0.2, 5.0, x.max() + 0.2 * span),
    )
    try:
        params, _ = curve_fit(
            logistic4,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        params, _ = curve_fit(logistic4, x, y, p0=p0, maxfev=20000)

    y_hat = logistic4(x, *params)
    return FitResult(
        label=label,
        lower=float(params[0]),
        upper=float(params[1]),
        k=float(params[2]),
        x_mid=float(params[3]),
        r2=float(r2_score(y, y_hat)),
    )


def print_result(result: FitResult) -> None:
    direction = "убывающая" if result.k > 0 else "возрастающая"
    amplitude = result.upper - result.lower
    formula = (
        f"{result.label}(T) = {result.lower:.5f} "
        f"+ ({amplitude:.5f}) / (1 + exp({result.k:.5f} * (T - {result.x_mid:.3f})))"
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
    t_dense = np.linspace(t.min(), t.max(), 800)
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

    print("Параметризация: f(T) = lower + (upper - lower) / (1 + exp(k * (T - T_mid)))")
    print_result(fit1)
    print_result(fit2)

    plot_with_dev(t, alpha1, alpha2, dev_max, fit1, fit2)


if __name__ == "__main__":
    run_fit_and_plot()
