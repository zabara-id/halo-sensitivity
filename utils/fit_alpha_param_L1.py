"""
Подбор компактной аналитической аппроксимации для кривых Alpha1/Alpha2
из файла ``data/output/coefficients/koefficients_data_L1.csv``.

Параметризация: четырёхпараметрическая сигмоида

    f(z) = lower + (upper - lower) / (1 + exp(k * (z - z_mid)))

* z — высота z0 в тысячах километров (как на существующих графиках).
* k > 0 даёт убывающую кривую, k < 0 — возрастающую.

Скрипт печатает параметры и метрику R^2; по флагу ``--plot`` строит
наложение данных и аппроксимации.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utils.libration_sense import du2km


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "output" / "coefficients" / "koefficients_data_L1.csv"


def logistic4(z: np.ndarray, lower: float, upper: float, k: float, z_mid: float) -> np.ndarray:
    """Четырехпараметрическая сигмоида с независимыми асимптотами."""
    return lower + (upper - lower) / (1.0 + np.exp(k * (z - z_mid)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def load_alpha_data(path: Path = DEFAULT_DATA) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает z0 в тыс. км и значения Alpha1/Alpha2/Deviation Max из CSV.
    """
    df = pd.read_csv(path).sort_values("z0")
    z0_thousand_km = du2km(df["z0"].values) / 1000.0
    return (
        z0_thousand_km,
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
    z_mid: float
    r2: float

    def predict(self, z: np.ndarray) -> np.ndarray:
        return logistic4(z, self.lower, self.upper, self.k, self.z_mid)


def fit_alpha_curve(z: np.ndarray, y: np.ndarray, label: str) -> FitResult:
    """
    Строит сигмоидальную аппроксимацию через нелинейную регрессию.
    Для устойчивости используются разумные начальные приближения и bounds.
    """
    lower_guess = float(np.quantile(y, 0.05))
    upper_guess = float(np.quantile(y, 0.95))
    span = float(z.max() - z.min())
    p0 = (lower_guess, upper_guess, 0.05, float(np.median(z)))
    bounds = (
        (lower_guess - 0.2, lower_guess, -5.0, z.min() - 0.2 * span),
        (upper_guess, upper_guess + 0.2, 5.0, z.max() + 0.2 * span),
    )
    try:
        params, _ = curve_fit(
            logistic4,
            z,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        # На случай, если bounds мешают сходимости.
        params, _ = curve_fit(logistic4, z, y, p0=p0, maxfev=20000)

    y_hat = logistic4(z, *params)
    return FitResult(
        label=label,
        lower=float(params[0]),
        upper=float(params[1]),
        k=float(params[2]),
        z_mid=float(params[3]),
        r2=float(r2_score(y, y_hat)),
    )


def print_result(result: FitResult) -> None:
    direction = "убывающая" if result.k > 0 else "возрастающая"
    amplitude = result.upper - result.lower
    formula = (
        f"{result.label}(z) = {result.lower:.5f} "
        f"+ ({amplitude:.5f}) / (1 + exp({result.k:.5f} * (z - {result.z_mid:.2f})))"
    )
    print(f"{result.label}: {direction} сигмоида, R^2 = {result.r2:.4f}")
    print(f"    {formula}")


def plot_with_dev(
    z: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    dev_max: np.ndarray,
    fit1: FitResult,
    fit2: FitResult,
) -> None:
    """
    Строит тот же макет, что в graphL1: точки Alpha1/Alpha2, наши сигмоиды и d_max справа.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Точки данных
    ax1.scatter(z, alpha1, color="blue", s=20, alpha=0.6, label="Alpha1")
    ax1.scatter(z, alpha2, color="green", s=20, alpha=0.6, label="Alpha2")

    # Сигмоидальные аппроксимации
    z_dense = np.linspace(z.min(), z.max(), 600)
    line1, = ax1.plot(
        z_dense,
        fit1.predict(z_dense),
        color="blue",
        linewidth=2.5,
        label=f"alpha1 fit (R2={fit1.r2:.3f})",
    )
    line2, = ax1.plot(
        z_dense,
        fit2.predict(z_dense),
        color="green",
        linewidth=2.5,
        label=f"alpha2 fit (R2={fit2.r2:.3f})",
    )

    ax1.set_xlabel(r"$z_0$ [тыс. км]", fontsize=16)
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
        z,
        du2km(dev_max) / 1000.0,
        color="red",
        linestyle="-",
        marker="D",
        markersize=4,
        label=r"$d_{max}$",
    )[0]

    legend1 = ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="center left", fontsize=14)
    legend2 = ax1.legend([line3], [line3.get_label()], loc="right", fontsize=14)
    ax1.add_artist(legend1)

    plt.tight_layout()
    plt.minorticks_on()
    ax1.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    plt.show()


def run_fit_and_plot() -> None:
    """
    Загружает данные L1, строит сигмоиды, печатает параметры и сразу рисует график.
    Без аргументов из консоли.
    """
    z, alpha1, alpha2, dev_max = load_alpha_data(DEFAULT_DATA)
    fit1 = fit_alpha_curve(z, alpha1, "Alpha1")
    fit2 = fit_alpha_curve(z, alpha2, "Alpha2")

    print("Параметризация: f(z) = lower + (upper - lower) / (1 + exp(k * (z - z_mid)))")
    print_result(fit1)
    print_result(fit2)

    plot_with_dev(z, alpha1, alpha2, dev_max, fit1, fit2)


if __name__ == "__main__":
    run_fit_and_plot()
