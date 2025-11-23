import numpy as np
import matplotlib.pyplot as plt
from typing import Any


def plot_ellipsoid_and_vectors_pretty(
    pos_sigma: float,
    r: float,
    vecs: dict[str, np.ndarray],
    title: str | None = None,
    ax=None,
    show: bool = True,
    axis_labels: tuple[str, str, str] | None = None,
    units_label: str = "DU",
    legend_labels: dict[str, str] | None = None,
    show_legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Рисует эллипсоид (радиус r * pos_sigma) и набор векторов vecs в 3D.

    Можно передать готовую ось ax для встраивания в составную фигуру.
    Если show=False, plt.show() не вызывается (ответственность на вызывающем коде).
    axis_labels позволяет переопределить подписи осей; units_label дописывается в скобках.
    legend_labels позволяет переопределить подписи легенды, show_legend управляет её выводом.
    """
    a = b = c = r * pos_sigma

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    X = a * np.outer(np.cos(u), np.sin(v))
    Y = b * np.outer(np.sin(u), np.sin(v))
    Z = c * np.outer(np.ones_like(u), np.cos(v))

    from matplotlib.lines import Line2D

    created_fig = None
    if ax is None:
        created_fig = plt.figure()
        ax = created_fig.add_subplot(111, projection='3d')

    # Очень лёгкая поверхность (почти невидимая), чтобы не перекрывать векторы
    ax.plot_surface(X, Y, Z, color=(0.8, 0.8, 0.8, 0.10), linewidth=0, antialiased=True, shade=False)
    # Тонкая проволочная сетка для ориентира
    ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, color=(0, 0, 0, 0.05), linewidth=0.5)

    color_cycle = {
        'sampling': 'tab:blue',
        'linear ellipsoid': 'tab:orange',
        'DA opt (multistart)': 'tab:green',
        'IVP opt (multistart)': 'tab:red',
        'Floquet': 'tab:purple',
    }
    default_labels = {
        'sampling': r'$\delta\mathbf{r}_0^*$, sampling',
        'linear ellipsoid': r'$\delta\mathbf{r}_0^*$, linear + SVD',
        'DA opt (multistart)': r'$\delta\mathbf{r}_0^*$, DA-optimization',
        'IVP opt (multistart)': r'$\delta\mathbf{r}_0^*$, IVP-optimization',
        'Floquet': r'$\delta\mathbf{r}_0^*$, Floquet'
    }
    labels = legend_labels or default_labels
    palette_fallback = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']

    # Стрелки‑векторы с подписями в концах
    proxies: list[Line2D] = []
    for idx, (name, vec6) in enumerate(vecs.items()):
        p = np.asarray(vec6[:3], dtype=float)
        color = color_cycle.get(name, palette_fallback[idx % len(palette_fallback)])
        ax.quiver(
            0.0, 0.0, 0.0,
            p[0], p[1], p[2],
            arrow_length_ratio=0.12,
            color=color,
            linewidth=1.5,
            normalize=False,
        )
        proxies.append(Line2D([0], [0], color=color, lw=3, label=labels.get(name, name)))

    # Оси‑стрелки для ориентира
    lim = r * pos_sigma
    axis_len = lim
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='gray', arrow_length_ratio=0.02, alpha=0.5, linewidth=0.5)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='gray', arrow_length_ratio=0.02, alpha=0.5, linewidth=0.5)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='gray', arrow_length_ratio=0.02, alpha=0.5, linewidth=0.5)
    ax.text(axis_len, 0, 0, 'x', color='gray')
    ax.text(0, axis_len, 0, 'y', color='gray')
    ax.text(0, 0, axis_len, 'z', color='gray')

    # Оформление
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    labels = axis_labels or (r'$\delta x$', r'$\delta y$', r'$\delta z$')
    suffix = f" [{units_label}]" if units_label else ""
    ax.set_xlabel(labels[0] + suffix)
    ax.set_ylabel(labels[1] + suffix)
    ax.set_zlabel(labels[2] + suffix)
    if title:
        ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=35)
    if show_legend:
        lk = {'loc': 'upper left', 'frameon': False}
        if legend_kwargs:
            lk.update(legend_kwargs)
        ax.legend(handles=proxies, **lk)
    if show:
        plt.tight_layout()
        plt.show()
