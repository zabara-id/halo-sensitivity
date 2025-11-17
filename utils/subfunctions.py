import numpy as np
import matplotlib.pyplot as plt


def plot_ellipsoid_and_vectors_pretty(
    pos_sigma: float,
    r: float,
    vecs: dict[str, np.ndarray],
    title: str | None = None,
) -> None:
    """
    Более выразительная визуализация эллипсоида и 3D‑векторов на Matplotlib.

    Особенности (только MPL):
    - Стрелки‑векторы (`quiver`) и аккуратная легенда (без надписей у концов).
    - Очень лёгкая полупрозрачная поверхность эллипсоида + тонкая проволочная сетка,
      чтобы не перекрывать содержимое внутри.
    - Равный масштаб осей и вспомогательные оси‑стрелки.

    Параметры такие же, как у старой функции. `vecs` — словарь имя → 6D‑вектор;
    используются только первые 3 компоненты (dx, dy, dz).
    """
    a = b = c = r * pos_sigma

    # Параметризация сферы (a=b=c), чтобы остаться совместимыми с текущей логикой
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    X = a * np.outer(np.cos(u), np.sin(v))
    Y = b * np.outer(np.sin(u), np.sin(v))
    Z = c * np.outer(np.ones_like(u), np.cos(v))

    # --- Matplotlib implementation (enhanced static) ---
    from matplotlib.lines import Line2D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Очень лёгкая поверхность (почти невидимая), чтобы не перекрывать векторы
    ax.plot_surface(X, Y, Z, color=(0.8, 0.8, 0.8, 0.10), linewidth=0, antialiased=True, shade=False)
    # Тонкая проволочная сетка для ориентира
    ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, color=(0, 0, 0, 0.05), linewidth=0.5)

    # Цвета как в исходном коде
    color_cycle = {
        'sampling': 'tab:blue',
        'linear ellipsoid': 'tab:orange',
        'DA opt (multistart)': 'tab:green',
        'IVP opt (multistart)': 'tab:red',
        'Floquet': 'tab:purple',
    }
    # Подписи легенды на русском
    ru_labels = {
        'sampling': 'семплинг (эллипсоид)',
        'linear ellipsoid': 'линейный эллипсоид',
        'DA opt (multistart)': 'ДА‑опт (мультистарт)',
        'IVP opt (multistart)': 'ОДУ‑опт (мультистарт)',
        'Floquet': 'Флоке',
    }
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
            linewidth=1.1,
            normalize=False,
        )
        proxies.append(Line2D([0], [0], color=color, lw=3, label=ru_labels.get(name, name)))

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
    ax.set_xlabel('dx (DU)')
    ax.set_ylabel('dy (DU)')
    ax.set_zlabel('dz (DU)')
    if title:
        ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=35)
    ax.legend(handles=proxies, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()