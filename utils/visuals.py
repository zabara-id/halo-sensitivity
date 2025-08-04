import time

from daceypy import DA, array
import matplotlib.pyplot as plt

from libration_sense import initial_state_parser
from libration_sense import *


def dots_graph(orbit_type: str, number_of_orbit: int, derorder: int=3, number_of_turns: int=1) -> None:
    DA.init(derorder, 6)

    x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)

    initial_state = array.identity(6)
    initial_state[0] += x0  # x
    initial_state[2] += z0  # z
    initial_state[4] += vy0  # v_y

    with DA.cache_manager():  # optional, for efficiency
        xf = RK78(initial_state, 0.0, number_of_turns * T, CR3BP)

    # Количество точек в "облаке"
    num_points = 10000

    # Стандартные отклонения для координат и скоростей
    std_pos = km2du(8)  # от 0 до 8 км
    std_vel = kmS2vu(0.5e-3)  # от 0 до 0.05 м/с

    # Создаем массив стандартных отклонений для каждой компоненты
    std_devs = np.array([std_pos] * 3 + [std_vel] * 3)

    # Генерируем изменения (deltax0) для каждой компоненты
    deltax0 = np.random.normal(0, std_devs, (num_points, 6))

    x0_cons = initial_state.cons()  # координаты центральной точки (начальные условия)
    x0_new = x0_cons + deltax0  # отклонения от начального положения

    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0]) # изменённые конечные положения объектов для возмущённых начальных условий

    # Разделение данных на координаты X, Y и Z
    x0_coords = x0_new[:, 0]
    y0_coords = x0_new[:, 1]
    z0_coords = x0_new[:, 2]

    x_coords = evaluated_results[:, 0]
    y_coords = evaluated_results[:, 1]
    z_coords = evaluated_results[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x0_coords, y0_coords, z0_coords, c='blue', marker='o', s=1)
    ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o', s=1)
    ax.scatter(x0_cons[0], x0_cons[1], x0_cons[2], c='black', marker='o', s=1)

    ax.set_xlabel('X, безразм. ед.')
    ax.set_ylabel('Y, безразм. ед.')
    ax.set_zlabel('Z, безразм. ед.')

    plt.show()


def deviation_graph(orbit_type: str,
                    number_of_orbit: int,
                    derorder: int = 3,
                    number_of_turns: int = 1,
                    number_of_points: int = 10000,
                    grid_density: int = 50) -> None:

    DA.init(derorder, 6)
    
    x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)

    initial_state = array.identity(6)
    initial_state[0] += x0  # x
    initial_state[2] += z0  # z
    initial_state[4] += vy0  # v_y

    with DA.cache_manager():  # optional, for efficiency
        xfinal = RK78(initial_state, 0.0, number_of_turns * T, CR3BP)

    std_pos_values = np.linspace(0, km2du(3), grid_density)  # от 0 до 8 км
    std_vel_values = np.linspace(0, kmS2vu(0.03e-3), grid_density)  # от 0 до 0.05 м/с

    # Матрица для хранения результатов
    results = np.zeros((len(std_pos_values), len(std_vel_values)))

    row, val = 0, 0
    # Вычисление значений для графика функции 2-х переменных
    start = time.time()
    for i, std_pos in enumerate(std_pos_values):
        print(f"Ряд {row} просчитан")
        row += 1
        val = 0
        for j, std_vel in enumerate(std_vel_values):
            results[i, j] = du2km(get_maxdeviation_wo_integrate(orbit_type, number_of_orbit, xfinal, std_pos, std_vel,
                                                                derorder=derorder,
                                                                number_of_turns=number_of_turns,
                                                                amount_of_points=number_of_points))
            print(f"Значение {val} в ряде {row} вычислено")
            val += 1
    end = time.time()

    print("Время расчёта: ", end - start)


    plt.figure(figsize=(10, 8))

    # Создание изолиний без заполнения
    CS = plt.contour(vu2ms(std_vel_values), du2km(std_pos_values), results, levels=30, colors='k')
    plt.clabel(CS, inline=True, fontsize=11)
    
    plt.xlabel(r'$\sigma_{vel}$, $[m/s]$', fontsize=14)
    plt.ylabel(r'$\sigma_{pos}$, $[km]$', fontsize=14)
    plt.title(f"Deviation for orbit '{number_of_orbit}' around {orbit_type} with µ={np.around(MAX_MUL, 2 )}, [km]", fontsize=12)

    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.show()


# Примеры
def main1():
    dots_graph("L1", 173, derorder=5)


def main2():
    # dots_graph('L1', 192, number_of_turns=3, derorder=1)
    # print(deviation_finder('L1', 200, number_of_points=1000, number_of_turns=1))
    deviation_graph('L1', 183, number_of_points=10000, number_of_turns=1, grid_density=45)