import os
import csv

import numpy as np
from itertools import product
from daceypy import DA, array
from typing import Any, Callable
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from subfunctions import plot_ellipsoid_and_vectors_pretty


ORBIT_TYPES_NUMS = {'L1' : 251,
                    'L2' : 583}

MU = 0.012150585609624
VU = 1.024540192302405
DU = 3.84405e5
TU = 4.342564574695797  # дни


def vu2kms(vu_array: np.ndarray) -> np.ndarray:
    return vu_array * VU


def vu2ms(vu_array: np.ndarray) -> np.ndarray:
    return vu_array * VU * 1000


def du2km(du_array: np.ndarray) -> np.ndarray:
    return du_array * DU


def km2du(km_array):
    return km_array / DU


def kmS2vu(kms_array):
    return kms_array / VU


def tu2days(tu_array):
    return tu_array * TU



def initial_state_parser(orbit_type: str, number_of_orbit: int = 1) -> np.ndarray:
    if orbit_type not in ORBIT_TYPES_NUMS:
        raise ValueError("Incorrect orbit type. You need either 'L1' or 'L2'.")
    
    if (number_of_orbit < 1) or number_of_orbit > ORBIT_TYPES_NUMS[orbit_type]:
        raise ValueError(f"Incorrect orbit number. The range (1, {ORBIT_TYPES_NUMS[orbit_type]}) is available for {orbit_type}.")
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', 'qualified_ic_halo', f'HOPhaseVectorsEarthMoon{orbit_type}_qualified.csv')
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        
        selected_row = rows[number_of_orbit - 1]
        
        selected_row = [float(num) for num in selected_row]
    
    return np.array(selected_row)


def RK78_trajectory(Y0: array, X0: float, X1: float, f: Callable[[array, float], array]):
    
    Y0 = Y0.copy()
    N = len(Y0)

    H0 = 0.0001
    HS = 0.01
    H1 = 0.1

    # H0 = 0.00001
    # HS = 0.0001
    # H1 = 0.001

    EPS = 1.e-12
    BS = 20 * EPS

    Z = array.zeros((N, 16))
    Y1 = array.zeros(N)

    VIHMAX = 0.0

    HSQR = 1.0 / 9.0
    A = np.zeros(13)
    B = np.zeros((13, 12))
    C = np.zeros(13)
    D = np.zeros(13)

    A = np.array([
        0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0,
        93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0,
        1201146811.0/1299019798.0, 1.0, 1.0,
    ])

    B[1, 0] = 1.0/18.0
    B[2, 0] = 1.0/48.0
    B[2, 1] = 1.0/16.0
    B[3, 0] = 1.0/32.0
    B[3, 2] = 3.0/32.0
    B[4, 0] = 5.0/16.0
    B[4, 2] = -75.0/64.0
    B[4, 3] = 75.0/64.0
    B[5, 0] = 3.0/80.0
    B[5, 3] = 3.0/16.0
    B[5, 4] = 3.0/20.0
    B[6, 0] = 29443841.0/614563906.0
    B[6, 3] = 77736538.0/692538347.0
    B[6, 4] = -28693883.0/1125000000.0
    B[6, 5] = 23124283.0/1800000000.0
    B[7, 0] = 16016141.0/946692911.0
    B[7, 3] = 61564180.0/158732637.0
    B[7, 4] = 22789713.0/633445777.0
    B[7, 5] = 545815736.0/2771057229.0
    B[7, 6] = -180193667.0/1043307555.0
    B[8, 0] = 39632708.0/573591083.0
    B[8, 3] = -433636366.0/683701615.0
    B[8, 4] = -421739975.0/2616292301.0
    B[8, 5] = 100302831.0/723423059.0
    B[8, 6] = 790204164.0/839813087.0
    B[8, 7] = 800635310.0/3783071287.0
    B[9, 0] = 246121993.0/1340847787.0
    B[9, 3] = -37695042795.0/15268766246.0
    B[9, 4] = -309121744.0/1061227803.0
    B[9, 5] = -12992083.0/490766935.0
    B[9, 6] = 6005943493.0/2108947869.0
    B[9, 7] = 393006217.0/1396673457.0
    B[9, 8] = 123872331.0/1001029789.0
    B[10, 0] = -1028468189.0/846180014.0
    B[10, 3] = 8478235783.0/508512852.0
    B[10, 4] = 1311729495.0/1432422823.0
    B[10, 5] = -10304129995.0/1701304382.0
    B[10, 6] = -48777925059.0/3047939560.0
    B[10, 7] = 15336726248.0/1032824649.0
    B[10, 8] = -45442868181.0/3398467696.0
    B[10, 9] = 3065993473.0/597172653.0
    B[11, 0] = 185892177.0/718116043.0
    B[11, 3] = -3185094517.0/667107341.0
    B[11, 4] = -477755414.0/1098053517.0
    B[11, 5] = -703635378.0/230739211.0
    B[11, 6] = 5731566787.0/1027545527.0
    B[11, 7] = 5232866602.0/850066563.0
    B[11, 8] = -4093664535.0/808688257.0
    B[11, 9] = 3962137247.0/1805957418.0
    B[11, 10] = 65686358.0/487910083.0
    B[12, 0] = 403863854.0/491063109.0
    B[12, 3] = - 5068492393.0/434740067.0
    B[12, 4] = -411421997.0/543043805.0
    B[12, 5] = 652783627.0/914296604.0
    B[12, 6] = 11173962825.0/925320556.0
    B[12, 7] = -13158990841.0/6184727034.0
    B[12, 8] = 3936647629.0/1978049680.0
    B[12, 9] = -160528059.0/685178525.0
    B[12, 10] = 248638103.0/1413531060.0

    C = np.array([
        14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0,
        181606767.0/758867731.0, 561292985.0/797845732.0,
        -1041891430.0/1371343529.0, 760417239.0/1151165299.0,
        118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0,
    ])

    D = np.array([
        13451932.0/455176623.0, 0.0, 0.0, 0.0, 0.0, -808719846.0/976000145.0,
        1757004468.0/5645159321.0, 656045339.0/265891186.0,
        -3867574721.0/1518517206.0, 465885868.0/322736535.0,
        53011238.0/667516719.0, 2.0/45.0, 0.0,
    ])

    Z[:, 0] = Y0

    H = abs(HS)
    HH0 = abs(H0)
    HH1 = abs(H1)
    X = X0
    RFNORM = 0.0
    ERREST = 0.0

    # For saving the trajectory
    trajectory = []
    
    while X != X1:

        # compute new stepsize
        if RFNORM != 0:
            H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
        if abs(H) > abs(HH1):
            H = HH1
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * H > 0:
            H = X1 - X

        for j in range(13):
            for i in range(N):
                Y0[i] = 0.0
                for k in range(j):
                    Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]
                Y0[i] = H * Y0[i] + Z[i, 0]
            Y1 = f(Y0, X + H * A[j])
            for i in range(N):
                Z[i, j + 3] = Y1[i]

        for i in range(N):
            Z[i, 1] = 0.0
            Z[i, 2] = 0.0
            for j in range(13):
                Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
                Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

            Y1[i] = (Z[i, 2] - Z[i, 1]) * H
            Z[i, 2] = Z[i, 2] * H + Z[i, 0]

        Y1cons = Y1.cons()
        trajectory.append(Z[:, 0].cons())  # Save the current position
        
        RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
        if RFNORM > BS and abs(H / H0) > 1.2:
            H = H / 3.0
            RFNORM = 0
        else:
            for i in range(N):
                Z[i, 0] = Z[i, 2]
            X = X + H
            VIHMAX = max(VIHMAX, H)
            ERREST = ERREST + RFNORM

    if RFNORM != 0:
        H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
    if abs(H) > abs(HH1):
        H = HH1
    elif abs(H) < abs(HH0) * 0.99:
        H = HH0
        print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

    if (X + H - X1) * H > 0:
        H = X1 - X

    for j in range(13):
        for i in range(N):
            Y0[i] = 0.0
            for k in range(j):
                Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]
            Y0[i] = H * Y0[i] + Z[i, 0]
        Y1 = f(Y0, X + H * A[j])
        for i in range(N):
            Z[i, j + 3] = Y1[i]

    for i in range(N):
        Z[i, 1] = 0.0
        Z[i, 2] = 0.0
        for j in range(13):
            Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
            Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

        Y1[i] = (Z[i, 2] - Z[i, 1]) * H
        Z[i, 2] = Z[i, 2] * H + Z[i, 0]

    Y1cons = Y1.cons()
    trajectory.append(Z[:, 0].cons())  # Save the current position
    
    RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
    if RFNORM > BS and abs(H / H0) > 1.2:
        H = H / 3.0
        RFNORM = 0
    else:
        for i in range(N):
            Z[i, 0] = Z[i, 2]
        X = X + H
        VIHMAX = max(VIHMAX, H)
        ERREST = ERREST + RFNORM

    return np.array(trajectory)


def RK78(Y0: array, X0: float, X1: float, f: Callable[[array, float], array]):
    Y0 = Y0.copy()

    N = len(Y0)

    H0 = 0.001
    HS = 0.01
    H1 = 0.1
    EPS = 1.e-12
    BS = 20 * EPS

    Z = array.zeros((N, 16))
    Y1 = array.zeros(N)

    VIHMAX = 0.0

    HSQR = 1.0 / 9.0
    A = np.zeros(13)
    B = np.zeros((13, 12))
    C = np.zeros(13)
    D = np.zeros(13)

    A = np.array([
        0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0,
        93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0,
        1201146811.0/1299019798.0, 1.0, 1.0,
    ])

    B[1, 0] = 1.0/18.0
    B[2, 0] = 1.0/48.0
    B[2, 1] = 1.0/16.0
    B[3, 0] = 1.0/32.0
    B[3, 2] = 3.0/32.0
    B[4, 0] = 5.0/16.0
    B[4, 2] = -75.0/64.0
    B[4, 3] = 75.0/64.0
    B[5, 0] = 3.0/80.0
    B[5, 3] = 3.0/16.0
    B[5, 4] = 3.0/20.0
    B[6, 0] = 29443841.0/614563906.0
    B[6, 3] = 77736538.0/692538347.0
    B[6, 4] = -28693883.0/1125000000.0
    B[6, 5] = 23124283.0/1800000000.0
    B[7, 0] = 16016141.0/946692911.0
    B[7, 3] = 61564180.0/158732637.0
    B[7, 4] = 22789713.0/633445777.0
    B[7, 5] = 545815736.0/2771057229.0
    B[7, 6] = -180193667.0/1043307555.0
    B[8, 0] = 39632708.0/573591083.0
    B[8, 3] = -433636366.0/683701615.0
    B[8, 4] = -421739975.0/2616292301.0
    B[8, 5] = 100302831.0/723423059.0
    B[8, 6] = 790204164.0/839813087.0
    B[8, 7] = 800635310.0/3783071287.0
    B[9, 0] = 246121993.0/1340847787.0
    B[9, 3] = -37695042795.0/15268766246.0
    B[9, 4] = -309121744.0/1061227803.0
    B[9, 5] = -12992083.0/490766935.0
    B[9, 6] = 6005943493.0/2108947869.0
    B[9, 7] = 393006217.0/1396673457.0
    B[9, 8] = 123872331.0/1001029789.0
    B[10, 0] = -1028468189.0/846180014.0
    B[10, 3] = 8478235783.0/508512852.0
    B[10, 4] = 1311729495.0/1432422823.0
    B[10, 5] = -10304129995.0/1701304382.0
    B[10, 6] = -48777925059.0/3047939560.0
    B[10, 7] = 15336726248.0/1032824649.0
    B[10, 8] = -45442868181.0/3398467696.0
    B[10, 9] = 3065993473.0/597172653.0
    B[11, 0] = 185892177.0/718116043.0
    B[11, 3] = -3185094517.0/667107341.0
    B[11, 4] = -477755414.0/1098053517.0
    B[11, 5] = -703635378.0/230739211.0
    B[11, 6] = 5731566787.0/1027545527.0
    B[11, 7] = 5232866602.0/850066563.0
    B[11, 8] = -4093664535.0/808688257.0
    B[11, 9] = 3962137247.0/1805957418.0
    B[11, 10] = 65686358.0/487910083.0
    B[12, 0] = 403863854.0/491063109.0
    B[12, 3] = - 5068492393.0/434740067.0
    B[12, 4] = -411421997.0/543043805.0
    B[12, 5] = 652783627.0/914296604.0
    B[12, 6] = 11173962825.0/925320556.0
    B[12, 7] = -13158990841.0/6184727034.0
    B[12, 8] = 3936647629.0/1978049680.0
    B[12, 9] = -160528059.0/685178525.0
    B[12, 10] = 248638103.0/1413531060.0

    C = np.array([
        14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0,
        181606767.0/758867731.0, 561292985.0/797845732.0,
        -1041891430.0/1371343529.0, 760417239.0/1151165299.0,
        118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0,
    ])

    D = np.array([
        13451932.0/455176623.0, 0.0, 0.0, 0.0, 0.0, -808719846.0/976000145.0,
        1757004468.0/5645159321.0, 656045339.0/265891186.0,
        -3867574721.0/1518517206.0, 465885868.0/322736535.0,
        53011238.0/667516719.0, 2.0/45.0, 0.0,
    ])

    Z[:, 0] = Y0

    H = abs(HS)
    HH0 = abs(H0)
    HH1 = abs(H1)
    X = X0
    RFNORM = 0.0
    ERREST = 0.0

    while X != X1:
        # compute new stepsize
        if RFNORM != 0:
            H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
        if abs(H) > abs(HH1):
            H = HH1
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * H > 0:
            H = X1 - X

        for j in range(13):

            for i in range(N):

                Y0[i] = 0.0
                # EVALUATE RHS AT 13 POINTS
                for k in range(j):
                    Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]

                Y0[i] = H * Y0[i] + Z[i, 0]

            Y1 = f(Y0, X + H * A[j])

            for i in range(N):
                Z[i, j + 3] = Y1[i]

        for i in range(N):

            Z[i, 1] = 0.0
            Z[i, 2] = 0.0
            # EXECUTE 7TH,8TH ORDER STEPS
            for j in range(13):
                Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
                Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

            Y1[i] = (Z[i, 2] - Z[i, 1]) * H
            Z[i, 2] = Z[i, 2] * H + Z[i, 0]

        Y1cons = Y1.cons()

        # ESTIMATE ERROR AND DECIDE ABOUT BACKSTEP
        RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
        if RFNORM > BS and abs(H / H0) > 1.2:
            H = H / 3.0
            RFNORM = 0
        else:
            for i in range(N):
                Z[i, 0] = Z[i, 2]
            X = X + H
            VIHMAX = max(VIHMAX, H)
            ERREST = ERREST + RFNORM


    # compute new stepsize
        if RFNORM != 0:
            H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
        if abs(H) > abs(HH1):
            H = HH1
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * H > 0:
            H = X1 - X

        for j in range(13):

            for i in range(N):

                Y0[i] = 0.0
                # EVALUATE RHS AT 13 POINTS
                for k in range(j):
                    Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]

                Y0[i] = H * Y0[i] + Z[i, 0]

            Y1 = f(Y0, X + H * A[j])

            for i in range(N):
                Z[i, j + 3] = Y1[i]

        for i in range(N):

            Z[i, 1] = 0.0
            Z[i, 2] = 0.0
            # EXECUTE 7TH,8TH ORDER STEPS
            for j in range(13):
                Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
                Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

            Y1[i] = (Z[i, 2] - Z[i, 1]) * H
            Z[i, 2] = Z[i, 2] * H + Z[i, 0]

        Y1cons = Y1.cons()

        # ESTIMATE ERROR AND DECIDE ABOUT BACKSTEP
        RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
        if RFNORM > BS and abs(H / H0) > 1.2:
            H = H / 3.0
            RFNORM = 0
        else:
            for i in range(N):
                Z[i, 0] = Z[i, 2]
            X = X + H
            VIHMAX = max(VIHMAX, H)
            ERREST = ERREST + RFNORM

    Y1 = Z[:, 0]

    return Y1


def CR3BP(x: array, t: float) -> array:
    """DACE-adapted RHS for three bodies problem

    The right-hand side of the equations of motion in the circular bounded problem
    of three bodies in a rotating coordinate system

    Args:
        x (array): State vector, dimensionless units of a rotating coordinate system.
        t (float): Time, dimensionless unit of a rotating coordinate system.

    Returns:
        array: RHS [dx, dy, dz, ddx, ddy, ddz]
    """
    pos: array = x[:3]
    vel: array = x[3:]

    r1_pos = array([MU + pos[0] - 1, pos[1], pos[2]])
    r2_pos = array([MU + pos[0], pos[1], pos[2]])
    
    r1 = r1_pos.vnorm()
    r2 = r2_pos.vnorm()
    
    acc_x = pos[0] + 2 * vel[1] - (MU * r1_pos[0]) / r1**3 - ((1 - MU) * r2_pos[0]) / r2**3
    acc_y = pos[1] - 2 * vel[0] - (MU * r1_pos[1]) / r1**3 - ((1 - MU) * r2_pos[1]) / r2**3
    acc_z = - (MU * r1_pos[2]) / r1**3 - ((1 - MU) * r2_pos[2]) / r2**3
    
    acc: array = array([acc_x, acc_y, acc_z])  # Вектор ускорений
    
    dx: array = vel.concat(acc)
    
    return dx


def get_xf(orbit_type: str,
              number_of_orbit: int,
              derorder: int = 3,
              number_of_turns: int = 1,
              number_of_halo_point: int | None = None) -> DA:
    DA.init(derorder, 6)

    if not number_of_halo_point:
        x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)
        initial_state = array.identity(6)
        initial_state[0] += x0  # x
        initial_state[2] += z0  # z
        initial_state[4] += vy0  # v_y
    else:
        x0, z0, vy0, T, _, __ = initial_state_parser(orbit_type, number_of_orbit)
        initial_state1 = np.array([x0, 0., z0, 0., vy0, 0.])
        initial_state, T = halo_qualify(orbit_type, number_of_orbit)

        t_span = (0, T)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(cr3bp, t_span, initial_state, t_eval=t_eval, rtol=1e-13, atol=1e-13, method='LSODA')
        halo_orbit_dots = sol.y.T
        halo_orbit_dot = halo_orbit_dots[number_of_halo_point]
        print(halo_orbit_dot)

        initial_state = array.identity(6)
        initial_state[0] += halo_orbit_dot[0]
        initial_state[1] += halo_orbit_dot[1]
        initial_state[2] += halo_orbit_dot[2]
        initial_state[3] += halo_orbit_dot[3]
        initial_state[4] += halo_orbit_dot[4]
        initial_state[5] += halo_orbit_dot[5]

    with DA.cache_manager():  # optional, for efficiency
        xf = RK78(initial_state, 0.0, number_of_turns * T, CR3BP)

    return xf


def get_maxdeviation(orbit_type: str,
                     number_of_orbit: int,
                     std_pos: float,
                     std_vel: float,
                     derorder: int = 3,
                     number_of_turns: int = 1,
                     number_of_points: int = 10000) -> float:
    DA.init(derorder, 6)
    
    x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)

    initial_state = array.identity(6)
    initial_state[0] += x0  # x
    initial_state[2] += z0  # z
    initial_state[4] += vy0  # v_y

    with DA.cache_manager():  # optional, for efficiency
        xf = RK78(initial_state, 0.0, number_of_turns * T, CR3BP)

    # Стандартные отклонения для координат и скоростей
    std_dev_positions = std_pos # 10 см
    std_dev_velocities =  std_vel # 1 см/с

    # Создаем массив стандартных отклонений для каждой компоненты
    std_devs = np.array([std_dev_positions] * 3 + [std_dev_velocities] * 3)

    # Отклонения в начальный момент времени для каждой компоненты фазового вектора
    deltax0 = np.random.normal(0, std_devs, (number_of_points, 6))

    # Клиппинг
    limits = np.array([3 * std_dev_positions] * 3 + [3 * std_dev_velocities] * 3)
    deltax0 = np.clip(deltax0, -limits, limits)

    x0_cons = initial_state.cons()  # координаты центральной точки (начальные условия)
    x0_coords = x0_cons[:3]  # только положения

    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0]) # изменённые конечные положения объектов для возмущённых начальных условий

    # Евклидовы расстояния от каждой точки в evaluated_results до x0_coords
    distances = np.sqrt(np.sum((evaluated_results - x0_coords) ** 2, axis=1))

    return np.max(distances)


def get_maxdeviation_initial_state(initial_state: DA,
                                   T: float,
                                   std_pos: float,
                                   std_vel: float,
                                   derorder: int = 3,
                                   number_of_turns: int = 1,
                                   number_of_points: int = 10000) -> float:
    DA.init(derorder, 6)

    with DA.cache_manager():  # optional, for efficiency
        xf = RK78(initial_state, 0.0, number_of_turns * T, CR3BP)

    # Стандартные отклонения для координат и скоростей
    std_dev_positions = std_pos # 10 см
    std_dev_velocities =  std_vel # 1 см/с

    # Создаем массив стандартных отклонений для каждой компоненты
    std_devs = np.array([std_dev_positions] * 3 + [std_dev_velocities] * 3)

    # Отклонения в начальный момент времени для каждой компоненты фазового вектора
    deltax0 = np.random.normal(0, std_devs, (number_of_points, 6))

    # Клиппинг
    limits = np.array([3 * std_dev_positions] * 3 + [3 * std_dev_velocities] * 3)
    deltax0 = np.clip(deltax0, -limits, limits)

    x0_cons = initial_state.cons()  # координаты центральной точки (начальные условия)
    x0_coords = x0_cons[:3]  # только положения

    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0]) # изменённые конечные положения объектов для возмущённых начальных условий

    # Евклидовы расстояния от каждой точки в evaluated_results до x0_coords
    distances = np.sqrt(np.sum((evaluated_results - x0_coords) ** 2, axis=1))

    return np.max(distances)



def get_maxdeviation_and_delta_from_initial_state(
    initial_state: DA,
    T: float,
    std_pos: float,
    std_vel: float,
    derorder: int = 3,
    number_of_turns: int = 1,
    number_of_points: int = 10000,
) -> tuple[float, np.ndarray]:
    DA.init(derorder, 6)

    with DA.cache_manager():  # optional, for efficiency
        xf = RK78(initial_state, 0.0, number_of_turns * T, CR3BP)

    # Стандартные отклонения для координат и скоростей
    std_dev_positions = std_pos  # 10 см
    std_dev_velocities = std_vel  # 1 см/с

    # Создаем массив стандартных отклонений для каждой компоненты
    std_devs = np.array([std_dev_positions] * 3 + [std_dev_velocities] * 3)

    # Отклонения в начальный момент времени
    deltax0 = np.random.normal(0, std_devs, (number_of_points, 6))

    # Клиппинг
    limits = np.array([3 * std_dev_positions] * 3 + [3 * std_dev_velocities] * 3)
    deltax0 = np.clip(deltax0, -limits, limits)

    x0_cons = initial_state.cons()  # начальные условия (центральная точка)
    x0_coords = x0_cons[:3]  # только положения

    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0])

    # Вычисляем расстояния
    distances = np.sqrt(np.sum((evaluated_results - x0_coords) ** 2, axis=1))

    # Находим индекс максимального отклонения
    max_index = np.argmax(distances)
    max_distance = distances[max_index]
    max_delta = deltax0[max_index]

    # И минимального
    min_index = np.argmin(distances)
    min_delta = deltax0[min_index]


    return max_distance, max_delta, min_delta


def cr3bp(t: float, x: np.ndarray) -> np.ndarray:
    """
    RHS for the Circular Restricted Three-Body Problem (CR3BP)
    in a rotating reference frame using numpy.

    Args:
        x (np.ndarray): State vector [x, y, z, vx, vy, vz].
        t (float): Time (not used here, but kept for compatibility with ODE solvers).

    Returns:
        np.ndarray: Derivative of the state vector [vx, vy, vz, ax, ay, az].
    """
    pos = x[:3]
    vel = x[3:]

    r1_pos = np.array([MU + pos[0] - 1, pos[1], pos[2]])
    r2_pos = np.array([MU + pos[0], pos[1], pos[2]])

    r1 = np.linalg.norm(r1_pos)
    r2 = np.linalg.norm(r2_pos)

    acc_x = pos[0] + 2 * vel[1] - (MU * r1_pos[0]) / r1**3 - ((1 - MU) * r2_pos[0]) / r2**3
    acc_y = pos[1] - 2 * vel[0] - (MU * r1_pos[1]) / r1**3 - ((1 - MU) * r2_pos[1]) / r2**3
    acc_z = - (MU * r1_pos[2]) / r1**3 - ((1 - MU) * r2_pos[2]) / r2**3

    acc = np.array([acc_x, acc_y, acc_z])
    dx = np.concatenate((vel, acc))

    return dx


def halo_qualify(orbit_type: str, number_of_orbit: int):
    """
    Уточняет НУ для более строгой замкнутости гало-орбит из таблицы.
    """
    x0, z0, vy0, T, _, __ = initial_state_parser(orbit_type, number_of_orbit)
    p0 = np.array([x0, vy0, T])

    def objective(p: np.ndarray) -> float:
        # x0, vy0, T = p
        initial_state = np.array([p[0], 0., z0, 0., p[1], 0.])
        t_span = (0, p[2])
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(cr3bp, t_span, initial_state, t_eval=t_eval, rtol=1e-13, atol=1e-13, method='LSODA')
        last_state = sol.y.T[-1]
        return np.power(last_state - initial_state, 2).sum()

    res = minimize(
        objective,
        p0,
        method='Nelder-Mead',
        options={
            'disp': False,
            'xatol': 1e-13,
            'fatol': 1e-13,
            'maxiter': 10000,
            'maxfev': 10000,
        }
    )

    x0_res, vy0_res, T_res = res.x

    return np.array([x0_res, 0., z0, 0., vy0_res, 0.]), T_res


def get_maxdev_sampling_no_integrate(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,
    std_pos: float,
    std_vel: float,
    derorder: int = 3,
    number_of_halo_point: int | None = None,
    amount_of_points: int = 10000,
    unit_deltas: np.ndarray | None = None,
    seed: int | None = None,
) -> float:
    DA.init(derorder, 6)
    if not number_of_halo_point:
        x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)
        initial_state = array.identity(6)
        initial_state[0] += x0  # x
        initial_state[2] += z0  # z
        initial_state[4] += vy0  # v_y
    else:
        x0, z0, vy0, T, _, __ = initial_state_parser(orbit_type, number_of_orbit)
        initial_state1 = np.array([x0, 0., z0, 0., vy0, 0.])
        initial_state, T = halo_qualify(orbit_type, number_of_orbit)

        t_span = (0, T)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(cr3bp, t_span, initial_state, t_eval=t_eval, rtol=1e-13, atol=1e-13, method='LSODA')
        halo_orbit_dots = sol.y.T
        halo_orbit_dot = halo_orbit_dots[number_of_halo_point]

        initial_state = array.identity(6)
        initial_state[0] += halo_orbit_dot[0]
        initial_state[1] += halo_orbit_dot[1]
        initial_state[2] += halo_orbit_dot[2]
        initial_state[3] += halo_orbit_dot[3]
        initial_state[4] += halo_orbit_dot[4]
        initial_state[5] += halo_orbit_dot[5]


    # Стандартные отклонения для координат и скоростей
    std_dev_positions = std_pos
    std_dev_velocities = std_vel

    # Создаем массив стандартных отклонений для каждой компоненты
    std_devs = np.array([std_dev_positions] * 3 + [std_dev_velocities] * 3)

    # Генерируем изменения (deltax0) для каждой компоненты
    # Если переданы unit_deltas (единичные нормальные шумы), переиспользуем их для воспроизводимости
    if unit_deltas is not None:
        if unit_deltas.shape != (amount_of_points, 6):
            raise ValueError(
                f"unit_deltas must have shape ({amount_of_points}, 6), got {unit_deltas.shape}"
            )
        deltax0 = unit_deltas * std_devs  # масштабирование по std_pos/std_vel
    else:
        if seed is not None:
            rng = np.random.default_rng(seed)
            deltax0 = rng.normal(0.0, 1.0, (amount_of_points, 6)) * std_devs
        else:
            deltax0 = np.random.normal(0.0, 1.0, (amount_of_points, 6)) * std_devs

    # Клиппинг
    limits = np.array([3 * std_dev_positions] * 3 + [3 * std_dev_velocities] * 3)
    deltax0 = np.clip(deltax0, -limits, limits)

    x0_cons = initial_state.cons()  # координаты центральной точки (начальные условия)
    x0_coords = x0_cons[:3]  # только положения

    x0_new = x0_cons + deltax0  # отклонения от начального положения

    # Применение формулы Тейлора
    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0]) # изменённые конечные положения объектов для возмущённых начальных условий

    # вычисление Евклидова расстояния от каждой точки в evaluated_results до x0_coords
    distances = np.sqrt(np.sum((evaluated_results - x0_coords) ** 2, axis=1))

    return np.quantile(distances, 0.999)


def get_maxdev_sampling_ellipsoid(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,
    std_pos: float,
    std_vel: float,
    derorder: int = 3,
    number_of_halo_point: int | None = None,
    amount_of_points: int = 10000,
    unit_deltas: np.ndarray | None = None,
    seed: int | None = None,
    radius: float = 3.0,
) -> float:
    """
    Семплирование отклонений с проекцией шумов на эллипсоид в "whitened"‑пространстве.

    Отличие от get_maxdev_sampling_no_integrate(): вместо покомпонентного клиппинга
    выполняется проекция каждого семпла на эллипсоид уровня

        (dx_1/σ_pos)^2 + (dx_2/σ_pos)^2 + (dx_3/σ_pos)^2
      + (dx_4/σ_vel)^2 + (dx_5/σ_vel)^2 + (dx_6/σ_vel)^2 ≤ radius^2.

    Реализовано через переход к переменной u = D^{-1} x, D = diag([σ_pos]*3 + [σ_vel]*3)
    и радиальную проекцию на шар ||u|| ≤ radius: если ||u|| > radius, то
    u ← radius * u / ||u||, затем x ← D u.

    Возвращает 0.999-квантиль нормы отклонения положения (DU).
    """
    DA.init(derorder, 6)

    # Базовое состояние (центральная точка) — как в get_maxdev_sampling_no_integrate
    if not number_of_halo_point:
        x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)
        initial_state = array.identity(6)
        initial_state[0] += x0  # x
        initial_state[2] += z0  # z
        initial_state[4] += vy0  # v_y
    else:
        x0, z0, vy0, T, _, __ = initial_state_parser(orbit_type, number_of_orbit)
        initial_state_np, T = halo_qualify(orbit_type, number_of_orbit)

        t_span = (0, T)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(cr3bp, t_span, initial_state_np, t_eval=t_eval, rtol=1e-13, atol=1e-13, method='LSODA')
        halo_orbit_dots = sol.y.T
        halo_orbit_dot = halo_orbit_dots[number_of_halo_point]

        initial_state = array.identity(6)
        initial_state[0] += halo_orbit_dot[0]
        initial_state[1] += halo_orbit_dot[1]
        initial_state[2] += halo_orbit_dot[2]
        initial_state[3] += halo_orbit_dot[3]
        initial_state[4] += halo_orbit_dot[4]
        initial_state[5] += halo_orbit_dot[5]

    # Стандартные отклонения для координат и скоростей и вектор весов
    std_dev_positions = std_pos
    std_dev_velocities = std_vel
    d = np.array([std_dev_positions] * 3 + [std_dev_velocities] * 3)

    # Генерация шумов (единичные нормальные, далее масштабируем d)
    if unit_deltas is not None:
        if unit_deltas.shape != (amount_of_points, 6):
            raise ValueError(
                f"unit_deltas must have shape ({amount_of_points}, 6), got {unit_deltas.shape}"
            )
        deltax0 = unit_deltas * d
    else:
        if seed is not None:
            rng = np.random.default_rng(seed)
            deltax0 = rng.normal(0.0, 1.0, (amount_of_points, 6)) * d
        else:
            deltax0 = np.random.normal(0.0, 1.0, (amount_of_points, 6)) * d

    # Проекция на эллипсоид: u = D^{-1} x, если ||u|| > radius, то u <- radius * u / ||u||
    # Безопасное деление: std_pos/std_vel могут быть нулями → соответствующие компоненты u ставим в 0
    u = np.divide(deltax0, d, out=np.zeros_like(deltax0), where=d != 0)  # whitened
    norms = np.linalg.norm(u, axis=1)
    mask = norms > radius
    scales = np.ones_like(norms)
    scales[mask] = radius / norms[mask]
    u_proj = u * scales[:, None]
    deltax0_proj = d * u_proj

    # Центральная точка для вычисления отклонений в координатном пространстве
    x0_cons = initial_state.cons()
    x0_coords = x0_cons[:3]

    # Оценка через DA-карту (xf)
    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0_proj])

    # Нормы отклонений и возврат максимума
    distances = np.sqrt(np.sum((evaluated_results - x0_coords) ** 2, axis=1))
    return np.max(distances)


def get_maxdev_sampling_ellipsoid_with_vector(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,
    std_pos: float,
    std_vel: float,
    derorder: int = 3,
    number_of_halo_point: int | None = None,
    amount_of_points: int = 10000,
    unit_deltas: np.ndarray | None = None,
    seed: int | None = None,
    radius: float = 3.0,
) -> tuple[float, np.ndarray]:
    """
    Как get_maxdev_sampling_ellipsoid, но помимо максимального значения возвращает
    и 6D-вектор начального отклонения x (в DU/VU), который дал этот максимум.
    Для визуализации обычно берём позиции x[:3].
    """
    DA.init(derorder, 6)

    # Базовое состояние
    if not number_of_halo_point:
        x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)
        initial_state = array.identity(6)
        initial_state[0] += x0
        initial_state[2] += z0
        initial_state[4] += vy0
    else:
        x0, z0, vy0, T, _, __ = initial_state_parser(orbit_type, number_of_orbit)
        initial_state_np, T = halo_qualify(orbit_type, number_of_orbit)

        t_span = (0, T)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(cr3bp, t_span, initial_state_np, t_eval=t_eval, rtol=1e-13, atol=1e-13, method='LSODA')
        halo_orbit_dots = sol.y.T
        halo_orbit_dot = halo_orbit_dots[number_of_halo_point]

        initial_state = array.identity(6)
        initial_state[0] += halo_orbit_dot[0]
        initial_state[1] += halo_orbit_dot[1]
        initial_state[2] += halo_orbit_dot[2]
        initial_state[3] += halo_orbit_dot[3]
        initial_state[4] += halo_orbit_dot[4]
        initial_state[5] += halo_orbit_dot[5]

    # Диагонали std
    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)

    # Семплы
    if unit_deltas is not None:
        if unit_deltas.shape != (amount_of_points, 6):
            raise ValueError(
                f"unit_deltas must have shape ({amount_of_points}, 6), got {unit_deltas.shape}"
            )
        deltax0 = unit_deltas * d
    else:
        rng = np.random.default_rng(seed)
        deltax0 = rng.normal(0.0, 1.0, (amount_of_points, 6)) * d

    # Проекция на эллипсоид (whitened радиальная)
    # Безопасное деление, чтобы std_pos/std_vel = 0 не давали nan/inf
    u = np.divide(deltax0, d, out=np.zeros_like(deltax0), where=d != 0)
    norms = np.linalg.norm(u, axis=1)
    mask = norms > radius
    scales = np.ones_like(norms)
    scales[mask] = radius / norms[mask]
    u_proj = u * scales[:, None]
    deltax0_proj = d * u_proj

    x0_cons = initial_state.cons()
    x0_coords = x0_cons[:3]
    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0_proj])
    distances = np.sqrt(np.sum((evaluated_results - x0_coords) ** 2, axis=1))

    idx = int(np.argmax(distances))
    return float(distances[idx]), deltax0_proj[idx]


def get_monodromy_matrix(orbit_type: str,
                         number_of_orbit: int):
    DA.init(3, 6)
    
    x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)

    initial_state = array.identity(6)
    initial_state[0] += x0  # x
    initial_state[2] += z0  # z
    initial_state[4] += vy0  # v_y

    with DA.cache_manager():  # optional, for efficiency
        xf = RK78(initial_state, 0.0, T, CR3BP)

    monodromy_matrix = np.array(xf.linear())

    return monodromy_matrix


def get_maxdev_floquet_ellipsoid(
    orbit_type: str,
    number_of_orbit: int,
    std_pos: float,
    std_vel: float,
    radius: float = 3.0,
) -> float:
    """
    Классическая (Флоке) оценка максимального отклонения на эллипсоиде неопределенности.

    Шаги:
      1) Находим максимальный по модулю множитель монодромии (берём MAX_MUL из CSV)
         и соответствующий ему собственный вектор матрицы монодромии M.
      2) Проецируем этот собственный вектор на границу эллипсоида неопределенности
         в 6D: (x_1/σ_pos)^2 + ... + (x_3/σ_pos)^2 + (x_4/σ_vel)^2 + ... ≤ radius^2.
         То есть нормируем векторально в «whitened»‑норме и масштабируем на radius.
      3) Умножаем полученный вектор на MAX_MUL, берём норму координатной части (3D).

    Возвращает норму позиции конечного отклонения (в DU).
    """
    # 0) Данные орбиты и монодромия
    x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)
    M = get_monodromy_matrix(orbit_type, number_of_orbit)

    # 1) Собственные значения/векторы; берём тот, чья |λ| ближе всего к MAX_MUL
    eigvals, eigvecs = np.linalg.eig(M)
    idx = int(np.argmin(np.abs(np.abs(eigvals) - float(MAX_MUL))))
    v = eigvecs[:, idx]
    lam = float(eigvals[idx])

    # Делаем вектор вещественным (ожидается реальный; берём действительную часть)
    if np.iscomplexobj(v):
        v = np.real(v)

    # print()
    # print("!!!")
    # print("MAX_MUL = ", MAX_MUL)
    # print("LAM = ", lam)
    # print("FOUND EIG VEC = ", v)
    # print()

    # 2) Проекция на границу эллипсоида в whitened‑норме
    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)
    denom = float(np.linalg.norm(v / d))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    v_proj = (float(radius) / denom) * v  # начальное отклонение на границе эллипсоида

    # 3) «Проход через период» по Флоке: масштабирование на MAX_MUL
    delta_final = lam * v_proj
    return float(np.linalg.norm(delta_final[:3]))


def get_maxdev_floquet_ellipsoid_with_vector(
    orbit_type: str,
    number_of_orbit: int,
    std_pos: float,
    std_vel: float,
    radius: float = 3.0,
) -> tuple[float, np.ndarray]:
    """
    Как get_maxdev_floquet_ellipsoid, но возвращает также 6D-вектор v_proj (DU/VU)
    — начальное возмущение на границе эллипсоида, дающее максимум в модели Флоке.
    Возвращаемое значение — ||lambda * v_proj|| по позиционной части.
    """
    x0, z0, vy0, T, JACOBI, MAX_MUL = initial_state_parser(orbit_type, number_of_orbit)
    M = get_monodromy_matrix(orbit_type, number_of_orbit)

    eigvals, eigvecs = np.linalg.eig(M)
    idx = int(np.argmin(np.abs(np.abs(eigvals) - float(MAX_MUL))))
    v = eigvecs[:, idx]
    lam = float(eigvals[idx])
    if np.iscomplexobj(v):
        v = np.real(v)

    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)
    denom = float(np.linalg.norm(v / d))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0, np.zeros(6)
    v_proj = (float(radius) / denom) * v

    delta_final = lam * v_proj
    val = float(np.linalg.norm(delta_final[:3]))
    return val, v_proj


def get_maxdev_optimization_no_integrate(
        orbit_type: str,
        number_of_orbit: int,
        xf: DA,
        std_pos: float,
        std_vel: float,
        verbose: bool = False,
):
    """
    Максимизирует норму отклонения в координатном пространстве через период,
    используя полиномиальную (DA) аппроксимацию потока без повторной интеграции.

    Замечание: xf — это результат get_xf(...), который уже содержит члены
    старших порядков (A1 x + A2 xx + A3 xxx + ...). Здесь мы напрямую
    используем оценку F(x) через xf.eval(x), что эквивалентно подстановке
    соответствующих подтензоров только для координатной (3-мерной) части.

    Для ускорения/стабильности мы используем multi-start по ограниченному
    боксу и метод L-BFGS-B (минимизируем -||F(x)||^2).

    Возвращает максимальную норму отклонения (float) в тех же безразмерных единицах.
    """

    # 1) Центральная точка (для вычитания в координатном пространстве)
    x0, z0, vy0, T, _, _ = initial_state_parser(orbit_type, number_of_orbit)
    initial_state = array.identity(6)
    initial_state[0] += x0
    initial_state[2] += z0
    initial_state[4] += vy0
    x0_coords = initial_state.cons()[:3]

    # 2) Границы на x (позиции и скорости отдельно)
    pos_lim = 3.0 * std_pos
    vel_lim = 3.0 * std_vel
    lower = np.array([-pos_lim, -pos_lim, -pos_lim, -vel_lim, -vel_lim, -vel_lim])
    upper = -lower

    bounds = [(lower[i], upper[i]) for i in range(6)]

    # 3) Целевая функция: минимизируем отрицательный квадрат нормы (эквив. максимизируем норму)
    def objective(x: np.ndarray) -> float:
        Fx = xf.eval(x)[:3] - x0_coords
        return -float(np.dot(Fx, Fx))

    # 4) Инициализации: несколько разумных стартов + экстремальные точки бокса
    inits: list[np.ndarray] = []
    inits.append(np.zeros(6))

    # Оси по отдельности (в плюс/минус)
    axes = [np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 1])]
    scales = np.array([pos_lim, pos_lim, pos_lim, vel_lim, vel_lim, vel_lim])
    for i in range(6):
        vec = np.zeros(6)
        vec[i] = scales[i]
        inits.append(vec)
        inits.append(-vec)

    # Грубая линейная эвристика для выбора «угла» бокса
    try:
        A_lin = np.array(xf.linear())[:3, :]  # позиционная подматрица монодромии
        Q = A_lin.T @ A_lin                   # матрица квадратичной формы
        # Собственный вектор для max направления
        w, V = np.linalg.eigh(Q)
        vmax = V[:, np.argmax(w)]
        corner = np.sign(vmax) * scales
        inits.append(corner)
        inits.append(-corner)
    except Exception:
        # На случай, если .linear() недоступен
        pass

    # Пара случайных стартов внутри бокса
    rng = np.random.default_rng(12345)
    for _ in range(8):
        r = rng.uniform(-1.0, 1.0, size=6)
        inits.append(r * scales)

    best_val = -np.inf
    best_x: np.ndarray | None = None
    # Итоговая норма (а не квадрат нормы)
    best_norm = 0.0

    for x0_init in inits:
        res = minimize(
            objective,
            x0_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        # objective возвращает отрицательный квадрат нормы
        val = -res.fun  # квадрат нормы
        if val > best_val:
            best_val = float(val)
            best_x = np.array(res.x, dtype=float)

    best_norm = np.sqrt(max(best_val, 0.0))

    if verbose and best_x is not None:
        # Диагностика решения: печать в удобных единицах и проверка ограничений
        limits = np.array([pos_lim] * 3 + [vel_lim] * 3)
        ratios = np.abs(best_x) / (limits + 1e-30)
        pos_km = du2km(best_x[:3])
        vel_ms = vu2ms(best_x[3:])
        Fx = xf.eval(best_x)[:3] - x0_coords
        print("[opt] argmax x (pos DU, vel VU):", np.array2string(best_x, precision=6))
        print("[opt] argmax x (pos km, vel m/s):",
              np.array2string(np.concatenate([pos_km, vel_ms]), precision=6))
        print("[opt] |x|/limits per-comp:", np.array2string(ratios, precision=3))
        print("[opt] within bounds (<=1+tol)?",
              bool(np.all(np.abs(best_x) <= limits + 1e-10)))
        print("[opt] F(x) (DU):", np.array2string(Fx, precision=6),
              "||F||=", float(np.linalg.norm(Fx)))
    return best_norm


def get_maxdev_linear_corner_max(
    xf: DA,
    std_pos: float,
    std_vel: float,
) -> float:
    """
    Линейная оценка максимума: max_x ||A_pos x|| на боксе
    |x[:3]| ≤ 3*std_pos, |x[3:6]| ≤ 3*std_vel.

    A_pos — верхний (3x6) блок линейной части DA-карты xf.
    Максимум выпуклой нормы на гиперпрямоугольнике достигается в одной из 2^6 вершин.
    Перебираем все 64 угла бокса.
    Возвращает максимум нормы в DU.
    """
    A_pos = np.array(xf.linear())[:3, :]
    pos_lim = 3.0 * std_pos
    vel_lim = 3.0 * std_vel
    scales = np.array([pos_lim] * 3 + [vel_lim] * 3)

    best = 0.0
    for signs in product((-1.0, 1.0), repeat=6):
        x = scales * np.array(signs)
        y = A_pos @ x
        val = float(np.linalg.norm(y))
        if val > best:
            best = val
    return best


def get_maxdev_optimization_ellipsoid(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,
    std_pos: float,
    std_vel: float,
    radius: float = 3.0,
    verbose: bool = False,
    n_random_starts: int = 6,
    init_strategy: str = 'mixed_multistart',  # 'multi' | 'linear' | 'mixed_multistart'
) -> float:
    """
    Максимизация ||F(x)|| при эллипсоидальном ограничении на начальное возмущение x.

    Формулировка:
      maximize  || F(x) ||,    F(x) = xf.eval(x)[:3] - x0_coords
      subject to  (x_1/σ_pos)^2 + (x_2/σ_pos)^2 + (x_3/σ_pos)^2
                + (x_4/σ_vel)^2 + (x_5/σ_vel)^2 + (x_6/σ_vel)^2  ≤  radius^2

    Реализация через замену x = D u, где
      D = diag([σ_pos, σ_pos, σ_pos, σ_vel, σ_vel, σ_vel]),  и ||u||_2 ≤ radius.

    Аргументы:
      - orbit_type (str): Тип орбиты, 'L1' или 'L2'.
      - number_of_orbit (int): Номер орбиты (1‑индексация) в таблице исходных НУ.
      - xf (DA): DA‑карта потока за период; используется для оценки F(x) без интегрирования.
      - std_pos (float): Стандартное отклонение по положениям в DU (безразмерные ед.).
      - std_vel (float): Стандартное отклонение по скоростям в VU (безразмерные ед.).
      - radius (float): Радиус эллипсоида в whitened‑пространстве (число σ). По умолчанию 3.0.
      - verbose (bool): Если True — печатает диагностику найденного максимума.
      - n_random_starts (int): Количество ДОПОЛНИТЕЛЬНЫХ случайных стартов в режимах
          init_strategy='multi' и init_strategy='mixed_multistart'. Помимо случайных стартов всегда добавляются
          детерминированные начальные точки: нулевая, а также ±radius вдоль
          каждой оси (итого 13 точек без учёта случайных). Случайные старты
          равномерно распределены по сфере ||u||=radius в whitened‑пространстве
          (нормальная выборка с последующей нормировкой), генератор с фиксированным
          seed=2025 для воспроизводимости. Установите 0, чтобы отключить случайные
          старты. Параметр игнорируется при init_strategy='linear'. Увеличение
          значения повышает шанс найти больший локальный максимум, но увеличивает
          время вычислений.
      - init_strategy (str): 'multi' — мультистарт (ноль, базисы, случайные на сфере);
                              'linear' — один старт из линейной оценки (правый сингулярный вектор A_pos D,
                              умноженный на radius);
                              'mixed_multistart' — мультистарт как в 'multi' плюс дополнительный старт
                              из линейной оценки.

    Возвращает:
      - float: Максимальная норма отклонения позиции через период (в DU).
    """
    x0, z0, vy0, T, _, _ = initial_state_parser(orbit_type, number_of_orbit)
    initial_state = array.identity(6)
    initial_state[0] += x0
    initial_state[2] += z0
    initial_state[4] += vy0
    x0_coords = initial_state.cons()[:3]

    # Диагональ масштаба (DU и VU)
    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)

    # Маска ненулевых компонент для устойчивого констрейнта, когда σ=0
    mask = d > 0
    r2 = float(radius * radius)

    def constraint_fun(u: np.ndarray) -> float:
        uu = u[mask]
        return r2 - float(np.dot(uu, uu))

    cons = ({'type': 'ineq', 'fun': constraint_fun},)

    def objective_u(u: np.ndarray) -> float:
        x = d * u
        Fx = xf.eval(x)[:3] - x0_coords
        val = float(np.dot(Fx, Fx))
        if not np.isfinite(val):
            # При выходе DA-полиномов из радиуса сходимости могут появиться NaN/inf.
            # Возвращаем большую положительную «антиприбыль», чтобы SLSQP отбросил эту точку.
            return 1e30
        return -val

    # Инициализация стартовых точек по стратегии:
    inits: list[np.ndarray] = []
    if init_strategy == 'linear':
        # Одно начальное приближение из линейной эллипсоидальной модели через helper
        v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
        u0 = float(radius) * v_hat
        inits = [u0]
    else:
        # Мультистарт: ноль, базисные направления и случайные точки на сфере
        inits.append(np.zeros(6))
        for i in range(6):
            e = np.zeros(6)
            e[i] = radius
            inits.append(e)
            inits.append(-e)

        rng = np.random.default_rng(2025)
        for _ in range(n_random_starts):
            u = rng.normal(0.0, 1.0, 6)
            nu = np.linalg.norm(u)
            if nu > 1e-12:
                u = (radius * u) / nu  # на сферу
            else:
                u = np.zeros(6)
            inits.append(u)

        if init_strategy == 'mixed_multistart':
            # Добавляем линейный старт к стандартному мультистарту
            v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
            u0 = float(radius) * v_hat
            inits.append(u0)

    best_val = -np.inf
    best_u: np.ndarray | None = None

    for u0 in inits:
        res = minimize(
            objective_u,
            u0,
            method='SLSQP',
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 500, 'disp': False},
        )
        if not np.isfinite(res.fun):
            continue  # численная ошибка оптимизатора, пробуем другой старт
        val = -res.fun
        if not np.isfinite(val):
            continue
        if val > best_val:
            best_val = float(val)
            best_u = np.array(res.x, dtype=float)

    # Если все старты «упали» (best_val так и остался -inf/отрицательный),
    # возвращаем устойчивую линейную оценку, чтобы избежать NaN/нулей на карте.
    if not np.isfinite(best_val) or best_val < 0:
        if not mask.any():
            return 0.0
        lin_val = get_maxdev_linear_ellipsoid(xf, std_pos, std_vel, radius)
        if verbose:
            print('[opt-ell] fallback to linear ellipsoid, val=', lin_val)
        return lin_val

    best_norm = np.sqrt(max(best_val, 0.0))

    if verbose and best_u is not None:
        x_best = d * best_u
        Fx = xf.eval(x_best)[:3] - x0_coords
        # относительное положение на сфере
        u_norm = float(np.linalg.norm(best_u[mask]))
        print('[opt-ell] ||u||/radius =', u_norm / radius if radius > 0 else np.nan)
        print('[opt-ell] u:', np.array2string(best_u, precision=6))
        print('[opt-ell] x (DU/VU):', np.array2string(x_best, precision=6))
        print('[opt-ell] x (km, m/s):',
              np.array2string(np.concatenate([du2km(x_best[:3]), vu2ms(x_best[3:])]), precision=6))
        print('[opt-ell] F(x) (DU):', np.array2string(Fx, precision=6), '||F||=', float(np.linalg.norm(Fx)))

    return best_norm


def get_maxdev_optimization_ellipsoid_with_vector(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,
    std_pos: float,
    std_vel: float,
    radius: float = 3.0,
    verbose: bool = False,
    n_random_starts: int = 6,
    init_strategy: str = 'mixed_multistart',  # 'multi' | 'linear' | 'mixed_multistart'
) -> tuple[float, np.ndarray]:
    """
    Вариант get_maxdev_optimization_ellipsoid, возвращающий и максимальное значение,
    и 6D-вектор начального отклонения x, который его достигает (для визуализации).
    """
    x0, z0, vy0, T, _, _ = initial_state_parser(orbit_type, number_of_orbit)
    initial_state = array.identity(6)
    initial_state[0] += x0
    initial_state[2] += z0
    initial_state[4] += vy0
    x0_coords = initial_state.cons()[:3]

    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)
    mask = d > 0
    r2 = float(radius * radius)

    def constraint_fun(u: np.ndarray) -> float:
        uu = u[mask]
        return r2 - float(np.dot(uu, uu))

    cons = ({'type': 'ineq', 'fun': constraint_fun},)

    def objective_u(u: np.ndarray) -> float:
        x = d * u
        Fx = xf.eval(x)[:3] - x0_coords
        return -float(np.dot(Fx, Fx))

    inits: list[np.ndarray] = []
    if init_strategy == 'linear':
        v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
        inits = [float(radius) * v_hat]
    else:
        inits.append(np.zeros(6))
        for i in range(6):
            e = np.zeros(6)
            e[i] = radius
            inits.append(e)
            inits.append(-e)
        rng = np.random.default_rng(2025)
        for _ in range(n_random_starts):
            u = rng.normal(0.0, 1.0, 6)
            nu = np.linalg.norm(u)
            u = (radius * u / nu) if nu > 1e-12 else np.zeros(6)
            inits.append(u)

        if init_strategy == 'mixed_multistart':
            v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
            inits.append(float(radius) * v_hat)

    best_val = -np.inf
    best_u: np.ndarray | None = None

    for u0 in inits:
        res = minimize(
            objective_u,
            u0,
            method='SLSQP',
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 500, 'disp': False},
        )
        if not np.isfinite(res.fun):
            continue
        val = -res.fun
        if not np.isfinite(val):
            continue
        if val > best_val:
            best_val = float(val)
            best_u = np.array(res.x, dtype=float)

    # Фолбэк на линейную оценку, если оптимизация не дала корректного результата.
    if not np.isfinite(best_val) or best_val < 0:
        if not mask.any():
            return 0.0, np.zeros(6)
        lin_val, lin_x = get_maxdev_linear_ellipsoid_with_vector(xf, std_pos, std_vel, radius)
        if verbose:
            print('[opt-ell/vec] fallback to linear ellipsoid, val=', lin_val)
        return lin_val, lin_x

    best_norm = np.sqrt(max(best_val, 0.0))
    x_best = d * (best_u if best_u is not None else np.zeros(6))

    if verbose and best_u is not None:
        Fx = xf.eval(x_best)[:3] - x0_coords
        u_norm = float(np.linalg.norm(best_u[mask]))
        print('[opt-ell/vec] ||u||/radius =', u_norm / radius if radius > 0 else np.nan)
        print('[opt-ell/vec] u:', np.array2string(best_u, precision=6))
        print('[opt-ell/vec] x (DU/VU):', np.array2string(x_best, precision=6))
        print('[opt-ell/vec] F(x) (DU):', np.array2string(Fx, precision=6), '||F||=', float(np.linalg.norm(Fx)))

    return best_norm, x_best


def get_maxdev_optimization_ellipsoid_integrate(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,  # not used; kept for API similarity with DA-based version
    std_pos: float,
    std_vel: float,
    radius: float = 3.0,
    verbose: bool = False,
    n_random_starts: int = 6,
    init_strategy: str = 'mixed_multistart',  # 'multi' | 'linear' | 'mixed_multistart'
) -> float:
    """
    Максимизация ||F(x)|| на эллипсоиде неопределенности с «честной» интеграцией.

    Вместо DA-аппроксимации (xf.eval(x)) целевая функция считается численно:
    интегрируем CR3BP из возмущённого состояния до периода T методом
    `solve_ivp(..., method='LSODA', rtol=1e-13, atol=1e-13)` и берём норму
    позиционного отклонения ||pos(T; x0+x) - pos(x0)||.

    Формулировка ограничения (эллипсоид в пространстве x):
      (x_1/sigma_pos)^2 + (x_2/sigma_pos)^2 + (x_3/sigma_pos)^2
    + (x_4/sigma_vel)^2 + (x_5/sigma_vel)^2 + (x_6/sigma_vel)^2 ≤ radius^2

    Переход к переменной u: x = D u, где D = diag([sigma_pos]*3 + [sigma_vel]*3), ||u||_2 ≤ radius.

    Аргументы:
      - orbit_type (str): Тип орбиты, 'L1' или 'L2'.
      - number_of_orbit (int): Номер орбиты (1-индексация) в таблице исходных НУ.
      - xf (DA): Не используется, сохранён для совместимости интерфейса.
      - std_pos (float): Стандартное отклонение по положениям в DU (безразмерные ед.).
      - std_vel (float): Стандартное отклонение по скоростям в VU (безразмерные ед.).
      - radius (float): Радиус эллипсоида в whitened-пространстве. По умолчанию 3.0.
      - verbose (bool): Если True — печатает диагностику найденного максимума.
      - n_random_starts (int): Количество ДОПОЛНИТЕЛЬНЫХ случайных стартов в режимах
          init_strategy='multi' и init_strategy='mixed_multistart'. Помимо случайных стартов всегда добавляются
          детерминированные начальные точки: нулевая, а также ±radius вдоль
          каждой оси (итого 13 точек без учёта случайных). Случайные старты
          равномерно распределены по сфере ||u||=radius в whitened-пространстве
          (нормальная выборка с последующей нормировкой), генератор с фиксированным
          seed=2025 для воспроизводимости. Установите 0, чтобы отключить случайные
          старты. Параметр игнорируется при init_strategy='linear'. Ввиду высокой
          стоимости интегрирования рекомендуется 0 или небольшие значения.
      - init_strategy (str): 'multi' — мультистарт;
                             'linear' — один старт из линейной оценки;
                             'mixed_multistart' — мультистарт + один старт из линейной оценки.

    Возвращает:
      - float: Максимальная норма отклонения позиции через период (в DU).
    """

    # Базовая (центральная) точка орбиты и период.
    initial_state_np, T = halo_qualify(orbit_type, number_of_orbit)
    x0_coords = initial_state_np[:3].copy()

    # Веса (стандартные отклонения) и радиус эллипсоида
    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)

    # Целевая функция в переменной u (так же как в DA-версии, но через интегрирование):
    # maximize ||F(D u)||^2  <=>  minimize -||F(D u)||^2,
    # где F(x) = pos(final(x0 + x)) - x0_coords.
    def objective_u(u: np.ndarray) -> float:
        x = d * u  # преобразование из whitened‑пространства
        y0 = initial_state_np + x
        t_span = (0.0, float(T))
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(
            cr3bp,
            t_span,
            y0,
            t_eval=t_eval,
            rtol=1e-13,
            atol=1e-13,
            method='LSODA',
        )
        # Берём последнюю точку (t ≈ T)
        last_state = sol.y.T[-1]
        Fx = last_state[:3] - x0_coords
        return -float(np.dot(Fx, Fx))

    # Ограничение: ||u||^2 ≤ radius^2  ->  radius^2 - ||u||^2 ≥ 0
    mask = np.ones(6, dtype=bool)
    cons = ({
        'type': 'ineq',
        'fun': lambda u: float(radius**2 - np.dot(u[mask], u[mask])),
    },)

    # Стартовые точки: выбираются по стратегии
    inits: list[np.ndarray] = []
    if init_strategy == 'linear':
        v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
        u0 = float(radius) * v_hat
        inits = [u0]
    else:
        inits = [np.zeros(6)]
        for i in range(6):
            e = np.zeros(6)
            e[i] = radius
            inits.append(e)
            inits.append(-e)

        rng = np.random.default_rng(2025)
        for _ in range(n_random_starts):
            u = rng.normal(0.0, 1.0, 6)
            nu = np.linalg.norm(u)
            if nu > 1e-12:
                u = (radius * u) / nu
            else:
                u = np.zeros(6)
            inits.append(u)

        if init_strategy == 'mixed_multistart':
            v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
            u0 = float(radius) * v_hat
            inits.append(u0)

    best_val = -np.inf
    best_u: np.ndarray | None = None

    for u0 in inits:
        res = minimize(
            objective_u,
            u0,
            method='SLSQP',
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 500, 'disp': False},
        )
        val = -res.fun
        if val > best_val:
            best_val = float(val)
            best_u = np.array(res.x, dtype=float)

    best_norm = np.sqrt(max(best_val, 0.0)) if np.isfinite(best_val) else np.nan

    if not np.isfinite(best_norm):
        # На случай полного срыва оптимизации возвращаем гарантированно конечную
        # линейную оценку, чтобы не отдавать NaN на тепловую карту.
        return get_maxdev_linear_ellipsoid(xf, std_pos, std_vel, radius)

    if verbose and best_u is not None:
        x_best = d * best_u
        # Перепропускаем для печати подробностей
        y0 = initial_state_np + x_best
        t_span = (0.0, float(T))
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(
            cr3bp,
            t_span,
            y0,
            t_eval=t_eval,
            rtol=1e-13,
            atol=1e-13,
            method='LSODA',
        )
        last_state = sol.y.T[-1]
        Fx = last_state[:3] - x0_coords
        u_norm = float(np.linalg.norm(best_u[mask]))
        print('[opt-ell-int] ||u||/radius =', u_norm / radius if radius > 0 else np.nan)
        print('[opt-ell-int] u:', np.array2string(best_u, precision=6))
        print('[opt-ell-int] x (DU/VU):', np.array2string(x_best, precision=6))
        print('[opt-ell-int] x (km, m/s):',
              np.array2string(np.concatenate([du2km(x_best[:3]), vu2ms(x_best[3:])]), precision=6))
        print('[opt-ell-int] F(x) (DU):', np.array2string(Fx, precision=6), '||F||=', float(np.linalg.norm(Fx)))

    return best_norm


def get_maxdev_optimization_ellipsoid_integrate_with_vector(
    orbit_type: str,
    number_of_orbit: int,
    xf: DA,  # kept for API similarity
    std_pos: float,
    std_vel: float,
    radius: float = 3.0,
    verbose: bool = False,
    n_random_starts: int = 6,
    init_strategy: str = 'mixed_multistart',
) -> tuple[float, np.ndarray]:
    """
    Как get_maxdev_optimization_ellipsoid_integrate, но возвращает и 6D-вектор x,
    который дал максимум (начальное возмущение в DU/VU).
    """
    initial_state_np, T = halo_qualify(orbit_type, number_of_orbit)
    x0_coords = initial_state_np[:3].copy()

    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)

    def objective_u(u: np.ndarray) -> float:
        x = d * u
        y0 = initial_state_np + x
        t_span = (0.0, float(T))
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(cr3bp, t_span, y0, t_eval=t_eval, rtol=1e-13, atol=1e-13, method='LSODA')
        last_state = sol.y.T[-1]
        Fx = last_state[:3] - x0_coords
        return -float(np.dot(Fx, Fx))

    cons = ({
        'type': 'ineq',
        'fun': lambda u: float(radius**2 - np.dot(u, u)),
    },)

    inits: list[np.ndarray] = []
    if init_strategy == 'linear':
        v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
        inits = [float(radius) * v_hat]
    else:
        inits = [np.zeros(6)]
        for i in range(6):
            e = np.zeros(6)
            e[i] = radius
            inits.append(e)
            inits.append(-e)
        rng = np.random.default_rng(2025)
        for _ in range(n_random_starts):
            u = rng.normal(0.0, 1.0, 6)
            nu = np.linalg.norm(u)
            inits.append((radius * u / nu) if nu > 1e-12 else np.zeros(6))

        if init_strategy == 'mixed_multistart':
            v_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
            inits.append(float(radius) * v_hat)

    best_val = -np.inf
    best_u: np.ndarray | None = None

    for u0 in inits:
        res = minimize(
            objective_u,
            u0,
            method='SLSQP',
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 500, 'disp': False},
        )
        val = -res.fun
        if val > best_val:
            best_val = float(val)
            best_u = np.array(res.x, dtype=float)

    best_norm = np.sqrt(max(best_val, 0.0))
    x_best = d * (best_u if best_u is not None else np.zeros(6))

    if verbose and best_u is not None:
        y0 = initial_state_np + x_best
        t_span = (0.0, float(T))
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(cr3bp, t_span, y0, t_eval=t_eval, rtol=1e-13, atol=1e-13, method='LSODA')
        last_state = sol.y.T[-1]
        Fx = last_state[:3] - x0_coords
        print('[opt-ell-int/vec] x:', np.array2string(x_best, precision=6), '||F||=', float(np.linalg.norm(Fx)))

    return best_norm, x_best

def get_maxdev_linear_ellipsoid(
    xf: DA,
    std_pos: float,
    std_vel: float,
    radius: float,
) -> float:
    """
    Линейная (аналитическая) оценка максимума на эллипсоиде неопределенности.

    Формула: d_max_lin = r * sigma_max(A_pos @ D),
      где A_pos — верхний 3x6 блок линейной части (монодромии),
          D = diag([σ_pos]*3 + [σ_vel]*3), r — радиус в whitened‑пространстве.

    Радиус r задаётся явно пользователем. Возвращает максимум ||A_pos x|| в DU.
    """
    A_pos = np.array(xf.linear())[:3, :]
    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)
    r = float(radius)

    M = A_pos @ np.diag(d)
    svals = np.linalg.svd(M, compute_uv=False, full_matrices=False)
    smax = float(svals[0]) if svals.size > 0 else 0.0
    return r * smax


def get_maxdev_linear_ellipsoid_argmax_vector(
    xf: DA,
    std_pos: float,
    std_vel: float,
) -> np.ndarray:
    """
    Возвращает направление u_hat (в whitened‑пространстве),
    максимизирующее линейную оценку ||A_pos D u|| при ||u||≤1.

    Здесь A_pos — верхний 3x6 блок линейной части, D=diag([σ_pos]*3+[σ_vel]*3).
    u_hat — первый правый сингулярный вектор матрицы M = A_pos @ D.
    Его можно масштабировать на требуемый радиус r, чтобы получить старт u0=r*u_hat.
    """
    A_pos = np.array(xf.linear())[:3, :]
    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)
    M = A_pos @ np.diag(d)
    try:
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        v1 = Vh[0, :] if Vh.ndim == 2 and Vh.shape[0] >= 1 else np.zeros(6)
        nrm = float(np.linalg.norm(v1))
        return (v1 / nrm) if nrm > 0 else np.zeros(6)
    except Exception:
        return np.zeros(6)


def get_maxdev_linear_ellipsoid_with_vector(
    xf: DA,
    std_pos: float,
    std_vel: float,
    radius: float,
) -> tuple[float, np.ndarray]:
    """
    Линейная эллипсоидальная оценка как значение + соответствующий 6D-вектор x,
    который достигает этого значения (начальное возмущение).

    x = D * (radius * u_hat), где u_hat — правый сингулярный вектор для
    матрицы M = A_pos @ D с наибольшим сингулярным значением.
    Значение — это ||A_pos x||_2.
    """
    A_pos = np.array(xf.linear())[:3, :]
    d = np.array([std_pos, std_pos, std_pos, std_vel, std_vel, std_vel], dtype=float)
    u_hat = get_maxdev_linear_ellipsoid_argmax_vector(xf, std_pos, std_vel)
    x = d * (float(radius) * u_hat)
    val = float(np.linalg.norm(A_pos @ x))
    return val, x


def main():
    """
    Сравнительный эксперимент:
      1) семплирование с эллипсоидной проекцией (0.999-квантиль),
      2) линейная эллипсоидальная оценка (аналитика),
      3) эллипсоидальная оптимизация (DA) с одним стартом от линейной оценки,
      4) эллипсоидальная оптимизация (интегрирование) с одним стартом от линейной оценки.

    Печатает значения в DU и км для наглядного сравнения.
    """
    # Конфигурация эксперимента
    orbit_type, orbit_num = "L1", 79
    std_pos, std_vel = km2du(3.), kmS2vu(0.03e-3)
    radius = 3.0
    derorder = 3

    # DA‑карта через период (для методов 1–3)
    xf = get_xf(orbit_type, orbit_num, derorder=derorder)

    # Визуализация эллипсоида и аргмакс-векторов для пяти методов
    #  - sampling (ellipsoid)
    #  - linear ellipsoid
    #  - DA optimization (multistart)
    #  - IVP optimization (multistart)
    #  - Floquet (monodromy eigen)

    # 1) 
    dev_sampling_val, vec_sampling = get_maxdev_sampling_ellipsoid_with_vector(
        orbit_type,
        orbit_num,
        xf,
        std_pos,
        std_vel,
        derorder=derorder,
        amount_of_points=100_000,
        radius=radius,
    )

    # Линейная эллипсоидальная оценка с вектором
    dev_linear_val, vec_linear = get_maxdev_linear_ellipsoid_with_vector(
        xf,
        std_pos,
        std_vel,
        radius=radius,
    )

    # DA optimization (multistart) с вектором
    dev_da_val, vec_da = get_maxdev_optimization_ellipsoid_with_vector(
        orbit_type,
        orbit_num,
        xf,
        std_pos,
        std_vel,
        radius=radius,
        verbose=False,
        n_random_starts=10,
        init_strategy='mixed_multistart',
    )

    # IVP optimization (multistart) с вектором
    dev_ivp_val, vec_ivp = get_maxdev_optimization_ellipsoid_integrate_with_vector(
        orbit_type,
        orbit_num,
        xf,
        std_pos,
        std_vel,
        radius=radius,
        verbose=False,
        n_random_starts=10,
        init_strategy='mixed_multistart',
    )

    # Floquet с вектором (v_proj)
    dev_floq_val, vec_floq = get_maxdev_floquet_ellipsoid_with_vector(
        orbit_type,
        orbit_num,
        std_pos,
        std_vel,
        radius=radius,
    )

    print()
    print("=== Comparison: Ellipsoid-Based Deviation Estimates ===")
    print()
    print(f"orbit={orbit_type} #{orbit_num}, derorder={derorder}, radius={radius}")
    print(f"sigmas: pos={du2km(std_pos):.3f} km, vel={vu2ms(std_vel):.6f} m/s")
    print()
    print(f"1) sampling (ellipsoid): {du2km(dev_sampling_val):.6f} km")
    print(f"2) linear ellipsoid (semi-analytic formula): {du2km(dev_linear_val):.6f} km")
    print(f"3) DA optimization (multistart): {du2km(dev_da_val):.6f} km")
    print(f"4) IVP optimization (multistart): {du2km(dev_ivp_val):.6f} km")
    print(f"5) Floquet (monodromy eigen): {du2km(dev_floq_val):.6f} km")
    print()

    vecs_map = {
        'sampling': vec_sampling,
        'linear ellipsoid': vec_linear,
        'DA opt (multistart)': vec_da,
        'IVP opt (multistart)': vec_ivp,
        'Floquet': vec_floq,
    }

    # Векторы в скоростном пространстве (последние 3 компоненты)
    vecs_vel = {name: np.asarray(vec)[3:] for name, vec in vecs_map.items()}

    # Совмещённый рисунок: слева — координатные отклонения, справа — скоростные
    fig = plt.figure(figsize=(12, 6))
    ax_pos = fig.add_subplot(121, projection='3d')
    ax_vel = fig.add_subplot(122, projection='3d')

    plot_ellipsoid_and_vectors_pretty(
        pos_sigma=std_pos,
        r=radius,
        vecs=vecs_map,
        ax=ax_pos,
        show=False,
        legend_kwargs={
            'loc': 'upper center',
            'bbox_to_anchor': (0.5, 1.08),
            'ncol': 2,
            'borderaxespad': 0.2,
        },
    )
    velocity_labels = {
        'sampling': r'$\delta\mathbf{v}_0^*$, sampling',
        'linear ellipsoid': r'$\delta\mathbf{v}_0^*$, linear + SVD',
        'DA opt (multistart)': r'$\delta\mathbf{v}_0^*$, DA-optimization',
        'IVP opt (multistart)': r'$\delta\mathbf{v}_0^*$, IVP-optimization',
        'Floquet': r'$\delta\mathbf{v}_0^*$, Floquet'
    }
    plot_ellipsoid_and_vectors_pretty(
        pos_sigma=std_vel,
        r=radius,
        vecs=vecs_vel,
        ax=ax_vel,
        show=False,
        axis_labels=(r'$\delta v_x$', r'$\delta v_y$', r'$\delta v_z$'),
        units_label="VU",
        legend_labels=velocity_labels,
        legend_kwargs={
            'loc': 'upper center',
            'bbox_to_anchor': (0.5, 1.08),
            'ncol': 2,
            'borderaxespad': 0.2,
        },
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def main_check_predictions(
    orbit_config1: tuple[str, int] = ("L1", 192),
    orbit_config2: tuple[str, int] | None = None,
    std_pos: float = km2du(3.),
    std_vel: float = kmS2vu(0.03e-3),
    radius: float = 3.0,
    derorder: int = 3,
    plot_scale: float | None = None,
    auto_scale_threshold: float = 20.0,
    auto_scale_min_dev: float = 1e-3,
):
    orbit_configs: list[tuple[str, int]] = [orbit_config1]
    if orbit_config2 is not None:
        orbit_configs.append(orbit_config2)

    color_cycle = {
        "central": "gray",
        "sampling (ellipsoid)": "tab:blue",
        "linear ellipsoid (semi-analytic)": "tab:orange",
        "DA optimization (multistart)": "tab:green",
        "IVP optimization (multistart)": "tab:red",
        "Floquet (monodromy eigen)": "tab:purple",
    }
    labels = {
        "central": "Halo-orbit",
        "sampling (ellipsoid)": "Sampling pert. solution",
        "linear ellipsoid (semi-analytic)": "Linear + SVD pert. solution",
        "DA optimization (multistart)": "DA-opimization pert. solution",
        "IVP optimization (multistart)": "IVP-opimization pert. solution",
        "Floquet (monodromy eigen)": "Floquet pert. solution",
    }

    def integrate_traj(y0: np.ndarray, period: float, npts: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        t_span = (0.0, float(period))
        t_eval = np.linspace(t_span[0], t_span[1], npts)
        sol = solve_ivp(
            cr3bp,
            t_span,
            y0,
            t_eval=t_eval,
            rtol=1e-14,
            atol=1e-14,
            method='LSODA',
        )
        if not sol.success:
            print(f"Warning: integration failed (message: {sol.message})")
        traj = sol.y.T
        return traj, traj[-1]

    def compute_orbit_data(orbit_type: str, orbit_num: int) -> dict[str, Any]:
        xf = get_xf(orbit_type, orbit_num, derorder=derorder)

        x0, z0, vy0, T, _, MAX_MUL = initial_state_parser(orbit_type, orbit_num)
        central_ic_for_traj_build = np.array([x0, 0., z0, 0., vy0, 0.], dtype=float)

        dev_sampling_val, vec_sampling = get_maxdev_sampling_ellipsoid_with_vector(
            orbit_type,
            orbit_num,
            xf,
            std_pos,
            std_vel,
            derorder=derorder,
            amount_of_points=100_000,
            radius=radius,
        )

        dev_linear_val, vec_linear = get_maxdev_linear_ellipsoid_with_vector(
            xf,
            std_pos,
            std_vel,
            radius=radius,
        )

        dev_da_val, vec_da = get_maxdev_optimization_ellipsoid_with_vector(
            orbit_type,
            orbit_num,
            xf,
            std_pos,
            std_vel,
            radius=radius,
            verbose=False,
            n_random_starts=10,
            init_strategy='mixed_multistart',
        )

        dev_ivp_val, vec_ivp = get_maxdev_optimization_ellipsoid_integrate_with_vector(
            orbit_type,
            orbit_num,
            xf,
            std_pos,
            std_vel,
            radius=radius,
            verbose=False,
            n_random_starts=10,
            init_strategy='mixed_multistart',
        )

        dev_floq_val, vec_floq = get_maxdev_floquet_ellipsoid_with_vector(
            orbit_type,
            orbit_num,
            std_pos,
            std_vel,
            radius=radius,
        )

        methods: list[tuple[str, np.ndarray, float]] = [
            ("sampling (ellipsoid)", vec_sampling, dev_sampling_val),
            ("linear ellipsoid (semi-analytic)", vec_linear, dev_linear_val),
            ("DA optimization (multistart)", vec_da, dev_da_val),
            ("IVP optimization (multistart)", vec_ivp, dev_ivp_val),
            ("Floquet (monodromy eigen)", vec_floq, dev_floq_val),
        ]

        results: dict[str, dict[str, Any]] = {}
        max_dev_seen = 0.0

        traj_central, last_central = integrate_traj(central_ic_for_traj_build, T)
        results["central"] = {
            "traj": traj_central,
            "last": last_central,
            "final_diff_pos": last_central[:3] - central_ic_for_traj_build[:3],
            "final_diff_norm_du": float(np.linalg.norm(last_central[:3] - central_ic_for_traj_build[:3])),
        }

        for name, vec6, dev_pred_du in methods:
            y0 = central_ic_for_traj_build + np.asarray(vec6, dtype=float)
            traj, last = integrate_traj(y0, T)
            diff_pos = last[:3] - central_ic_for_traj_build[:3]
            diff_norm_du = float(np.linalg.norm(diff_pos))
            results[name] = {
                "traj": traj,
                "last": last,
                "vec": vec6,
                "pred_dev_du": float(dev_pred_du),
                "final_diff_pos": diff_pos,
                "final_diff_norm_du": diff_norm_du,
            }
            max_dev_seen = max(max_dev_seen, diff_norm_du)

        return {
            "orbit_type": orbit_type,
            "orbit_num": orbit_num,
            "max_mul": float(MAX_MUL),
            "period": T,
            "central_ic": central_ic_for_traj_build,
            "results": results,
            "method_names": [name for name, _, _ in methods],
            "max_dev_seen": max_dev_seen,
        }

    orbit_data_list: list[dict[str, Any]] = [compute_orbit_data(*cfg) for cfg in orbit_configs]

    fig = plt.figure(figsize=(8 * len(orbit_data_list), 6))
    axes: list[Axes3D] = []
    if len(orbit_data_list) == 1:
        axes = [fig.add_subplot(111, projection='3d')]
    else:
        for idx in range(len(orbit_data_list)):
            axes.append(fig.add_subplot(1, len(orbit_data_list), idx + 1, projection='3d'))

    flag = 0
    for ax, orbit_data in zip(axes, orbit_data_list):
        central_ic_for_traj_build = np.asarray(orbit_data["central_ic"], dtype=float)
        results = orbit_data["results"]

        tr = results["central"]["traj"]
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=color_cycle["central"], label=labels.get("central", "central"), linewidth=1.5, linestyle='--')
        ax.scatter(
            central_ic_for_traj_build[0],
            central_ic_for_traj_build[1],
            central_ic_for_traj_build[2],
            color='k',
            s=30,
            marker='o',
            zorder=5,
        )

        max_mul_val = float(orbit_data.get("max_mul", 0.0))
        scale_for_plot = plot_scale if plot_scale is not None else 1.0
        # Автомасштаб только для устойчивых орбит (малый MAX_MUL)
        if (
            plot_scale is None
            and max_mul_val < auto_scale_threshold
        ):
            scale_for_plot = 140.

        for name in orbit_data["method_names"]:
            if scale_for_plot != 1.0:
                # Масштабируем отклонение траектории относительно центральной, чтобы показать малые отличия.
                trm_raw = results[name]["traj"]
                delta = trm_raw - tr  # разница относительно центральной траектории
                trm = tr + delta * scale_for_plot
            else:
                trm = results[name]["traj"]
            ax.plot(trm[:, 0], trm[:, 1], trm[:, 2], color=color_cycle.get(name, None), label=labels.get(name, name), linewidth=1.2)

        ax.set_xlabel(r'$x$ [DU]')
        ax.set_ylabel(r'$y$ [DU]')
        ax.set_zlabel(r'$z$ [DU]')
        if flag == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.02), ncol=1, framealpha=0.9, fontsize=9)

        flag += 1
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()

    for orbit_data in orbit_data_list:
        print()
        print("=== Final-position differences after 1 period (validation) ===")
        print(f"orbit={orbit_data['orbit_type']} #{orbit_data['orbit_num']}, derorder={derorder}, radius={radius}, MAX_MUL={orbit_data['max_mul']}")
        print(f"sigmas: pos={du2km(std_pos):.3f} km, vel={vu2ms(std_vel):.6f} m/s")
        print()
        cen_du = orbit_data["results"]["central"]["final_diff_norm_du"]
        print(f"central: diff={du2km(float(cen_du)):.6f} km (should be about 0)")

        for name in orbit_data["method_names"]:
            diff_du = orbit_data["results"][name]["final_diff_norm_du"]
            pred_du = orbit_data["results"][name]["pred_dev_du"]
            print(f"{name}: predicted={du2km(float(pred_du)):.6f} km, integrated={du2km(float(diff_du)):.6f} km")


if __name__ == "__main__":
    main_check_predictions(orbit_config1=("L2", 21), orbit_config2=("L2", 345))
