import os
import csv

import numpy as np
from daceypy import DA, array
from typing import Callable
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


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
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', f'HOPhaseVectorsEarthMoon{orbit_type}.csv')
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        
        selected_row = rows[number_of_orbit]
        
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


def get_maxdeviation_wo_integrate(
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

    # # Клиппинг
    # limits = np.array([3 * std_dev_positions] * 3 + [3 * std_dev_velocities] * 3)
    # deltax0 = np.clip(deltax0, -limits, limits)

    x0_cons = initial_state.cons()  # координаты центральной точки (начальные условия)
    x0_coords = x0_cons[:3]  # только положения

    x0_new = x0_cons + deltax0  # отклонения от начального положения

    # Применение формулы Тейлора
    evaluated_results = np.array([xf.eval(delta)[:3] for delta in deltax0]) # изменённые конечные положения объектов для возмущённых начальных условий

    # вычисление Евклидова расстояния от каждой точки в evaluated_results до x0_coords
    distances = np.sqrt(np.sum((evaluated_results - x0_coords) ** 2, axis=1))

    return np.max(distances)


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


# Примеры
def main1():
    num = 192
    std_pos = km2du(8)  # от 0 до 8 км
    std_vel = kmS2vu(0.05e-3)  # от 0 до 0.05 м/с
    print(f"Max deviation orbit {num}: ", du2km(get_maxdeviation("L1", num, std_pos, std_vel, number_of_points=100_000)))

def main2():
    M = get_monodromy_matrix('L1', 192)


if __name__ == "__main__":
    main1()
