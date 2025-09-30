"""
Скрипт уточнения НУ для построения замкнутых гало-орбит.

@since: 30.09.2025
@version: 1.0
"""

import os
import csv

import numpy as np
from tqdm import tqdm

from utils.libration_sense import halo_qualify, initial_state_parser, ORBIT_TYPES_NUMS


def qualify_orbittype_ic(orbit_type: str) -> None:
    rows_refined = []
    for i in tqdm(
        np.arange(1, ORBIT_TYPES_NUMS[orbit_type] + 1),
        desc=f"{orbit_type} processing:"
        ):
        x0, z0, vy0, period, jacobi, max_multiplier = initial_state_parser(orbit_type, i)
        refined_state, refined_period = halo_qualify(orbit_type, i)

        x0_refined = refined_state[0]
        z0_refined = refined_state[2]
        vy0_refined = refined_state[4]
        period_refined = refined_period
        jacobi_refined = jacobi
        max_multiplier_refined = max_multiplier

        refined_row = [x0_refined, z0_refined, vy0_refined, period_refined, jacobi_refined, max_multiplier_refined]
        rows_refined.append(refined_row)
        del refined_row

    result_filepath = f"data/output/qualified_ic_halo/HOPhaseVectorsEarthMoon{orbit_type}_qualified.csv"
    os.makedirs(os.path.dirname(result_filepath), exist_ok=True)
    with open(result_filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_refined)


def main() -> None:
    qualify_orbittype_ic("L1")
    qualify_orbittype_ic("L2")


if __name__ == "__main__":
    main()
