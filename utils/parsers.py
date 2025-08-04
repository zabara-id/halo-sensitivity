import numpy as np
import os
import csv


ORBIT_TYPES_NUMS = {'L1' : 251,
                    'L2' : 583}


def initial_state_parser(orbit_type: str, number_of_orbit: int = 0) -> np.ndarray:
    """Parses the initial state for a specific orbit from a CSV file.

    This function retrieves a tuple of six numbers representing the initial 
    state for a specific orbit, based on the provided orbit type ('L1' or 'L2') 
    and orbit number. The data is loaded from a corresponding CSV file located in 
    the 'data' directory.

    Args:
        orbit_type (str): 'L1' or 'L2'
        number_of_orbit (int, optional): Orbit number. 1-251 for 'L1' 1-583 for 'L1'.  Defaults to 0.

    Raises:
        ValueError: If the orbit type is not 'L1' or 'L2'.
        ValueError: If the orbit number is outside the valid range for the selected orbit type.

    Returns:
        np.ndarray: A numpy array containing six floating-point numbers representing
        the initial state vector of the specified orbit.
    """
    if orbit_type not in ORBIT_TYPES_NUMS:
        raise ValueError("Incorrect orbit type. You need either 'L1' or 'L2'.")
    
    if (number_of_orbit < 1) or number_of_orbit > ORBIT_TYPES_NUMS[orbit_type]:
        raise ValueError(f"Incorrect orbit number. The range (1, {ORBIT_TYPES_NUMS[orbit_type]}) is available for {orbit_type}.")
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'HOPhaseVectorsEarthMoon{orbit_type}.csv')
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        
        selected_row = rows[number_of_orbit - 1]
        
        selected_row = [float(num) for num in selected_row]
    
    return np.array(selected_row)

def main():
    initial_state = initial_state_parser('L1', 200)
    print(initial_state)


if __name__ == "__main__":
    main()