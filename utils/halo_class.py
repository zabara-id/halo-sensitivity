from libration_sense import get_xf, km2du, kmS2vu, du2km, get_maxdev_sampling_no_integrate
from formula_creators import n_finder, alpha_finder_of_n


class HaloOrbitSensitivityProcessor:

    STD_POS_DEFAULT: float = 1                              # Стандартное отклонение по положению по умолчанию [км]
    STD_VEL_DEFAULT: float = 0.01e-3                        # Стандартное отклонение по скорости по умолчанию [см / с]
    STD_POS_DEFAULT_DU: float = km2du(STD_POS_DEFAULT)      # Стандартное отклонение по положению по умолчанию [DistUnits]
    STD_VEL_DEFAULT_VU: float = kmS2vu(STD_VEL_DEFAULT)     # Стандартное отклонение по скорости по умолчанию [VelocityUnits]

    GRID_DENSITY_DEFAULT: int = 15                          # Квадратный корень из числа вычисленных максимальных изохронных отклонений через виток [-]
    NOISE_REUSABILITY_DEFAULT: bool = True                  # Флаг переиспользования генерируемых отклонения от начального положения [-]
    AMOUNT_OF_POOINTS_DEFAULT: int = 15000                  # Количество генерируемых отклонений от начальных условий [-]


    def __init__(self, type: str, num: int) -> None:
        """
        Args:
            type (str):  Точка либрации ("L1" или "L2").
            num (int): Номер гало-орбиты из каталога.
        """
        self._type = type
        self._num = num

    @property
    def type(self):
        return self._type
    
    @property
    def num(self):
        return self._num
    
