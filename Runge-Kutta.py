import numpy as np
from scipy.integrate import trapezoid

#константы задачи
n_exo, T_exo = 100, 1000
k_B, m = 1.4 * 10**(-23), 2
c_exo = ((2*k_B*T_exo)/m)**(1/2)
#распределение Максвелла
def f_m(V):
    return n_exo/(c_exo * (np.pi)**0.5) * np.exp(-V**2 / (c_exo)**2)

MU_EARTH = 3.986004418e14
R_EARTH  = 6_371_000.0     

def R_exo(h_exo_m: float = 500_000.0) -> float:
    """
    Радиус экзобазы R_exo = R_earth + h_exo.
    По умолчанию высота экзобазы 500 км (можешь поправить под свою постановку).
    Возвращает значение в тех же единицах, в каких задан R_EARTH и h_exo_m (SI).
    """
    return R_EARTH + h_exo_m


def v_escape(r: float, mu: float = MU_EARTH) -> float:
    """Вторая космическая скорость на радиусе r: v_esc = sqrt(2*mu/r)."""
    return np.sqrt(2.0 * mu / r)

def v_exo(r: float, v_tau: float, v_r: float, dt: float = 10.0,
          h_exo_m: float = 500_000.0, mu: float = MU_EARTH) -> float:
    """
    Возвращает модуль скорости частицы на экзобазе V, если её баллистическая траектория
    пересекает экзобазу. Иначе — np.nan.

    Формула: V^2 = v^2 + v_esc^2(R_exo) - v_esc^2(r).
    Проверки: условие пересечения экзобазы (см. (19)) и v < v_esc(r) (исключаем внешние прилёты).
    """
    Rexo = R_exo(h_exo_m)

    v2        = v_r*v_r + v_tau*v_tau
    vesc_Rexo = v_escape(Rexo, mu)
    vesc_r    = v_escape(r, mu)

    V2 = v2 + vesc_Rexo*vesc_Rexo - vesc_r*vesc_r
    # численная защита от нуля
    if V2 <= 0:
        return 0.0
    return np.sqrt(V2)

#  Постоянный коэффициент для функции распределения в фазовом пространстве
F_AT_R_FACTOR = 4 * np.round(np.pi**2, 3)

def f_at_r(r, v_tau, v_r, dt=10):
    """
    Вычисляет значение функции распределения в фазовом пространстве
    для точки с координатами [r, v_tau, v_r].

    Параметры:
    ----------
    r : float
        Радиус (координата в пространстве).
    v_tau : float
        Тангенциальная компонента скорости.
    v_r : float
        Радиальная компонента скорости.
    dt : float, optional
        Временной шаг (по умолчанию 10).

    Возвращает:
    -----------
    float : значение функции распределения.
    """
    return F_AT_R_FACTOR * np.abs(v_tau) * r**2 * f_m(v_exo(r, v_tau, v_r, dt=dt))

# Векторизованная версия функции для работы с массивами
f_at_r_vectorized = np.vectorize(f_at_r, signature='(), (), () -> ()')


def gen_data(n_r_points, resolution_tau, v_max, v_min, r_max_factor=2):
    """
    Генерирует значения функции распределения в сетке по радиусу и скоростям.

    Параметры:
    ----------
    n_r_points : int
        Количество точек по радиусу.
    resolution_tau : int
        Количество точек по тангенциальной скорости.
    v_max : float
        Максимальное значение радиальной скорости.
    v_min : float
        Минимальное значение радиальной скорости.
    r_max_factor : float, optional
        Множитель для максимального радиуса (по умолчанию 2).

    Возвращает:
    -----------
    tuple :
        - data : ndarray
            Значения функции распределения на сетке.
        - r : ndarray
            Сетка по радиусу.
        - v_grid : list[ndarray]
            Сетка по скоростям (v_r, v_tau).
    """
    r = np.linspace(R_exo, r_max_factor * R_exo, n_r_points)
    v_tau = np.linspace(-v_min, v_min, resolution_tau)
    v_r = np.linspace(-v_min, v_max, resolution_tau * (v_max + v_min) // (2 * v_min))

    v_grid = np.meshgrid(v_r, v_tau, sparse=True)

    data = f_at_r_vectorized(
        r=r[:, None, None],
        v_tau=v_tau[None, :, None],
        v_r=v_r[None, None, :]
    )
    return data, r, v_grid


def concentration(dataset, r, v_grid):
    """
    Вычисляет концентрацию (интеграл от функции распределения по скоростям)
    для разных радиусов.

    Параметры:
    ----------
    dataset : ndarray
        Значения функции распределения для всех r и скоростей.
    r : ndarray
        Сетка по радиусу.
    v_grid : list[ndarray]
        Сетка по скоростям (v_r, v_tau).

    Возвращает:
    -----------
    ndarray : концентрации для каждого значения r.
    """

    concentrations = []
    for dataset_at_r in dataset:
        # 1. Интегрируем по радиальной скорости (ось 0).
        fst_int = trapezoid(dataset_at_r, v_grid[0].reshape(-1), axis=1)

        # 2. Интегрируем по тангенциальной скорости (ось 1).
        snd_int = trapezoid(fst_int, v_grid[1].reshape(-1), axis=0)

        concentrations.append(snd_int)

    # Нормируем результат
    return np.array(concentrations) / (4 * np.pi * r**2)

