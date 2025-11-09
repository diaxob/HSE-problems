import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

phi_deg = np.array([
    0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0,
    33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0, 54.0, 57.0, 60.0
])
P = np.array([
    146.5, 140.0, 147.6, 142.0, 139.5, 128.5, 139.7, 132.5, 142.2, 136.0,
    120.0, 83.8, 88.0, 28.0, 13.0, 40.6, 87.2, 98.5, 103.0, 96.5, 94.0
])

baseline_mask = (phi_deg < 30) | (phi_deg > 51)
phi_baseline = phi_deg[baseline_mask]
P_baseline = P[baseline_mask]

poly_fit = Polynomial.fit(phi_baseline, P_baseline, deg=3)
baseline = poly_fit(phi_deg)

P_corrected = P - baseline

def lorentzian(theta, I0, theta0, gamma):
    return I0 / (1 + ((theta - theta0)/gamma)**2)

initial_guess = [-140, 42, 3]

popt, pcov = curve_fit(lorentzian, phi_deg, P_corrected, p0=initial_guess)

I0, theta0, gamma = popt
dI0, dtheta0, dgamma = np.sqrt(np.diag(pcov))

font_size = 16
title_size = 18
legend_size = 15

plt.figure(figsize=(10, 5))
plt.plot(phi_deg, P, 'bo', label='Экспериментальные точки')
plt.plot(phi_deg, baseline, 'r--', label='Аппроксимация полиномом 3-й степени')
plt.xlabel(r'Угол $\theta$ (градусы)', fontsize=font_size)
plt.ylabel('Интенсивность (мкА)', fontsize=font_size)
plt.title('Двухэтапная аппроксимация', fontsize=title_size)
plt.legend(fontsize=legend_size)
plt.grid(True)
plt.tick_params(axis='both', labelsize=font_size)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(phi_deg, P_corrected, 'go', label='Очищенные данные (без фона)')
plt.plot(phi_deg, lorentzian(phi_deg, *popt), 'k-', label='Аппроксимация функцией Лоренца')
plt.xlabel(r'Угол $\theta$ (градусы)', fontsize=font_size)
plt.ylabel('Интенсивность (мкА)', fontsize=font_size)
plt.title('Двухэтапная аппроксимация', fontsize=title_size)
plt.legend(fontsize=legend_size)
plt.grid(True)
plt.tick_params(axis='both', labelsize=font_size)
plt.tight_layout()
plt.show()
