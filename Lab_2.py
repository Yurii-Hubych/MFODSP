import numpy as np
import matplotlib.pyplot as plt
import time

def fourier_coefficient(f, k):
    N = len(f)
    n = np.arange(N)
    Ak = np.sum(f * np.cos(2 * np.pi * k * n / N))
    Bk = np.sum(f * np.sin(2 * np.pi * k * n / N))
    Ck = Ak - 1j * Bk
    num_multiplications = 8 * N + 1
    num_additions = 2 * (N - 1)
    print("С_" + str(k) + " = " + str(Ck))
    return Ck, num_multiplications, num_additions

def fourier_coefficients(f):
    N = len(f)
    C = np.zeros(N, dtype=complex)
    total_multiplications = 0
    total_additions = 0
    for k in range(N):
        C[k], num_multiplications, num_additions = fourier_coefficient(f, k)
        total_multiplications += num_multiplications
        total_additions += num_additions
    return C, total_multiplications, total_additions

N = 50
f = np.random.rand(N)

start_time = time.time()
C, total_multiplications, total_additions = fourier_coefficients(f)
end_time = time.time()

print("Час обчислення: ", end_time - start_time)
print("Кількість операцій множення: ", total_multiplications)
print("Кількість операцій додавання: ", total_additions)
print("Загальна к-сть операцій: ", total_additions + total_multiplications)

amplitude_spectrum = np.abs(C)
phase_spectrum = np.angle(C)

plt.plot
plt.stem(amplitude_spectrum)
plt.title("Спектр амплітуд")
plt.show()

plt.plot
plt.stem(phase_spectrum)
plt.title("Спектр фаз")
plt.show()