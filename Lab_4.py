import numpy as np
import matplotlib.pyplot as plt
import statistics

def generate_test_sequence(N, A, n, phi):
    x = np.linspace(0, np.pi/4.5, N)
    y = A * np.sin(n * x + phi) + n
    noise = np.random.uniform(-0.05 * A, 0.05 * A, size=N)
    y_noisy = y + noise
    return x, y, y_noisy

def arithmetic_mean(sequence):
    return np.mean(sequence)

def harmonic_mean(sequence):
    return len(sequence) / sum(1 / x for x in sequence)

def geometric_mean(sequence):
    if np.any(sequence < 0):
        return np.nan
    return np.prod(sequence) ** (1 / len(sequence))


def exact_value(x, A, n, phi):
    return A * np.sin(n * x + phi) + n

def compare_approximate_with_exact(approximate, exact):
    absolute_error = np.abs(approximate - exact)
    relative_error = np.zeros_like(absolute_error)
    mask = (exact != 0)
    relative_error[mask] = absolute_error[mask] / exact[mask]
    return absolute_error, relative_error


N = 40000
A = 2
n = 4
phi = 0


x, y, y_noisy = generate_test_sequence(N, A, n, phi)
arithmetic = np.mean(y_noisy)
geometric = statistics.geometric_mean(y_noisy)
harmonic = statistics.harmonic_mean(y_noisy)

print(f"Арифметичне середнє: {arithmetic}")
print(f"Гармонійне середнє: {harmonic}")
print(f"Геометричне середнє: {geometric}")


exact = exact_value(x, A, n, phi)
arithmetic_absolute_error, arithmetic_relative_error = compare_approximate_with_exact(arithmetic, exact)
harmonic_absolute_error, harmonic_relative_error = compare_approximate_with_exact(harmonic, exact)
geometric_absolute_error, geometric_relative_error = compare_approximate_with_exact(geometric, exact)

print(f"Максимальна абсолютна похибка (арифметичне середнє): {np.max(arithmetic_absolute_error)}")
print(f"Мінімальна абсолютна похибка (арифметичне середнє): {np.min(arithmetic_absolute_error)}")
print(f"Максимальна відносна похибка (арифметичне середнє): {np.max(arithmetic_relative_error)}")
print(f"Мінімальна відносна похибка (арифметичне середнє): {np.min(arithmetic_relative_error)}")
print()
print(f"Максимальна абсолютна похибка (гармонійне середнє): {np.max(harmonic_absolute_error)}")
print(f"Мінімальна абсолютна похибка (гармонійне середнє): {np.min(harmonic_absolute_error)}")
print(f"Максимальна відносна похибка (гармонійне середнє): {np.max(harmonic_relative_error)}")
print(f"Мінімальна відносна похибка (гармонійне середнє): {np.min(harmonic_relative_error)}")
print()
print(f"Максимальна абсолютна похибка (геометричне середнє): {np.max(geometric_absolute_error)}")
print(f"Мінімальна абсолютна похибка (геометричне середнє): {np.min(geometric_absolute_error)}")
print(f"Максимальна відносна похибка (геометричне середнє): {np.max(geometric_relative_error)}")
print(f"Мінімальна відносна похибка (геометричне середнє): {np.min(geometric_relative_error)}")

plt.plot(x, y)
plt.plot(x, y_noisy)
plt.legend(['Точна послідовність', 'Згенерована послідовність'], loc='lower center', ncol=2)
plt.show()


plt.plot(x, y)
plt.plot(x, y_noisy)
plt.plot([x[0], x[-1]], [arithmetic, arithmetic], label='Арифметичне середнє')
plt.plot([x[0], x[-1]], [harmonic, harmonic], label='Гармонійне середнє')
plt.plot([x[0], x[-1]], [geometric, geometric], label='Геометричне середнє')
plt.legend(['Точна послідовність', 'Згенерована послідовність', 'Арифметичне середнє', 'Гармонійне середнє', 'Геометричне середнє'], loc='lower center', ncol=2)
plt.show()