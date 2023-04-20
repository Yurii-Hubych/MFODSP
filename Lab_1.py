import numpy as np
import matplotlib.pyplot as plt

# Define the function to be approximated
def f(x):
    return x**4 * np.exp(-x**2 / 4)

# Define the range of x values
x = np.linspace(-np.pi, np.pi, 1000)
max_err = 0
plot_arr = []
error_arr = []
# Define the number of terms to use in the Fourier series
for n_terms in range(20):
    # Calculate the Fourier series coefficients
    a0 = np.sum(f(x)) / len(x)
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)

    for i in range(1, n_terms+1):
        an[i-1] = (2/len(x)) * np.sum(f(x) * np.cos(i * np.pi * x / max(x)))
        bn[i-1] = (2/len(x)) * np.sum(f(x) * np.sin(i * np.pi * x / max(x)))



    # Define the Fourier series approximation

    fourier_series = np.zeros(len(x))
    # Calculate the relative error at each point
    error = np.abs((f(x) - (fourier_series + a0)) / f(x))
    avg_error = np.mean(np.abs((f(x) - (fourier_series + a0)) / np.max(np.abs(f(x)))))

    fourier_series = np.zeros(len(x))
    for i in range(n_terms):
        fourier_series += an[i] * np.cos((i+1) * np.pi * x / max(x)) + bn[i] * np.sin((i+1) * np.pi * x / max(x))
    plot_arr.append(fourier_series)
    plt.figure(figsize=(10, 8))

# Print Fourier coefficients
print((a0 + sum(fourier_series)))
print("2. a0 = {:.4f}".format(a0))
for i in range(n_terms):
    print("a{} = {:.4f}, b{} = {:.4f}".format(i+1, an[i], i+1, bn[i]))
# Print the average relative error
print("\n5. Середня відносна похибка отриманого наближення = {:.4f}".format(avg_error))

plt.plot(x, f(x), label='Початкова функція')
for fourier_series in plot_arr:
    plt.plot(x, fourier_series + a0)
plt.legend()
plt.show()

plt.figure()
plt.plot(x, error)
plt.title(f'Error for n_terms={n_terms}')
plt.xlabel('x')
plt.ylabel('Error')
plt.show()

# Calculate the Fourier coefficients magnitude for k=0,1,...,N
N = 10
ak = np.zeros(N + 1)
bk = np.zeros(N + 1)
for k in range(N + 1):
    if k == 0:
        ak[k] = a0
        bk[k] = 0
    else:
        ak[k] = 2 / len(x) * np.sum(f(x) * np.cos(k * np.pi * x / max(x)))
        bk[k] = 2 / len(x) * np.sum(f(x) * np.sin(k * np.pi * x / max(x)))

# Plot the original function and its Fourier approximation


plt.plot
plt.stem(range(N + 1), np.sqrt(ak ** 2 + bk ** 2), label='Магнітуда')
plt.stem(range(N + 1), ak, markerfmt='C1o', linefmt='C1--', label='Дійсна частина')
plt.stem(range(N + 1), bk, markerfmt='C2o', linefmt='C2--', label='Уявна частина')
plt.legend()
plt.show()

# Write data to file
with open("data.txt", "w", encoding='utf-8') as file:
    # Write the number of terms to use in the Fourier series
    file.write("Порядок N: {}\n".format(n_terms))

    # Write all the Fourier coefficients
    file.write("\na0 = {:.4f}\n".format(a0))
    for i in range(n_terms):
        file.write("a{} = {:.4f}, b{} = {:.4f}\n".format(i + 1, an[i], i + 1, bn[i]))

    # Write the average relative error
    file.write("\nПохибка наближення: {:.4f}\n".format(avg_error))





x = np.linspace(-3*np.pi, -np.pi, 1000)
max_err = 0
plot_arr = []
error_arr = []
# Define the number of terms to use in the Fourier series
for n_terms in range(20):
    # Calculate the Fourier series coefficients
    a0 = np.sum(f(x)) / len(x)
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)

    for i in range(1, n_terms+1):
        an[i-1] = (2/len(x)) * np.sum(f(x) * np.cos(i * np.pi * x / max(x)))
        bn[i-1] = (2/len(x)) * np.sum(f(x) * np.sin(i * np.pi * x / max(x)))



    # Define the Fourier series approximation

    fourier_series = np.zeros(len(x))
    # Calculate the relative error at each point
    error = np.abs((f(x) - (fourier_series + a0)) / f(x))
    avg_error = np.mean(np.abs((f(x) - (fourier_series + a0)) / np.max(np.abs(f(x)))))

    fourier_series = np.zeros(len(x))
    for i in range(n_terms):
        fourier_series += an[i] * np.cos((i+1) * np.pi * x / max(x)) + bn[i] * np.sin((i+1) * np.pi * x / max(x))
    plot_arr.append(fourier_series)
    plt.figure(figsize=(10, 8))

# Print Fourier coefficients
print((a0 + sum(fourier_series)))
print("2. a0 = {:.4f}".format(a0))
for i in range(n_terms):
    print("a{} = {:.4f}, b{} = {:.4f}".format(i+1, an[i], i+1, bn[i]))
# Print the average relative error
print("\n5. Середня відносна похибка отриманого наближення = {:.4f}".format(avg_error))

plt.plot(x, f(x), label='Початкова функція')
for fourier_series in plot_arr:
    plt.plot(x, fourier_series + a0)
plt.legend()
plt.show()

x = np.linspace(np.pi, 3*np.pi, 1000)
max_err = 0
plot_arr = []
error_arr = []
# Define the number of terms to use in the Fourier series
for n_terms in range(20):
    # Calculate the Fourier series coefficients
    a0 = np.sum(f(x)) / len(x)
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)

    for i in range(1, n_terms+1):
        an[i-1] = (2/len(x)) * np.sum(f(x) * np.cos(i * np.pi * x / max(x)))
        bn[i-1] = (2/len(x)) * np.sum(f(x) * np.sin(i * np.pi * x / max(x)))



    # Define the Fourier series approximation

    fourier_series = np.zeros(len(x))
    # Calculate the relative error at each point
    error = np.abs((f(x) - (fourier_series + a0)) / f(x))
    avg_error = np.mean(np.abs((f(x) - (fourier_series + a0)) / np.max(np.abs(f(x)))))

    fourier_series = np.zeros(len(x))
    for i in range(n_terms):
        fourier_series += an[i] * np.cos((i+1) * np.pi * x / max(x)) + bn[i] * np.sin((i+1) * np.pi * x / max(x))
    plot_arr.append(fourier_series)
    plt.figure(figsize=(10, 8))

# Print Fourier coefficients
print((a0 + sum(fourier_series)))
print("2. a0 = {:.4f}".format(a0))
for i in range(n_terms):
    print("a{} = {:.4f}, b{} = {:.4f}".format(i+1, an[i], i+1, bn[i]))
# Print the average relative error
print("\n5. Середня відносна похибка отриманого наближення = {:.4f}".format(avg_error))

plt.plot(x, f(x), label='Початкова функція')
for fourier_series in plot_arr:
    plt.plot(x, fourier_series + a0)
plt.legend()
plt.show()




x = np.linspace(-3*np.pi, 3*np.pi, 1000)
x = x[(x < np.pi) | (x > np.pi)]
max_err = 0
plot_arr = []
error_arr = []
# Define the number of terms to use in the Fourier series
for n_terms in range(20):
    # Calculate the Fourier series coefficients
    a0 = np.sum(f(x)) / len(x)
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)

    for i in range(1, n_terms+1):
        an[i-1] = (2/len(x)) * np.sum(f(x) * np.cos(i * np.pi * x / max(x)))
        bn[i-1] = (2/len(x)) * np.sum(f(x) * np.sin(i * np.pi * x / max(x)))



    # Define the Fourier series approximation

    fourier_series = np.zeros(len(x))
    # Calculate the relative error at each point
    error = np.abs((f(x) - (fourier_series + a0)) / f(x))
    avg_error = np.mean(np.abs((f(x) - (fourier_series + a0)) / np.max(np.abs(f(x)))))

    fourier_series = np.zeros(len(x))
    for i in range(n_terms):
        fourier_series += an[i] * np.cos((i+1) * np.pi * x / max(x)) + bn[i] * np.sin((i+1) * np.pi * x / max(x))
    plot_arr.append(fourier_series)
    plt.figure(figsize=(10, 8))

# Print Fourier coefficients
print((a0 + sum(fourier_series)))
print("2. a0 = {:.4f}".format(a0))
for i in range(n_terms):
    print("a{} = {:.4f}, b{} = {:.4f}".format(i+1, an[i], i+1, bn[i]))
# Print the average relative error
print("\n5. Середня відносна похибка отриманого наближення = {:.4f}".format(avg_error))

plt.plot(x, f(x), label='Початкова функція')
for fourier_series in plot_arr:
    plt.plot(x, fourier_series + a0)
plt.legend()
plt.show()
