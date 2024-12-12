import math
import numpy as np
import matplotlib.pyplot as plt

# Пример функции
# Новая функция
def f(x):
    return x ** 2 - 2 * x - 2 * math.cos(x)

# Первая производная функции
def f_prime(x):
    return 2 * x - 2 + 2 * math.sin(x)

# Вторая производная функции
def f_double_prime(x):
    return 2 + 2 * math.cos(x)

# Метод нулевого порядка: Метод бисекции с дельтой
def bisection_method_with_delta(a, b, epsilon=1e-6, delta=1e-7):
    iteration = 0
    while abs(f_prime((a + b) / 2)) > epsilon:
        iteration += 1
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        if f(x1) > f(x2):
            a = x1
        else:
            b = x2
    xmin = (a + b) / 2
    return xmin, f(xmin), iteration

# Метод касательных (метод секущих)
def secant_method(f_prime, x0, x1, epsilon=1e-6, max_iter=100):
    iteration = 0
    for _ in range(max_iter):
        iteration += 1
        f_prime_x0 = f_prime(x0)
        f_prime_x1 = f_prime(x1)
        if abs(f_prime_x1 - f_prime_x0) < epsilon:
            return x2, f(x2), iteration
        x2 = x1 - f_prime_x1 * (x1 - x0) / (f_prime_x1 - f_prime_x0)
        if abs(x2 - x1) < epsilon:
            return x2, f(x2), iteration
        x0, x1 = x1, x2
    return x1, f(x1), iteration

# Метод второго порядка: Метод Ньютона
def newton_method(f_prime, f_double_prime, start, epsilon=1e-6, max_iter=100):
    x = start
    iteration = 0
    for _ in range(max_iter):
        iteration += 1
        f_prime_val = f_prime(x)
        f_double_prime_val = f_double_prime(x)
        if abs(f_double_prime_val) < epsilon:
            return x, f(x), iteration
        new_x = x - f_prime_val / f_double_prime_val
        if abs(new_x - x) < epsilon:
            return x, f(x), iteration
        x = new_x
    return x, f(x), iteration

# Пример использования методов

# Метод бисекции с дельтой
a = 0.5
b = 1
xmin_bisect, fmin_bisect, iter_bisect = bisection_method_with_delta(a, b)
print(f"Метод бисекции: минимум достигается при x = {xmin_bisect:.7f}, f(x) = {fmin_bisect:.7f}, число итераций: {iter_bisect}")

# Метод касательных (метод секущих)
x0_secant = 0.5
x1_secant = 1
xmin_secant, fmin_secant, iter_secant = secant_method(f_prime, x0_secant, x1_secant)
print(f"Метод секущих: минимум достигается при x = {xmin_secant:.7f}, f(x) = {fmin_secant:.7f}, число итераций: {iter_secant}")

# Метод Ньютона
x0_newton = 0.5
xmin_newton, fmin_newton, iter_newton = newton_method(f_prime, f_double_prime, x0_newton)
print(f"Метод Ньютона: минимум достигается при x = {xmin_newton:.7f}, f(x) = {fmin_newton:.7f}, число итераций: {iter_newton}")

# Построение графика функции и отображение точек минимума
x_values = np.linspace(-100, 100, 1000)
y_values = [f(x) for x in x_values]

plt.plot(x_values, y_values, label='f(x)')
plt.scatter([xmin_bisect, xmin_secant, xmin_newton],
            [fmin_bisect, fmin_secant, fmin_newton],
            color='red', zorder=5, label='Минимумы')

plt.title('График функции и точки минимума')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()