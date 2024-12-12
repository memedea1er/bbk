import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Функция для минимизации
def function(x1, x2):
    return np.exp(x1 ** 2) + (x1 + x2) ** 2


# Градиент функции
def gradient(x1, x2):
    df_dx1 = 2 * x1 * np.exp(x1 ** 2) + 2 * (x1 + x2)
    df_dx2 = 2 * (x1 + x2)
    return np.array([df_dx1, df_dx2])


# Метод Хука-Дживса
def hooke_jeeves(f, x0, step_size=0.5, tol=1e-6, max_iter=1000):
    x = np.array(x0)
    iters = 0
    points = [x.copy()]  # Для сохранения траектории

    while iters < max_iter:
        iters += 1
        # Поиск по образцу
        new_x = exploratory_search(f, x, step_size)

        # Проверка, улучшилась ли функция
        if f(*new_x) < f(*x):
            x = new_x
            step_size *= 1.2  # Увеличиваем шаг
        else:
            step_size *= 0.5  # Уменьшаем шаг

        points.append(x.copy())

        # Условие остановки
        if step_size < tol:
            break

    return x, f(*x), iters, points

# анализ некоторой окрестности базисной точки, и из этой окрестности определяется направление,
# вдоль которого значение искомой функции убывает – этот этап называется «исследующим поиском».

# перемещение вдоль выбранного направления к новой базисной точке – «поиск по образцу».

# Функция для поиска по образцу в методе Хука-Дживса
def exploratory_search(f, x, step_size):
    new_x = np.array(x)
    for i in range(len(x)):
        for delta in [step_size, -step_size]:
            test_x = new_x.copy()
            test_x[i] += delta
            if f(*test_x) < f(*new_x):
                new_x = test_x
                break
    return new_x


# Метод градиентного спуска
def gradient_descent(f, grad, x0, step_size=0.5, tol=1e-6, max_iter=1000):
    x = np.array(x0)
    iters = 0
    points = [x.copy()]  # Для сохранения траектории

    while iters < max_iter:
        iters += 1
        grad_value = grad(*x)
        new_x = x - step_size * grad_value

        # Условие остановки
        if np.linalg.norm(new_x - x) < tol:
            break

        # Проверка, улучшилась ли функция
        if f(*new_x) < f(*x):
            x = new_x
            step_size *= 1.1  # Увеличиваем шаг
        else:
            step_size *= 0.5  # Уменьшаем шаг

        points.append(x.copy())

    return x, f(*x), iters, points


# Визуализация функции и траекторий обоих методов
def plot_function_and_paths(f, paths, method_names):
    x1_vals = np.linspace(-0.5, 1.25, 100)
    x2_vals = np.linspace(-0.5, 1.25, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(X1, X2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

    # Траектории оптимизации для каждого метода
    for path, method_name, color in zip(paths, method_names, ['red', 'blue']):
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], f(path[:, 0], path[:, 1]), color=color, marker='o', label=method_name)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Пути оптимизации методами Хука-Дживса и градиентного спуска')
    ax.legend()
    plt.show()


# Основная часть программы
x0 = [1.0, 1.0]  # Начальное приближение

# Метод Хука-Дживса
result_hooke_jeeves, f_min_hooke_jeeves, iters_hooke_jeeves, points_hooke_jeeves = hooke_jeeves(function, x0)
print(
    f"Метод Хука-Дживса: Минимум в точке {result_hooke_jeeves} со значением {f_min_hooke_jeeves}, количество итераций: {iters_hooke_jeeves}")

# Метод градиентного спуска
result_gradient, f_min_gradient, iters_gradient, points_gradient = gradient_descent(function, gradient, x0)
print(
    f"Градиентный спуск: Минимум в точке {result_gradient} со значением {f_min_gradient}, количество итераций: {iters_gradient}")

# Отрисовка графика с обеими траекториями
plot_function_and_paths(function, [points_hooke_jeeves, points_gradient], ["Hooke-Jeeves", "Gradient Descent"])
