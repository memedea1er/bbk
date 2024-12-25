import numpy as np

def simplex_method(c, A, b):
    """
    Реализация симплекс-метода для решения задачи линейного программирования.

    max F = c^T * x
    A * x <= b, x >= 0

    :param c: Коэффициенты целевой функции (вектор).
    :param A: Матрица ограничений (левая часть).
    :param b: Вектор ограничений (правая часть).
    :return: Оптимальное решение x и значение целевой функции.
    """
    m, n = A.shape

    # Добавляем искусственные переменные для преобразования в равенства
    A_eq = np.hstack([A, np.eye(m)])
    c_eq = np.hstack([c, np.zeros(m)])

    # Инициализация базиса
    basis = list(range(n, n + m))

    # Симплекс-таблица
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :-1] = A_eq
    tableau[:m, -1] = b
    tableau[-1, :-1] = -c_eq

    while True:
        # Проверка на оптимальность
        if all(tableau[-1, :-1] >= 0):
            break

        # Выбор входящей переменной (по правилу минимального значения в последней строке)
        pivot_col = np.argmin(tableau[-1, :-1])

        # Проверка на неограниченность
        if all(tableau[:-1, pivot_col] <= 0):
            raise ValueError("Задача не имеет ограниченного решения.")

        # Выбор выходящей переменной (по правилу минимального отношения)
        ratios = []
        for i in range(m):
            if tableau[i, pivot_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
        pivot_row = np.argmin(ratios)

        # Обновление базиса
        basis[pivot_row] = pivot_col

        # Приведение ведущего элемента к 1
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]

        # Обнуление остальных элементов в ведущем столбце
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Извлечение результата
    x = np.zeros(n + m)
    x[basis] = tableau[:-1, -1]
    return x[:n], tableau[-1, -1]

# Коэффициенты целевой функции
c = np.array([7, -2])

# Коэффициенты ограничений
A = np.array([
    [5, -2],
    [-1, -1],
    [-3, 1],
    [2, 1]
])

# Правая часть ограничений
b = np.array([3, -1, 3, 4])

# Решение задачи
try:
    solution, max_value = simplex_method(c, A, b)
    print(f"x1 = {solution[0]:.6f}")
    print(f"x2 = {solution[1]:.6f}")
    print(f"Максимальное значение F = {max_value:.6f}")
except ValueError as e:
    print(e)
