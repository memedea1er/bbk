from scipy.optimize import linprog

# Коэффициенты целевой функции (нужно минимизировать -F, т.к. SciPy решает задачи минимизации)
c = [-7, 2]  # Для максимизации F = 7x1 - 2x2

# Коэффициенты ограничений
A = [
    [5, -2],  # 5x1 - 2x2 <= 3
    [-1, -1], # -x1 - x2 <= -1 (эквивалентно x1 + x2 >= 1)
    [-3, 1],  # -3x1 + x2 <= 3
    [2, 1]    # 2x1 + x2 <= 4
]

# Правая часть ограничений
b = [3, -1, 3, 4]

# Границы переменных
x_bounds = (0, None)  # x1 >= 0
y_bounds = (0, None)  # x2 >= 0

# Решение задачи
result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='highs')

# Вывод результата
if result.success:
    print("Оптимальное решение найдено!")
    print(f"x1 = {result.x[0]:.6f}")
    print(f"x2 = {result.x[1]:.6f}")
    print(f"Максимальное значение F = {7 * result.x[0] - 2 * result.x[1]:.6f}")
else:
    print("Решение не найдено.")
