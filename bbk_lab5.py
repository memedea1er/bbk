def convert_to_standard_form(c, A, b, inequality_signs):
    M = -1e6  # Огромный коэффициент для искусственных переменных
    num_constraints = len(A)

    # Преобразуем ограничения
    for i in range(num_constraints):
        if inequality_signs[i] == '>=':  # Для ограничений вида >= добавляем искусственную переменную
            artificial_var = [0] * num_constraints
            artificial_var[i] = -1  # Искусственная переменная с коэффициентом -1
            A[i].extend(artificial_var)  # Добавляем искусственную переменную в матрицу A_eq
            artificial_var_index = i
        else:
            artificial_var = [0] * num_constraints
            artificial_var[i] = 1  # Искусственная переменная
            A[i].extend(artificial_var)  # Добавляем искусственную переменную в матрицу A_eq

    for i in range(num_constraints):
        if i == artificial_var_index:
            A[i].extend([1])
        else:
            A[i].extend([0])

    # Обновляем целевую функцию
    c.extend([0] * num_constraints)  # Добавляем нули для новых переменных (искусственных и базисных)
    c.append(M)  # Добавляем большой коэффициент для искусственной переменной

    return c, A, b


def simplex_method_artificial_basis(c, A, b):

    # Инициализация переменных
    num_vars = len(c)
    num_constraints = len(b)

    # Создаем начальную таблицу симплекс-метода
    tableau = []
    for i in range(num_constraints):
        tableau.append(A[i] + [b[i]])

    tableau.append(c + [0])

    # Функция для вывода текущей таблицы
    def print_tableau():
        print("Текущая симплекс-таблица:")
        for row in tableau:
            print(row)
        print()

    # Итерации симплекс-метода
    while True:
        print_tableau()

        # Поиск ведущего столбца (наибольший положительный коэффициент в строке цели)
        pivot_col = max(range(num_vars), key=lambda j: tableau[-1][j])
        if tableau[-1][pivot_col] <= 0:
            break  # Оптимальное решение найдено

        # Поиск ведущей строки
        ratios = []
        for i in range(num_constraints):
            if tableau[i][pivot_col] > 0:
                ratios.append(tableau[i][-1] / tableau[i][pivot_col])
            else:
                ratios.append(float('inf'))

        pivot_row = min(range(num_constraints), key=lambda i: ratios[i])
        if ratios[pivot_row] == float('inf'):
            print("Решение не ограничено.")
            return

        # Приведение ведущего элемента к 1
        pivot_element = tableau[pivot_row][pivot_col]
        tableau[pivot_row] = [x / pivot_element for x in tableau[pivot_row]]

        # Приведение остальных строк
        for i in range(len(tableau)):
            if i != pivot_row:
                factor = tableau[i][pivot_col]
                tableau[i] = [tableau[i][j] - factor * tableau[pivot_row][j] for j in range(len(tableau[i]))]

    # Вывод результата
    print("Оптимальное решение найдено:")
    print_tableau()
    solution = [0] * num_vars
    for i in range(num_constraints):
        for j in range(num_vars):
            if tableau[i][j] == 1 and all(tableau[k][j] == 0 for k in range(len(tableau)) if k != i):
                solution[j] = tableau[i][-1]
                break

    return solution, tableau[-1][-1]


# Исходные данные
c = [7, -2]
A = [
    [5, -2],
    [1, 1],
    [-3, 1],
    [2, 1]
]
b = [3, 1, 3, 4]
inequality_signs = ['<=', '>=', '<=', '<=']  # Знаки неравенств

# Решение задачи
try:
    c, A_eq, b_eq = convert_to_standard_form(c, A, b, inequality_signs)
    solution, optimal_value = simplex_method_artificial_basis(c, A, b)
    print("Оптимальное решение найдено:")
    print(f"Значения переменных: {solution}")
    print(f"Оптимальное значение целевой функции: {-optimal_value}")
except ValueError as e:
    print(str(e))