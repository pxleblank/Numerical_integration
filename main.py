import math
import numpy as np
from scipy.special import roots_legendre
from scipy import integrate
import decimal
import fractions


def f(x):
    return x ** 2 + (2 / (1 - np.cos(1))) * np.sin(x)


a = 0
b = 1
N = 5
N2 = N * 2
n_gauss2u = 2  # Количество узлов для метода Гаусса
n_gauss3u = 3  # Количество узлов для метода Гаусса
n_gauss2u2 = n_gauss2u * 2
n_gauss3u2 = n_gauss3u * 2

main_integral = float(7 / 3)


# Метод левых прямоугольников
def left_rectangles(N):
    global a, b
    h = (b - a) / N
    s = 0
    for i in range(0, N):
        s += f(a + i * h) * h
    return s


# Метод центральных прямоугольников
def mid_rectangles(N):
    global a, b
    h = (b - a) / N
    s = 0
    for i in range(0, N):
        s += f(a + i * h + h / 2) * h
    return s


# Метод трапеций
def trapezoidal(N):
    global a, b
    h = (b - a) / N
    s = f(a) + f(b)
    for i in range(1, N):
        s += 2 * f(a + i * h)
    s = s * (h / 2)
    return s


# Метод Симпсона
def simpson(N):
    global a, b
    h = (b - a) / N
    s = 0

    for i in range(0, N):
        x_i = a + i * h  # Левый узел
        x_ip1 = a + (i + 1) * h  # Правый узел
        x_mid = (x_i + x_ip1) / 2  # Центральная точка между x_i и x_{i+1}

        s += (f(x_i) + 4 * f(x_mid) + f(x_ip1))

    s = s * (h / 6)
    return s


# Метод Гаусса (с использованием n узлов)
def gauss(n, N):
    global a, b
    h = (b - a) / N
    result = 0.
    result_ = 0.

    # Цикл по каждому подотрезку
    for j in range(0, N):
        # Определяем границы подотрезка
        a_j = a + j * h
        b_j = a + (j + 1) * h

        # Половина и середина подотрезка
        half = float(b_j - a_j) / 2.
        mid = (a_j + b_j) / 2.

        x = [np.cos(np.pi * (4 * i + 3) / (4 * n + 2)) for i in range(n)]

        # Цикл по каждому узлу
        for i in range(0, n):
            # Вычисление узлов и весов для текущего узла
            xi = x[i]  # Начальное приближение для узла

            # Метод Ньютона для уточнения узла
            for _ in range(100):
                Pn = legendre_polynomial(n, xi)  # Полином Лежандра P_n(xi)
                Pn_prime = legendre_derivative(n, xi)  # Производная полинома P'_n(xi)

                dx = Pn / Pn_prime  # Смещение узла по методу Ньютона
                xi = xi - dx  # Обновляем значение узла

                # Проверяем сходимость
                if abs(dx) < 1e-7:
                    break

            # Вычисляем вес для текущего узла
            wi = 2 / ((1 - xi ** 2) * (legendre_derivative(n, xi) ** 2))

            X = half * xi + mid  # Преобразование узла

            # Обновляем вклад в интеграл для текущего узла
            result_ += wi * f(X)  # Преобразуем узел на интервал [a, b]

        result_ *= half  # Масштабируем результат на длину интервала
        result += result_
        result_ = 0.

    return result


def gauss2(N):
    global a, b
    h = (b - a) / N
    result = 0.
    # на отрезке [-1, 1]
    x1 = -1 / np.sqrt(3)
    x2 = 1 / np.sqrt(3)
    a1 = 1
    a2 = 1

    for j in range(0, N):
        # Определяем границы подотрезка
        a_j = a + j * h
        b_j = a_j + h

        # Половина и середина подотрезка
        half = float(b_j - a_j) / 2.

        x1m = half * (x1 + 1) + a_j
        x2m = half * (x2 + 1) + a_j

        a1m = half * a1
        a2m = half * a2

        result += a1m * f(x1m) + a2m * f(x2m)

    return result


def gauss3(N):
    global a, b
    h = (b - a) / N
    result = 0.
    # на отрезке [-1, 1]
    x1 = -np.sqrt(0.6)
    x2 = 0.
    x3 = np.sqrt(0.6)
    a1 = float(5/9)
    a2 = float(8/9)
    a3 = float(5/9)

    for j in range(0, N):
        # Определяем границы подотрезка
        a_j = a + j * h
        b_j = a_j + h

        # Половина и середина подотрезка
        half = float(b_j - a_j) / 2.

        x1m = half * (x1 + 1) + a_j
        x2m = half * (x2 + 1) + a_j
        x3m = half * (x3 + 1) + a_j

        a1m = half * a1
        a2m = half * a2
        a3m = half * a3

        result += a1m * f(x1m) + a2m * f(x2m) + a3m * f(x3m)

    return result


def legendre_polynomial(n, x):
    """Вычисляет значение полинома Лежандра P_n(x) для данного n и x"""
    if n == 0:
        return 1  # P_0(x) = 1
    elif n == 1:
        return x  # P_1(x) = x
    else:
        P0 = 1  # P_0(x)
        P1 = x  # P_1(x)
        for i in range(2, n + 1):
            P2 = ((2 * i - 1) * x * P1 - (i - 1) * P0) / i  # Трехчленная рекурсия
            P0 = P1  # Обновляем предыдущие значения
            P1 = P2
        return P1  # Возвращаем P_n(x)


def legendre_derivative(n, x):
    """Вычисляет производную полинома Лежандра P'_n(x)"""
    Pn = legendre_polynomial(n, x)  # P_n(x)
    Pn_1 = legendre_polynomial(n - 1, x)  # P_{n-1}(x)
    return n * (Pn_1 - x * Pn) / (1 - x ** 2)


integral_left = left_rectangles(N)
integral_mid = mid_rectangles(N)
integral_trap = trapezoidal(N)
integral_simp = simpson(N)
integral_gauss2 = gauss2(N)
integral_gauss3 = gauss3(N)

integral_left2 = left_rectangles(N2)
integral_mid2 = mid_rectangles(N2)
integral_trap2 = trapezoidal(N2)
integral_simp2 = simpson(N2)
integral_gauss2u2 = gauss2(N2)
integral_gauss3u2 = gauss3(N2)


def improve_integral(In, I2n, p):
    alpha = -1. / (2 ** p - 1)
    result = alpha * In + (1 - alpha) * I2n
    return result


integral_left_improve = improve_integral(integral_left, integral_left2, 1)
integral_mid_improve = improve_integral(integral_mid, integral_mid2, 4)
integral_trap_improve = improve_integral(integral_trap, integral_trap2, 4)
integral_simp_improve = improve_integral(integral_simp, integral_simp2, 4)
integral_gauss2_improve = improve_integral(integral_gauss2, integral_gauss2u2, 4)
integral_gauss3_improve = improve_integral(integral_gauss3, integral_gauss3u2, 6)

print(f'Значение интеграла: {main_integral:.1f}')
print(' ' * 28 + 'Погр. N:' + ' ' * 11 + 'Погр. 2N:' + ' ' * 22 + 'Отношение:' + ' ' * 10 + 'Уточнение(погр.):')
print(
    f'Левые прямоугольники:       {abs(main_integral - integral_left):.10f}  |    {abs(main_integral - integral_left2):.10f}  | {abs(main_integral - integral_left2):.5e} |     {(abs(main_integral - integral_left) / abs(main_integral - integral_left2)):7.2f}       |   {abs(main_integral - integral_left_improve)}\n'
    f'Центральные прямоугольники: {abs(main_integral - integral_mid):.10f}  |    {abs(main_integral - integral_mid2):.10f}  | {abs(main_integral - integral_mid2):.5e} |     {(abs(main_integral - integral_mid) / abs(main_integral - integral_mid2)):7.2f}       |   {abs(main_integral - integral_mid_improve)}\n'
    f'Трапеции:                   {abs(main_integral - integral_trap):.10f}  |    {abs(main_integral - integral_trap2):.10f}  | {abs(main_integral - integral_trap2):.5e} |     {(abs(main_integral - integral_trap) / abs(main_integral - integral_trap2)):7.2f}       |   {abs(main_integral - integral_trap_improve)}\n'
    f'Метод Симпсона:             {abs(main_integral - integral_simp):.10f}  |    {abs(main_integral - integral_simp2):.10f}  | {abs(main_integral - integral_simp2):.5e} |     {(abs(main_integral - integral_simp) / abs(main_integral - integral_simp2)):7.2f}       |   {abs(main_integral - integral_simp_improve)}\n'
    f'Метод Гаусса (2):           {abs(main_integral - integral_gauss2):.10f}  |    {abs(main_integral - integral_gauss2u2):.10f}  | {abs(main_integral - integral_gauss2u2):.5e} |     {(abs(main_integral - integral_gauss2) / abs(main_integral - integral_gauss2u2)):7.2f}       |   {abs(main_integral - integral_gauss2_improve)}\n'
    f'Метод Гаусса (3):           {abs(main_integral - integral_gauss3):.10f}  |    {abs(main_integral - integral_gauss3u2):.10f}  | {abs(main_integral - integral_gauss3u2):.5e} |     {(abs(main_integral - integral_gauss3) / abs(main_integral - integral_gauss3u2)):7.2f}       |   {abs(main_integral - integral_gauss3_improve)}\n')
