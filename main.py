import math
import numpy as np
from scipy.special import roots_legendre
from scipy import integrate


def f(x):
    return 12 * x ** 5 - 10 * x ** 3 + 3 * x ** 2


a = 0
b = 1
N = 20
N2 = N * 2
n_gauss2u = 2  # Количество узлов для метода Гаусса
n_gauss3u = 3  # Количество узлов для метода Гаусса
n_gauss2u2 = n_gauss2u * 2
n_gauss3u2 = n_gauss3u * 2

main_integral = 0.5


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
def gauss(n):
    global a, b
    half = float(b - a) / 2.
    mid = (a + b) / 2.
    [x, w] = gauss_legendre_nodes_weights(n)  # Узлы и веса для квадратур Гаусса
    result = 0.
    for i in range(n):
        result += w[i] * f(half * x[i] + mid)
    result *= half
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


def gauss_legendre_nodes_weights(n, tol=1e-14):
    """Вычисляет узлы и веса для квадратур Гаусса-Лежандра"""

    # Начальное приближение для узлов (равномерное распределение по отрезку [-1, 1])
    x = [np.cos(np.pi * (4 * i - 1) / (4 * n + 2)) for i in range(1, n + 1)]

    # Метод Ньютона для уточнения корней
    for _ in range(100):  # Ограничим количество итераций 100
        Pn = [legendre_polynomial(n, xi) for xi in x]  # Значения P_n(x) в точках x
        Pn_prime = [legendre_derivative(n, xi) for xi in x]  # Значения P'_n(x)

        # Обновляем значения x методом Ньютона
        dx = [Pn[i] / Pn_prime[i] for i in range(n)]
        x = [x[i] - dx[i] for i in range(n)]

        # Проверяем, если сходимость достигнута
        if max(abs(dx_i) for dx_i in dx) < tol:
            break

    # Вычисление весов
    w = [2 / ((1 - xi ** 2) * (legendre_derivative(n, xi) ** 2)) for xi in x]

    return x, w


integral_left = left_rectangles(N)
integral_mid = mid_rectangles(N)
integral_trap = trapezoidal(N)
integral_simp = simpson(N)
integral_gauss = gauss(n_gauss2u)
integral_gauss2 = gauss(n_gauss3u)

integral_left2 = left_rectangles(N2)
integral_mid2 = mid_rectangles(N2)
integral_trap2 = trapezoidal(N2)
integral_simp2 = simpson(N2)
integral_gauss2 = gauss(n_gauss2u2)
integral_gauss23 = gauss(n_gauss3u2)


print(f'Значение интеграла: {main_integral:.1f}')
print(' ' * 28 + 'Погр. N:' + ' ' * 11 + 'Погр. 2N:' + ' ' * 10 + 'Отношение:')
print(
    f'Левые прямоугольники:       {abs(main_integral - integral_left):.10f}  |    {abs(main_integral - integral_left2):.10f}  |     {(abs(main_integral - integral_left) / abs(main_integral - integral_left2)):7.2f}\n'
    f'Центральные прямоугольники: {abs(main_integral - integral_mid):.10f}  |    {abs(main_integral - integral_mid2):.10f}  |     {(abs(main_integral - integral_mid) / abs(main_integral - integral_mid2)):7.2f}\n'
    f'Трапеции:                   {abs(main_integral - integral_trap):.10f}  |    {abs(main_integral - integral_trap2):.10f}  |     {(abs(main_integral - integral_trap) / abs(main_integral - integral_trap2)):7.2f}\n'
    f'Метод Симпсона:             {abs(main_integral - integral_simp):.10f}  |    {abs(main_integral - integral_simp2):.10f}  |     {(abs(main_integral - integral_simp) / abs(main_integral - integral_simp2)):7.2f}\n'
    f'Метод Гаусса (2):           {abs(main_integral - integral_gauss):.10f}  |    {abs(main_integral - integral_gauss2):.10f}  |     {(abs(main_integral - integral_gauss) / abs(main_integral - integral_gauss2)):7.2f}\n'
    f'Метод Гаусса (3):           {abs(main_integral - integral_gauss2):.10f}  |    {abs(main_integral - integral_gauss23):.10f}  |     {(abs(main_integral - integral_gauss2) / abs(main_integral - integral_gauss23)):7.2f}\n')
