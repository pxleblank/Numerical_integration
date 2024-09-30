import math
import numpy as np
from scipy.special import roots_legendre
from scipy import integrate


def f(x):
    return 1 / (x * math.sqrt(x ** 2 - 1))


a = 2
b = np.pi
N = 10
n_gauss = 3  # Количество узлов для метода Гаусса

main_integral = 0.1997


# Метод левых прямоугольников
def left_rectangles(N):
    global a, b
    h = (b - a) / N
    return sum(f(a + i * h) for i in range(N)) * h


# Метод центральных прямоугольников
def mid_rectangles(N):
    global a, b
    h = (b - a) / N
    return sum(f(a + i * h + h / 2) for i in range(N)) * h


# Метод трапеций
def trapezoidal(N):
    global a, b
    h = (b - a) / N
    return (f(a) + 2 * sum(f(a + i * h) for i in range(1, N)) + f(b)) * h / 2


# Метод Симпсона
def simpson(N):
    global a, b
    if N % 2 == 1:  # Число N должно быть четным для метода Симпсона
        N += 1
    h = (b - a) / N
    return (f(a) + 4 * sum(f(a + i * h) for i in range(1, N, 2)) +
            2 * sum(f(a + i * h) for i in range(2, N, 2)) + f(b)) * h / 3


# Метод Гаусса (с использованием n узлов)
def gauss(n):
    global a, b
    [nodes, weights] = roots_legendre_(n)  # Узлы и веса для квадратур Гаусса
    result = 0
    for i in range(n):
        x_i = 0.5 * (b - a) * nodes[i] + 0.5 * (b + a)
        result += weights[i] * f(x_i)
    return 0.5 * (b - a) * result


# Функция для нахождения корней полинома Лежандра и весов
def roots_legendre_(n):
    # Начальные приближения для корней - используем корни полинома Лежандра
    beta = np.arange(1, n) / np.sqrt(4 * np.arange(1, n) ** 2 - 1)  # Коэффициенты Лежандра
    T = np.diag(beta, 1) + np.diag(beta, -1)

    # Собственные значения этой матрицы дадут корни полинома Лежандра
    x, v = np.linalg.eigh(T)
    nodes = x  # Корни

    # Вычисляем веса
    weights = 2 * v[0, :] ** 2

    return nodes, weights


integral_left = left_rectangles(N)
integral_mid = mid_rectangles(N)
integral_trap = trapezoidal(N)
integral_simp = simpson(N)
integral_gauss = gauss(n_gauss)

print(f'Значение интеграла: {main_integral:.4f}')
print(f'Левые прямоугольники: {integral_left:.4f}\nЦентральные прямоугольники: {integral_mid:.4f}\n'
      f'Трапеции: {integral_trap:.4f}\nМетод Симпсона: {integral_simp:.4f}\nМетод Гаусса: {integral_gauss:.4f}\n')

print(' ' * 22 + 'Погрешность:')
print(
    f'Левые прямоугольники: {abs(main_integral - integral_left):.4f}\nЦентральные прямоугольники: {abs(main_integral - integral_mid):.4f}\n'
    f'Трапеции: {abs(main_integral - integral_trap):.4f}\nМетод Симпсона: {abs(main_integral - integral_simp):.4f}\nМетод Гаусса: {abs(main_integral - integral_gauss):.4f}\n')
