def bisection(f, a, b, eps):
    """Решает уравнение f(x) = 0 методом бисекции."""

    if eps <= 0:
        raise ValueError("eps должна быть положительной")

    if a >= b:
        raise ValueError("a должно быть меньше b")

    fa = f(a)
    fb = f(b)

    # Проверяем, не является ли конец отрезка корнем
    if fa == 0:
        return a

    if fb == 0:
        return b

    # Проверяем наличие корня на концах отрезка
    if fa * fb >= 0:
        raise ValueError(
            "f(a) и f(b) должны иметь разные знаки"
        )

    while True:
        # Делим отрезок пополам
        c = a + (b - a) / 2
        fc = f(c)

        # Проверяем условия остановки
        if fc == 0 or abs(fc) < eps or b - a < eps:
            return c

        # Выбираем половину отрезка со сменой знака
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc


def f(x):
    # Тестовое уравнение: x^3 - x - 2 = 0
    return x**3 - x - 2


def main():
    a = 1.0
    b = 2.0
    eps = 1e-6

    root = bisection(f, a, b, eps)

    print("Метод бисекции")
    print(f"Интервал: [{a}, {b}]")
    print(f"Точность: {eps}")
    print(f"Найденный корень: {root:.10f}")
    print(f"Значение функции в корне: {f(root):.10e}")


if __name__ == "__main__":
    main()