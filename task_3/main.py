import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# 1. ПАРАМЕТРЫ ЗАДАЧИ (ТАБЛИЦА 1)

m1 = 1.0
m2 = 1.0

L1 = 1.0
L2 = 1.0

g = 9.81

t0 = 0.0
t1 = 20.0

N = 1000

rtol = 1e-8
atol = 1e-8

delta = 1e-6


# Начальные условия A
# y = (theta1, omega1, theta2, omega2)

y0_A = np.array([
    np.pi / 2,
    0.0,
    np.pi / 2 + 0.01,
    0.0
])

# Начальные условия B

y0_B = np.array([
    np.pi / 2,
    0.0,
    np.pi / 2 + 0.01 + delta,
    0.0
])

# 2. ПРАВАЯ ЧАСТЬ СИСТЕМЫ ОДУ

def right_hand_side(t, y):
    """
    Правая часть системы дифференциальных уравнений
    двойного маятника по формуле (5.5).
    
    Вектор состояния:
    y = (theta1, omega1, theta2, omega2)
    """

    theta1, omega1, theta2, omega2 = y

    # Разность углов
    dtheta = theta1 - theta2

    sin_dtheta = np.sin(dtheta)
    cos_dtheta = np.cos(dtheta)

    # Общий знаменатель
    denominator = (
        2 * m1 + m2
        - m2 * np.cos(2 * dtheta)
    )

    # Угловое ускорение первого маятника
    omega1_dot = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        -m2 * g * np.sin(theta1 - 2 * theta2)
        -2 * sin_dtheta * m2 * (
            omega2**2 * L2
            + omega1**2 * L1 * cos_dtheta
        )
    ) / (L1 * denominator)

    # Угловое ускорение второго маятника
    omega2_dot = (
        2 * sin_dtheta * (
            omega1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * L2 * m2 * cos_dtheta
        )
    ) / (L2 * denominator)

    return np.array([
        omega1,
        omega1_dot,
        omega2,
        omega2_dot
    ])

# 3. ЧИСЛЕННОЕ РЕШЕНИЕ СИСТЕМЫ ОДУ

def solve_motion(y0, rtol=1e-8, atol=1e-8):
    """
    Решает задачу Коши методом RK45.
    """

    # 1000 равноотстоящих точек времени
    t_eval = np.linspace(t0, t1, N)

    solution = solve_ivp(
        right_hand_side,
        (t0, t1),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    if solution.y.shape[1] != N:
        raise RuntimeError(
            "Получено неверное количество точек решения."
        )

    return solution

# 4. ПЕРЕВОД В ДЕКАРТОВЫ КООРДИНАТЫ

def to_cartesian(state):
    """
    Переводит углы маятника в декартовы координаты
    по формулам (5.1).
    """

    theta1 = state[0]
    theta2 = state[2]

    # Координаты первого груза
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    # Координаты второго груза
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2

# 5. ПОЛНАЯ МЕХАНИЧЕСКАЯ ЭНЕРГИЯ

def total_energy(state):
    """
    Вычисляет полную механическую энергию E = T + V.
    Формулы (5.2) и (5.3).
    """

    theta1 = state[0]
    omega1 = state[1]

    theta2 = state[2]
    omega2 = state[3]

    # Кинетическая энергия
    T = (
        0.5 * (m1 + m2) * L1**2 * omega1**2
        + 0.5 * m2 * L2**2 * omega2**2
        + m2 * L1 * L2 * omega1 * omega2
        * np.cos(theta1 - theta2)
    )

    # Потенциальная энергия
    V = (
        -(m1 + m2) * g * L1 * np.cos(theta1)
        -m2 * g * L2 * np.cos(theta2)
    )

    return T + V

# 6. ГРАФИК ТРАЕКТОРИИ НИЖНЕГО ГРУЗА

def plot_trajectory(solution):
    """
    Строит траекторию нижнего груза (x2, y2).
    """

    x1, y1, x2, y2 = to_cartesian(solution.y)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        x2,
        y2,
        linewidth=1.2,
        label="Траектория нижнего груза"
    )

    ax.set_title(
        "Траектория нижнего груза двойного маятника"
    )

    ax.set_xlabel("x2, м")
    ax.set_ylabel("y2, м")

    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

# 7. АНИМАЦИЯ ДВОЙНОГО МАЯТНИКА

def animate_motion(solution):
    """
    Создаёт анимацию движения двойного маятника.

    На каждом кадре показываются:
    - точка крепления O;
    - первый стержень;
    - второй стержень;
    - след нижнего груза.
    """

    x1, y1, x2, y2 = to_cartesian(solution.y)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Стержни и грузы
    rods, = ax.plot(
        [],
        [],
        "o-",
        linewidth=2,
        markersize=7
    )

    # След нижнего груза
    trail, = ax.plot(
        [],
        [],
        linewidth=1,
        alpha=0.7
    )

    limit = L1 + L2 + 0.2

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")

    ax.set_title("Анимация двойного маятника")

    ax.grid(True)

    # Начальное состояние анимации
    def init():
        rods.set_data([], [])
        trail.set_data([], [])

        return rods, trail

    # Обновление кадра
    def update(frame):

        # Положение точки O и двух грузов
        rods.set_data(
            [0, x1[frame], x2[frame]],
            [0, y1[frame], y2[frame]]
        )

        # След нижнего груза
        trail.set_data(
            x2[:frame + 1],
            y2[:frame + 1]
        )

        return rods, trail

    animation = FuncAnimation(
        fig,
        update,
        frames=len(solution.t),
        init_func=init,
        interval=20,
        blit=True,
        repeat=True
    )

    plt.tight_layout()

    return animation

# 8. СРАВНЕНИЕ НАЧАЛЬНЫХ УСЛОВИЙ A И B


def plot_sensitivity(solution_A, solution_B):
    """
    Строит две траектории нижнего груза
    для почти одинаковых начальных условий.
    """

    _, _, x2_A, y2_A = to_cartesian(solution_A.y)
    _, _, x2_B, y2_B = to_cartesian(solution_B.y)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True
    )

    # Случай A
    axes[0].plot(x2_A, y2_A)

    axes[0].set_title(
        f"Случай A: theta2(0) = {y0_A[2]:.8f}"
    )

    # Случай B
    axes[1].plot(x2_B, y2_B)

    axes[1].set_title(
        f"Случай B: theta2(0) = {y0_B[2]:.8f}"
    )

    for ax in axes:
        ax.set_xlabel("x2, м")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

    axes[0].set_ylabel("y2, м")

    fig.suptitle(
        "Чувствительность к начальным условиям"
    )

    plt.tight_layout()



# 9. ПРОВЕРКА СОХРАНЕНИЯ ЭНЕРГИИ

def plot_energy_error(solution_1e8, solution_1e5):
    """
    Вычисляет и строит относительное изменение энергии.
    Сравнение точностей 1e-8 и 1e-5.
    """

    # Вычисляем энергию
    energy_1e8 = total_energy(solution_1e8.y)
    energy_1e5 = total_energy(solution_1e5.y)

    # Начальная энергия
    E0_1e8 = energy_1e8[0]
    E0_1e5 = energy_1e5[0]

    print("\nПроверка энергии:")
    print(f"E(0), точность 1e-8: {E0_1e8:.10f}")
    print(f"E(0), точность 1e-5: {E0_1e5:.10f}")

    # Проверка начальной энергии
    if abs(E0_1e8) < 1e-12:
        raise ValueError("Начальная энергия близка к нулю.")

    if abs(E0_1e5) < 1e-12:
        raise ValueError("Начальная энергия близка к нулю.")

    # Относительное изменение энергии
    error_1e8 = (
        np.abs(energy_1e8 - E0_1e8) / abs(E0_1e8)
    )

    error_1e5 = (
        np.abs(energy_1e5 - E0_1e5) / abs(E0_1e5)
    )

    # Максимальные погрешности
    print("Максимальная погрешность при 1e-8:",
          np.max(error_1e8))

    print("Максимальная погрешность при 1e-5:",
          np.max(error_1e5))

    # Построение графика
    fig, ax = plt.subplots(figsize=(9, 5))

    # Первую точку t = 0 не отображаем,
    # так как логарифмическая шкала не отображает ноль.

    ax.semilogy(
        solution_1e8.t[1:],
        error_1e8[1:],
        label="rtol = atol = 1e-8"
    )

    ax.semilogy(
        solution_1e5.t[1:],
        error_1e5[1:],
        label="rtol = atol = 1e-5"
    )

    ax.set_xlabel("Время t, с")
    ax.set_ylabel("|E(t) - E(0)| / |E(0)|")

    ax.set_title("Относительное изменение полной энергии")

    ax.grid(True, which="both")
    ax.legend()

    plt.tight_layout()



# 10. ГЛАВНАЯ ФУНКЦИЯ

def main():

    print("Домашнее задание № 3")
    print("Двойной маятник")
    print()


    # ЗАДАНИЕ 1. ЧИСЛЕННОЕ РЕШЕНИЕ И АНИМАЦИЯ
    
    print("Задание 1. Численное решение...")

    # Решение с начальными условиями A
    solution_A = solve_motion(
        y0_A,
        rtol=1e-8,
        atol=1e-8
    )

    # Статичный график траектории
    plot_trajectory(solution_A)

    # Анимация движения
    animation = animate_motion(solution_A)

    print("Задание 1 выполнено.")

    
    # ЗАДАНИЕ 2. ПРОВЕРКА НЕУСТОЙЧИВОСТИ


    print("Задание 2. Проверка неустойчивости...")

    # Решение с начальными условиями B
    solution_B = solve_motion(
        y0_B,
        rtol=1e-8,
        atol=1e-8
    )

    # Сравнение двух траекторий
    plot_sensitivity(
        solution_A,
        solution_B
    )

    print("Проверка сохранения энергии...")

    # Решение случая A с точностью 1e-5
    solution_A_1e5 = solve_motion(
        y0_A,
        rtol=1e-5,
        atol=1e-5
    )

    # Сравнение изменения энергии
    plot_energy_error(
        solution_A,
        solution_A_1e5
    )

    print("\nВсе расчёты завершены.")

    # Сохраняем ссылку на анимацию
    animation

    # Показ всех графиков и анимации
    plt.show()


if __name__ == "__main__":
    main()

