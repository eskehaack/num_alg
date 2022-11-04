import numpy as np


def comp_trap(f, a, b, n):
    x = np.linspace(a, b, n + 1, endpoint=True)
    h = (b - a) / n
    edge_vals = 0.5 * (f(x[0]) + f(x[-1]))
    return h * (edge_vals + sum([f(x_i) for x_i in x[1:-1]]))


def MySimpson(f, a, b, n):
    if n % 2 == 1:
        print(f"ERROR: n={n} is not even")
        return float("NaN")
    h = (b - a) / n
    nodes = np.linspace(a, b, n + 1, endpoint=True)
    comp_simpson = (h / 3) * (f(a) + f(b))

    comp_simpson += ((4 * h) / 3) * sum(
        [f(a + (2 * i - 1) * h) for i in range(1, int(n / 2) + 1)]
    )
    comp_simpson += ((2 * h) / 3) * sum(
        [f(a + 2 * i * h) for i in range(1, int((n - 2) / 2) + 1)]
    )

    return comp_simpson


def simpson(a, b, y):
    h = (b - a) / 2
    return (h / 3) * (y[0] + 4 * y[1] + y[2])


def trapezoid(a, b, y):
    h = b - a
    return (h / 2) * (y[0] + y[-1])
