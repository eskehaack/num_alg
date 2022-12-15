# exercise 1.1 - Horners algorithm
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from functionality.roots import bisection, newton, secant, heron

# def horner(
#     a: list,
#     x: float
#     ) -> float:

#     p = a[-1]
#     for ai in a[-2::-1]:
#         p = ai + x * p
#     return p

# def nested_multiplicaiton(x):
#     real = 5*math.exp(3*x)+7*math.exp(2*x)+9*math.exp(x) + 11
#     nested = 11+math.exp(x)*(9+math.exp(x)*(7+math.exp(x)*5))
#     return real-nested

# data = loadmat("FitData.mat")

# x_array = np.array(data["x"])
# y_array = np.array(data["y"])


# def my_func(x, c0, c1, c2, c3, c4, c5):
#     return (
#         c0
#         + c1 * x
#         + c2 * np.sin(x)
#         + c3 * np.cos(x)
#         + c4 * np.sin(2 * x)
#         + c5 * np.cos(2 * x)
#     )


# coefs, _ = curve_fit(my_func, x_array.flatten(), y_array.flatten())

# sorted_inx = np.argsort(x_array.flatten())

# plt.plot(x_array, y_array, "r.")
# plt.plot(sorted(x_array.flatten()[sorted_inx]), my_func(x_array, *coefs)[sorted_inx])
# plt.show()

# -------------------------------------------------------------------

# x = np.array([11, 25, 26, 31, 33, 36, 47, 58, 75, 79])
# y = np.array([160, 140, 138, 130, 125, 120, 95, 72, 27, 17])

# w = 2 * math.pi / 365


# def my_func(x, c0, c1, c2):
#     return c0 + c1 * np.sin(w * x) + c2 * np.cos(w * x)


# coefs, _ = curve_fit(my_func, x, y)

# sorted_inx = np.argsort(x.flatten())
# x_plot = np.arange(366)

# longest_day = np.argmin(my_func(x_plot, *coefs))


# plt.plot(x, y, ".")
# plt.plot(x_plot, my_func(x_plot, *coefs))
# plt.show()


# f = lambda x: x**2 - 0.8 * x - 0.2  # ((x - 2) * x + 1) * x - 3
# f_div = lambda x: 2 * x - 0.8  # (3 * x - 4) * x + 1

# xs = newton(3, 6, f, f_div)
# err = [abs(1 - x) for x in xs]
# rat = [abs(err[i] / (err[i - 1] ** 2)) for i in range(1, len(err))]
# rat.insert(0, "-")
# df = pd.DataFrame({"x": xs, "error": err, "ratio": rat})
# print(df)

# x_plot = np.arange(4)
# plt.plot(x_plot, f(x_plot))
# plt.show()


# f = lambda x: x**5 + x**3 + 3
# f_div = lambda x: 5 * x**4 + 3 * x**2
# sec = secant(8, 7, 24, f)
# bi = bisection(-4, 9, 24, f)
# newt = newton(8, 24, f, f_div)

# print(pd.DataFrame({"secant": sec, "bisection": bi, "newton": newt}))

# x_plot = np.linspace(-2, 2, 30, endpoint=True)
# plt.plot(x_plot, f(x_plot))
# plt.ylim(top=0.5, bottom=-0.5)
# plt.show()


# --------------------------------------------------------------

# my_func = lambda x, c1, c2: c1 + c2 * x
# x = [-1, 0, 0.6, 1]
# y = [-0.5, 0.1, 0.3, 0.24]
# coefs, _ = curve_fit(my_func, x, y)
# print(coefs)

# f = lambda x: math.log(x) + x
# f_div = lambda x: 1 / x + 1
# print(newton(1.4, 2, f, f_div))

# f = lambda x: math.sin(x)
# print(secant(math.pi / 2 - 0.1, math.pi / 2 + 0.1, 4, f))


# data = loadmat("data/Grovdata.mat")
# x, y = data["x"].flatten(), data["y"].flatten()
# plt.plot(x, y, ".")


# def my_func(x, c0, c1, c2, c3, c4, c5, c6, c7, c8):
#     return (
#         c0
#         + c1 * x
#         + c2 * x**2
#         + c3 * x**3
#         + c4 * x**4
#         + c5 * x**5
#         + c6 * x**6
#         + c7 * x**7
#         + c8 * x**8
#     )


# coefs, _ = curve_fit(my_func, x, y)
# x_plot = np.linspace(0, max(x), len(x), endpoint=True)
# plt.plot(x_plot, my_func(x, *coefs))

# err = [abs(y[i] - my_func(x[i], *coefs)) for i in range(len(x))]
# big_error = [i for i in range(len(x)) if err[i] > 5 * np.mean(err)]

# plt.plot(x, err)
# plt.show()

# x = np.delete(x, big_error)
# y = np.delete(y, big_error)

# coefs, _ = curve_fit(my_func, x, y)
# x_plot = np.linspace(0, max(x), len(x), endpoint=True)
# plt.plot(x_plot, my_func(x, *coefs))

# err = [abs(y[i] - my_func(x[i], *coefs)) for i in range(len(x))]
# plt.plot(x, err)

# plt.plot(x, y)
# plt.show()

# print(heron(1, 2, 5) - math.sqrt(2))

# f = lambda x: x**2 - 2
# for i in range(50):
#     print(bisection(1, 2, i, f)[-1] - math.sqrt(2))
