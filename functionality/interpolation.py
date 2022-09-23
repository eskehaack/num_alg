import numpy as np
import matplotlib.pyplot as plt


class PolynomialInterpolation:
    """
    Newtons method for polynomial interpolation
    """

    def __init__(self, x_0, y_0):
        self.n = 0
        self.xs = [x_0]
        self.constants = [y_0]

    def evaluate(self, x: float) -> float:
        """
        Evaluate x value through function f

        Parameters
        ----------
        x
            x-value at point

        Returns
        ------
        float
            y-value at point (approximation)
        """
        p = self.constants[-1]
        for i in range(self.n - 1, -1, -1):
            p = self.constants[i] + (x - self.xs[i]) * p
        return p

    def denominator(self, x: float) -> float:
        """
        Find denominator for evaluating c constant
        """
        p = x - self.xs[-1]
        for i in range(self.n):
            p *= x - self.xs[i]
        return p

    def next_pair(self, x_i: float, y_i: float) -> None:
        """
        Insert new pair of x,y in function
        """
        c = (y_i - self.evaluate(x_i)) / self.denominator(x_i)
        self.xs.append(x_i)
        self.constants.append(c)
        self.n += 1


def newt_inter(xs: list, ys: list) -> object:
    """
    Wrapper for newton interpolation method
    Makes it easier to parse values as lists
    """
    temp = PolynomialInterpolation(xs[0], ys[0])
    for x, y in zip(xs[1:], ys[1:]):
        temp.next_pair(x, y)
    return temp


def cardinal_polynomials(nodes: list, i: int, t: list) -> float:
    """
    Creates n-1 (where n = len(nodes)) cardinal polynomials from the i'th node

    Parameters
    ----------
    nodes
        x-values

    i
        which node (by index) the cardinal polynomials should be made from

    t
        x-interval for evaluating

    Returns
    --------
    list
        values of l_i at t
    """
    nodes = np.array(nodes)
    t = np.array(t)

    x_i = nodes[i]
    p = 1
    for x_j in np.delete(nodes, i, axis=0):
        p *= (t - x_j) / (x_i - x_j)
    return p


def lagrange_interpolation(x: list, y: list, t: list) -> list:
    """
    creates values for lagrange interpolation in x interval = t
    """
    return sum([np.array(cardinal_polynomials(x, i, t)) * y[i] for i in range(len(x))])


def lagrange_plot(x: list, y: list, t: list) -> None:
    """
    plots lagrange function, cardinal polynomials and the given x,y values
    """
    for i in range(len(x)):
        plt.plot(t, cardinal_polynomials(x, i, t), "g--")

    plt.plot(t, lagrange_interpolation(x, y, t))
    plt.plot(x, y, ".")
    plt.show()
