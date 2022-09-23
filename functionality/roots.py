def newton(starting_point: float, nmax: int, f: type, f_div: type) -> list:
    """
    Newtons method for finding roots in funcitons

    Parameters
    -----------
    starting point
        point for initializing the algorithm

    nmax
        number of itterations

    f
        function to be estimated

    f_div
        dirivative of f

    Returns
    --------
    list
        list of guesses the algoritm takes, last one being the best one
    """

    x = starting_point
    xs = [x]
    for _ in range(nmax):
        fx = f(x)
        fp = f_div(x)
        x = x - (fx / fp)
        xs.append(x)
    return xs


def secant(a: float, b: float, nmax: int, f: type) -> list:
    """
    secant method for finding roots in funtions

    Parameters
    ---------

    a
        the first starting point

    b
        second starting point

    nmax
        number of itterations

    f
        funtion to be estimated

    Returns
    ---------
    list
        list of guesses the algoritm takes, last one being the best one

    """
    fa = f(a)
    fb = f(b)
    x = [a, b]
    for _ in range(nmax - 1):
        d = fb * (b - a) / (fb - fa)
        a = b
        fa = fb
        b = b - d
        fb = f(b)
        x.append(b)
    return x


def bisection(a: float, b: float, nmax: int, f: type) -> list:
    """
    bisection method for finding roots in functions

    Parameters
    ----------

    a
        the first starting point

    b
        second starting point

    nmax
        number of itterations

    f
        funtion to be estimated

    Returns
    -------
    list
        list of guesses the algoritm takes, last one being the best one
    """

    fa = f(a)
    x = []
    for _ in range(nmax):
        c = 1 / 2 * (a + b)
        fc = f(c)
        x.append(c)
        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc
    x.append((a + b) / 2)
    return x


def heron(x0, R, nmax):
    for _ in range(nmax):
        x0 = 0.5 * (x0 + (R / x0))
    return x0
