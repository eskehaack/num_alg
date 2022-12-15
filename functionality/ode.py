import numpy as np
from sympy import *

def RK4(f, t, x, h, n):
    """
    Parameters
    ----------
    f
        function to approximate
        
    t
        start of interval
    
    x
        initial value of x (x(t) = x)
    
    h
        (end interval - start interval) / 2 or (b-a)/2
    
    n
        itterations
    
    Returns
    ---------
    x
        Aprroximated value at the end of the interval (for differential equations)
    """
    ta = t
    h2 = h / 2
    for j in range(1, n):
        k1 = h * f(t, x)
        k2 = h * f(t + h2, x + 0.5 * k1)
        k3 = h * f(t + h2, x + 0.5 * k2)
        k4 = h * f(t + h, x + k3)
        
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = ta + j * h
    return x


def RK4system(fs, t, xs, h, n):
    """
    Parameters
    ----------
    f
        system to approximate
        
    t
        start of interval
    
    x
        initial value of x (x(t) = x)
    
    h
        (end interval - start interval) / 2 or (b-a)/2
    
    n
        itterations
    
    Returns
    ---------
    xs
        Aprroximated value at the end of the interval (for differential equations)
    """
    xs = np.array(xs)
    ta = t
    h2 = h / 2
    for j in range(1, n):
        k1 = h * np.array([f(t, xs[i]) for i, f in enumerate(fs)])
        k2 = h * np.array([f(t + h2, xs[i] + 0.5 * k1) for i, f in enumerate(fs)])
        k3 = h * np.array([f(t + h2, xs[i] + 0.5 * k2) for i, f in enumerate(fs)])
        k4 = h * np.array([f(t + h, xs[i] + k3) for i, f in enumerate(fs)])
        
        xs += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = ta + j * h
    return xs


def FdF20_1(f, X):
    x1,x2,x3 = symbols("x1 x2 x3")
    f = Lambda((x1, x2, x3), np.array([[x1 + x2 + x3], [x1**2 + x2**2 + x3**2 - 2], [x1 * (x2 + x3) + 1]]))

    jac = Lambda((x1,x2,x3), np.array([[diff(f(x1,x2,x3)[0], x1), diff(f(x1,x2,x3)[0], x2), diff(f(x1,x2,x3)[0], x3)], 
                                       [diff(f(x1,x2,x3)[1], x1), diff(f(x1,x2,x3)[1], x2), diff(f(x1,x2,x3)[1], x3)],
                                       [diff(f(x1,x2,x3)[2], x1), diff(f(x1,x2,x3)[2], x2), diff(f(x1,x2,x3)[2], x3)]]).reshape(-1, 3))

    return np.array(f(X[0], X[1], X[2]), dtype=('float64')), np.array(jac(X[0], X[1], X[2]), dtype=('float64'))


def order_2_taylor(a, b, n, x_0, df, ddf):
    """
    Parameters
    ----------
    a, b
        interval edges
    
    n 
        itterations
    
    x_0
        initial value of x (x(a) = x_0)
    
    df, ddf
        derivatives of function
    """
    h = (b-a)/n
    t = a
    x = x_0
    vals = [x_0]
    for k in range(1,n+1):
        df_val = df(t, x)
        ddf_val = ddf(df_val, x)
        
        x = x + h * (df_val + (1/2) * h * ddf_val)
        
        vals.append(x)
        t = (a+k*h)
        
    return vals