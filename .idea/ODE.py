from scipy.integrate import odeint
import matplotlib.pyplot as pl
import numpy as np

y0=5.0
x = np.linspace(0, 100, 10001)

def f(y,x):
    function=-12*x+10*np.math.sin(x)
    return function

sol=odeint(f,y0,x)

pl.plot(x,sol)
pl.show()