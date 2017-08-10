import numpy as np
import sdeint
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

num_of_Dtime=10001
tspan = np.linspace(0,5,num=num_of_Dtime)

dimension=2

x0 = np.array([5.0,9.0])

ro=1.1

def f(x, t):
    return 0.5*(1+np.math.sin(t)**2)*np.math.exp(t)*np.array([x[0]**2,
                                                              x[1]**2])

def G(x, t):
    return np.math.sqrt(2*ro*np.math.exp(t)*np.math.sqrt(x[0]**2+x[1]**2)+1)*np.array([[x[0],0],
                                                                                       [x[1],0]])

# result = sdeint.itoint(f, G, x0, tspan)
result =  sdeint.itoSRI2(f, G, x0, tspan)

pl.plot(tspan,result)
pl.show()
#
# fig = pl.figure()
# ax = Axes3D(fig)
# X = resultx1
# Y = resultx2
# Z = resultx3
# for t in tspan:
#     t=(int(t)-1)
#     if t%2!=1:
#         continue
#     ax.scatter(X[t], Y[t], Z[t], c='r', marker='^')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#