import numpy as np
import sdeint
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


num_of_Dtime=10000
tspan = np.linspace(0,30,num=num_of_Dtime)

dimension=9

x0 = np.array([1,1,1,1,10,1,0,2,3])
wx=1
a=0.15
p=0.2
c=10
D1=0
D2=10

wu=0.95
epsilo=0.8

def f(x, t):
    return np.array([-wx*x[1]-x[2],
                     wx*x[0]+a*x[1],
                     p+x[2]*(x[0]-c),
                     -wu*x[4]-x[5]-epsilo*(x[0]-x[3]),
                     wu*x[3]+a*x[4],
                     p+x[5]*(x[3]-c),
                     -wu * x[7] - x[8] - epsilo * (x[0] - x[6]),
                     wu * x[6] + a * x[7],
                     p + x[8] * (x[6] - c),
                     ])

def G(x, t):
    return np.array([[D1,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
                     [0,epsilo*D2,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
                     [0, epsilo*D2, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],])

# result = sdeint.itoint(f, G, x0, tspan)
result =  sdeint.itoSRI2(f, G, x0, tspan)

result_reshape= result.reshape(dimension*num_of_Dtime,1)

resultx1=result_reshape[0:-9:dimension]
resultx2=result_reshape[1:-8:dimension]
resultx3=result_reshape[2:-7:dimension]
resultx4=result_reshape[3:-6:dimension]
resultx5=result_reshape[4:-5:dimension]
resultx6=result_reshape[5:-4:dimension]
resultx7=result_reshape[6:-3:dimension]
resultx8=result_reshape[7:-2:dimension]
resultx9=result_reshape[8:-1:dimension]

pl.subplot(4,1,3)
pl.plot(tspan[0:-1],resultx6-resultx9)

pl.subplot(4,1,2)
pl.plot(tspan[0:-1],resultx5-resultx8)

pl.subplot(4,1,1)
pl.plot(tspan[0:-1],resultx4-resultx7)

pl.subplot(4,1,4)
for t in tspan:
    t=(int(t)-1)
    if t%2!=1:
        continue
    pl.scatter(resultx5[t], resultx8[t])
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