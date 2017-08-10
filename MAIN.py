import numpy as np
import sdeint
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


num_of_Dtime=100000
tspan = np.linspace(0,1000,num=num_of_Dtime)

dimension=9

x0 = np.array([1,1,1,1,5,1,1,1,5],dtype=float)
ar=12.8
R=1
be=19.1
a=0.6
b=-1.1
c=0.45
epsilo=0

D2=0


def fun(x):
    result=b*x**2+c*x**3
    return result

def f(x, t):
    return np.array([ar*(x[1]-x[0]-fun(x[0])),
                     x[0]-x[1]+R*x[2],
                     -be* x[1],

                     ar * (x[4] - x[3] - fun(x[3]))+epsilo*(x[3]-x[0]),
                     x[3] - x[4] + R * x[5],
                     -be * x[4],

                     ar * (x[7] - x[6] - fun(x[6])),
                     x[6] - x[7] + R * x[8],
                     -be * x[7]
                     ])

def G(x, t):
    return np.array([[0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
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
pl.xlabel('x6-x9')
pl.ylabel('error')
pl.title('error term')
pl.grid(True)
pl.plot(tspan[0:-1],resultx6-resultx9)

pl.subplot(4,1,2)
pl.plot(tspan[0:-1],resultx5-resultx8)

pl.subplot(4,1,1)
pl.plot(tspan[0:-1],resultx4-resultx7)

# pl.subplot(4,1,4)
# for t in tspan:
#     t=(int(t)-1)
#     if t%2!=1:
#         continue
#     pl.scatter(resultx5[t], resultx8[t])
pl.show()
#
fig = pl.figure()
ax = Axes3D(fig)
X = resultx1
Y = resultx2
Z = resultx3
for t in tspan:
    t=(int(t)-1)
    if t%2!=1:
        continue
    ax.scatter(X[t], Y[t], Z[t], c='r', marker='^')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
pl.show()
