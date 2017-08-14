import numpy as np
import sdeint
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


num_of_Dtime=10000
tspan = np.linspace(0,50,num=num_of_Dtime)

dimension=9

x0 = np.array([0.7,0,0,
               0.5,1,0.7,
               0.5,1,3],dtype=float)
ar=16
R=1
be_1=28
be_2=30

a=-1.1
b=-0.7

epsilo=-3

D11=np.math.sqrt(20/7)*0
D22=np.math.sqrt(50/7)*0
D33=np.math.sqrt(110/7)*0

def fun(x):
    result=b*x+0.5*(a-b) * (np.math.fabs(x+1)-np.math.fabs(x-1))
    return result

def f(x, t):
    return np.array([-ar*(x[0]-x[1]+fun(x[0])),
                     x[0]-x[1]+R*x[2],
                     -be_1* x[1],

                     -ar * (x[3] - x[4] + fun(x[3]))+epsilo*(x[3]-x[0]),
                     x[3] - x[4] + R * x[5]+epsilo*(x[4]-x[1]),
                     -be_2 * x[4]+epsilo*(x[5]-x[2]),

                     -ar * (x[6] - x[7] +fun(x[6]))+epsilo*(x[6]-x[0]),
                     x[6] - x[7] + R * x[8]+epsilo*(x[7]-x[1]),
                     -be_2 * x[7]+epsilo*(x[8]-x[2])
                     ])

def G(x, t):
    return np.array([[0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],

                     [D11*(x[3]-x[0]),0,0,0,0,0,0,0,0],
                     [0,D22*(x[4]-x[1]),0,0,0,0,0,0,0],
                     [0,0,D33*(x[5]-x[2]),0,0,0,0,0,0],

                     [D11*(x[6]-x[0]),0, 0, 0, 0, 0, 0, 0, 0],
                     [0, D22*(x[7]-x[1]), 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, D33*(x[8]-x[2]), 0, 0, 0, 0, 0, 0],])

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


# pl.subplot(4,1,1)
# pl.plot(tspan[0:-1],resultx1)
#
# pl.subplot(4,1,2)
# pl.plot(tspan[0:-1],resultx2)
#
# pl.subplot(4,1,3)
# pl.plot(tspan[0:-1],resultx3)




# pl.subplot(4,1,4)
# for t in tspan:
#     t=(int(t)-1)
#     if t%2!=1:
#         continue
#     pl.scatter(resultx5[t], resultx8[t])
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


pl.subplot(4,1,3)
pl.xlabel('x6-x9')
pl.ylabel('error')
pl.title('error term')
pl.grid(True)
pl.plot(tspan[0:-1],resultx4-resultx1)
pl.subplot(4,1,2)
pl.plot(tspan[0:-1],resultx5-resultx2)
pl.subplot(4,1,1)
pl.plot(tspan[0:-1],resultx6-resultx3)
pl.show()



pl.plot(resultx8,resultx9)
pl.plot(resultx5,resultx6)
pl.plot(resultx2,resultx3)
pl.show()
