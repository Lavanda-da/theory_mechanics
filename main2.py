import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

psi = np.cos(0.2*t) + 3*np.sin(1.8*t)
phi = 2*np.sin(1.7*t) + 5*np.cos(1.2*t)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.axis('equal')
ax.set(xlim=[-2, 2], ylim=[-2, 2], zlim=[-2, 2])

x = np.zeros(100)
y = np.zeros(100)
z = np.zeros(100)
for i in range(len(z)):
    z[i] = i - 50

Draw_Line = ax.plot3D(x, y, z, "black")

CircleR = 1

alfa = np.linspace(0, 2*math.pi, 100)
X_Circle = CircleR*np.sin(alfa)*np.cos(phi[0])
Y_Circle = CircleR*np.sin(alfa)*np.sin(phi[0])
Z_Circle = CircleR*np.cos(alfa)

Drawed_Circle, = ax.plot3D(X_Circle, Y_Circle, Z_Circle, "blue")

X_A = CircleR*np.sin(psi[0])*np.cos(phi[0])
Y_A = CircleR*np.sin(psi[0])*np.sin(phi[0])
Z_A = CircleR*np.cos(psi[0])

Point_A, = ax.plot(X_A, Y_A, Z_A, marker='o')

SpringR = 0.05

alfa1 = np.linspace(0, 8*math.pi, 100)
X_Spring = SpringR * np.sin(alfa1)
Y_Spring = SpringR * np.cos(alfa1)
Z_Spring = np.linspace(CircleR, Z_A, 100)
k1 = (X_A - X_Spring[-1]) / 100
k2 = (Y_A - Y_Spring[-1]) / 100
for j in range(100):
    X_Spring[j] += (j * k1)
    Y_Spring[j] += (j * k2)
Drawed_Spring, = ax.plot3D(X_Spring, Y_Spring, Z_Spring, "red")

def anima(i):
    ax.clear()
    ax.axis('equal')
    ax.set(xlim=[-2, 2], ylim=[-2, 2], zlim=[-2, 2])
    Draw_Line = ax.plot3D(x, y, z, "black")
    X_Circle = CircleR*np.sin(alfa)*np.cos(phi[i])
    Y_Circle = CircleR*np.sin(alfa)*np.sin(phi[i])
    Drawed_Circle, = ax.plot3D(X_Circle, Y_Circle, Z_Circle, "blue")
    X_A = CircleR * np.sin(psi[i]) * np.cos(phi[i])
    Y_A = CircleR * np.sin(psi[i]) * np.sin(phi[i])
    Z_A = CircleR * np.cos(psi[i])
    Point_A, = ax.plot(X_A, Y_A, Z_A, marker='o')
    X_Spring = SpringR * np.sin(alfa1)
    Y_Spring = SpringR * np.cos(alfa1)
    k1 = (X_A - X_Spring[-1]) / 100
    k2 = (Y_A - Y_Spring[-1]) / 100
    for j in range(100):
        X_Spring[j] += (j * k1)
        Y_Spring[j] += (j * k2)
    Z_Spring = np.linspace(CircleR, Z_A, 100)
    Drawed_Spring, = ax.plot3D(X_Spring, Y_Spring, Z_Spring, "red")
    return Draw_Line, Drawed_Circle, Point_A, Drawed_Spring

anim = FuncAnimation(fig, anima, frames=len(t), interval=40, repeat=False)

plt.show()