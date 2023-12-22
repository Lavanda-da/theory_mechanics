import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint

def odesys(y, t, M, m, R, c, a, l, g):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    dy[2] = (a * y[0] ** 2 - 2 * m * R ** 2 * y[2] * y[3] * np.sin(4 * y[1])) / ((M / 2 + m * (np.sin(2 * y[1])) ** 2) * R ** 2)
    dy[3] = 1 / 4 * ((2 * c * (2 * R * np.cos(y[1]) - l) * np.sin(y[1]) - 2 * m * g * np.sin(2 * y[1])) / (m * R) + y[2] ** 2 * np.sin(4 * y[1]))

    return dy

M = 1
m = 0.1
R = 0.3
c = 20
a = 0.001
l = 0.2
g = 9.8

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

psi0 = 0
phi0 = np.pi / 6
dpsi0 = 0.5
dphi0 = 0
y0 = [psi0, phi0, dpsi0, dphi0]

Y = odeint(odesys, y0, t, (M, m, R, c, a, l, g))
psi = Y[:, 0]
phi = Y[:, 1]
dpsi = Y[:, 2]
dphi = Y[:, 3]
ddpsi = [odesys(y, t, M, m, R, c, a, l, g)[2] for y, t in zip(Y, t)]
ddphi = [odesys(y, t, M, m, R, c, a, l, g)[3] for y, t in zip(Y, t)]

fig_for_graphs = plt.figure(figsize=[13, 7])

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, psi, color='Blue')
ax_for_graphs.set_title("psi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, phi, color='Green')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

n1 = m * R * (4 * dphi ** 2 + dpsi ** 2 * (np.sin(2 * phi) ** 2)) + m * g * np.cos(2 * phi) - c * (2 * R * np.cos(phi) - l) * np.cos(phi)
n2 = m * R * (ddpsi * np.sin(2 * phi) + 4 * dpsi * dphi * np.cos(2 * phi))

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, n1, color='Black')
ax_for_graphs.set_title("N1(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(t, n2, color='Grey')
ax_for_graphs.set_title("N2(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

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