import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D as l2
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def rotate_2d(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY

v_scl = 0.1
a_scl = v_scl**2
ar_scl = 0.025
frame_num = 1000
frame_frequency = 10
t_end = 12 * math.pi / 6
t = sp.Symbol('t')

r = 1 + sp.sin(8 * t)
phi = t + 0.5 * sp.sin(8 * t)

x = r * sp.sin(phi)
y = r * sp.cos(phi)
v_x = sp.diff(x, t)
v_y = sp.diff(y, t)
a_x = sp.diff(v_x, t)
a_y = sp.diff(v_y, t)

T = np.linspace(0, t_end, frame_num)
X = np.zeros_like(T)
Y = np.zeros_like(T)
V_X = np.zeros_like(T)
V_Y = np.zeros_like(T)
A_X = np.zeros_like(T)
A_Y = np.zeros_like(T)
RC_X = np.zeros_like(T)
RC_Y = np.zeros_like(T)
RC_R = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    V_X[i] = sp.Subs(v_x, t, T[i])
    V_Y[i] = sp.Subs(v_y, t, T[i])
    A_X[i] = sp.Subs(a_x, t, T[i])
    A_Y[i] = sp.Subs(a_y, t, T[i])
    RC_X[i] = (V_X[i]**2 + V_Y[i]**2) / (A_X[i] * V_Y[i] - A_Y[i] * V_X[i])
    RC_Y[i] = RC_X[i] * V_Y[i]
    RC_X[i] *= V_X[i]
    RC_R[i] = math.sqrt(RC_X[i]**2 + RC_Y[i]**2)

V_X *= v_scl
V_Y *= v_scl
A_X *= a_scl
A_Y *= a_scl

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-3, 3], ylim=[-3, 3])

P, = ax1.plot(X[0], Y[0], color='red', marker='*')
V, = ax1.plot([X[0], X[0] + V_X[0]], 
              [Y[0], Y[0] + V_Y[0]], color='red')
R, = ax1.plot([0, X[0]], [0, Y[0]], color='black')
A, = ax1.plot([X[0], X[0] + A_X[0]], 
              [Y[0], Y[0] + A_Y[0]], color='green')

Alpha = np.linspace(0, math.pi * 2, 100)
RC, = ax1.plot([X[0], X[0] + RC_Y[0]],
               [Y[0], Y[0] - RC_X[0]], color='blue')

ArrowX = np.array([-2 * ar_scl, 0, -2 * ar_scl])
ArrowY = np.array([ar_scl, 0, -ar_scl])

VArrowX, VArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(V_Y[0], V_X[0]))
VArrow, = ax1.plot(VArrowX + X[0] + V_X[0],
                   VArrowY + Y[0] + V_Y[0], color='red')

RArrowX, RArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
RArrow, = ax1.plot(RArrowX + X[0], RArrowY + Y[0], color='black')

AArrowX, AArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(A_Y[0], A_X[0]))
AArrow, = ax1.plot(AArrowX + X[0] + A_X[0], 
                   AArrowY + Y[0] + A_Y[0], color='green')

RCArrowX, RCArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(-RC_X[0], RC_Y[0]))
RCArrow, = ax1.plot(RCArrowX + X[0] + RC_Y[0], 
                    RCArrowY + Y[0] - RC_X[0], color='blue')

ax1.plot(X, Y, 'grey')

def anim(i):
    P.set_data(X[i], Y[i])
    V.set_data([X[i], X[i] + V_X[i]], 
               [Y[i], Y[i] + V_Y[i]])
    R.set_data([0, X[i]], [0, Y[i]])
    A.set_data([X[i], X[i] + A_X[i]], 
               [Y[i], Y[i] + A_Y[i]])
    VArrowX, VArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(V_Y[i], V_X[i]))
    VArrow.set_data(VArrowX + X[i] + V_X[i],
                    VArrowY + Y[i] + V_Y[i])
    RArrowX, RArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RArrow.set_data(RArrowX + X[i], RArrowY + Y[i])
    AArrowX, AArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(A_Y[i], A_X[i]))
    AArrow.set_data(AArrowX + X[i] + A_X[i],
                    AArrowY + Y[i] + A_Y[i])
    RC.set_data([X[i], X[i] + RC_Y[i]],
                [Y[i], Y[i] - RC_X[i]])
    RCArrowX, RCArrowY = rotate_2d(ArrowX, ArrowY, math.atan2(-RC_X[i], RC_Y[i]))
    RCArrow.set_data(RCArrowX + X[i] + RC_Y[i],
                     RCArrowY + Y[i] - RC_X[i])
    return P, V, VArrow, R, RArrow, A, AArrow, RC, RCArrow

anim1 = FuncAnimation(fig, anim, frames=frame_num, interval=frame_num / frame_frequency, blit=True)

custom_lines = [l2([0], [0], color='grey'),
                l2([0], [0], color='black'),
                l2([0], [0], color='red'),
                l2([0], [0], color='green'),
                l2([0], [0], color='blue')]
ax1.legend(custom_lines, ['Траектория', 'Радиус-вектор', 'Скорость', 'Ускорение', 'Радиус кривизны'], loc='lower left')

plt.show()
