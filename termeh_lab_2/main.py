import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

x = 1.5 * np.sin(1.8 * t)
s = 0.6 * np.sin(1.2 * t)

Alpha = np.pi / 4
BoxM, CylM, WeightM, WheelM = 10, 3, 2, 1
X0Right = 15
BoxW, BoxH = 6, 3
BoxHRight = BoxH / 4
BoxWUp = BoxW - (BoxH - BoxHRight) / np.tan(Alpha)
WheelR, CylR, BlockR = 0.5, 0.4, 0.2
WeightW, WeightH = 0.8, 2 * BlockR
ThreadLen = 4
S0 = 1.2
InclStart = ThreadLen - BoxWUp + S0

# O - left bottom of the box
X_O = X0Right - BoxW - x
Y_O = 2 * WheelR
# A - center of the cylinder
X_A = X_O + S0 + s
Y_A = Y_O + BoxH + CylR
# C1 & C2 - centers of the wheels
X_C1 = X0Right - 4 * BoxW / 5 - x
Y_C1 = WheelR
X_C2 = X0Right - BoxW / 5 - x
Y_C2 = WheelR
# C3 - center of the block
X_C3 = X_O + BoxWUp - BlockR
Y_C3 = Y_O + BoxH + BlockR
# B1 - top of the block
X_B1 = X_C3
Y_B1 = Y_C3 + BlockR
# B2 - top-right of the block
X_B2 = X_B1 + BlockR * np.cos(np.pi / 4)
Y_B2 = Y_B1 - BlockR * (1 - np.sin(np.pi / 4))
# D - left bottom of the weight
X_D = X_O + BoxWUp + np.cos(Alpha) * (InclStart + s)
Y_D = Y_O + BoxH - np.sin(Alpha) * (InclStart + s)
# F - left center of the weight
X_F = X_D + WeightH / 2 * np.sin(Alpha)
Y_F = Y_D + WeightH / 2 * np.cos(Alpha)

X_Ground = [17.5, 17.5, 0]
Y_Ground = [6, 0, 0]

X_Box = np.array([0, 0, BoxWUp, BoxW, BoxW, 0])
Y_Box = np.array([0, BoxH, BoxH, BoxHRight, 0, 0])
X_Weight = np.array([
    0, WeightH * np.sin(Alpha), 
    WeightH * np.sin(Alpha) + WeightW * np.cos(Alpha),
    WeightW * np.cos(Alpha), 0
])
Y_Weight = np.array([
    0, WeightH * np.cos(Alpha),
    WeightH * np.cos(Alpha) - WeightW * np.sin(Alpha),
    -WeightW * np.sin(Alpha), 0
])

psi = np.linspace(0, 2 * np.pi, 20)
X_Wheel = WheelR * np.sin(psi)
Y_Wheel = WheelR * np.cos(psi)
X_Block = BlockR * np.sin(psi)
Y_Block = BlockR * np.cos(psi)
X_Cyl = CylR * np.sin(psi)
Y_Cyl = CylR * np.cos(psi)

fig = plt.figure(figsize=[15, 7])

ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-5, 20], ylim=[-4, 10])

ax.plot(X_Ground, Y_Ground, color='black', linewidth=3)
Drawed_Wheel1 = ax.plot(X_C1[0] + X_Wheel, Y_C1 + Y_Wheel)[0]
Drawed_Wheel2 = ax.plot(X_C2[0] + X_Wheel, Y_C2 + Y_Wheel)[0]
Drawed_Block = ax.plot(X_C3[0] + X_Block, Y_C3 + Y_Block)[0]
Drawed_Cyl = ax.plot(X_A[0] + X_Cyl, Y_A + Y_Cyl)[0]
Drawed_Box = ax.plot(X_O[0] + X_Box, Y_O + Y_Box)[0]
Drawed_Weight = ax.plot(X_D[0] + X_Weight, Y_D[0] + Y_Weight)[0]
Line_A_B1 = ax.plot([X_A[0], X_B1[0]], [Y_A, Y_B1], color=Drawed_Block.get_color())[0]
Line_B2_F = ax.plot([X_B2[0], X_F[0]], [Y_B2, Y_F[0]], color=Drawed_Block.get_color())[0]

Point_A = ax.plot(X_A[0], Y_A, marker='o', markersize=5, color=Drawed_Block.get_color())[0]

AlphaWheel = -x / WheelR
Drawed_WheelD1 = ax.plot([X_C1[0] + WheelR * np.sin(AlphaWheel[0]), X_C1[0] - WheelR * np.sin(AlphaWheel[0])],
                         [Y_C1 + WheelR * np.cos(AlphaWheel[0]), Y_C1 - WheelR * np.cos(AlphaWheel[0])])[0]
Drawed_WheelD2 = ax.plot([X_C2[0] + WheelR * np.sin(AlphaWheel[0] + 1), X_C2[0] - WheelR * np.sin(AlphaWheel[0] + 1)],
                         [Y_C2 + WheelR * np.cos(AlphaWheel[0] + 1), Y_C2 - WheelR * np.cos(AlphaWheel[0] + 1)])[0]

def anima(i):
    Point_A.set_data(X_A[i], Y_A)
    Line_A_B1.set_data([X_A[i], X_B1[i]], [Y_A, Y_B1])
    Line_B2_F.set_data([X_B2[i], X_F[i]], [Y_B2, Y_F[i]])
    Drawed_Box.set_data(X_O[i] + X_Box, Y_O + Y_Box)
    Drawed_Cyl.set_data(X_A[i] + X_Cyl, Y_A + Y_Cyl)
    Drawed_Block.set_data(X_C3[i] + X_Block, Y_C3 + Y_Block)
    Drawed_Weight.set_data(X_D[i] + X_Weight, Y_D[i] + Y_Weight)
    Drawed_Wheel1.set_data(X_C1[i] + X_Wheel, Y_C1 + Y_Wheel)
    Drawed_Wheel2.set_data(X_C2[i] + X_Wheel, Y_C2 + Y_Wheel)
    Drawed_WheelD1.set_data([X_C1[i] + WheelR * np.sin(AlphaWheel[i]), X_C1[i] - WheelR * np.sin(AlphaWheel[i])],
                            [Y_C1 + WheelR * np.cos(AlphaWheel[i]), Y_C1 - WheelR * np.cos(AlphaWheel[i])])
    Drawed_WheelD2.set_data([X_C2[i] + WheelR * np.sin(AlphaWheel[i] + 1), X_C2[i] - WheelR * np.sin(AlphaWheel[i] + 1)],
                            [Y_C2 + WheelR * np.cos(AlphaWheel[i] + 1), Y_C2 - WheelR * np.cos(AlphaWheel[i] + 1)])


    return [Point_A, Drawed_Box, Drawed_Wheel1, Drawed_Wheel2, Drawed_WheelD1, Drawed_WheelD2]

anim = FuncAnimation(fig, anima, frames=len(t), interval=10)

plt.show()