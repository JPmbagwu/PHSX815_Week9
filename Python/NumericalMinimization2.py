#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:34:55 2023

@author: johnpaulmbagwu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


result = minimize(rosenbrock, x0=[-1, 1.2], method='BFGS')
print(f"Minimum: f({result.x[0]}, {result.x[1]}): {result.fun}")

# Plot Rosenbrock function surface
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 5, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
