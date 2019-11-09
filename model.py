from abc import ABC, abstractmethod
from math import exp
import matplotlib.pyplot as plt
import numpy as np

'''
    defines the grid and the range for the solution
    range = (x0, X)
    N - number of grid cells
    h - size of the grid cell
'''


class Grid:
    def __init__(self, x0, y0, x, n):
        self.x0 = x0
        self.y0 = y0
        self.X = x
        self.N = n
        self.h = (x - x0) / n
        self.x = []
        i = x0
        for k in range(self.N + 1):
            self.x.append(i)
            i += self.h

    def cell_size(self):
        return self.h


class Function(ABC):
    @abstractmethod
    def compute(self, x, y):
        pass


'''
    y' = f(x, y)
    compute() calculates f(x, y) at the given point
'''


class MyFunction(Function):
    def compute(self, x, y):
        return 2 * exp(x) - y


'''
    y = g(x) - solution for the DE
'''


class SolutionFunc(Function):
    def __init__(self):
        self.c = 0

    # calculates y value for a given x
    def compute(self, x, y=None):
        return exp(x) - self.c * exp(-x)

    # solves the ivp for a given y(x0) = y0
    def calculate_const(self, x0, y0):
        c = (y0 - exp(x0)) / exp(-x0)
        self.c = c
        return c


class Solution(ABC):
    def __init__(self, grid: Grid, function: Function):
        self.grid = grid
        self.function = function
        self.y = []

    @abstractmethod
    def solve(self):
        pass


'''
    y stores the exact solution for a given grid
'''


class ExactSolution(Solution):
    def __init__(self, grid: Grid, function: Function):
        super().__init__(grid, function)

    # solves the ivp and then calculates y values based on the solution function y=g(x)
    def solve(self):
        c = self.function.calculate_const(self.grid.x0, self.grid.y0)
        for i in self.grid.x:
            self.y.append(self.function.compute(i))


'''
    function defines f(x, y) if y' = f(x, y)
    y stores approximated solutions for the given grid
    le stores local approximation error at each step
    ge stores global approximation error at each step
'''


class NumericalMethod(Solution):
    def __init__(self, grid: Grid, function: Function):
        super().__init__(grid, function)
        self.le = []
        self.ge = []
        self.y.append(self.grid.y0)

    @abstractmethod
    def local_error(self, exact_solution: ExactSolution):
        pass

    # global error[i] = y(x[i]) - y[i]
    def global_error(self, exact_solution: ExactSolution):
        for i in range(self.grid.N + 1):
            self.ge.append(-self.y[i] + exact_solution.y[i])


class EulerMethod(NumericalMethod):
    def __init__(self, grid: Grid, function: Function):
        super().__init__(grid, function)

    # y[i] = y[i-1] + h*f(x[i-1], y[i-1])
    def solve(self):
        for i in range(1, self.grid.N + 1):
            self.y.append(self.y[i - 1] + self.grid.cell_size() *
                          self.function.compute(self.grid.x[i - 1], self.y[i - 1]))

    # local error[i] = y(x[i]) - y(x[i-1]) - h*f(x[i-1], y(x[i-1]))
    def local_error(self, exact_solution: ExactSolution):
        self.le.append(0)
        for i in range(1, self.grid.N + 1):
            self.le.append(exact_solution.y[i] - exact_solution.y[i - 1] -
                           self.grid.h * self.function.compute(self.grid.x[i - 1], exact_solution.y[i - 1]))


class ImprovedEulerMethod(NumericalMethod):
    def __init__(self, grid: Grid, function: Function):
        super().__init__(grid, function)

    # y[i] = y[i-1] + h*(k1+k2)/2
    # k1 = f(x[i-1], y[i-1])
    # k2 = f(x[i], y[i-1] + h * k1)
    def solve(self):
        for i in range(1, self.grid.N + 1):
            k1 = self.function.compute(self.grid.x[i - 1], self.y[i - 1])
            k2 = self.function.compute(self.grid.x[i], self.y[i - 1] + self.grid.h * k1)
            self.y.append(self.y[i - 1] + self.grid.h / 2 * (k1 + k2))

    # local error[i] = y(x[i]) - y(x[i-1]) - h * (k1 + k2) / 2
    def local_error(self, exact_solution: ExactSolution):
        self.le.append(0)
        for i in range(1, self.grid.N + 1):
            k1 = self.function.compute(self.grid.x[i - 1], exact_solution.y[i - 1])
            k2 = self.function.compute(self.grid.x[i], exact_solution.y[i - 1] + self.grid.h * k1)
            self.le.append(exact_solution.y[i] - exact_solution.y[i - 1] - self.grid.h / 2 * (k1 + k2))


class RungeKuttaMethod(NumericalMethod):
    def __init__(self, grid: Grid, function: Function):
        super().__init__(grid, function)

    # y[i] = y[i-1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # k1 = f(x[i-1], y[i-1])
    # k2 = f(x[i-1] + h / 2, y[i-1] + h * k1 / 2)
    # k3 = f(x[i-1] + h / 2, y[i-1] + h * k2 / 2)
    # k4 = f(x[i], y[i-1] + h * k3)
    def solve(self):
        for i in range(1, self.grid.N + 1):
            k1 = self.function.compute(self.grid.x[i - 1], self.y[i - 1])
            k2 = self.function.compute(self.grid.x[i - 1] + self.grid.h / 2, self.y[i - 1] + self.grid.h / 2 * k1)
            k3 = self.function.compute(self.grid.x[i - 1] + self.grid.h / 2, self.y[i - 1] + self.grid.h / 2 * k2)
            k4 = self.function.compute(self.grid.x[i], self.y[i - 1] + self.grid.h * k3)
            self.y.append(self.y[i - 1] + self.grid.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    # local error[i] = y(x[i]) - y(x[i-1]) - h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    def local_error(self, exact_solution: ExactSolution):
        self.le.append(0)
        for i in range(1, self.grid.N + 1):
            k1 = self.function.compute(self.grid.x[i - 1], exact_solution.y[i - 1])
            k2 = self.function.compute(self.grid.x[i - 1] + self.grid.h / 2,
                                       exact_solution.y[i - 1] + self.grid.h / 2 * k1)
            k3 = self.function.compute(self.grid.x[i - 1] + self.grid.h / 2,
                                       exact_solution.y[i - 1] + self.grid.h / 2 * k2)
            k4 = self.function.compute(self.grid.x[i], exact_solution.y[i - 1] + self.grid.h * k3)
            self.le.append(exact_solution.y[i] - exact_solution.y[i - 1] -
                           self.grid.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
