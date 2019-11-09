import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

from model import *


class Controller:
    def __init__(self):
        self.view = View()
        self.calculate_button = tk.Button(self.view.init, text='calculate', command=self.on_calc_button_click, state='normal')
        self.calculate_button.grid(row=10, column=11)
        self.view.init.mainloop()

    def on_calc_button_click(self):

        # read entered numerical values
        x0 = self.view.x0_textbox.get()
        if (x0 != '') and (str.isdigit(x0)):
            self.x0 = int(x0)
        else:
            messagebox.showwarning("wrong value", "please, enter the correct value for x0")
            return
        y0 = self.view.y0_textbox.get()
        if (y0 != '') and (str.isdigit(y0)):
            self.y0 = int(y0)
        else:
            messagebox.showwarning("wrong value", "please, enter the correct value for y0")
            return
        x = self.view.x_textbox.get()
        if (x != '') and (str.isdigit(x)):
            self.x = int(x)
        else:
            messagebox.showwarning("wrong value", "please, enter the correct value for X")
            return
        n = self.view.n_textbox.get()
        if (n != '') and (str.isdigit(n)):
            self.n = int(n)
        else:
            messagebox.showwarning("wrong value", "please, enter the correct value for N")
            return
        ti = self.view.n_s_textbox.get()
        if (ti != '') and (str.isdigit(ti)):
            self.ti = int(ti)
        else:
            messagebox.showwarning("wrong value", "please, enter the correct value for starting N")
            return
        tf = self.view.n_f_textbox.get()
        if (tf != '') and (str.isdigit(tf)):
            self.tf = int(tf)
        else:
            messagebox.showwarning("wrong value", "please, enter the correct value for final N")
            return

        # read checkboxes
        em = self.view.em.get()
        iem = self.view.iem.get()
        rkm = self.view.rkm.get()
        le = self.view.le.get()
        ge = self.view.ge.get()
        if (em + iem + rkm == 0):
            messagebox.showwarning('checkboxes', 'you must choose at least one numerical method to show')
            return

        # exact solution
        self.grid = Grid(self.x0, self.y0, self.x, self.n)
        sol_function = SolutionFunc()
        self.exact_solution = ExactSolution(self.grid, sol_function)
        self.exact_solution.solve()

        deriv_function = MyFunction()
        self.subplot_num = 2
        # euler method if chosen
        if (em):
            self.euler_method = EulerMethod(self.grid, deriv_function)
            self.euler_method.solve()
            if (le):
                self.subplot_num += 1
                self.euler_method.local_error(self.exact_solution)
            if (ge):
                self.subplot_num += 1
                self.euler_method.global_error(self.exact_solution)
        # improved euler method if chosen
        if (iem):
            self.ie_method = ImprovedEulerMethod(self.grid, deriv_function)
            self.ie_method.solve()
            if (le):
                self.ie_method.local_error(self.exact_solution)
            if (ge):
                self.ie_method.global_error(self.exact_solution)
        # runge-kutta method if chosen
        if (rkm):
            self.rk_method = RungeKuttaMethod(self.grid, deriv_function)
            self.rk_method.solve()
            if (le):
                self.rk_method.local_error(self.exact_solution)
            if (ge):
                self.rk_method.global_error(self.exact_solution)

        # total errors
        self.em_total_error = []
        self.iem_total_error = []
        self.rkm_total_error = []
        for i in range(self.ti, self.tf + 1):
            temp_grid = Grid(self.x0, self.y0, self.x, i)
            temp_exact = ExactSolution(temp_grid, sol_function)
            temp_exact.solve()
            if (em):
                temp_euler = EulerMethod(temp_grid, deriv_function)
                temp_euler.solve()
                temp_euler.local_error(temp_exact)
                t_le = np.amax(np.absolute(np.array(temp_euler.le)))
                self.em_total_error.append(t_le)
            if (iem):
                temp_iem = ImprovedEulerMethod(temp_grid, deriv_function)
                temp_iem.solve()
                temp_iem.local_error(temp_exact)
                t_le = np.amax(np.absolute(np.array(temp_iem.le)))
                self.iem_total_error.append(t_le)
            if (rkm):
                temp_rkm = RungeKuttaMethod(temp_grid, deriv_function)
                temp_rkm.solve()
                temp_rkm.local_error(temp_exact)
                t_le = np.amax(np.absolute(np.array(temp_rkm.le)))
                self.rkm_total_error.append(t_le)

        #displaying plots
        self.figure = plt.Figure(figsize=(6, 8), dpi=90)
        if (le):
            self.le_plot = self.figure.add_subplot(self.subplot_num, 1, 2)
            self.le_plot.set_title("local errors", fontsize='x-small')
        if (ge):
            if (le):
                self.ge_plot = self.figure.add_subplot(self.subplot_num, 1, 3)
            else:
                self.ge_plot = self.figure.add_subplot(self.subplot_num, 1, 2)
            self.ge_plot.set_title("global errors", fontsize='x-small')
        self.te_plot = self.figure.add_subplot(self.subplot_num, 1, self.subplot_num)
        self.plot_exact_solution()
        if (em):
            self.plot_euler()
            self.plot_euler_total_error()
            if (le):
                self.plot_euler_le()
            if (ge):
                self.plot_euler_ge()
        if (iem):
            self.plot_improved_euler()
            self.plot_iem_total_error()
            if (le):
                self.plot_improved_euler_le()
            if (ge):
                self.plot_improved_euler_ge()
        if (rkm):
            self.plot_runge_kutta()
            self.plot_rkm_total_error()
            if (le):
                self.plot_runge_kutta_le()
            if (ge):
                self.plot_runge_kutta_ge()
        #displaying legends
        self.graph_plot.legend(loc='best', fontsize='xx-small')
        self.te_plot.legend(loc='best', fontsize='xx-small')
        if (le):
            self.le_plot.legend(loc='best', fontsize='xx-small')
        if (ge):
            self.ge_plot.legend(loc='best', fontsize='xx-small')
        #changing font sizes
        for label in (self.graph_plot.get_xticklabels() + self.graph_plot.get_yticklabels()):
            label.set_fontsize('x-small')
        for label in (self.te_plot.get_xticklabels() + self.te_plot.get_yticklabels()):
            label.set_fontsize('x-small')
        if (le):
            for label in (self.le_plot.get_xticklabels() + self.le_plot.get_yticklabels()):
                label.set_fontsize('x-small')
        if (ge):
            for label in (self.ge_plot.get_xticklabels() + self.ge_plot.get_yticklabels()):
                label.set_fontsize('x-small')
        #outputting graphs
        self.canvas = FigureCanvasTkAgg(self.figure, self.view.init)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=15, columnspan=8,  sticky="WENS")

    def plot_exact_solution(self):
        y = self.exact_solution.y
        x = self.grid.x
        self.graph_plot = self.figure.add_subplot(self.subplot_num, 1, 1)
        self.graph_plot.set_title('graphs', fontsize='x-small')
        self.graph_plot.plot(x, y, label='exact solution',)

    def plot_euler(self):
        y = self.euler_method.y
        x = self.grid.x
        self.graph_plot.plot(x, y, label="euler's method")

    def plot_improved_euler(self):
        y = self.ie_method.y
        x = self.grid.x
        self.graph_plot.plot(x, y, label="improved euler's method")

    def plot_runge_kutta(self):
        y = self.rk_method.y
        x = self.grid.x
        self.graph_plot.plot(x, y, label="runge-kutta method")

    def plot_euler_le(self):
        y = self.euler_method.le
        x = self.grid.x
        self.le_plot.plot(x, y, label="euler's method")

    def plot_improved_euler_le(self):
        y = self.ie_method.le
        x = self.grid.x
        self.le_plot.plot(x, y, label="improved euler's method")

    def plot_runge_kutta_le(self):
        y = self.rk_method.le
        x = self.grid.x
        self.le_plot.plot(x, y, label="runge-kutta method")

    def plot_euler_ge(self):
        y = self.euler_method.ge
        x = self.grid.x
        self.ge_plot.plot(x, y, label="euler's method")

    def plot_improved_euler_ge(self):
        y = self.ie_method.ge
        x = self.grid.x
        self.ge_plot.plot(x, y, label="improved euler's method")

    def plot_runge_kutta_ge(self):
        y = self.rk_method.ge
        x = self.grid.x
        self.ge_plot.plot(x, y, label="runge-kutta method")

    def plot_euler_total_error(self):
        y = self.em_total_error
        x = np.arange(self.ti, self.tf + 1)
        self.te_plot.plot(x, y, label="euler's method")

    def plot_iem_total_error(self):
        y = self.iem_total_error
        x = np.arange(self.ti, self.tf + 1)
        self.te_plot.plot(x, y, label="improved euler's method")

    def plot_rkm_total_error(self):
        y = self.rkm_total_error
        x = np.arange(self.ti, self.tf + 1)
        self.te_plot.plot(x, y, label="runge-kutta method")


class View:

    def __init__(self):
        self.init = tk.Tk()
        self.init.title('the best solver')

        tk.Label(self.init, text='x0: ').grid(row=1, column=1)
        tk.Label(self.init, text='y0: ').grid(row=2, column=1)
        tk.Label(self.init, text='X: ').grid(row=3, column=1)
        tk.Label(self.init, text='N: ').grid(row=4, column=1)
        tk.Label(self.init, text='starting N for TE: ').grid(row=5, column=1)
        tk.Label(self.init, text='finishing N for TE: ').grid(row=6, column=1)

        self.x0_textbox = tk.Entry(self.init)
        self.x0_textbox.grid(row=1, column=2)
        self.y0_textbox = tk.Entry(self.init)
        self.y0_textbox.grid(row=2, column=2)
        self.x_textbox = tk.Entry(self.init)
        self.x_textbox.grid(row=3, column=2)
        self.n_textbox = tk.Entry(self.init)
        self.n_textbox.grid(row=4, column=2)
        self.n_s_textbox = tk.Entry(self.init)
        self.n_s_textbox.grid(row=5, column=2)
        self.n_f_textbox = tk.Entry(self.init)
        self.n_f_textbox.grid(row=6, column=2)

        self.em = tk.IntVar()
        tk.Checkbutton(self.init, text="Euler's method", variable=self.em).grid(row=2, column=11)
        self.iem = tk.IntVar()
        tk.Checkbutton(self.init, text="improved Euler's method", variable=self.iem).grid(row=3, column=11)
        self.rkm = tk.IntVar()
        tk.Checkbutton(self.init, text="Runge-Kutta method", variable=self.rkm).grid(row=4, column=11)
        self.le = tk.IntVar()
        tk.Checkbutton(self.init, text="local errors", variable=self.le).grid(row=5, column=11)
        self.ge = tk.IntVar()
        tk.Checkbutton(self.init, text="global errors", variable=self.ge).grid(row=6, column=11)


controller = Controller()
'''
self.error_plot = self.figure.add_subplot(2, 1, 2)
        self.error_plot.set_title('errors')
        self.canvas = FigureCanvasTkAgg(self.figure, self.view.init)
        self.canvas.get_tk_widget().grid(row=1, column=3, rowspan=10)
'''