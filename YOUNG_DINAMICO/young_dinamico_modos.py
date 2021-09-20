from sympy import symbols, solve, nsolve, cos, cosh, sin, sinh, exp, lambdify, latex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
# Para trabajar simbolicamente:
k, y, x = symbols("k y x", real = True)

# Longitud de la barra:
L = 5

# Calculo numericamente los valores que puede tomar k (frecuencia espacial):
results = [] 
for x_0 in np.arange(0, 100, 1/L):
    try:
        result = np.round(float(nsolve(cos(k*L)*cosh(k*L) + 1, k, x_0)), 5)
        if result not in results:
            results.append(result)
    except:
        pass

# Estos son los valores que puede tomar k:
results = np.array(results)

# Parte espacial de la solucion:
y = sin(k*x)-sinh(k*x)-((sin(k*L) + sinh(k*L))/(cos(k*L) + cosh(k*L)))*(cos(k*x)-cosh(k*x))

# Grafico los modos encontrados más arriba:

with plt.style.context('seaborn-whitegrid'):
    
    fig, ax = plt.subplots(figsize = (12, 6))
    plt.subplots_adjust(left = .25, bottom = .25)

    K = results[0]
    x_vals = np.linspace(0, L, 1000)

    # lamdify is a function that converts sympy expressions to numeric modules. In this case numpy.
    lam_x = lambdify([x, k], y, modules = ['numpy']) # The first agument must be an iterable
    l, = plt.plot(x_vals, lam_x(x_vals, K), linewidth = 2)

    ax.margins(x = 0)
    ax.set_xlim([-1, L])
    ax.set_ylim([-2.75, 2.75])
    ax.set_xlabel(r'$x$', fontsize = 16)
    ax.set_ylabel(r'${}$'.format('y(x)'), fontsize = 16)
    axcolor = 'lightgoldenrodyellow'
    axk = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor = axcolor) # left, bottom, width, height

    sk = Slider(axk,
                 label = 'k: frecuencia espacial',
                 valmin = results[0], 
                 valmax = results[-1], 
                 valinit = results[0], 
                 valstep = results)

    def update(val):
        k = sk.val
        l.set_ydata(lam_x(x_vals, k))
        fig.canvas.draw_idle()
    sk.on_changed(update)
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    def reset(event):
        sk.reset()
    button.on_clicked(reset)
    plt.show()