from re import A
from sympy import symbols, solve, nsolve, cos, cosh, sin, sinh, exp, lambdify, latex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
# Es el path al directorio contenedor de ticks.py
file = sys.path.append("C:/repos/labo_4/") 
from ticks import *
#=======================================================================================
# Voy a trabajar con sympy para encontrar los modos fundamentales de una vara
# en voladizo. Además incluyo un gráfico interactivo para visualizarlos.
#=======================================================================================

# Para trabajar simbolicamente:
k, y, x = symbols("k y x", real = True)

# Longitud de la barra:
L = .38
# L = .316
# Calculo numericamente los valores que puede tomar k (frecuencia espacial):
results = [] 
for x_0 in np.arange(0, 100, 1/L):
    try:
        result = np.round(float(nsolve(cos(k*L)*cosh(k*L) + 1, k, x_0)), 9)
        if (result not in results) and (-result not in results):
            results.append(result)
    except:
        pass

# Estos son los valores que puede tomar k:
results = np.array(results)
re_tex = str(results)
print('Los modos normales son: ' + re_tex.strip('[').strip(']'))


#=======================================================================================
# Lo que sigue es sólo una visualización de los modos.
#=======================================================================================


# Parte espacial de la solucion:
y = sin(k*x)-sinh(k*x)-((sin(k*L) + sinh(k*L))/(cos(k*L) + cosh(k*L)))*(cos(k*x)-cosh(k*x))

# Grafico los modos encontrados más arriba:
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(figsize = (12, 6))
    fig.subplots_adjust(left = .25, bottom = .25)

    # Inicializo el parámetro a variar y tomo los valores de x:
    K = results[0]
    x_vals = np.linspace(0, L, 1000)

    # Lamdify convierte valores de sympy a paquetes numéricos como numpy.
    # El primer argumento debe ser un iterable
    lam_x = lambdify([x, k], y, modules = ['numpy'])
    l, = ax.plot(x_vals, lam_x(x_vals, K), linewidth = 2)

    ax.margins(x = 0)
    ax.set_xlim([-L/6, L])
    ax.set_ylim([-2.75, 2.75])
    ax.set_xlabel(r'$x$', fontsize = 16)
    ax.set_ylabel(r'${}$'.format('y(x)'), fontsize = 16)
    xticks, xticks_labels = multiplos_pi(0, L, L/5, tick_label = 'L', tick_value = L)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)
    axcolor = 'lightgoldenrodyellow'
    
    # Creo el deslizador:
    axk = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor = axcolor) # left, bottom, width, height

    # Lo hago interactivo:
    sk = Slider(axk,
                 label = 'Frecuencia espacial:',
                 valmin = results[0], 
                 valmax = results[-1], 
                 valinit = results[0], 
                 valstep = results) # hago que los pasos sean los que encontre más arriba

    # Una funcion que actualiza los valores para el gráfico que creamos
    def update(val):
        k = sk.val
        l.set_ydata(lam_x(x_vals, k))
        fig.canvas.draw_idle()
    
    sk.on_changed(update)
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color = axcolor, hovercolor = '0.975')
    
    # Un botón para volver a la configuración inicial
    def reset(event):
        sk.reset()
    button.on_clicked(reset)
    fig.show()
