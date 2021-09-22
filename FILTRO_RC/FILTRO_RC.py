# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:05:28 2021

@author: jocta & sofi
"""
print(__doc__)
import time
import pyvisa as visa
import numpy as np
from matplotlib import pyplot as plt

# =============================================================================
# Chequeo los instrumentos que están conectados por USB
# =============================================================================

rm = visa.ResourceManager()
instrumentos = rm.list_resources()  

# =============================================================================
# Printear la variable instrumentos para chequear que están enchufados en el mismo orden.
# Sino hay que rechequear la numeración de gen y osc.
# =============================================================================

gen = rm.open_resource(instrumentos[0])

osc = rm.open_resource(instrumentos[1])

# =============================================================================
# Si vas a repetir la adquisicion muchas veces sin cambiar la escala es util definir una
# funcion que mida y haga las cuentas. 'WFMPRE': waveform preamble.
# =============================================================================

def definir_medir(inst, channel):
    xze, xin, yze, ymu, yoff = inst.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?;', separator = ';')
    # Creamos una function auxiliar
    def _medir(channel):
        """
        Adquiere los datos del canal channel (es un int) y los devuelve en un array de numpy
        La siguiente linea puede no funcionar, de ser así comentarla y toma por defecto el 
        channel 1
        """
        inst.write('DATa:SOUrce CH' + str(channel)) #
        datos = inst.query_binary_values('CURV?', datatype = 'B', container = np.array)
        data = (datos - yoff)*ymu + yze
        tiempo = xze + np.arange(len(data)) * xin
        return tiempo, data
    # Devolvemos la funcion auxiliar que "sabe" la escala
    return _medir

# =============================================================================
# Entonces ahora, cuando quiera ver la pantalla del osciloscopio (siempre y cuando esté bien
# seteado) se puede correr las siguientes lineas
# =============================================================================

medir = definir_medir(osc, 1)

N = 10 # número de secuencias de la pantalla
for n in range(N):
    tiempo, data = medir()
    plt.figure()
    plt.plot(tiempo, data);
    plt.xlabel('Tiempo [s]');
    plt.ylabel('Tensión [V]');
    time.sleep(1)

#%% PARA MEDIR EL FILTRO RC
# =============================================================================
# Seteo las frecuencias que quiero [Hz] y tomo mediciones percatándome de ajustar el 
# osciloscopio por c/ medicion. Para escala de tiempos por división:
#     'HORizontal:MAIn:SCAle <escala>', donde <escala> está en segundos 
# <escala> puede ser: 1  ,  2.5  , 5    con un exponente
# Ej: 2.5E-3 son 2.5 ms
#       5E-6 son   5 us
# =============================================================================

channel_1 = []
channel_2 = []

# =============================================================================
# Estoy asumiendo que el generador está en High Z y que alimento con Vp2p.
# La siguiente lista está para fijar la tensión del osci más adelante, dentro
# de la función.
# =============================================================================

graduacion_vertical = [50e-3, 100e-3, 200e-3, 500e-3, .1e1, .2e1, .5e1]


for tension in np.arange(1, 4):
    """
    Seteo la resolucion vertical de los dos canales, asumiendo que ch2 es la de
    referencia. Los valores posibles son {50mV, 100mV, 200mV, 500mV, 1V, 2V, 5V}
    y algunos más chicos o grandes (pero no creo que usemos)
    """
    gen.write(f'VOLT {tension}')
    """
    La escala vertical la seteo como para que en 6 cuadraditos entre las dos
    crestas de la onda. Para esto creo un diccionario que tiene como values
    c/graduación posible de la escala vertical y como keys la distancias de
    dichas graduaciones a la deseada (6 cuadraditos entre las dos crestas). 
    Elijo la graduación que tiene menor distancia a la deseada.
    """
    aux = {}
    for grad in graduacion_vertical:
        aux[abs(grad-tension/6)] = grad
    escala = aux[min(aux.keys())]
    
    # No sé donde meter los time sleep
    time.sleep(1)
    osc.write(f'CH{1}:SCAle {escala:3.1E}')    
    osc.write(f'CH{2}:SCAle {escala:3.1E}')    
    time.sleep(1)
    
    for freq in np.logspace(start = 1, stop = 4, num = 20, base = 10):
        time.sleep(1/freq)
        gen.write(f'FREQ {freq}')
        """
        Seteo la escala temporal como para que entren 4 períodos (T = 1/freq)
        en el ancho completo de la pantalla (que tiene 10 cuadraditos)
        El valor al cual queda fijado la escala es el más cercano a los valores 
        predefinidos.
        """
        escala_t = (4/10)*(1/freq)
        osc.write(f'HORizontal:MAIn:SCAle {escala_t:3.1E}')
        time.sleep(1) # entiendo si va o no
        osc.write('MEASUrement:IMMed:SOU CH1; TYPe PK2k')
        channel_1.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])
        osc.write('MEASUrement:IMMed:SOU CH2; TYPe PK2k')
        channel_2.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])

# =============================================================================
# Lo que sigue es para guardar lo medido en c/ canal. Importante notar que las 
# mediciones estan en Volts y que se tomaron tres sets de tensiones: {1,2,3}V.
# =============================================================================

# np.savetxt('canal1.txt', channel_1)
# np.savetxt('canal2.txt', channel_2)

# %% Ploteamos los datos para ver que onda
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Importamos lo guardado arriba. Cambiar el path acorde a dónde queden
# guardados los archivos
# =============================================================================

file1 = open(f'C:/Users/jocta/Google Drive/Laboratorio 4/FILTRO RC/canal1.txt', 'r')
file2 = open(f'C:/Users/jocta/Google Drive/Laboratorio 4/FILTRO RC/canal2.txt', 'r')
canal1 = file1.readlines()
canal2 = file2.readlines()

# =============================================================================
# Transformamos a float los datos y después convertimos np
# =============================================================================

channel1 = [float(i) for i in canal1]
channel2 = [float(i) for i in canal2]
   
channel_1 = np.array(channel1)  # Convertimos la lista en un array para poder calcular       
channel_2 = np.array(channel2)  # la atenuacion

transferencia = abs(channel_1/channel_2)
atenuacion = 20*np.log10(transferencia)

with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
    axs[0].plot(np.logspace(start = 1, stop = 4, num = 20, base = 10), transferencia[:20], label = '1V de entrada')
    axs[0].plot(np.logspace(start = 1, stop = 4, num = 20, base = 10), transferencia[20:40], label = '1V de entrada')
    axs[0].plot(np.logspace(start = 1, stop = 4, num = 20, base = 10), transferencia[40:], label = '3V de entrada')
    axs[0].set_title('Transferencia')
    axs[0].set_xlabel('Frecuancia [Hz]')
    axs[0].set_ylabel('Transferencia')
    axs[0].grid('True')
    axs[0].legend()
 
    axs[1].plot(np.logspace(start = 1, stop = 4, num = 20, base = 10), atenuacion[:20], label = '1V de entrada')
    axs[1].plot(np.logspace(start = 1, stop = 4, num = 20, base = 10), atenuacion[20:40], label = '1V de entrada')
    axs[1].plot(np.logspace(start = 1, stop = 4, num = 20, base = 10), atenuacion[40:], label = '3V de entrada')
    axs[1].set_xscale('log')
    axs[1].set_title('Atenuación')
    axs[1].set_xlabel('Frecuencia [Hz]')
    axs[1].set_ylabel('Atenuación [dB]')
    axs[1].grid('True')
    axs[1].legend()
fig.tight_layout()
fig.show()
fig.savefig('Google Drive/Laboratorio 4/FILTRO RC/Graficos_combinados.png', dpi = 1100)