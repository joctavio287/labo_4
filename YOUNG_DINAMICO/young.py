# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:12:37 2021

@author: Publico
"""
# =============================================================================
# Puede haber dos DAQS (Data acquisition systems) posibles:
# NI-DAQmx Python Documentation: https://nidaqmx-python.readthedocs.io/en/latest/index.html
# NI USB-621x User Manual: https://www.ni.com/pdf/manuals/371931f.pdf
# =============================================================================

import visa, time, nidaqmx, pandas as pd, numpy as np, matplotlib.pyplot as plt

# =============================================================================
# Chequeo los instrumentos que están conectados por USB
# =============================================================================

rm = visa.ResourceManager()
instrumentos = rm.list_resources()  
osc = rm.open_resource(instrumentos[0]) # modificar por numero
# =============================================================================
# Para saber el ID de la placa conectada (DevX):
# =============================================================================

system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)

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

medir = definir_medir(osc, 1)

tiempo, data = medir(1)
plt.figure()
plt.plot(tiempo, data);
plt.xlabel('Tiempo [s]');
plt.ylabel('Tensión [V]');
df = pd.DataFrame({'datos': data.tolist()})
df.to_csv('datita5.csv')
for i in range(1,6):
    df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/datita{}.csv'.format(str(i)))
    plt.plot(np.linspace(0,.5,len(df.datos)), df.datos)
    plt.show()
plt.clf()
df.datos[0]
(4/10)*(1/freq)
# =============================================================================
# Para setear (y preguntar) el modo y rango de un canal analógico:
# 'nidaqmx.Task()' is a method to create a task.
#'.ai_channels' gets the collection of analog input (ai) channels for this task.
#"Dev1/ai1" is the input channel.
# =============================================================================

with nidaqmx.Task() as task:  
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev3/ai1", max_val = 10, min_val = -10)
    print(ai_channel.ai_term_cfg)  # specifies the terminal configuration for the channel.
    print(ai_channel.ai_max) # it returns the coerced maximum value that the device can measure with the current settings
    print(ai_channel.ai_min) # idem with minimum	
	

# =============================================================================
# Medicion por tiempo/samples de una sola vez:
# 'cfg_samp_clk_timing' sets the source of the Sample Clock, the rate of the Sample Clock,
# and the number of samples to acquire or generate. Specifies the sampling rate in samples
# per channel per second.
# =============================================================================

def medir_daq(duracion, fs):
    # medir = definir_medir(osc, 1) # preparo la función de medición del osciloscopio
    cant_puntos = duracion*fs    
    with nidaqmx.Task() as task:
        modo = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev3/ai1", terminal_config = modo)
        task.timing.cfg_samp_clk_timing(fs, samps_per_chan = cant_puntos,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE)
        datos = task.read(number_of_samples_per_channel = nidaqmx.constants.READ_ALL_AVAILABLE)
    datos = np.asarray(datos)    
    return datos

# =============================================================================
# Ejemplo:
# =============================================================================

medir = definir_medir(osc, 1)
freq_medicion = 1000 # Hz
N = 1000 # número de secuencias de la pantalla
duracion = 1 # duración de la medición en segundos
fs = 250000 # frecuencia de muestreo (Hz) del daq
datas  = []
for n in range(N):
    tiempo, data = medir()
    datas.append(tiempo,data)
    # plt.figure()
    # plt.plot(tiempo, data);
    # plt.xlabel('Tiempo [s]');
    # plt.ylabel('Tensión [V]');
    time.sleep(1/freq_medicion)


datita = medir_daq(duracion, fs)

# =============================================================================
# Medición continua:
# =============================================================================
task = nidaqmx.Task()
modo = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
task.ai_channels.add_ai_voltage_chan("Dev3/ai1", terminal_config = modo)
task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
task.start()
t0 = time.time()
total = 0
for i in range(10):
    time.sleep(0.1)
    datos = task.read(number_of_samples_per_channel = nidaqmx.constants.READ_ALL_AVAILABLE)           
    t1 = time.time() # time in seconds since the 'epoch': January 1, 1970, 00:00:00 (UTC)
    total = total + len(datos)
    print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0)))    
task.stop()
task.close()

plt.plot(np.linspace(0, 1/fs*len(datos), len(datos)), datos)
# =============================================================================
# Guardo los datos:
# =============================================================================

#señal_de_fondo = np.mean(datos) SEÑAL DE FONDO
laser_de_lleno = np.mean(datos)
df = pd.DataFrame({'daq': y_D, osc: y_O})
source = ''
filename = source + 'mediciones' + '-'.join([str(time.localtime()[i]) for i in np.arange(6)])+ '.csv'
df.to_csv(filename)
