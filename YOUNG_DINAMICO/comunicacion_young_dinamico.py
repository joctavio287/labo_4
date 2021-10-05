# =============================================================================
# Puede haber dos DAQS (Data acquisition systems) posibles:
# NI-DAQmx Python Documentation: https://nidaqmx-python.readthedocs.io/en/latest/index.html
# NI USB-621x User Manual: https://www.ni.com/pdf/manuals/371931f.pdf
# =============================================================================

import visa, time, math, nidaqmx, pandas as pd, numpy as np, matplotlib.pyplot as plt

# =============================================================================
# Chequeo los instrumentos que están conectados por USB
# =============================================================================

rm = visa.ResourceManager()
instrumentos = rm.list_resources()  
# osc = rm.open_resource(instrumentos[0]) # modificar por numero

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


# =============================================================================
# OSICLOSCOPIO: definimos la función para medir y la usamos. Está pensado para golpear la barra
# y frenar la pantalla del osci cuando se vea una buena cantidad de oscilaciones:
# =============================================================================

medir = definir_medir(osc, 1)

tiempo, data = medir(1)
plt.figure(0)
plt.plot(tiempo, data)
plt.xlabel('Tiempo [s]')
plt.ylabel('Tensión [V]')
plt.show()

# Agarramos la medición que tomamos de la pantalla y creamos un DataFrame con dos columnas: tiempo y tensión.
df = pd.DataFrame({'tiempo': tiempo.tolist() ,'tension': data.tolist()})

# 'path' es el camino donde guardamos las mediciones (copiar del explorador de windows de la carpeta en la cual
# estes trabajando). El ultimo troncho después de la barra es el nombre con el que guardamos el archivo. Adentro
# de format hay que PONER MANUALMENTE EL NUMERO DE LA MEDICION PARA NO PISAR VIEJAS (OJALDRE).
path = 'C:/repos/labo_4/YOUNG_DINAMICO/medicio_osci{}.csv'.format()
df.to_csv(path)

# =============================================================================
# Para setear (y preguntar) el modo y rango de un canal analógico:
# 'nidaqmx.Task()' is a method to create a task.
#'.ai_channels' gets the collection of analog input (ai) channels for this task.
#"Dev1/ai1" is the input channel.
# =============================================================================

with nidaqmx.Task() as task:  
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev1/ai1", max_val = 10, min_val = -10)
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
    cant_puntos = duracion*fs    
    with nidaqmx.Task() as task:
        modo = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo)
        task.timing.cfg_samp_clk_timing(fs, samps_per_chan = cant_puntos,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE)
        datos = task.read(number_of_samples_per_channel = nidaqmx.constants.READ_ALL_AVAILABLE)
    datos = np.asarray(datos)    
    return datos


# =============================================================================
# DAQ en modo finito (en el cual ajustamos nosotres las frecuencia de sampleo y el tiempo
# de medición). Creo que va a ser más conveniente que el otro. El chiste es que le ponemos
# un tiempo largo (10s por ej), le pegas a la barra y dejas que oscile. Cuando termine vas
# a tener una señal larga que va a tener el tiempo previo a la pegada que la descartaremos
# cuando analicemos los datos:
# =============================================================================
duracion = 10 # duración de la medición en segundos. Si se cambia a lo largo de las mediciones es un
# numero que hay que guardar para cuando procesemos los datos.

fs = 250000 # frecuencia de muestreo (Hz) del daq (tal vez conviene bajarla, dado que 
# la frecuencia mínima de sampleo es mucho menor y no necesitamos una transformada tan fina)
datita = medir_daq(duracion, fs)

plt.figure(1)
plt.plot(duracion, datita)
plt.xlabel('Tiempo [s]')
plt.ylabel('Tensión [V]')
plt.show()

# Para guardar la datita no sabría como hacer porque no sé en qué formato viene. Probar type(datita),
# si es una lista, array o algo por el estilo se puede copiar el guardado de arriba. Tal vez, si son 
# demasiados datos, conviene guardar de otra forma que no sea .csv. También esta la opción de bajar la
# frecuencia de muestreo (no descartar, puede ser muy util, sobre todo para tomar mediciones 
# preliminares)

# =============================================================================
# Otra opción: medición continua. La idea es la misma, la diferencia es que cuando vos frenas
# la task, va a dejar de medir. Es medio desprolijo, creo que conviene usar lo de arriba. Ade
# más no tenemos con precisión cuál es la escala horizontal
# =============================================================================
task = nidaqmx.Task()
modo = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo)
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

plt.figure(2)
plt.plot(datita)
plt.xlabel('Tiempo [s]')
plt.ylabel('Tensión [V]')
plt.show()