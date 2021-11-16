import numpy as np, matplotlib.pyplot as plt, pandas as pd

# ===================================================================================
# Creo un diccionario para asociar masas a número de imagen. Las masas estaban
# númeradas. El diccionario masas se lee 'numero':'masa' [g]. El error de las 
# masas medidas es de 0.0002 g.
# ===================================================================================

######################## DIA1 #########################

masas_1 = {'base':14.6570,7:20.0345,10:4.7468,4:2.1346,1:2.1157,6:5.0945,11:9.9158,14:2.1471,8:2.1254}
imagenes_1 = {
'solo': 0,
'base': masas_1['base'],
'173602775': masas_1['base']+masas_1[4],
'173702132': masas_1['base']+masas_1[4]+masas_1[8],
'173743569': masas_1['base']+masas_1[4]+masas_1[8]+masas_1[1],
'174106806': masas_1['base']+masas_1[4]+masas_1[8]+masas_1[1]+masas_1[14],
'174200450': masas_1['base']+masas_1[4]+masas_1[8]+masas_1[1]+masas_1[14]+masas_1[10],
'174321652': masas_1['base']+masas_1[4]+masas_1[8]+masas_1[1]+masas_1[14]+masas_1[10]+masas_1[11],
'174637541': masas_1['base']+masas_1[4]+masas_1[8]+masas_1[1]+masas_1[14]+masas_1[10]+masas_1[11]+masas_1[6],
'174820367': masas_1['base']+masas_1[4]+masas_1[8]+masas_1[1]+masas_1[14]+masas_1[10]+masas_1[11]+masas_1[6]+masas_1[7]}

mediciones_1 = {key: None for key in imagenes_1.keys()}
nombres_1 = list(mediciones_1.keys())

######################## DIA2 #########################

masas_2 = {'base':14.6624,1:2.1160,2:2.0399,3:1.6966,4:10.046,5:2.1535,
6:5.0952,7:20.0350,10:4.7474,11:9.9163,12:2.1593,15:2.1468}

# Imagenes sacadas con grana angular:
imagenes_2_grana = {
'164151598': 'fondo',
'171219177': 0,
'171340262': masas_2['base'],
'171442093': masas_2['base']+masas_2[3],
'172921941': masas_2['base']+masas_2[10],
'172631960': masas_2['base']+masas_2[11],
'171539519': masas_2['base']+masas_2[3]+masas_2[5],
'172723255': masas_2['base']+masas_2[11]+masas_2[4],
'171740093': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15],
'171839967': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7],
'171921081': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6],
'172116591': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1],
'172238962': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2],
'172421706': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2]+masas_2[12],
}

# Imagenes sacadas sin grana angular
imagenes_2_estandar = {
'173134073': 'fondo',
'173805029': 0,
'173855360': masas_2['base'],
'174023733': masas_2['base']+masas_2[3],
'174120856': masas_2['base']+masas_2[3]+masas_2[5],
'174202073': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15],
'174548787': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7],
'174632748': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6],
'174707741': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1],
'174818967': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2],
'174952519': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2]+masas_2[12] #1.5 mm
}

mediciones_2_grana = {key:None for key in imagenes_2_grana.keys()}
mediciones_2_estandar = {key: None for key in imagenes_2_estandar.keys()}
nombres_2_grana, nombres_2_estandar = list(imagenes_2_grana.keys()), list(imagenes_2_estandar.keys())

# ===================================================================================
# Vamos a trabajar manualmente sobre cada imagen, para eso vamos a copy pastear un
# trozo de texto por c/ medicion.
# ===================================================================================

colormap = {0:plt.cm.Reds_r, 1:plt.cm.Greens_r , 2:plt.cm.Blues_r}


######################## DIA2-GRANA #########################

# En esta sección debería haber 14 mediciones efectuadas:

# Medicion '164151598'. Este es el fondo. Entonces el resultado de esta medicion es la escala:
l = 0
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 0].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:400,360:400].sum(axis = 1))
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Regla de tres para sacar las dimensiones de un pixel:
pixel_cm = .5/media
error_pixel_cm = .01/(media*np.sqrt(len(deltas)))

mediciones_2_grana[nombres_2_grana[l]] = pixel_cm, error_pixel_cm
mediciones_2_grana.values()

# Medicion '171219177'. Esta sería sin peso (ni siquiera la base):
l = 1
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 0].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[120:520,680:730].sum(axis = 1))
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))

# Medicion '171340262'. Esta sería con 14.6624 g:
l = 2
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 1].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:580, 640:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas)) 


# Medicion '171442093'. Esta sería con 16.359:
l = 3
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')
# imagenes_2_grana[nombres_2_grana[l]]

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[0:700, 500:900, i], cmap = colormap[i])
    axs[i].set_axis_off()
fig.subplots_adjust( 
    left  = 0.01,  # the left side of the subplots of the figure
    right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    bottom = 0.01,   # the bottom of the subplots of the figure
    top = 0.99,      # the top of the subplots of the figure
    wspace = 0.00015,   # the amount of width reserved for blank space between subplots
    hspace = 0.075) 
fig.show()
fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/tricolor.png', dpi = 1200)
# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 1].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:620, 660:760].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))

# Medicion '172921941'. Esta sería con 19.4098:
l = 4
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')
# imagenes_2_grana[nombres_2_grana[l]]

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 2].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:600, 660:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas)) 

# Medicion '172631960'. Esta sería con 24.578699999999998:
l = 5
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 1].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:600, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas)) 


# Medicion '171539519'. Esta sería con 18.512500000000003:
l = 6
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 1].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:600, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()

# Medicion '172723255'. Esta sería con 34.6247:
l = 7
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 2].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:600, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()


# Medicion '171740093'. Esta sería con 20.6593:
l = 8
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 2].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[100:600, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()


# Medicion '171839967'. Esta sería con 40.6943:
l = 9
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 2].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[120:480, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()


# Medicion '171921081'. Esta sería con 45.7895:
l = 10
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 2].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[120:480, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()


# Medicion '172116591'. Esta sería con 47.9054999999996:
l = 11
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 2].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[120:480, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()

# Medicion '172238962'. Esta sería con 49.9454:
l = 12
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 1].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[120:480, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()


# Medicion '172421706'. Esta sería con 52.1047:
l = 13
# nombres_2_grana[l],imagenes_2_grana[nombres_2_grana[l]]
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

# Veo los tres colores (RGB) y decido cuál es el que mejor se vé para analizar: 
fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Lo elijo y selecciono el rango para hacer el bineo:
imagen = imagen[:, :, 2].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
# Seteo los ejes muy densos para poder encontrar los margenes propicios:
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[120:480, 680:740].sum(axis = 1))
# plt.show(block = False)
# Valores negativos en n y timeout es igual a indefinidos (para cortar usar click con la ruedita):
clicks = fig.ginput(n = -1, timeout = -1)

# Marque pasos de (.50 +-.01) cm
clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

# Media sería la cantidad de pixeles que representan el paso de .5
media = np.array(deltas).mean()

# Guardo la medición en pixeles
mediciones_2_grana[nombres_2_grana[l]] = media*pixel_cm, error_pixel_cm/np.sqrt(len(deltas))
mediciones_2_grana.values()

# ===========================================================================
# Guardo los datos para no perder todas las mediciones:
# ===========================================================================
index = pd.Index(mediciones_2_grana.keys(), name = 'Foto')
df = pd.DataFrame(data = {'Mediciones(valor, error)[cm]':list(mediciones_2_grana.values())}, index = index, columns = ['Mediciones(valor, error)[cm]'])
df.to_csv('C:/repos/labo_4/YOUNG_ESTATICO/mediciones_grana_angular_3.csv')
