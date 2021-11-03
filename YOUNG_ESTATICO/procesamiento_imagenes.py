import numpy as np, matplotlib.pyplot as plt
from scipy import ndimage, signal

# ===================================================================================
# Creo un diccionario para asociar masas a número de imagen. Las masas estaban
# númeradas. El diccionario masas se lee 'numero':'masa' [g]. El error de las 
# masas medidas es de 0.0002 g.
# ===================================================================================
colormap = {0:plt.cm.Reds_r, 1:plt.cm.Greens_r , 2:plt.cm.Blues_r}
masas = {'base':14.6570,7:20.0345,10:4.7468,4:2.1346,1:2.1157,6:5.0945,11:9.9158,14:2.1471,8:2.1254}
imagenes = {
'solo': 0,
'base': masas['base'],
'173602775': masas['base']+masas[4],
'173702132': masas['base']+masas[4]+masas[8],
'173743569': masas['base']+masas[4]+masas[8]+masas[1],
'174106806': masas['base']+masas[4]+masas[8]+masas[1]+masas[14],
'174200450': masas['base']+masas[4]+masas[8]+masas[1]+masas[14]+masas[10],
'174321652': masas['base']+masas[4]+masas[8]+masas[1]+masas[14]+masas[10]+masas[11],
'174637541': masas['base']+masas[4]+masas[8]+masas[1]+masas[14]+masas[10]+masas[11]+masas[6],
'174820367': masas['base']+masas[4]+masas[8]+masas[1]+masas[14]+masas[10]+masas[11]+masas[6]+masas[7]}

mediciones = {'solo': None, 'base': None, '173602775': None, '173702132': None, '173743569': None, '174106806': None,
 '174200450': None, '174321652': None, '174637541': None, '174820367':None}
nombres = list(mediciones.keys())

# Medicion 'solo':
l = 0
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones - alteradas/' + nombres[l] + '.jpeg')

fig, axs = plt.subplots(nrows = 1, ncols = 3)
for i in range(3):
    axs[i].imshow(imagen[:, :, i])#, cmap = colormap[i])
    axs[i].set_axis_off()
fig.tight_layout()
fig.show()

# Parece que en la imagen roja se vé mejor la intensidad lumínica
imagen = imagen[:, :, 0].copy()
fig = plt.figure()
plt.imshow(imagen)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.get_xlim()
fig.show()

# Bineo sumando todas las columnas de la imagen:
fig = plt.figure()
plt.plot(imagen[:,1120:1260].sum(axis=1))
clicks = fig.ginput(n = 20)#, timeout = 40, show_clicks=True)
fig.show()

clicks = [x for x,y in clicks]
deltas = []
for i in range(1, len(clicks)):
    delta = clicks[i] - clicks[i-1]
    deltas.append(delta)

media = np.array(deltas).mean()
mediciones[nombres[l]] = media

# ===================================================================================
# Creo un diccionario para asociar masas a número de imagen. Las masas estaban
# númeradas. El diccionario masas se lee 'numero':'masa' [g]. El error de las 
# masas medidas es de 0.0002 g.
# ===================================================================================

























# def diff(v):
#     v_diff = []
#     for i in range(1, len(v) - 1):
#         v_diff.append(v[i][0] - v[i - 1][0])
        
#     return v_diff
# v = imagen[:,2,0]
# len(v)
# for i in range(len(v)):
#     i
# #funcion para calibrar la escala del calibre. Toma un vector y devuelve el promedio de las diferencias entre elementos de ese vector
# #junto con su desviación estandar
# def calibrateScale(x):
#     v_diff = []      
#     v_diff = diff(x)
    
#     mm = np.mean(v_diff)
#     mm_err = 2*np.std(v_diff)
#     return mm, mm_err


# #kernel en el codigo de Richi (de donde sale este kernel? Porque usando este kernel la convolucion es tan ruidosa?)
# s = [[1, 2, 1],  
#      [0, 0, 0], 
#      [-1, -2, -1]]


# #cargamos la imagen a analizar
# AA = imageio.imread('image1.png')

# #layer de la imagen como matriz con valores de 0-255
# A = AA[:,:,1]

# #grafica la imagen original
# plt.figure(figsize = (5, 5))
# plt.imshow(A)
# plt.show()

# #calcula la convolucion de la imagen con el kernel (H aplica el filtro en la dirección x y V en la dirección y)
# H = signal.convolve2d(A, s)
# V = signal.convolve2d(A, np.transpose(s))
# R = (H**2 + V**2)**0.5


# #cgrafica dicha convolución
# plt.figure(figsize = (5, 5))
# plt.imshow(R)
# plt.show()


# #se rota 1 grado en la dirección de las agujas del reloj para alinear mejor la imagen
# R2 = ndimage.rotate(R, -1)

# #esta linea modifica el valor de una linea en la imagen
# R2[:,850] = ndimage.maximum(R[:])

# #se grafica la imagen modificada
# plt.figure(figsize = (5, 5))
# plt.imshow(R2)
# plt.show()

# perf2 = R2[100:300, 851]
# plt.figure(figsize = (5, 5))
# plt.plot(perf2)
# plt.show(block = False)
# perf2b = A[100:300, 851]
# plt.figure(figsize = (5, 5))
# plt.plot(perf2b)
# plt.show(block= False)


# #selecciona 10 puntos en alguna figura y guarda las coordenadas (para ser usado en el grafico de perf2b)
# x = plt.ginput(10)


# escala, error_escala = calibrateScale(x)


# # En adelante no es completamente equivalente al codigo Matlab original
# # resta comparar para completar diferencias

# #esta linea binariza la imagen (a todo valor mayor a 55 le asigna un 1 y un 0 a todos los demas valores)
# Rbin = (R2 > 55)

    
# [a,b] = len(R2), len(R2[0])
# x_s = [i/escala for i in range(0, a)]
# y_s = [i/escala for i in range(0, b)]

# #grafica la imagen con la escala correcta. La imagen sale de costado porque el origen de plt.imshow define el origen en
# #otro lado con respecto a plt.contourf
# #plt.contourf les permite graficar cualquier par x, y, z
# plt.figure(figsize = (5, 5))
# plt.contourf(x_s, y_s, np.transpose(Rbin))

# #lineas aisladas
# perf4 = R2[:,100]
# perf5 = R2[:,793]

# plt.figure(figsize = (5, 5))
# plt.plot(x_s, perf4)
# plt.plot(x_s, perf5)



