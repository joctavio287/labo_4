import matplotlib.pyplot as plt, numpy as np
from funciones import *
import ast

masas_2 = {'base':14.6624,1:2.1160,2:2.0399,3:1.6966,4:10.046,5:2.1535,6:5.0952,7:20.0350,10:4.7474,11:9.9163,12:2.1593,15:2.1468}
imagenes_2_grana = {'164151598': 'fondo','171219177': 0,'171340262': masas_2['base'],'171442093': masas_2['base']+masas_2[3],'172921941': masas_2['base']+masas_2[10],'172631960': masas_2['base']+masas_2[11],'171539519': masas_2['base']+masas_2[3]+masas_2[5],'172723255': masas_2['base']+masas_2[11]+masas_2[4],'171740093': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15],'171839967': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7],'171921081': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6],'172116591': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1],'172238962': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2],'172421706': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2]+masas_2[12]}
mediciones_2_grana = {key:None for key in imagenes_2_grana.keys()}
nombres_2_grana = list(imagenes_2_grana.keys())
colormap = {0:plt.cm.Reds_r, 1:plt.cm.Greens_r , 2:plt.cm.Blues_r}

imagenes_2_grana[nombres_2_grana[5]]
# Gráfico tricolor
l = 5
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')
 
# with plt.style.context('seaborn-whitegrid'):
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (8,4))
for i in range(3):
    axs[i].imshow(imagen[:,:,0][100:550,600:820], cmap = colormap[i])
    axs[i].set_axis_off()
    # axs[i].grid(color = 'white')
    # axs[i].set_xticks(np.arange(0,220,10))
    # axs[i].set_yticks(np.arange(0,450,10))
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])
fig.subplots_adjust( 
left  = 0.0001,  # the left side of the subplots of the figure
right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
bottom = 0.01,   # the bottom of the subplots of the figure
top = 0.99,      # the top of the subplots of the figure
wspace = 0.05,   # the amount of width reserved for blank space between subplots
hspace = 0.00001)   # the amount of height reserved for white space between subplots
fig.show()
# fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/colores_2.png', dpi=1200)

# Grafico escala
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + '164151598' + '.jpg')
 
# with plt.style.context('seaborn-whitegrid'):
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (8,4))
for i in range(3):
    axs[i].imshow(imagen[:,:,0][90:390,310:430], cmap = colormap[i])
    # axs[i].imshow(imagen)
    # axs[i].set_axis_off()
    axs[i].grid(color = 'white')
    axs[i].set_xticks(np.arange(0,120,10))
    axs[i].set_yticks(np.arange(0,300,10))
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])
fig.subplots_adjust( 
left  = 0.0001,  # the left side of the subplots of the figure
right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
bottom = 0.01,   # the bottom of the subplots of the figure
top = 0.99,      # the top of the subplots of the figure
wspace = 0.05,   # the amount of width reserved for blank space between subplots
hspace = 0.00001)   # the amount of height reserved for white space between subplots
fig.show()
# fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/escala_2.png', dpi=1200)


# Grafico escala sola
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + '164151598' + '.jpg')
 
# with plt.style.context('seaborn-whitegrid'):
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (8,4))
axs.imshow(imagen[90:390,310:430])#, cmap = colormap[i])
axs.grid(color = 'white')
axs.set_xticks(np.arange(0,120,10))
axs.set_yticks(np.arange(0,300,10))
axs.set_xticklabels([])
axs.set_yticklabels([])
fig.subplots_adjust( 
left  = 0.0001,  # the left side of the subplots of the figure
right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
bottom = 0.01,   # the bottom of the subplots of the figure
top = 0.99,      # the top of the subplots of the figure
wspace = 0.05,   # the amount of width reserved for blank space between subplots
hspace = 0.00001)   # the amount of height reserved for white space between subplots
fig.show()
fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/escala_1.png', dpi=1200)

# Grafico perfil escala
l = 5
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
    axs[0].set_xlabel('Píxeles eje horizontal', fontsize = 12)
    axs[0].set_ylabel('Píxeles eje vertical', fontsize = 12)
    axs[0].imshow(imagen[:,:,0][90:390,310:430], cmap = colormap[0])
    axs[0].set_xticks(np.arange(0,120,10))
    axs[0].set_xticklabels([])
    axs[0].set_yticks(np.arange(0,300,10))
    axs[0].set_yticklabels([])    
    axs[0].grid('white')
    axs[1].plot(imagen[:,:, 0][90:390,371:372].sum(axis = 1))
    axs[1].set_ylabel('Suma de píxeles sobre el eje horizontal', fontsize = 12)
    axs[1].set_xlabel('Píxeles eje vertical', fontsize = 12)
    # clicks = plt.ginput(n = -1, timeout = -1)
    # for i,j  in clicks:
    #     plt.plot(i,j, '.', color = 'r')
    axs[1].legend([r'$\propto$ Intensidad lumínica', 'Mínimos relativos'], loc = 'best', fontsize = 10)
    fig.subplots_adjust( 
    left  = 0.0001,  # the left side of the subplots of the figure
    right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    bottom = 0.12,   # the bottom of the subplots of the figure
    top = 0.99,      # the top of the subplots of the figure
    wspace = 0.00015,   # the amount of width reserved for blank space between subplots
    hspace = 0.075)   # the amount of height reserved for white space between subplots

    fig.show()
    fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/escala_4.png', dpi=1200)

# Patron

# Gráfico perfil:
l = 5
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
    axs[0].set_xlabel('Píxeles eje horizontal', fontsize = 12)
    axs[0].set_ylabel('Píxeles eje vertical', fontsize = 12)
    axs[0].imshow(imagen[:,:, 1][100:550,600:800], cmap = colormap[1])
    axs[0].set_xticks(np.arange(0,200,10))
    axs[0].set_xticklabels([])
    axs[0].set_yticks(np.arange(0,450,10))
    axs[0].set_yticklabels([])    
    axs[0].grid('white')
    axs[1].plot(imagen[:,:, 1][100:550,706:708].sum(axis = 1))
    axs[1].set_ylabel('Intensidad columna del patrón', fontsize = 12)
    axs[1].set_xlabel('Píxeles eje vertical', fontsize = 12)
    # clicks = plt.ginput(n = -1, timeout = -1)
    # for i,j  in clicks:
    #     plt.plot(i,j, '.', color = 'r')
    # axs[1].legend([r'$\propto$ Intensidad lumínica', 'Mínimos relativos'], loc = 'best', fontsize = 10)
    fig.subplots_adjust( 
    left  = 0.0001,  # the left side of the subplots of the figure
    right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    bottom = 0.12,   # the bottom of the subplots of the figure
    top = 0.99,      # the top of the subplots of the figure
    wspace = 0.00015,   # the amount of width reserved for blank space between subplots
    hspace = 0.075)   # the amount of height reserved for white space between subplots

    fig.show()
    fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/haciendo.png', dpi=1200)

# Muchos perfiles


imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (8,4), sharex = True)
    fig.supylabel('Suma de píxeles sobre el eje horizontal', fontsize = 10)
    fig.supxlabel('Píxeles eje vertical', fontsize = 10)
    axs = axs.flatten()
    imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[2] + '.jpg')
    axs[0].plot(imagen[:,:, 1][100:550,650:770].sum(axis = 1))
    # axs[0].set_ylabel('Suma de píxeles sobre el eje horizontal', fontsize = 12)
    # axs[0].set_xlabel('Píxeles eje vertical', fontsize = 12)
    axs[0].legend([f'Masa: {np.round(imagenes_2_grana[nombres_2_grana[2]],4)} g'], loc = 'best', fontsize = 10)

    imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[5] + '.jpg')
    axs[1].plot(imagen[:,:, 1][100:550,650:770].sum(axis = 1))
    # axs[1].set_ylabel('Suma de píxeles sobre el eje horizontal', fontsize = 12)
    # axs[1].set_xlabel('Píxeles eje vertical', fontsize = 12)
    axs[1].legend([f'Masa: {np.round(imagenes_2_grana[nombres_2_grana[5]],4)} g'], loc = 'best', fontsize = 10)

    imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[7] + '.jpg')
    axs[2].plot(imagen[:,:, 1][100:550,650:770].sum(axis = 1))
    # axs[2].set_ylabel('Suma de píxeles sobre el eje horizontal', fontsize = 12)
    # axs[2].set_xlabel('Píxeles eje vertical', fontsize = 12)
    axs[2].legend([f'Masa: {np.round(imagenes_2_grana[nombres_2_grana[7]],4)} g'], loc = 'best', fontsize = 10)

    imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[13] + '.jpg')
    axs[3].plot(imagen[:,:, 1][100:550,650:770].sum(axis = 1))
    # axs[3].set_ylabel('Suma de píxeles sobre el eje horizontal', fontsize = 12)
    # axs[3].set_xlabel('Píxeles eje vertical', fontsize = 12)
    axs[3].legend([f'Masa: {np.round(imagenes_2_grana[nombres_2_grana[13]],4)} g'], loc = 'best', fontsize = 10)
    fig.subplots_adjust( 
    left  = 0.1,  # the left side of the subplots of the figure
    right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    bottom = 0.12,   # the bottom of the subplots of the figure
    top = 0.99,      # the top of the subplots of the figure
    wspace = 0.25,   # the amount of width reserved for blank space between subplots
    hspace = 0.35)   # the amount of height reserved for white space between subplots

    fig.show()
    fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/perfiles.png', dpi=1200)