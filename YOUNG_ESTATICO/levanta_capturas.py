import imageio
import matplotlib.pyplot as plt, numpy as np
# Tanto imageio como matplotlib tienen funciones imread. La de imageio carga enteros entre 0 y 255,
# mientras que matplotlib carga entre 0 y 1. Pueden usar cualquiera de las dos.
imagen = imageio.imread('captura1.png')
imagen.shape # R, G, B. Son
for i in range(3):
    imagen1 = imagen[:, :, 3] - imagen[:, :, i] 
    fig, ax = plt.figure(), plt.imshow(imagen1)
    fig.colorbar(ax)
    fig.show()

# Deteccion de bordes:

# Núcleo de convolución
# Operador de Sobel
s = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])