#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:21:00 2020

@author: nico
"""
import imageio
import matplotlib.pyplot as plt 
# Tanto imageio como matplotlib tienen funciones imread. La de imageio carga enteros entre 0 y 255,
# mientras que matplotlib carga entre 0 y 1. Pueden usar cualquiera de las dos.

imagen = imageio.imread('YOUNG_ESTATICO/captura1.png')
imagen.shape
imagen1 = imagen[:, :, 0]
imagen1.shape
plt.figure()
plt.imshow(imagen1)
plt.colorbar()

plt.figure()

plt.plot(imagen1[480, :])

plt.show()










