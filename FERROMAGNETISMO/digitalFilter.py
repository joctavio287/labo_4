#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:39:42 2021

@author: lolo
"""

from numpy import * 
from matplotlib import pyplot as plt



# Script para practicar con distintos tipos de algoritmos 
# integradores: primero lo más simple, suma acumulada de datos
# (cumsum) esto tiene el problema de que si hay un offset, 
# la integración va derivando. Se puede solucionar pero tengo
# que restar el valor medio, y para eso preciso toda la señal 
# de antemano
# Después hay filtros recursivos, que se pueden aplicar 
# "en tiempo real", es decir a medida que van llegando los datos


DT  = 1e-5;     #frecuencia de sampleo
t   = arange(0,0.05,DT)

# defino mi "señal": una onda cuadrada
T   = 1/1000;  
x0  = sin(2*pi*1/T*t);    
x   = sign(x0)+0.15;



# lo más simple: suma acumulada

w   = cumsum(x)*DT;

fig, axx = plt.subplots(2,1,   constrained_layout=True , sharex=True)
ax1,ax2  = axx

ax1.plot( t[1:1000] , x[1:1000] )
#plt.xlabel('t[s]')
ax1.set_ylabel('amplitud [V]')

ax2.plot( t[1:1000] , w[1:1000])
ax2.set_xlabel('t[s]')
ax2.set_ylabel('amplitud [V]' )
fig.show()
# esto no funciona como quiero, porque la señal tiene un offset

#  le sacamos el valor medio


w2   = cumsum(x-mean(x))*DT;

plt.figure(33 , constrained_layout=True)

plt.plot( t[1:1000] ,  w[1:1000] )
plt.plot( t[1:1000] , w2[1:1000] )
plt.xlabel('t[s]')
plt.ylabel('amplitud [V]' )
plt.show(block=False)

#
# filtros definidos en forma recursiva:
# frecuencia de corte del pasabajos (el integrador)

RC2   = 1/(2*pi*20);  
# la condicion es que RC2>>T el período de la señal que quiero integrar

# frecuencia de corte del pasaaltos (el DC-block que saca el offset)
RC1=1/(2*pi*2); 

f_hig = 1/RC2
f_low = 1/RC1

# Es decir: el filtro 2 deja pasar frecuencias por debajo de 1/RC2
# y el filtro 1 por arriba de 1/RC1

z   = zeros(len(t))
y   = zeros(len(t))
z2  = zeros(len(t))


for n in range(2,len(t)):
    # sólo un filtro pasa bajos (el integrador)
    y[n]  = x[n]
    z[n]  = z[n-1]*RC2/(RC2+DT)+y[n]*DT/(RC2+DT) 
    
    # le agrego un fitro para bloquear DC, pasa altos:
#    y[n]  = RC1/(RC1+DT)*y[n-1]+RC1/(RC1+DT)*(x[n]-x[n-1]) 
#    z[n]  = z[n-1]*RC2/(RC2+DT)+y[n]*DT/(RC2+DT) 

    # UN INTEGRADOR con opamp "IDEAL" EN CAMBIO, ES ASÍ
#    y[n]  = x[n];
#    z[n]  = z[n-1]-y[n]*DT/RC2;
    


# plt.figure(33 , constrained_layout=True)
plt.figure()

plt.plot( t ,  x )
plt.plot( t ,  z )
plt.plot( t ,  y )
plt.xlabel('t[s]')
plt.ylabel('amplitud [V]' )
plt.show(block=False)

#%%
# Qué aspecto tiene la función de transferencia de estos filtros?:

f      = arange(0.1,1000,0.1)

# Respuesta en frecuencia de estos tres filtros:
# pasabajo
Vout1  =  (1/RC2)/(2*pi*1j*f+(1/RC2));

# pasabajo + DC block
Vout2  =  (RC1*2*pi*1j*f)/(1+RC1*2*pi*f*1j)*(1/RC2)/(2*pi*1j*f+(1/RC2))

# opamp (invierte!!)
Vout3  = -1/(2*pi*1j*f*RC2);


plt.figure(44 , constrained_layout=True)

plt.loglog( f ,  abs(Vout1) , label='LOWPASS' )
plt.loglog( f ,  abs(Vout2) , label='LOWPAS+DCBLOCK' )
plt.loglog( f ,  abs(Vout3) , label='OPAMP' )

plt.xlabel('frecuencia [Hz]')
plt.ylabel('$|V_{out}/V_{in}|$')
leg = plt.legend()

plt.xlim(0.1, 1E3)
plt.ylim(1E-4, 200)

plt.grid(b=True,linestyle=':',color='lightgray')


#%%

# un pasaaltos de 4o orden, pero con frecuencia de corte f=1 (la mitad):



RC3   = 1/(2*pi*1)
Vout4 = ((RC3*2*pi*1j*f)/(1+RC3*2*pi*f*1j))**4

plt.loglog( f ,  abs(Vout4) , label='orden4' )


leg.remove()
leg = plt.legend()

