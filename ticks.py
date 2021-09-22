# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:05:33 2021

@author: jocta
"""
import numpy as np
import fractions

# In order to use the function in another script, use the following code:
# import sys
# file = sys.path.append("C:/repos/codigos_utiles/")
# from ticks import *
# multiplos_pi(3*np.pi, 15*np.pi, np.pi/2)


def multiplos_pi(start, stop, step, tick_value = np.pi, tick_label = '\pi'):
    '''
    INPUT: multiples of np.pi. For ex, 3*np.pi, np.pi/2, etc. \n
    OUTPUT: touple of two elements: ticks (a list of ticks), ticks_labels (a list of ticks labels)
    '''
    # The method '.limit_denominator(self, max_denominator = 1000000)' gives the closest fraction. For ex,
    # >>> Fraction('3.141592653589793').limit_denominator(10)
    # Fraction(22, 7)
    # The variable frac is a list with numerator and denominator of the fraction of the corresponding tick
    ticks = np.arange(start, stop, step).tolist()
    ticks.append(stop)
    ticks_labels = []
    for tick in ticks:
        frac = str(fractions.Fraction(tick/tick_value).limit_denominator()).split('/')
        
        if len(frac) == 2:
            if frac[0] == '-1':
                ticks_labels.append(r'$\frac{' + str(tick_value) + '}{' + str(frac[1]) + '}$')
            elif frac[0] != '1':
                ticks_labels.append(r'$\frac{' + str(frac[0]) + tick_label + '}{' + str(frac[1]) + '}$')
            else:
                ticks_labels.append(r'$\frac{' + tick_label + '}{' + str(frac[1]) + '}$')   
        else:
            if frac[0] == '0':
                ticks_labels.append('0')
            elif frac[0] == '1':
                ticks_labels.append('${}$'.format(tick_label))
            else:
                ticks_labels.append(r'${}{}$'.format(frac[0], tick_label))
    return ticks, ticks_labels
