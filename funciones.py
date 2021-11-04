import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sympy import symbols,lambdify,latex,diff,sqrt
import scipy.stats as sp
from sklearn.preprocessing import PolynomialFeatures
# ===========================================================================
# Este archivo va a tener funciones y clases útiles para el procesamiento de 
# datos del laboratorio
#===========================================================================

# Clase para hacer regresiones lineales:

class regresion_lineal:

    def __init__(self, x, y, cov_y = None, n = 1, ordenada = True) -> None:
        '''
        INPUT: 
        -x e y --> np.array: son los datos para ajustar.
        -'cov_y' --> np.array: es la matriz de covarianza de los datos en y, de no haber errores, 
        por defecto se toma la identidad.
        -n --> int: orden del ajuste lineal.
        -'ordenada' --> bool: si queremos o no obtener la ordenada del ajuste.
        '''
        self.x, self.y = x.reshape(-1,1), y.reshape(-1,1)
        self.ordenada, self.n, self.cov_y = ordenada, n, cov_y
        self.parametros = None
        self.cov_parametros = None
        self.vander = None
        self.r = None
        self.R2 = None
    
    def __str__(self):
        texto  = f'Ajuste lineal:\n -Parámetros de ajuste: {self.parametros}.\n'
        texto += f' -Matriz de covarianza de los parámetros: {self.cov_parametros}.\n'
        texto += f' -Bondad del ajuste:\n  *Correlación lineal: {self.r}'
        print(texto)
        return texto

    def fit(self):
        '''
        OUTPUT: 
        -actualiza los coeficientes del ajuste, su matriz de covarianza y la matriz de Vandermonde.
        '''        
        # Matriz de Vandermonde:
        pfeats = PolynomialFeatures(degree = self.n, include_bias = self.ordenada)
        A = pfeats.fit_transform(self.x)
        if self.cov_y is None:
            # Es la identidad:
            self.cov_y = np.identity(len(self.y)) 
        # Calculos auxilares:
        inversa_cov = np.linalg.inv(self.cov_y)
        auxiliar = np.linalg.inv(np.dot(np.dot(A.T,inversa_cov),A))
        # Parámetros [At.Cov-1. A]-1.At.Cov-1.y:
        parametros = np.dot(np.dot(np.dot(auxiliar, A.T), inversa_cov), self.y) 
        # Matriz de covarianza de los parámetros [At.Cov-1.A]-1:
        cov_parametros = np.linalg.inv(np.dot(A.T, np.dot(inversa_cov, A)))
        self.parametros, self.cov_parametros, self.vander= parametros, cov_parametros, A
    
    def bondad(self):
        # Matriz de correlación lineal de los datos:
        self.r = np.corrcoef(self.x.flatten(), self.y.flatten())
        # Coeficiente de determincación: 1 - sigma_r**2/sigma_y**2
        sigma_r = self.y - np.dot(self.vander, self.parametros)
        sigma_y = np.sqrt(np.diag(self.cov_y))
        self.R2 = float(1 - np.dot(sigma_r.T, sigma_r)/np.dot(sigma_y.T, sigma_y))

# Transformadas de Fourier:
def fft(tiempo = None, señal = None):
    '''
    INPUT:
    tiempo[opt.]-->np.array: eje x de la función que queremos transformar.
    señal-->np.array: eje y de la función que queremos transformar.

    OUTPUT:
    touple de np.arrays(): frecuencias, amplitud espectral.
    '''
    if señal is None:
        print('error')
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo 
    señal_fft, N = np.fft.fft(señal), len(señal)
    # El dividido dos está porque según el teorema del muestreo de Nyquist,
    # la mayor frecuencia que se puede registrar es la mitad de la frecuencia
    # de sampleo. 

    # Si el sampleo fuese por lo menos dos veces más lento que la
    # propia de la señal veríamos aliasing.

    # Si tomamos todo el rango espectral, el algoritmo lo completa espejando 
    # el resultado respecto a fsamp/2.
    return np.linspace(0, fsamp/2, int(N/2)), 2*np.abs(señal_fft[:N//2])/N


# Clase para hacer propagación de errores:

class propagacion_errores:

    def __init__(self, data: dict) -> None:
        '''
        INPUT: 
        -data --> dict: es un diccionario con dos keys: 
        
            *'variables'--> lista de tuplas de tres elementos (es necesario pasar 
            tres elementos aunque no haya errores para dicha variable): 
            (simbolo de la variable, valor, error).
        
            *'expr'--> tupla de dos elementos: (simbolo de la formula a predecir, formula).
        '''
        self.parametros = data
        self.valor = None
        self.error = None

    def __str__(self):
        texto  = f'El valor obtenido es: ({self.parametros} ± {self.error})'
        print(texto)
        return texto
    
    @staticmethod
    def norma(vector):
        '''
        INPUT:
        - vector --> list: lista de sympy.symbols al cual se le quiere calcular la norma.
        OUTPUT:
        - float: norma simbólica de la lista como si fuese un vector.
        '''
        suma = 0
        for el in vector:
            suma += el**2
        return sqrt(suma)

    def fit(self):
        '''
        OUTPUT:
        Actualiza los valores de self.valor y self.error.
        '''
        simbolos = [i for i,j,k in self.parametros['variables']]
        valores = [j for i,j,k in self.parametros['variables']]
        errores = [k for i,j,k in self.parametros['variables']]
        # Defino el símbolo de la expresión:
        locals()[self.parametros['expr'][0]] = symbols(self.parametros['expr'][0], real = True)
        # Hago lo mismo para las variables de las cuales depende la expresión:
        for sim in simbolos:
            locals()[sim] = symbols(sim, real = True)
        # Defino la expresión simbólica:
        locals()[self.parametros['expr'][0]] = eval(self.parametros['expr'][1])
        
        # Calculo la norma del vector 'terminos', cuyos elementos son las
        # derivadas parciales multiplicadas por el respectivo error de la
        # variable respecto a la cual derivamos:
        terminos = []
        for sim, error in zip(simbolos, errores):
            derivada_parcial =  diff(eval(self.parametros['expr'][0]), eval(sim))
            terminos.append(derivada_parcial*error)
        # Convierto la expresión simbólica en un módulo numérico (numpy) para poder reemplazar:
        lambd_err = lambdify(simbolos, propagacion_errores.norma(terminos), modules = ['numpy'])
        lambd_val = lambdify(simbolos, locals()[self.parametros['expr'][0]], modules = ['numpy'])
        self.valor, self.error = lambd_val(*valores), lambd_err(*valores)



dic = {
    'variables': [('f', 14.78, 0), ('a',-.052710,.00009), ('d',5/1000,0.05/1000), ('m',88.85/1000,.01/1000), ('l',.36,1/1000)],
    'expr': ('E', '((f**2)*4*np.pi**2+a**2)/((((np.pi*(d)**4)/64)/(m/l)*4.934484391**4))')
}
propaga = propagacion_errores(dic)
propaga.fit()
propaga.valor/1e9, propaga.error/1e9
