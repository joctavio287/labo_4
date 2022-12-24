import numpy as np
from sympy import symbols,lambdify,latex,diff,sqrt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures

# ===========================================================================
# Este archivo va a tener funciones y clases útiles para el procesamiento de 
# datos del laboratorio
#===========================================================================

# Funcion de prueba:
def suma(a:int, b:int):
    '''
    suma de dos numeros
    '''
    return a + b

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
        self.x, self.y, self.sigma_y = x.reshape(-1,1), y.reshape(-1,1), None
        self.ordenada, self.n, self.cov_y = ordenada, n, cov_y
        self.vander = None
        self.parametros = None
        self.cov_parametros = None
        self.r = None
        self.R2 = None
        self.chi_2 = None
        self.reduced_chi_2 = None
    
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
        self.vander = A.copy()
        
        if self.cov_y is None:
            # Es la identidad:
            self.cov_y = np.identity(len(self.y)) 
        self.sigma_y = np.sqrt(np.diag(self.cov_y).reshape(-1,1))

        # Calculos auxilares:
        inversa_cov = np.linalg.inv(self.cov_y)
        auxiliar = np.linalg.inv(np.dot(np.dot(A.T,inversa_cov),A))

        # Parámetros [At.Cov-1. A]-1.At.Cov-1.y = [a_0, a_1, ..., a_n]t:
        parametros = np.dot(np.dot(np.dot(auxiliar, A.T), inversa_cov), self.y) 

        # Matriz de covarianza de los parámetros [At.Cov-1.A]-1:
        cov_parametros = np.linalg.inv(np.dot(A.T, np.dot(inversa_cov, A)))
        self.parametros, self.cov_parametros = parametros, cov_parametros
        self.y_modelo = np.dot(self.vander, self.parametros)
    
    def bondad(self):
        # Matriz de correlación lineal de los datos:
        self.r = np.corrcoef(self.x.flatten(), self.y.flatten())

        # Coeficiente de determinación: 1 - sigma_r**2/sigma_y**2
        sigma_r = self.y - np.dot(self.vander, self.parametros)
        sigma_y = np.sqrt(np.diag(self.cov_y))
        self.R2 = float(1 - np.dot(sigma_r.T, sigma_r)/np.dot(sigma_y.T, sigma_y))

        # Chi^2:
        # El valor esperado para chi es len(y_data) - # de datos ajustados ± sqrt(len - #).
        # Un chi alto podría indicar error subestimado o que y_i != f(x_i)
        # Un chi bajo podría indicar error sobrestimado
        self.chi_2 = np.sum(((self.y - self.y_modelo)/self.sigma_y)**2, axis = 0)
        self.reduced_chi_2 = self.chi_2/(len(self.y)-len(self.parametros))

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
    # El dividido dos está porque según el teorema del muestreo de Nyquist, la mayor frecuencia
    # que se puede registrar es la mitad de la frecuencia de sampleo. Si el sampleo fuese por l_
    # o menos dos veces más lento que la propia de la señal veríamos aliasing. Si tomamos todo 
    # el rango espectral, el algoritmo lo completa espejando el resultado respecto a fsamp/2.
    return np.linspace(0, fsamp/2, int(N/2)), 2*np.abs(señal_fft[:N//2])/N

# Clase para hacer propagación de errores:

class propagacion_errores:

    def __init__(self, data: dict, covarianza: np.array = None) -> None:
        '''
        INPUT: 
        -data --> dict: es un diccionario con dos keys: 
        
            *'variables'--> list de tuples de tres elementos (es necesario pasar 
            tres elementos aunque no haya errores para dicha variable): (simbolo de la variable, valor, error).
        
            *'expr'--> str: formula
        -covarianza --> np.array of shape n x n, donde n es la cantidad de variables
        '''
        self.parametros = data
        self.valor = None
        self.error = None
        self.covarianza = covarianza

    def __str__(self):
        texto  = f'El valor obtenido es: ({self.parametros} ± {self.error})'
        print(texto)
        return texto
    
    def fit(self):
        '''
        OUTPUT:
        Actualiza los valores de self.valor y self.error.
        '''
        simbolos = [i for i,j,k in self.parametros['variables']]
        valores = [j for i,j,k in self.parametros['variables']]
        errores = np.array([k for i,j,k in self.parametros['variables']])
        # Defino como símbolos las variables de las cuales depende la expresión:
        for sim in simbolos:
            globals()[sim] = symbols(sim, real = True)

        # Defino la expresión simbólica:
        formula = eval(self.parametros['expr'])
        
        # De no estar definida, defino la matriz de covarianza de las variables dependientes:
        if self.covarianza is None: 
            self.covarianza = np.diag(errores**2)   

        # Calculo la aproximación lineal de la covarianza de el i-ésimo dato con el j-ésimo y sumo:
        derivadas_parciales = [diff(formula, eval(sim)) for sim in simbolos]
        covarianza_resultado = 0
        for i in range(len(simbolos)):
            for j in range(len(simbolos)):
                covarianza_resultado += derivadas_parciales[i]*self.covarianza[i, j]*derivadas_parciales[j]
        error_simbolico = sqrt(covarianza_resultado)

        # Convierto la expresión simbólica en un módulo numérico (numpy) para poder reemplazar:
        lambd_err = lambdify(simbolos, error_simbolico, modules = ['numpy'])
        lambd_val = lambdify(simbolos, formula, modules = ['numpy'])
        self.valor, self.error = lambd_val(*valores), lambd_err(*valores)
        return self.valor, self.error

if __name__== '__main__':
    # Prueba regresión lineal
    dic = {'variables': [
    ('f', 14.78, 0), 
    ('a',-.052710,.00009), 
    ('d',5/1000,0.05/1000),
    ('m',88.85/1000,.01/1000), 
    ('l',.36,1/1000)],
    'expr': '((f**2)*4*np.pi**2+a**2)/((((np.pi*(d)**4)/64)/(m/l)*4.934484391**4))'}
    propaga = propagacion_errores(dic)
    propaga.fit()

    # Prueba de regresión lineal y Chi_2
    auxiliar = []
    auxiliar_2 = []
    for i in range(1000):
        x = np.arange(0, 100, 1)
        y = x + norm.rvs(loc=0, scale=2, size=x.shape, random_state = None)
        sigma = np.full(y.shape, 4)

        regr = regresion_lineal(x, y, cov_y = np.diag(sigma), n = 1, ordenada = True)
        regr.fit()
        regr.bondad()
        auxiliar.append(regr.chi_2)
        auxiliar_2.append(regr.reduced_chi_2)
    print(f'Estos valores deberían ser {len(regr.y) -len(regr.parametros)} y 1:', np.array(auxiliar).mean(),np.array(auxiliar_2).mean())

    # Prueba fft
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks  
    t = np.linspace(0, 5*np.pi, 1000)
    señal = np.sin(3*2*np.pi*t) + 4*np.sin(5*2*np.pi*t)
    frecuencias, amplitud_espectral = fft(tiempo = t, señal = señal)
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8,5))
    ax[0].plot(t, señal)
    ax[1].plot(frecuencias, amplitud_espectral)
    picos = frecuencias[find_peaks(amplitud_espectral)[0]]
    print(f'Las frecuencias de la transformada estan en : {picos[0]} Hz y {picos[1]} Hz.')
    fig.show()
