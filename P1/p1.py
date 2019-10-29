#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:02:21 2019
@title: VC - P1
@author: Miguel Lentisco Ballesteros
"""
# Librerias
import numpy as np
import cv2
import math

# Funciones auxiliares

""" Si una imagen "img" es plana (si tiene formato de dimension (filas, columnas)) """
def es_plana(img):
    return len(img.shape) == 2

""" Normaliza la imagen "img" al intervalo [0, 1]. Se aplica la fórmula de 
normalización a cada valor x_nor = (x_nor - min) / (max - min), se aplica por
cada canal de color buscando su máximo y su mínimo. """
def normalizar(img):
  # Pasamos a flotantes
  img = img.astype("float64")
  # Normalizamos por cada banda
  for i in range(img.shape[2]):
    # Cogemos la banda i-ésima
    banda_color = img[:, :, i]
    # Buscamos max y min
    v_max = np.amax(banda_color)
    v_min = np.amin(banda_color)
    diff = v_max - v_min
    # Si v_max > v_min podemos normalizar (si no es que todos tienen el mismo valor)
    if diff > 0:
      # Normalizamos
      banda_color[:, :] = (banda_color[:, :] - v_min) / diff
  return img

""" Toma la imagen "img", la normaliza y la escala al intervalo [0, 255] lista para
ser mostrada. Si "usa_abs"es True entonces se toma el valor absoluto.
Es importante usar esta función antes de imprimir para que se vea bien. """
def reescalar(img, usa_abs=False):
    if usa_abs:
        img = np.abs(img)
    img = img[:, :, None] if es_plana(img) else img
    return np.round(255 * normalizar(img)).astype("uint8")

""" Toma la imagen "img" y la muestra con la ventana de título "titulo". """
def muestraImagen(titulo, img):
    # Mostramos la imágen
    cv2.imshow(titulo, img)
    # Esperamos a una tecla
    cv2.waitKey(0)
    # Borramos las ventanas abiertas
    cv2.destroyAllWindows()
    
# Funciones principales

""" Aplica a la imagen "img" la convolucion1D con las máscaras1D "kernel1" y
"kernel". Para ello primero se realizan las convoluciones por filas con "kernel1"
y al resultado las convoluciones por columnas con "kernel2". Se puede cambiar 
el modo de borde con "modo_borde": "constant", "edge", "reflect",
"symmetric", "wrap", "mean"... """
def convolve1Dfast(img, kernel1, kernel2, modo_borde="reflect"):
    # Ponemos las máscaras1D aplanadas y le damos la vuelta
    kernel1 = np.flip(kernel1.flatten())
    kernel2 = np.flip(kernel2.flatten())
    # Pasamos a float
    img = img.astype(float)
    # Dimensiones de la imagen (filas/columnas/canales)
    N, M, C = img.shape
    # Cantidad de padding
    pad = round((kernel1.shape[0] - 1) / 2.0)
    # Por cada canal de color aplicamos la convolución
    for k in range(C):
        # Rellenamos los bordes con el modo indicado
        banda = np.pad(img[:, :, k], (pad, pad), modo_borde)
        # Convolución por las filas con el kernel1 (TODAS las filas)
        for i in range(N + 2 * pad):
            banda[i, pad:(M + pad)] = np.convolve(kernel1, banda[i], 'valid')
        # Convolución por columnas con el kernel2 (columnas SIN RELLENO)
        for j in range(pad, M + pad):
            banda[pad:(N + pad), j] = np.convolve(kernel2, banda[:, j], 'valid')
        # Actualizamos el valor sin el relleno
        img[:, :, k] = banda[pad:(N + pad), pad:(M + pad)]
    return img

""" Rellena la matriz2D "canal" replicando los bordes, añadiendo dimensiones
a las filas "relleno_fil" * 2 y columnas "relleno_col" * 2. Se repiten los bordes
para conseguir esto. El resultado por ej: aaaaa | abcdefgh | hhhhh """
def edge(canal, relleno_fil, relleno_col):
    # Replicamos el borde superior
    fil_sup = np.repeat([canal[0]], relleno_fil, axis=0)
    # Replicamos el borde superior
    fil_inf = np.repeat([canal[canal.shape[0] - 1]], relleno_fil, axis=0)
    # Lo juntamos todo
    res = np.vstack((fil_sup, canal, fil_inf))
    # Replicamos el borde izquierdo
    col_izq = np.repeat(res[:, 0][:, None], relleno_col, axis=1)
    # Replicamos el borde derecho
    col_der = np.repeat(res[:, res.shape[1] - 1][:, None], relleno_col, axis=1)
    # Lo juntamos todo
    res = np.hstack((col_izq, res, col_der))
    return res

""" Rellena la matriz2D "canal" añadiendo un valor constante "valor_cte" según
con 2 * "relleno_fil" filas y 2 * "relleno_col" columnas. El resultado por ej
sería: xxxx | abcdefgh | xxxx con x cualquier valor preasignado """
def constant(canal, relleno_fil, relleno_col, valor_cte):
    # Fila constante
    f_const = np.repeat(valor_cte, canal.shape[1])
    # Creamos los bordes superior e inferior
    fil_sup = fil_inf = np.repeat([f_const], relleno_fil, axis=0)
    # Juntamos
    res = np.vstack((fil_sup, canal, fil_inf))
    # Columna constante
    c_const = np.repeat(valor_cte, res.shape[0])
    # Creamos los bordes izq y derecha
    fil_izq = fil_der = np.repeat(c_const[:, None], relleno_col, axis=1)
    # Juntamos
    res = np.hstack((fil_izq, res, fil_der))
    return res

""" Rellena la matriz2D "canal" reflejando la imagen sobre los bordes, añadiendo
2 * "relleno_fil" filas y 2 * "relleno_col" columnas. Un ejemplo de resultado
sería: gfedcb | abcdefgh | gfedcb """
def reflect(canal, relleno_fil, relleno_col):
    # Submatriz desde la segunda fila con nº relleno_fil filas y le damos la vuelta
    m_sup = np.flip(canal[1:(relleno_fil + 1)], axis=0)
    # Igual con la penúltima fila
    m_inf = np.flip(canal[(canal.shape[0] - 1 - relleno_col):(canal.shape[0] - 1)], axis=0)
    # Juntamos todo
    res = np.vstack((m_sup, canal, m_inf))
    # Submatriz desde la segunda columna con nº relleno_col columnas y le damos la vuelta
    m_izq = np.flip(res[:, 1:(relleno_col + 1)], axis=1)
    # Igual con la penúltima columna
    m_der = np.flip(res[:, (res.shape[1] - 1 - relleno_col):(res.shape[1] - 1)], axis=1)
    # Juntamos todo
    res = np.hstack((m_izq, res, m_der))
    return res

    
""" Rellena la matriz2D "canal" añadiendo 2 * "relleno_fil" filas y
2 * "relleno_col" columnas con el modo índicado "modo_borde" y para el modo
"constant" se usa "valor_cte" """
def rellenar(canal, relleno_fil, relleno_col, modo_borde="reflect", valor_cte=0):
    if modo_borde == "reflect":
        return reflect(canal, relleno_fil, relleno_col)
    elif modo_borde == "edge":
        return edge(canal, relleno_fil, relleno_col)
    elif modo_borde == "constant":
        return constant(canal, relleno_fil, relleno_col, valor_cte)
    # Si no es ningun modo de los creados invoca a pad de numpy
    else:
        return np.pad(canal, (relleno_fil, relleno_col), modo_borde)
    

""" Aplica a la imagen "img" la convolucion1D con las máscaras1D "kernel1" y
"kernel". Para ello primero se realizan las convoluciones por filas con "kernel1"
y al resultado las convoluciones por columnas con "kernel2". Se puede cambiar 
el modo de borde con "modo_borde": "constant", "edge", "reflect",
"symmetric", "wrap", "mean"... """
def convolve1Dbonus(img, kernel1, kernel2, modo_borde="reflect"):
    # Ponemos las máscaras1D aplanadas y le damos la vuelta
    kernel1 = np.flip(kernel1.flatten())
    kernel2 = np.flip(kernel2.flatten())
    # Pasamos a float
    img = img.astype(float)
    # Dimensiones de la imagen (filas/columnas/canales)
    N, M, C = img.shape
    # Cantidad de padding
    pad = round((kernel1.shape[0] - 1) / 2.0)
    # Por cada canal de color aplicamos la convolución
    for k in range(C):
        # Rellenamos los bordes con el modo indicado
        banda = rellenar(img[:, :, k], pad, pad, modo_borde)
        # Convolución por las filas con el kernel1 (TODAS las filas)
        for i in range(N + 2 * pad):
            fila = np.copy(banda[i])
            for j in range(pad, M + pad):
                banda[i, j] = np.sum(np.multiply(fila[(j - pad):(j + pad + 1)], kernel1))
        # Convolución por columnas con el kernel2 (columnas SIN RELLENO)
        for j in range(pad, M + pad):
            columna = np.copy(banda[:, j])
            for i in range(pad, N + pad):
                banda[i, j] = np.sum(np.multiply(columna[(i - pad):(i + pad + 1)], kernel2))
        # Actualizamos el valor sin el relleno
        img[:, :, k] = banda[pad:(N + pad), pad:(M + pad)]
    return img
    
""" Aplica a la imagen "img" las máscaras1D separables "kernel1" y "kernel2"
con el modo de borde "modo_borde". Se puede ajustar si se quiere el modo
rápido con "fast", de manera que se usa la función optimizada de numpy; en caso
contrario se aplica la convolución hecha por mi en python """
def filtro1D(img, kernel1, kernel2, modo_borde="reflect", fast=True):
    if fast:
        return convolve1Dfast(img, kernel1, kernel2, modo_borde)
    return convolve1Dbonus(img, kernel1, kernel2, modo_borde)    

""" Aplica a una imagen "img" una convolución2D con la máscara2D "kernel",
esta función está pensada por si se quiere aplicar la convolución a una
máscara2D no separable pero en principio no va a usarse ya que se prefiere
la ventaja de menos cálculos que produce hacer la convolución1D. Se puede
cambiar el modo de borde con "modo_borde" """
def convolve2D(img, kernel, modo_borde="reflect"):
    # Pasamos a float
    img = img.astype(float)
    # Dimensiones de la imagen (filas/columnas/canales)
    N, M, C = img.shape
    # Volteamos el kernel
    kernel = np.flip(kernel)
    # Cantidad de relleno
    pad = round((kernel.shape[0] - 1) / 2.0)
    # Por cada canal de color
    for k in range(C):
        # Rellenamos según modo_borde
        banda = rellenar(img[:, :, k], pad, pad, modo_borde)
        # Aplicamos la convolución
        for i in range(pad, pad + N):
            for j in range(pad, pad + M):
                matriz = banda[(i - pad):(i + pad + 1), (j - pad):(j + pad + 1)]
                img[i - pad, j - pad, k] = np.sum(np.multiply(matriz, kernel))
    return img

""" Separa y devuelve dos máscaras1D por los que se puede descomponer "kernel"
que debe ser de rango 1 para ser descompuesto y se aprovecha la descomposición
SVD para esto. Se puede ver que al descomponer la matriz "kernel" en la
multiplicación de 3 matrices USV.T con S diagonal, la matriz se puede expresar
como la sumatoria de u_i * s_i * v_i.T, entonces si el rango es 1 la matriz
solo tendrá un valor singular no cero (s_1) el resto será 0 o casi cero,
entonces podremos expresar matriz = u_1 * s_1 * v_1.T """
def separarMatriz2D(kernel):
    # Para ser separable la matriz debe tener rango 1
    assert(np.linalg.matrix_rank(kernel) == 1), "ERROR: La matriz no es separable"
    # Hacemos descomposición SVD
    u, s, vT = np.linalg.svd(kernel)
    # La descomposición se hace con las dos primeras columnas de U y V
    # junto a la normalización dividiendo entre la raíz del 1er valor singular
    k1 = math.sqrt(s[0]) * u[:, 0]
    k2 = math.sqrt(s[0]) * vT.T[:, 0]
    return k1, k2

""" Aplica a una imagen "img", la máscara "kernel" de manera que si la máscara
es separable (tiene rango 1), las separa y aplica convolución separable. Si no
es, entonces aplica la convolución con la máscara2D, aunque en principio no
debería pasar nunca ya que se van a usar máscaras separables. Se puede elegir 
el modo de borde al convolucionar con "modo_borde" """
def filtro2D(img, kernel, modo_borde="reflect", fast=True):
    # Si la matriz es separable convolución1D
    if np.linalg.matrix_rank(kernel) == 1:
        # Descomposición de la matriz (svd)
        k1, k2 = separarMatriz2D(kernel)
        # Convolucionamos la imagen
        return filtro1D(img, k1, k2, modo_borde, fast)
    # Si no es separable convolución2D
    return convolve2D(img, kernel, modo_borde)

# Ejercicio 1.1) Filtros gaussianos y derivadas

""" Devuelve la máscara1D de la función gaussiana con mu=0, sigma="sigma"
haciendo una discretización en el intervalo [-3 * sigma, 3 * sigma] con
6 *sigma + 1 elementos. Simplemente se toman los elementos en el intervalo, 
se aplica la función gaussiana al vector y se normaliza. """
def getKernelGaussian(sigma):
    # Radio del intervalo
    radio = math.floor(3 * sigma)
    # Tomamos los elementos en el intervalo (2 * radio + 1 elementos que es impar)
    kernel1D = np.arange(-radio, radio + 1)
    # Aplicamos la gaussiana sin el coef (lo vamos a normalizar)
    kernel1D = np.exp(- 0.5 / (sigma * sigma) * kernel1D ** 2)
    # Normalizamos
    kernel1D = kernel1D / kernel1D.sum()    
    # Devolvemos la matriz
    return kernel1D

""" Aplica a la imagen "img" el filtro gaussiano con sigma "sigma". Para ello toma
la máscara1D proporcionada por la función anterior. Se puede elegir el modo de borde
al convolucionar con "modo_borde" """
def aplicaGaussiana(img, sigma, modo_borde="reflect"):
    # Obtenemos la máscara1D gaussiana
    gauss_ker = getKernelGaussian(sigma)
    # Aplicamos el filtro a la imagen
    return filtro1D(img, gauss_ker, gauss_ker, modo_borde)

""" Aplica a la imagen "img" el filtro de la derivada de orden "dx" respecto x y la
derivada de orden "dy" respecto y de tamaño "mask_size". Para ello toma de OpenCV
las máscaras1D y las aplica. Se puede elegir el modo de borde al convolucionar
con "modo_borde" """
def aplicaDeriv(img, dx, dy, mask_size, modo_borde="reflect"):
    # Obtenemos las máscaras1D de las derivadass
    d1, d2 = cv2.getDerivKernels(dx, dy, mask_size)
    # Aplicamos el filtro a la imagen
    return filtro1D(img, d1, d2, modo_borde)

""" Bonus1: Probamos convolución con máscara2D separable """
def bonus1(imgs, sigma):
    # Probemos mi convolve con una gaussiana y comparada con la de OpenCV
    gauss_ker1D = getKernelGaussian(sigma)
    gauss_ker2D = gauss_ker1D * gauss_ker1D[:, None]
    for img in imgs:
        # Convolve con máscara2D separable
        img1 = reescalar(filtro2D(img, gauss_ker2D, fast=False))
        # GaussianBlur 
        img2 = reescalar(cv2.GaussianBlur(img, gauss_ker2D.shape, sigma))
        res = cv2.hconcat([img1, img2])
        muestraImagen("Convolve1D a mano", res)

""" Ej1.1: Aplicamos filtros gaussianos y de derivadas """
def ej11(imgs, mask_size, sigma1, sigma2, dx1, dy1, dx2, dy2, borde1, borde2):
    # Filtros gaussianos
    for img in imgs:
        # Distintos ejemplos con diferentes sigma y bordes
        img1 = reescalar(aplicaGaussiana(img, sigma1, borde1))
        img2 = reescalar(aplicaGaussiana(img, sigma2, borde1))
        img3 = reescalar(aplicaGaussiana(img, sigma1, borde2))
        img4 = reescalar(aplicaGaussiana(img, sigma2, borde2))
        # Mostramos las imágenes
        h1 = cv2.hconcat([img1, img2])
        h2 = cv2.hconcat([img3, img4])
        res = cv2.vconcat([h1, h2])
        muestraImagen("Gaussiana", res) 
        
    # Filtros derivadas
    for img in imgs:
        # Ejemplos de derivadas
        img1 = reescalar(aplicaDeriv(img, dx1, dy1, mask_size, borde1))
        img2 = reescalar(aplicaDeriv(img, dx2, dy2, mask_size, borde1))
        img3 = reescalar(aplicaDeriv(img, dx1, dy1, mask_size, borde2))
        img4 = reescalar(aplicaDeriv(img, dx2, dy2, mask_size, borde2))
        # Mostramos las imágenes
        h1 = cv2.hconcat([img1, img2])
        h2 = cv2.hconcat([img3, img4])
        res = cv2.vconcat([h1, h2])
        muestraImagen("Derivadas 1er orden", res)

# Ejercicio 1.2) Laplaciana de gaussiana

""" Aplica a la imagen "img" la laplaciana, que consiste en la suma de las
derivadas segundas respecto x e y. Para ello se obtienen las máscaras1D de
las derivadas se convolucionan con la imagen y se suman. Se puede cambiar el
tamaño de las máscaras1D con "mask_size" y el modo borde con "modo_borde".
Se hace la suma ya que la suma de matrices separables no es separable en general
y por tanto es más rápido hacer 2 convoluciones con máscaras separables que
1 con una no separable. """
def laplaciana(img, mask_size, modo_borde="reflect"):
    # Obtenemos las máscaras1D de la segunda derivada respecto x e y
    kerX1, kerX2 = cv2.getDerivKernels(2, 0, mask_size)
    kerY1, kerY2 = cv2.getDerivKernels(0, 2, mask_size)
    # Convolucionamos la imagen gaussiana con cada derivada respectiva
    img_x = filtro1D(img, kerX1, kerX2, modo_borde)
    img_y = filtro1D(img, kerY1, kerY2, modo_borde)
    # Devolvemos la suma de las derivadas segundas
    return img_x + img_y

""" Aplica a la imagen "img" la laplaciana de la gaussiana. Se aplica primero 
la gaussiana con sigma="sigma" y borde "modo_borde" y al resultado se le hace
la laplaciana con tamaño "mask_size" y borde "modo_borde". Finalmente se
normaliza multiplicando el resultado por sigma^2. """
def aplicaLoG(img, mask_size, sigma, modo_borde="reflect"):
    # Aplicamos la gaussiana
    img_g = aplicaGaussiana(img, sigma, modo_borde)
    # Aplicamos la laplaciana a la gaussiana y normalizamos
    return (sigma * sigma) * laplaciana(img_g, mask_size, modo_borde)

""" Ej 1.2: aplicamos la laplaciana de gaussiana """
def ej12(imgs, mask_size, sigma1, sigma2, borde1, borde2):
    # Laplaciana de gaussiana
    for img in imgs:
        # Ejemplos de la Lapaciana de Gaussiana
        img11 = aplicaLoG(img, mask_size, sigma1, borde1)
        img12 = aplicaLoG(img, mask_size, sigma2, borde1)
        img13 = aplicaLoG(img, mask_size, sigma1, borde2)
        img14 = aplicaLoG(img, mask_size, sigma2, borde2)
        
        # Sin valor absoluto
        img1 = reescalar(img11)
        img2 = reescalar(img12)
        img3 = reescalar(img13)
        img4 = reescalar(img14)
        h1 = cv2.hconcat([img1, img2])
        h2 = cv2.hconcat([img3, img4])
        res = cv2.vconcat([h1, h2])    
        muestraImagen("LoG sin abs", res)
        
        # Con valor absoluto
        img1 = reescalar(img11, True)
        img2 = reescalar(img12, True)
        img3 = reescalar(img13, True)
        img4 = reescalar(img14, True)
        h1 = cv2.hconcat([img1, img2])
        h2 = cv2.hconcat([img3, img4])
        res = cv2.vconcat([h1, h2])
        muestraImagen("LoG con abs", res)

""" Ej1: aplicar gaussiana/derivadas y laplaciana de gaussiana """
def ej1(imgs, mask_size, sigma1, sigma2, dx1, dy1, dx2, dy2, borde1, borde2):
    # Gaussiana y derivadas
    ej11(imgs, mask_size, sigma1, sigma2, dx1, dy1, dx2, dy2, borde1, borde2)
    # LoG
    ej12(imgs, mask_size, sigma1, sigma2, borde1, borde2)

# Ejercicio 2.1) Piramide Gaussiana

""" Aplica un downsampling a la imagen "img" por filas según indique "ratio_filas"
y por columnas según indique "ratio_columnas". Simplemente se toma la subimagen
de tantos saltos como se diga. """
def downsampling(img, ratio_filas=2, ratio_columnas=2):
    return img[::ratio_filas, ::ratio_columnas]

""" Crea la pirámide gaussiana que consiste en aplicar a la imagen "img" la gaussiana
con sigma="sigma" y después hacer un downsampling con ratio "ratio_sampling". 
Al resultado se le vuelve a realizar la misma operación y así hasta obtener tantas imagénes 
como "niveles" indique. Se cambia el modo de borde con "modo_borde". """
def piramideGaussiana(img, sigma, niveles, modo_borde="reflect", ratio_sampling=2):
    # Donde guardamos las imágenes de la piramide gaussiana
    piramide = [img]
    # Hacemos una imagen por nivel
    for i in range(niveles):
        # Tomamos la actual y le aplicamos la gaussiana
        img_gauss = aplicaGaussiana(piramide[i], sigma, modo_borde)
        # Añadimos la gaussiana con un downsampling a la mitad
        piramide.append(downsampling(img_gauss, ratio_sampling, ratio_sampling))
    return piramide

""" Crea a partir de "piramide" una estructura en forma de escalera para poder
mostrar una pirámide, se pone una a la izquierda y a forma de escalera el
resto a la derecha. Se puede cambiar si usar los valores absolutos "v_abs" """
def crearPiramideImagen(piramide, v_abs=False):
    # Primer nivel que irá a la izquierda
    orig = reescalar(piramide[0], v_abs)
    # Color de fondo relleno
    COLOR = [220, 220, 220]
    # Escalera a la derecha 
    escalera = [reescalar(piramide[1], v_abs)]
    # Por cada imagen le añadimos relleno en las columnas
    for i in range(2, len(piramide)):
        esc = reescalar(piramide[i], v_abs)
        esc = cv2.copyMakeBorder(esc, 0, 0, 0,  escalera[i - 2].shape[1] - esc.shape[1], cv2.BORDER_CONSTANT, value=COLOR)
        escalera.append(esc)
    # Juntamos la escalera y la adjustamos
    escalera = cv2.vconcat(escalera)
    escalera = cv2.copyMakeBorder(escalera, 0, orig.shape[0] - escalera.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=COLOR)
    # Juntamos todo
    res = cv2.hconcat([orig, escalera])
    return res

""" Ej 2.1: piramide gaussiana """
def ej21(imgs, sigma, niveles, borde1, borde2):
    # Piramides gaussianas
    for img in imgs:
        # Obtenemos la piramides gaussianas
        piramide1 = piramideGaussiana(img, sigma, niveles, borde1)
        piramide2 = piramideGaussiana(img, sigma, niveles, borde2)
        # Forma de escalera
        escalera1 = crearPiramideImagen(piramide1)
        escalera2 = crearPiramideImagen(piramide2)
        # Juntamos
        res = cv2.vconcat([escalera1, escalera2])
        muestraImagen("Piramide gaussiana", res)
        
# Ejercicio 2.2) Piramide Laplaciana

""" Aplica un upsampling a "img" por filas según indique "ratio_filas" y
por columnas según indique "ratio_columnas". La función duplica las filas y las
columnas tantas veces como se indique. """
def upsampling(img, ratio_filas=2, ratio_columnas=2):
    # Copiamos la imagen
    img = np.copy(img)
    # Nº filas y columnas nuevas
    filas = ratio_filas * img.shape[0]
    columnas = ratio_filas * img.shape[1]
    # Guardaremos el resultado aqui
    res = np.zeros((filas, columnas, img.shape[2]), img.dtype)
    # Por cada banda aplicamos
    for k in range(img.shape[2]):
        # Vector vacío
        aux = np.empty((0, img.shape[1]), img.dtype)
        # Por cada fila de la imagen se añaden "ratio_filas" filas
        for i in range(img.shape[0]):
            for _ in range(ratio_filas):
                aux = np.vstack([aux, img[i, :, k]])
        # Vector vacío
        aux2 = np.empty((filas, 0), img.dtype)
        # Por cada columna de la imagen se añaden "ratio_columnas" columnas
        for j in range(img.shape[1]):
            for _ in range(ratio_columnas):
                aux2 = np.hstack([aux2, aux[:, j][:, None]])
        # Se añade a res el resultado de la banda
        res[:, :, k] = aux2
    return res

""" Hace la piramide laplaciana de "img" con sigma "sigma", borde "modo_borde",
de "niveles" niveles y con un sampling de "ratio_sampling". Se realiza
la diferencia entre las imagenes de la pirámide gaussiana y la siguiente
upsampleada a su mismo tamaño y efectuando la diferencia. """
def piramideLaplaciana(img, sigma, niveles, modo_borde="reflect", ratio_sampling=2):
    # Donde guardamos la piramide laplaciana
    piramide_lap = []
    # Obtenemos la piramide gaussiana sin la original
    piramide_gauss = piramideGaussiana(img, sigma, niveles, modo_borde, ratio_sampling)
    for i in range(niveles):
        # La i-ésima imagen de la pirámide gaussiana
        img_g = piramide_gauss[i]
        # La (i+1)-ésima imagen de la pirámide gaussiana upsampleada 
        img_up = upsampling(piramide_gauss[i + 1], ratio_sampling, ratio_sampling)
        # Si alguna imagen no era divisible, reajusta la dimensión para que coincida
        if img_g.shape != img_up.shape:
            img_up = cv2.resize(img_up, (img_g.shape[1], img_g.shape[0]))
            # Redimensionamos
            if es_plana(img_up): img_up = img_up[:, :, None]
        # Añadimos la diferencia
        piramide_lap.append(img_g - img_up)
    return piramide_lap

""" Devuelve la primera entrada de la pirámide gaussiana de "img" reconstruyendola
a partir de laplaciana y la última de la gaussiana, Piramide gaussiana
    # con el sigma "sigma, borde
"modo_borde", niveles "niveles" y sampling "ratio_sampling" """
def recuperarImagen(img, sigma, niveles, modo_borde="reflect", ratio_sampling=2):
    # Obtenemos piramides laplaciana y gaussiana
    piramide_lap = piramideLaplaciana(img, sigma, niveles, modo_borde, ratio_sampling)
    # Solo la última para reconstruir
    res = piramideGaussiana(img, sigma, niveles, modo_borde, ratio_sampling)[niveles]
    for i in range(niveles - 1, -1, -1):
        # La i-ésima imagen de la pirámide laplaciana
        img_l = piramide_lap[i]
        # Upsampling del g_i actual
        res = upsampling(res, ratio_sampling, ratio_sampling)
        # Si alguna imagen no era divisible, reajusta la dimensión para que coincida
        if img_l.shape != res.shape:
            res = cv2.resize(res, (img_l.shape[1], img_l.shape[0]))
            # Redimensionamos
            if es_plana(res): res = res[:, :, None]
        # Sumamos el g_i actual con la laplaciana
        res = img_l + res
    return res

""" Ej 2.2: piramide laplaciana """
def ej22(imgs, sigma, niveles, borde1, borde2):
    # Piramides Laplacianas
    for img in imgs:
        # Obtenemos las piramides laplaciana
        piramide1 = piramideLaplaciana(img, sigma, niveles, borde1)
        piramide2 = piramideLaplaciana(img, sigma, niveles, borde2)
        # Creamos las escaleras con v_abs y sin
        escalera1 = crearPiramideImagen(piramide1)
        escalera1_abs = crearPiramideImagen(piramide1, True)
        escalera2 = crearPiramideImagen(piramide2)
        escalera2_abs = crearPiramideImagen(piramide2, True)
        h1 = cv2.hconcat([escalera1, escalera1_abs])
        h2 = cv2.hconcat([escalera2, escalera2_abs])
        res = cv2.vconcat([h1, h2])
        muestraImagen("Piramide laplaciana", res)
    
        # Reconstrucción de la imagen
        rec1 = reescalar(recuperarImagen(img, sigma, niveles, borde1))
        rec2 = reescalar(recuperarImagen(img, sigma, niveles, borde2))
        res = cv2.hconcat([reescalar(img), rec1, rec2])
        muestraImagen("Reconstruccion", res)

# Ejercicio 2.3)
    
""" Aplica la supresión de no máximos a la imagen "img", es decir si un pixel
no es el máximo de su entorno entonces es puesto a 0. Para ello se va iterando
y mirando si el pixel es mayor que el subcubo de orden 3 con centro él (se mira
por arriba y por abajo de las escalas) """
def supresionNoMaximos(escala, escala_abajo=None, escala_arriba=None):
    # Copia
    img = np.copy(escala)
    # Tamaño del "cubo"
    cubos = 1
    # Posición de la escala primaria
    pos = 0
    # Si tiene una escala por abajo unimos
    if escala_abajo is not None:
        escala = np.concatenate((escala_abajo, escala), axis=2)
        cubos += 1
        pos = 1
    # Si tiene una escala por arriba unimos
    if escala_arriba is not None:
        escala = np.concatenate((escala, escala_arriba), axis=2)
        cubos += 1
    # Dimensiones
    N, M, C = img.shape
    # Por cada banda
    for k in range(C):
        # Rellenamos para los bordes
        canal = np.pad(escala[:, :, k::C], (1, 1), "constant")[:, :, 1:(cubos + 1)]
        # Por cada pixel comprobamos
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Obtenemos el máximo del entorno (en el cubo)
                v_max = np.amax(canal[(i - 1):(i + 2), (j - 1):(j + 2), 0:cubos])
                # Si no es el máximo local pone a 0
                if v_max > canal[i, j, pos]: img[i - 1, j - 1] = 0
    return img

""" Dibuja circulos considerando el radio como  K * "sigma", en la imagen "img" 
tomando como centros los puntos en "img_max" donde su valor sea mayor que "UMBRAL". """
def dibujarCirculos(img, img_max, sigma):
    # Grosor del círculo
    GROSOR = 1
    # Color del circulo
    COLOR = 250 if img.shape[2] == 1 else [0, 255, 0]
    # Constante proporcionalidad al radio del círculo
    K = 2
    # Dimensiones
    N, M, C = img.shape
    # Umbral para considerar centro
    UMBRAL = 124
    # Radio del círculo
    radio = round(K * sigma)
    # Buscamos los píxeles con valor > UMBRAL
    for i in range(N):
        for j in range(M):
            # Con que en algún canal sea > UMBRAL nos vale
            if np.sum(img_max[i, j]) > UMBRAL:
                # Dibujamos el círculo
                cv2.circle(img, (j, i), radio, color=COLOR, thickness=GROSOR)
    return img
    
""" Crea "n_escalas" escalas laplacianas, tomando como base "img" y empezando a aplicar
filtro LoG de sigma "sigma" y tamaño "mask_size" con borde "modo_borde". Se crean las
escalas aumentando el sigma cada vez, después se aplica supresión de no-máximos y
finalmente se imprime en la imagen original todas las zonas encontradas. """
def escalasLaplaciano(img, sigma, mask_size, n_escalas, modo_borde="reflect"):
    assert(n_escalas > 2), "ERROR: nº de escalas laplacianas debe ser al menos 3 niveles"
    # Aumento de sigma
    K = 1.4
    # Guardamos las escalas
    escalas = []
    # Guardamos los sigmas
    sigmas = []
    # Guardamos las supresiones
    supresiones = []
    # Creamos N escalas
    for _ in range(n_escalas):
        # LoG al cuadrado
        escala = np.square(aplicaLoG(img, mask_size, sigma, modo_borde))
        # Guardamos la escala
        escalas.append(escala)
        # Guardamos el sigma
        sigmas.append(sigma)
        # Aumentamos el sigma
        sigma *= K 
    # Primera supresión
    supresiones.append(supresionNoMaximos(escalas[0], None, escalas[1]))
    # Resto de supresiones
    for i in range(1, len(escalas) - 1):
            # Aplicamos supresión de no-máximos y reescalamos
            supresiones.append(supresionNoMaximos(escalas[i], escalas[i - 1], escalas[i + 1]))
    # Última supresión
    supresiones.append(supresionNoMaximos(escalas[n_escalas - 1], escalas[n_escalas - 2], None))
    # Copiamos la imagen
    res = np.copy(img)
    # Dibujamos en la imagen las regiones encontradas por cada escala
    for i in range(n_escalas):
        res = dibujarCirculos(res, reescalar(supresiones[i]), sigmas[i])
    return res

""" Ej 2.3: blob detection en escalas laplacianas """
def ej23(imgs, sigma, mask_size, n_escalas, borde):
    for img in imgs:
        res = escalasLaplaciano(img, sigma, mask_size, n_escalas, borde)
        muestraImagen("Escalas laplacianas", res)
        
""" Ej 2: Piramide gaussiana, laplaciana y blob detection en escalas laplacianas """
def ej2(imgs, imgs2, sigma1, sigma2, niveles, borde1, borde2, mask_size, n_escalas):
    # Piramide gaussiana
    #ej21(imgs, sigma1, niveles, borde1, borde2)
    # Piramide laplaciana
    #ej22(imgs, sigma1, niveles, borde1, borde2)
    # Blob detection
    ej23(imgs2, sigma2, mask_size, n_escalas, borde1)
    
# Ejercicio 3)
    
""" Mezcla "img1" con "img2" para formar una imagen híbrida, a "img1" se le
pasa un filtro gaussiano (paso bajo) con "sigma_bajo", a "img" se le pasa
un filtro de paso alto (1 - filtro gaussiano) con "sigma_alto"; y con
modo de borde "modo_borde". Se muestra la baja, alta y la híbrida. """
def hibridar(img1, img2, sigma_bajo, sigma_alto, modo_borde="reflect"):
    # Aplicamos un filtro de paso bajo
    img_bajo = aplicaGaussiana(img1, sigma_bajo, "reflect")
    # Aplicamos un filtro de paso alto (I - G2I)
    img_alto = img2 - aplicaGaussiana(img2, sigma_alto, "reflect")
    # Imagen híbrida suma de ambas
    img_hibrida = img_bajo + img_alto
    res = cv2.hconcat([reescalar(img_bajo), reescalar(img_alto), reescalar(img_hibrida)])
    muestraImagen("Imagen hibrida", res)
    return img_hibrida

""" Ej3: hibridar con las parejas, mostrarlas y pirámide gaussiana de la híbrida """
def ej3():
    # Gato y perro
    sigma_bajo1 = 8.5
    sigma_alto1 = 2.2
    img1 = cv2.imread("./imagenes/cat.bmp", 0)[:, :, None]
    img2 = cv2.imread("./imagenes/dog.bmp", 0)[:, :, None]
    h1 = hibridar(img1, img2, sigma_bajo1, sigma_alto1)
    res = crearPiramideImagen(piramideGaussiana(h1, sigma_bajo1, 4))
    muestraImagen("Piramide gaussiana hibrida", res)
    
    # Pajaro y avión
    sigma_bajo2 = 5.0
    sigma_alto2 = 1.5
    img2 = cv2.imread("./imagenes/bird.bmp", 0)[:, :, None]
    img1 = cv2.imread("./imagenes/plane.bmp", 0)[:, :, None]
    h2 = hibridar(img1, img2, sigma_bajo2, sigma_alto2)
    res = crearPiramideImagen(piramideGaussiana(h2, sigma_bajo2, 4))
    muestraImagen("Piramide gaussiana hibrida", res)
    
    # Einstein y marilyn
    sigma_bajo3 = 5.0
    sigma_alto3 = 1.5
    img1 = cv2.imread("./imagenes/marilyn.bmp", 0)[:, :, None]
    img2 = cv2.imread("./imagenes/einstein.bmp", 0)[:, :, None]
    h3 = hibridar(img1, img2, sigma_bajo3, sigma_alto3)
    res = crearPiramideImagen(piramideGaussiana(h3, sigma_bajo3, 4))
    muestraImagen("Piramide gaussiana hibrida", res)
    
    # Pez y submarino
    sigma_bajo4 = 5.0
    sigma_alto4 = 1.0
    img2 = cv2.imread("./imagenes/fish.bmp", 0)[:, :, None]
    img1 = cv2.imread("./imagenes/submarine.bmp", 0)[:, :, None]
    h4 = hibridar(img1, img2, sigma_bajo4, sigma_alto4)
    res = crearPiramideImagen(piramideGaussiana(h4, sigma_bajo4, 4))
    muestraImagen("Piramide gaussiana hibrida", res)
    
    # Bici y moto
    sigma_bajo5 = 9.0
    sigma_alto5 = 1.4
    img2 = cv2.imread("./imagenes/motorcycle.bmp", 0)[:, :, None]
    img1 = cv2.imread("./imagenes/bicycle.bmp", 0)[:, :, None]
    h5 = hibridar(img1, img2, sigma_bajo5, sigma_alto5)
    res = crearPiramideImagen(piramideGaussiana(h5, sigma_bajo5, 4))
    muestraImagen("Piramide gaussiana hibrida", res)
    
    
""" Bonus2: hibridar con color """
def bonus2():
    # Gato y perro
    sigma_bajo1 = 8.5
    sigma_alto1 = 3.0
    img1 = cv2.imread("./imagenes/cat.bmp", 1)
    img2 = cv2.imread("./imagenes/dog.bmp", 1)
    hibridar(img1, img2, sigma_bajo1, sigma_alto1)
    
    # Pájaro y avión
    sigma_bajo2 = 8.0
    sigma_alto2 = 1.1
    img2 = cv2.imread("./imagenes/bird.bmp", 1)
    img1 = cv2.imread("./imagenes/plane.bmp", 1)
    hibridar(img1, img2, sigma_bajo2, sigma_alto2)
    
    # Einstein y marilyn
    sigma_bajo3 = 5.0
    sigma_alto3 = 1.5
    img1 = cv2.imread("./imagenes/marilyn.bmp", 1)
    img2 = cv2.imread("./imagenes/einstein.bmp", 1)
    hibridar(img1, img2, sigma_bajo3, sigma_alto3)
    
    # Pez y submarino
    sigma_bajo4 = 5.0
    sigma_alto4 = 1.0
    img2 = cv2.imread("./imagenes/fish.bmp", 1)
    img1 = cv2.imread("./imagenes/submarine.bmp", 1)
    hibridar(img1, img2, sigma_bajo4, sigma_alto4)
    
    # Bici y moto
    sigma_bajo5 = 7.0
    sigma_alto5 = 2.0
    img1 = cv2.imread("./imagenes/bicycle.bmp", 1)
    img2 = cv2.imread("./imagenes/motorcycle.bmp", 1)
    hibridar(img1, img2, sigma_bajo5, sigma_alto5)
    
""" Bonus3: hibridar con dos imágenes búscadas """
def bonus3():
    img2 = cv2.imread("./imagenes/zapatero.png", 1)
    img1 = cv2.imread("./imagenes/atkinson.png", 1)
    sigma_bajo = 6.0
    sigma_alto = 2.0
    hibridar(img1, img2, sigma_bajo, sigma_alto)

""" Main """
def main():
    # Leemos imágenes en grises y en color
    img_g = cv2.imread("./imagenes/cat.bmp", 0)[:, :, None]
    img_c = cv2.imread("./imagenes/fish.bmp", 1)
    img_c2 = cv2.imread("./imagenes/bird.bmp", 1)
    imgs = [img_g, img_c]
    imgs2 = [img_g, img_c2]
     
    # Bonus1
    sigma = 2.0
    #bonus1(imgs, sigma)
    
    # Ej1
    mask_size = 5
    sigma1 = 1.0
    sigma2 = 3.0
    dx1 = 1  # Índices del nº de derivadas
    dy1 = 0
    dx2 = 0
    dy2 = 1
    borde1 = "edge"
    borde2 = "constant"
    #ej1(imgs, mask_size, sigma1, sigma2, dx1, dy1, dx2, dy2, borde1, borde2)
    
    # Ej2
    sigma1 = 3.0
    sigma2 = 1.2
    niveles = 6
    borde1 = "reflect"
    borde2 = "edge"
    mask_size = 5
    n_escalas = 6
    ej2(imgs, imgs2, sigma1, sigma2, niveles, borde1, borde2, mask_size, n_escalas)

    # Ej3
    #ej3()
    
    # Bonus 2
    #bonus2()
    
    # Bonus 3
    #bonus3()
    
if __name__ == "__main__":
    main()
    
    