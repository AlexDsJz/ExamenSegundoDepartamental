""""
    Autor: Dueñas Jiménez Cristian Alexis
    Grupo: 5BM1
    Matería; Visión Artificial
    Fecha: 09 de diciembre de 2022
"""

import numpy as np
import random
import math
import sys
import cv2
from functionsGauss import *
from two_dimentions import draw_lines

"""""
    Se aplicó un filtro Gaussiano para poder eliminar un poco el brillo que tenían los jitomates, dado que eso es un poco de ruido y podías afectar a nuestro resultado.
    Se crearon 2 archivos aparte para poder tener un mejor control de los procesos que se hacen. 
    Uno gestiona el proceso del filtro Gaussiano y el otro el análisis 2D

    NOTA:PARA EJECUTAR EL PROGRAMA TARDA BASTANTE, DEBIDO AL TAMAÑO QUE TIENE. VIENE ADJUNTA UNA IMAGEN AJUSTADA A UN TAMAÑO MAS PEQUEÑO POR SI SE QUIERE HACER UNA PRUEBA RAPIDA.
    DE LO CONTRARIO, SÓLO CAMBIAR EL NOMBRE DE LA IMAGEN QUE SE LEERÁ.
"""""

""""
    Creamos variables fijas, en este caso es el tamaño de nuestro Kernel y el valor de sigma.
    Además tenemos algunas variables como son el valor de intensidad más alto y el más bajo, para la binarización.
"""
MAX = 255
MIN = 0
n = 5
sigma = 1.05

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Clase que almacena todos los elementos de nuestros pixeles
class Point:
    def __init__(self, x, y, z, i=-1, j=-1):
        self.x = x
        self.y = y
        self.z = z
        self.i = i
        self.j = j
        self.cluster = -1
        self.min_distance = sys.float_info.max

    def __str__(self):
        return f"[{self.i}, {self.j}, ({self.x}, {self.y}, {self.z})]"

    def array(self):
        return np.array([self.x, self.y, self.z])


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creamos una función para calcular la distancia Euclidiana, los parámetros son 2 puntos, y se calcula la distancia para cada uno de los ejes.
#Es decir, respecto a X, Y y para Z. (X = Valor de color Rojo, Y = Valor de color verde, Z = Valor de color azul)
def euclidean_distance(point1, point2):
    x = point1.x - point2.x #Asignamos a la variable X la resta de las coordenadas X
    y = point1.y - point2.y #Lo mismo para Y
    z = point1.z - point2.z #Finalmente para Z

    return math.sqrt(x**2 + y**2 + z**2) # D = Raíz((x1-x2)^2 + (x1-x2)^2 + (x1-x2)^2)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Función para el algoritmo KMeans, recibe como parámetro la imagen, el número de procesos que hará y el número de clusters.
def kmeans(image, epochs, clusters):

    #Creamos un conjunto de centroides ya fijos, para que no haya tanto ruido al final del proceso
    initial_centroids = np.array([
        Point(0, 0, 0),
        Point(0, 0, 255),
        Point(0, 255, 0),
        Point(255, 0, 0),
        Point(255, 255, 255),
    ])

    k = clusters #Le asignamos a K la cantidad de clusters
    centroids = {} 
    rows, cols, _ = image.shape #Obtenemos las propiedades de la imagen

    #El algoritmo KMeans determmina los centroides de manera aleatoria
    #En este caso, por cada cluster, vamos a tener un centroide
    for i in range(k): 
        x = random.randint(0,255) #Determinamos un valor aleatorio para cada uno de nuestros colores
        y = random.randint(0,255)
        z = random.randint(0,255)
        centroids[i] = Point(x, y, z, 0, 0) #Guardamos cada centroide generado 

    if initial_centroids is not None: centroids = initial_centroids

    points = []    
    #Recorremos nuestra imagen, le vamos a asignar puntos a nuestros centroides
    for i in range(rows):
        for j in range(cols):
            blue = image[i][j][0] 
            green = image[i][j][1] 
            red = image[i][j][2] 
            points.append(Point(blue, green, red, i, j)) #Se manda el valor de cada componente y la posición

    #Pasamos a la parte de las etapas, recorremos de 0 hasta el número de etapas definido
    for epoch in range(epochs):
        print(f"Etapa {epoch}")
        # Asignamos cada punto a un cluster
        for i in range(k): 
            centroid = centroids[i] #Obtenemos el centroide de cada cluster
            for point in points:
                #Para cada uno de nuestros puntos que obtuvimos, aplicamos la distancia euclidiana entre dicho punto y el centroide que obtuvimos
                distance = euclidean_distance(centroid, point)
                #Hacemos una comparativa para encontrar la distancia mínima, de igual manera es por cada cluster
                if distance < point.min_distance:
                    point.min_distance = distance
                    point.cluster = i
        
        # Calculamos los nuevos centroides
        total_centroids_points = np.zeros(k, dtype=int) #Creamos una matriz de Ceros, que almacena datos de tipo entero y de tamaño k
        sum_centroids_x = np.zeros(k, dtype=int) #Hacemos lo mismo pero para los centroides del color azul
        sum_centroids_y = np.zeros(k, dtype=int) #Para los centroides del color verde
        sum_centroids_z = np.zeros(k, dtype=int) #Y los centroides del color rojo

        for point in points: 
            cluster = point.cluster 
            total_centroids_points[cluster] += 1 
            sum_centroids_x[cluster] += point.x 
            sum_centroids_y[cluster] += point.y
            sum_centroids_z[cluster] += point.z

        total = np.zeros(clusters, dtype=int)
        sum_centroids = np.zeros((clusters, image.shape[2]), dtype=int)

        for point in points:
            cluster = point.cluster
            total[cluster] += 1
            sum_centroids[cluster] += point.array()

        # Asignamos los nuevos centroides
        for i in range(k): #Obtenemos el promedio de los elementos de cada color como nuevo centroide
            x = sum_centroids_x[i] / total_centroids_points[i] 
            y = sum_centroids_y[i] / total_centroids_points[i]
            z = sum_centroids_z[i] / total_centroids_points[i]
            centroids[i] = Point(x, y, z, 0 ,0) #Se asigna a cada cluster

    return points, centroids


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Binarización de imagen
def get_objects(img, umbral):
    rows, cols, _ = img.shape
    bImg = np.zeros((rows, cols), dtype = np.uint8)
    #Funciona muy parecido a la binarización de escala de grises, pero aquí analizamos por cada elemento RGB
    for i in range(rows):
        for j in range(cols):

            b = img[i][j][0]
            g = img[i][j][1]
            r = img[i][j][2]

            if (r > umbral):
                if(g > umbral):
                    if(b > umbral):
                        bImg[i][j] = MAX
            elif r > umbral:
                bImg[i][j] = MIN
            else:
                bImg[i][j] = MAX

    return bImg

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Lo que se busca con esta función es obtener un etiquetado de los objetos que forman parte de nuestra imagen. Es decir, saber en qué pixeles hay más que un blanco. Para ello, necesitamos recorrer
#toda nuestra imagen e ir comparando con los vecinos. 
def classification_elements(image, tag):
    
    rows, cols = image.shape
    mid1 = rows / 2
    mid2 = cols / 2
    active = 0
    flag = False
    dictionary = {}

    for i in range(rows):
        for j in range(cols):
            if i < mid1 and j < mid2: continue
            if image[i][j] == 0:
                if active:
                    tag[i][j] = flag
                else:
                    active = True
                    flag += 1
                    tag[i][j] = flag
            else:
                active = False

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if tag[i][j] != 0:
                        if tag[i + x][j + y] != 0: 
                            if tag[i][j] != tag[i + x][j + y]:
                                tag[i, j] = min(tag[i, j], tag[i + x][j + y])
                                tag[i + x][j + y] = tag[i][j]

    for i in range(rows):
        for j in range(cols):
            if tag[i][j] in dictionary and tag[i][j] != 0:
                dictionary[tag[i][j]]["points"].append(Point(i, j, 1))
                dictionary[tag[i][j]]["count"] += 1
            else:
                dictionary[tag[i][j]] = {
                                        "points": [Point(i, j, 1)],
                                         "count": 1
                                         }

    print("Obteniendo la distancia maxima del objeto 2 y 4...")
    return tag, dictionary


#De los elementos que se obtuvieron con la funcion anterior, se hace un calculo de distancias entre acada pixel con el resto, y estas distancias se guardan en un arreglo.
#Al final se hace una comparacion de todas con todas y la que seamayor e la que tomamos
def get_maximum_distance(elements):
    p = []
    distance_temp = 0
    for i, t in enumerate(elements):

        if elements[t]["count"] < 1000: 
            continue
        points = elements[t]["points"]
        maximum_distance_element = 0
        maximum_distance_element2 = 0
        
        for j in range(len(points)):
            for k in range(j+1, len(points)):
                tempelement = euclidean_distance(points[j], points[k])
                if tempelement > distance_temp:
                    distance_temp = tempelement
                    maximum_distance_element = points[j]
                    maximum_distance_element2 = points[k]
            
        p.append((maximum_distance_element, maximum_distance_element2))
        distance_temp = 0
    
    return p


#Nuestra función main
def main():

    original = cv2.imread("r2.jpg") #Recibimos la imagen
    rows, cols, _ = original.shape
    b,g,r = cv2.split(original)    

    umbral = 100
    clusters = 5
    iterations = 3
    
    """
        Filtro Gaussiano aplicado
    """
    kernel = createKernel(n, sigma)

    imgBB, imgBG, imgBR = padding(b,g,r)
    filterGaussB = np.zeros((rows,cols), dtype=np.uint8)
    filterGaussG = np.zeros((rows,cols), dtype=np.uint8)
    filterGaussR = np.zeros((rows,cols), dtype=np.uint8)
    filter_b,filter_g,filter_r = gaussian_filter(kernel,imgBB,imgBG,imgBR,n,filterGaussB,filterGaussG,filterGaussR,b)

    image_merge = cv2.merge([filter_b, filter_g, filter_r]) #Unimos las imagenes filtradas para cada nivel de intensidad, para poder tener una imagen a color


    """"
        Algoritmo KMeans aplicado
    """
    image_segmented = np.zeros(original.shape, dtype=np.uint8)
    points, centroids = kmeans(image_merge, iterations, clusters)
    centroids = np.array([np.round(centroid.array()) for centroid in centroids])

    for point in points:
        image_segmented[point.i][point.j] = centroids[point.cluster]


    """ Binarización de imagen """
    binarize_img = get_objects(image_segmented, umbral)


    """
        Detección de objetos
    """
    tag = np.zeros((rows, cols), np.uint8)
    et, elements = classification_elements(binarize_img, tag)
    points_getter = get_maximum_distance(elements)

    draw_lines(points_getter, et, rows, original)

    """ Impresión de imágenes """
    cv2.imshow("Original", original)
    cv2.imshow("Filtro Gaussiano", image_merge)
    cv2.imshow("Segmentacion Color", image_segmented)
    cv2.imshow("Objetos pintados", binarize_img)

    cv2.waitKey(0)

main()

