import math
import numpy as np

n = 5

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion para la formula de filtro Gaussiano
def getMaskValues(sigma, x, y):
    div = (1/(2 * math.pi * (sigma**2)))
    power = (-((x**2) + (y**2))/2*sigma)
    result = div * math.exp(power)
    return result

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creamos el Kernel
def createKernel(n, sigma):
    filter = np.zeros([n,n,1], np.float32)
    size = int((n-1)/2)
    sum = 0

    for i in range(size,-size-1,-1):
        for j in range(-size, size+1,+1):
            filter[i+size][j+size] = getMaskValues(sigma, i, j)
            sum += filter[i+size][j+size]

    return filter

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Aplicamos el filtro
def gaussian_filter(kernel,imgBB,imgBG,imgBR,n,imgFB,imgFG,imgFR,img):
    rows_filter,cols_filter = img.shape
    matrix_tb = np.zeros((n, n), dtype=np.float32)
    matrix_tg = np.zeros((n, n), dtype=np.float32)
    matrix_tr = np.zeros((n, n), dtype=np.float32)
    
    for i in range(rows_filter):
        for j in range(cols_filter):
            
                col_tb = 0
                col_tg = 0
                col_tr = 0
                matrix_tb = imgBB[i:i+n,j:j+n]
                matrix_tg = imgBG[i:i+n,j:j+n]
                matrix_tr = imgBR[i:i+n,j:j+n]


                for k in range(0,n):
                    for l in range(0,n):
                        col_tb = col_tb + (matrix_tb[k,l] * kernel[k,l])
                        col_tg = col_tg + (matrix_tg[k,l] * kernel[k,l])
                        col_tr = col_tr + (matrix_tr[k,l] * kernel[k,l])

                imgFB[i,j] = col_tb
                imgFG[i,j] = col_tg
                imgFR[i,j] = col_tr
        

    return imgFB,imgFG,imgFR

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Aplicamos bordes a cada imagen de cada color
def padding(imgGB, imgGG, imgGR): 
    r_gray, c_gray = imgGB.shape
    newimgB = np.zeros([n-1+r_gray, n-1+c_gray], dtype=np.uint8)
    newimgG = np.zeros([n-1+r_gray, n-1+c_gray], dtype=np.uint8)
    newimgR = np.zeros([n-1+r_gray, n-1+c_gray], dtype=np.uint8)

    mid = int((n-1)/2)

    for i in range(r_gray):
        for j in range(c_gray):
            newimgB[i+mid, j+mid] = imgGB[i,j]
            newimgG[i+mid, j+mid] = imgGG[i,j]
            newimgR[i+mid, j+mid] = imgGR[i,j]

    return newimgB, newimgG, newimgR