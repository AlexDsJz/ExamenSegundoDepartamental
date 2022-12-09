import numpy as np
import cv2
import math
import sys

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
def get_m(vector1, vector2):
    vector_r = np.cross(vector1, vector2)

    x = (vector_r[0]/vector_r[2])
    y = (vector_r[1]/vector_r[2])
    z = (vector_r[2]/vector_r[2])

    return Point(x,y,z)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creamos nuestra línea
def get_line(u, x):
    return "y = " + str((-u[0] / u[1])) + "x " + str(-u[2] / u[1])
    # return (-u[2] - u[0] * x) / u[1]


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def draw_lines(points_getter, et, rows, img):
    for points in points_getter:
        
        pm = Point((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2, 1)
        pm.x = math.floor(pm.x)
        pm.y = math.floor(pm.y)

        if pm.x < rows / 2: 

            i_l = pm.y
            i_r = pm.y

            while et[pm.x][i_l] != 0: 
                i_l -= 1
            while et[pm.x][i_r] != 0: 
                i_r += 1

            left = Point(pm.x, i_l, 1)
            right = Point(pm.x, i_r, 1)
            line = get_m([left.y, left.x, 1], [right.y, right.x, 1])

            print(f"Coordenadas para la medicion del Objeto 2: P1({left.y}, {left.x}))       P2({right.y}, {right.x})")
            long = pow((right.y-left.y)**2 + (right.x-left.x)**2, 1/2)
            print(f"Longitud de la recta: {long}")
            get_line([line.x, line.y, line.z], 0)
            cv2.line(img, (left.y, left.x), (right.y, right.x), (0, 255, 204), 1)

        else:

            i_l = pm.y
            j_l = pm.x

            i_r = pm.y
            j_r = pm.x

            while et[j_l][i_l] != 0:
                i_l -= 1
                j_l += 1

            while et[j_r][i_r] != 0:
                i_r += 1
                j_r -= 1

            left = Point(j_l, i_l, 1)
            right = Point(j_r, i_r, 1)
            line = get_m([left.y, left.x, 1], [right.y, right.x, 1])

            print(f"Coordenadas para la medicion del objeto 4: P1({left.y}, {left.x}))       P2({right.y}, {right.x})")
            long = pow((right.y-left.y)**2 + (right.x-left.x)**2, 1/2)
            print(f"Longitud de la recta: {long}")
            get_line([line.x, line.y, line.z], 0)
            cv2.line(img, (left.y, left.x), (right.y, right.x), (0, 255, 204), 1)
