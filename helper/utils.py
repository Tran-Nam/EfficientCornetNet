import numpy as np 

def calc_line_length(point1, point2):
    x1, y1 = point1 
    x2, y2 = point2 
    length = ((x1-x2)**2+(y1-y2)**2)**0.5
    return length 

