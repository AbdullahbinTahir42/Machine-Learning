import numpy as np

def euclidean_distance(p1,p2):
    return np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))

def manhathan_distance(p1,p2):
    return np.abs((p1[0]-p2[0]) + (p1[1]-p2[1]))

def Minkowski_distance(p1,p2,p):
    return (np.power(np.abs(p1[0]-p2[0]), p) + np.power(np.abs(p1[1]-p2[1]), p))**(1/p)