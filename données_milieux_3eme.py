import math as m
import numpy as np
import pandas as pd
from pandas.io.sql import table_exists

donnees= np.array([[30,3.62,9.8,1],[30,11.39,12,1],[50,4.25,8,2],[70,9.7,8,3],[30,2.08,8,4],[30,2.67,6,4]])

def f(tableau):
    L=[]
    for i in range(len(tableau)):
        pression=tableau[i][0]
        tableau[i][0]=pression*10**-4
        vitesse = tableau[i][2]*0.01/tableau[i][3]
        L.append(tableau[i][0]/vitesse)
    return L