"""
    File name: Tarea4.py
    Author: Maria Paula Rey, EAFIT University
    Email: mpreyb@eafit.edu.co
    Date last modified: 07/03/2022
    Python Version: 3.8
"""

import numpy as np

nump=5
f_0=5
w_0=2*np.pi*f_0
Mr=np.zeros([nump,nump])

t=np.linspace(-(1/f_0),1/f_0,10000)
deltat=t[-1]/len(t)


for n in range(1,nump+1):
    for m in range(1,nump+1):
        suma=0
        for ti in t:
            suma=np.exp((1j*w_0*ti)*(n-m))*deltat+suma
        Mr[n-1][m-1]=round(suma/(1/f_0))
