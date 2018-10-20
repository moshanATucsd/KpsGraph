#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 02:03:17 2018

@author: dinesh
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import norm
Folder='/home/dinesh/Research/car_render/'
files=glob.glob(Folder + '/*')

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
data_corr = [[0,34],[1,16],[2,35],[3,17],[4,19],[5,1],[6,24],[7,6],[8,21],[9,3],[10,22],[11,4]]

lengths = [[0,2],[2,5],[5,11],[11,8],[8,6],[6,0],[0,1]]
#lengths = [[0,1],[2,3],[5,4],[11,10],[8,9],[6,7],[0,1]]
all_values = list()
for num,name in enumerate(files):
    filename=name+'/anchors.txt'
    
    points3d_all = np.loadtxt(filename,delimiter=',')
    points3d =np.zeros((12,3))
    for r,t in enumerate(data_corr):
        points3d[t[0],:]=points3d_all[t[1],:]
        
    for p1,p2 in enumerate(lengths):
        d=np.linalg.norm(points3d[p2[1],:]-points3d[p2[0],:])
        all_values.append((p1,d))
between_wheel =[]
wheel_head =[]
head_top =[]
between_top =[]
top_back =[]
back_wheel =[]
width =[]

for r,t in enumerate(all_values):
    if t[0]==0:
        between_wheel.append(t[1])
    if t[0]==1:
        wheel_head.append(t[1])
    if t[0]==2:
        head_top.append(t[1])
    if t[0]==3:
        between_top.append(t[1])
    if t[0]==4:
        top_back.append(t[1])
    if t[0]==5:
        back_wheel.append(t[1])
    if t[0]==6:
        width.append(t[1])

plt.hist(between_wheel, bins=100, normed=True, edgecolor='none');
plt.show()
plt.hist(wheel_head, bins=100, normed=True, histtype='stepfilled', edgecolor='none');
plt.show()
plt.hist(head_top, bins=100, normed=True, histtype='stepfilled', edgecolor='none');
plt.show()
plt.hist(between_top, bins=100, normed=True, histtype='stepfilled', edgecolor='none');
plt.show()
plt.hist(top_back, bins=100, normed=True, histtype='stepfilled', edgecolor='none');
plt.show()
plt.hist(back_wheel, bins=100, normed=True, histtype='stepfilled', edgecolor='none');
plt.show()
plt.hist(width, bins=100, normed=True, histtype='stepfilled', edgecolor='none');
plt.show()
  
mean,std=norm.fit(width)
mean,std=norm.fit(between_wheel)

    