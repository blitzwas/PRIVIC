#Packages
from pyemd import emd
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import time
import pandas as pd 
import csv


from privic_functions import X_paris_lat, X_paris_long, X_paris_lat_wide, X_paris_long_wide, X_paris, X_paris_wide, N_X_paris_lat, N_X_paris_long, X_paris_possible, N_X_paris, p0_paris, C0_paris, CLap_paris
from privic_functions import X_sf_lat, X_sf_long, X_sf_lat_wide, X_sf_long_wide, X_sf, X_sf_wide, N_X_sf_lat, N_X_sf_long, X_sf_possible, N_X_sf, p0_sf, C0_sf, CLap_sf
from privic_functions import ‎eps_tight_paris, eps_tight_sf‎
from privic_functions import ‎PosToValParis, ValToPosParis, PosToValSF, ValToPosSF, BlahutArimotoParis, BlahutArimotoSF, obfuscate_paris, obfuscate_sf, IBU, LaplaceBetter, KWdist




f = open('Gowalla_Paris_(No head).csv', 'r', encoding='utf-8')
w_freq = open('Gowalla_Paris_grid_freq.csv', 'w', encoding='utf-8')
w_loc = open('Gowalla_Paris_grid.csv', 'w', encoding='utf-8')

rdr = csv.reader(f)
count={} #Frequency of each grid
grid_loc={} #Data in each grid
range_lat=sorted(list(np.arange(48.8286,48.8798,0.00427)),reverse=True) #Range of latitudes of Paris check-in
range_long=sorted(list(np.arange(2.2855,2.3909,0.0066)),reverse=True) #Range of longitudes of Paris check-in


for line in rdr:
    for i in range(0,len(range_lat)-1):
        for j in range(0,len(range_long)-1):
            if float(line[0])<=float(range_lat[i]) and float(line[0])>float(range_lat[i+1]):
                if float(line[1])<=float(range_long[j]) and float(line[1])> float(range_long[j+1]):
                    grid_loc[(line[0],line[1])]=np.array([i,j]) #Aggregating the location points
                    w_loc.write(str(i)+','+str(j)+'\n')
                    if (i,j) in count:
                        count[(i,j)]=count[(i,j)]+1
                    else:
                        count[(i,j)]=1

for i in range(0,len(range_lat)):
    for j in range(0,len(range_long)):
        if (i,24-j) not in count:
            count[(i,24-j)]=0
        w_freq.write(str(count[(i,24-j)])+'\n')

        
f.close()
w_freq.close()
w_loc.close()
