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





RDSlope=np.array([0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,
                  3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5])

#Setting 4 (vulnerable location -- isolated on an island) index: 93
#Setting 1 (strong location -- crowded around it) index: 52

str_loc_index=52 #Strong location in a crowd
vul_loc_index=93 #Vulnarable location in isolation

#List of the privacy channels for different levels of geo-indistinguishability
##Channels via BA algorithm
Channels_BA_Paris_island={} #BA mechanism for Paris island locations
#Channels via Laplace (tight, better) mechanism
Channels_Laplace_Paris_island={} #Laplace mechanism for Paris island locations


for i in range(len(RDSlope)):
    print ("Beta iteration: {}".format(i))
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    eps_paris=eps_tight_paris[i] #Geo-Ind parameter for Paris
    eps_sf=eps_tight_sf[i]
    C_BA_paris_island=BlahutArimotoParis(C0_paris, p0_paris_island, B=beta, R_BA=10, Num_X=N_X_paris, Num_Y=N_X_paris) #BA channel for Paris_island
    C_Lap_paris_island=LaplaceBetter(eps=eps_paris, X=X_paris, Y=X_paris_wide) #Laplace channel for Paris
    key="Beta {}".format(beta)
    Channels_BA_Paris_island[key]=C_BA_paris_island
    Channels_Laplace_Paris_island[key]=C_Lap_paris_island






#!!!#!!!#
##Getting the obfuscation probabilities

#Indices of X_paris_wide corresponding to X_paris
L_paris=[]
for i in range(len(X_paris_wide)):
    x1=np.array(X_paris_wide[i])
    for j in range(len((X_paris))):
        x2=np.array(X_paris[j])
        if ((x1==x2).all()):
            L_paris.append(i)


##Setting 1:
##Location of high probability in crowd
#!!PARIS ISLAND
density1_BA_Paris_island=[] #List of obfucation PDFs of strong location in Paris island under BA of different privacy level
density1_Lap_Paris_island=[] #List of obfucation PDFs of strong location in Paris island under Laplace of different privacy level

str_loc_index=52 #Index of strong location in a crowd
str_loc=(PosToValParis(str_loc_index)) #(Latitude, Longitude) of strong location in a crowd


##Setting 4:
##Location of high probability in isolation
#!!PARIS ISLAND
density4_BA_Paris_island=[] #List of obfucation PDFs of vulnerable location in Paris island under BA of different privacy level
density4_Lap_Paris_island=[] #List of obfucation PDFs of vulnerable location in Paris island under Laplace of different privacy level

vul_loc_index=93 #Index of vulnerable location in isolation
vul_loc=(PosToValParis(vul_loc_index)) #(Latitude, Longitude) of vulnerable location in isolation




for i in range(len(RDSlope)):
    print ("Beta iteration: {}".format(i))
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    eps_paris=eps_tight_paris[i] #Geo-Ind parameter for Paris
    key="Beta {}".format(beta)
    #Calling the channels
    CBAPar=Channels_BA_Paris_island[key]
    CLapPar=Channels_Laplace_Paris_island[key]
    ####
    ##Blahut-Arimoto obfuscation probabilities
    #Obfuscation location for Paris location in setting 1
    density1_BA_Paris_island.append(CBAPar[str_loc_index]) 
    #Obfuscation location for Paris location in setting 4
    density4_BA_Paris_island.append(CBAPar[vul_loc_index])
    ####
    
    ##Laplace obfuscation probabilities
    #Obfuscation location for Paris location in setting 1
    density1_Lap_Paris_island.append(CLapPar[str_loc_index][L_paris]) 
    #Obfuscation location for Paris location in setting 4
    density4_Lap_Paris_island.append(CLapPar[vul_loc_index][L_paris]) 
