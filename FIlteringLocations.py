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


#Importing and setting up
import pandas as pd 
gowalla_all = pd.read_csv (r'/Users/sayanbiswas/Desktop/Work/Projects/BAandIBU/Latest/gowalla_checkin.txt') #Importing Gowalla dataset
gowalla_values=gowalla_all.values #Converting gowalla dataset to an array

#Filtering Gowalla data for Paris
gowalla_paris=np.array([]) #To add (lat,long) pairs here
#if lat >=48.8286 and lat <=48.8798), if long <=2.3909 and long >=2.2855)
latitudes_Paris=[]
longitudes_Paris=[]
n_paris=0

for loc in gowalla_values:
    split_line=loc[0].split('\t')
    if(float(split_line[2])>=48.8286 and float(split_line[2])<=48.8798): #latitude
        if(float(split_line[3])<=2.3909 and float(split_line[3])>=2.2855): #longitude     
            latitudes_Paris.append(split_line[2])
            longitudes_Paris.append(split_line[3]) 
            n_paris+=1

w_P = open('Gowalla_Paris.csv', 'w', encoding='utf-8') #Creating file for Paris locations
for i in range(0,len(latitudes_Paris)):
    w_P.write(str(latitudes_Paris[i])+','+str(longitudes_Paris[i])+'\n')

#Filtering Gowalla data for San Francisco
gowalla_SF=np.array([]) #To add (lat,long) SF here
#latitudes (37.7228, 37.7946) and longitudes (-122.5153, -122.3789)
latitudes_SF=[]
longitudes_SF=[]
n_SF=0
for loc in gowalla_values:
    split_line=loc[0].split('\t')
    if(float(split_line[2])>=37.7228 and float(split_line[2])<=37.7946):
        if(float(split_line[3])<=-122.3789 and float(split_line[3])>=-122.5153):        
            latitudes_SF.append(split_line[2])
            longitudes_SF.append(split_line[3])
            n_SF+=1

w_SF = open('Gowalla_SF.csv', 'w', encoding='utf-8') #Creating file for SF locations
for i in range(0,len(latitudes_SF)):
    w_SF.write(str(latitudes_SF[i])+','+str(longitudes_SF[i])+'\n')
