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




#####################!!!!!!!#Comparing statistical utility

RDSlope=np.array([0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,
                  3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5])



###Obfuscation!!!----!!!
####Blahut Arimoto
##Paris
N_sim_paris=5
X=np.array(X_paris)
Noisy_BA_paris=[] #Paris data with BA channels
for i in range(len(RDSlope)):
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    C_paris=BlahutArimotoParis(C0_paris,p0_paris,B=beta,R_BA=10,Num_X=N_X_paris,Num_Y=N_X_paris) #BA channel with slope-parameter beta
    #Storing noisy data for different simulations for current privacy parameter
    noisy_BA_sim={} 
    #Obfuscating with BA
    for nsim in range(N_sim_paris):
        simkey="sim {}".format(nsim) #Referring to the number of simulation of the noisy data
        noisyParisBA=obfuscate_paris(X_samp=paris_loc, C=C_paris, Y=X_paris)[1] #Freq of noisy Paris data under BA mechanism
        noisy_BA_sim[simkey]=noisyParisBA
    Noisy_BA_paris.append(noisy_BA_sim)
 
#Saving Paris locations sanitized with BA
parBA = open('Noisy Paris BA.csv', 'w', encoding='utf-8')
for i in Noisy_BA_paris:
    parBA.write(str(i)+'\n')
parBA.close()
           
    
##SF
N_sim_sf=3
X=np.array(X_sf)
Noisy_BA_sf=[] #SF data with BA channels
for i in range(len(RDSlope)):
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    C_sf=BlahutArimotoSF(C0_sf,p0_sf,B=beta,R_BA=10,Num_X=N_X_sf,Num_Y=N_X_sf) #BA channel with slope-parameter beta
    #Storing noisy data for different simulations for current privacy parameter
    noisy_BA_sim={} 
    #Obfuscating with BA
    for nsim in range(N_sim_sf):
        simkey="sim {}".format(nsim) #Referring to the number of simulation of the noisy data
        noisySFBA=obfuscate_sf(X_samp=sf_loc, C=C_sf, Y=X_sf)[1] #Freq of noisy SF data under BA mechanism
        noisy_BA_sim[simkey]=noisySFBA
    Noisy_BA_sf.append(noisy_BA_sim)

#Saving SF locations sanitized with BA
sfBA = open('Noisy SF BA.csv', 'w', encoding='utf-8')
for i in Noisy_BA_sf:
    sfBA.write(str(i)+'\n')
sfBA.close()
    
######Laplace
##Paris
N_sim_paris=5
X=np.array(X_paris)
Noisy_lap_tight_paris=[] #Paris data with tight Laplace channels
for i in range(len(RDSlope)):
    e_T=eps_tight_paris[i] #Empirical tight epsilon for Paris
    CLap_paris_tight=LaplaceBetter(eps=e_T, X=X_paris, Y=X_paris_wide) #Laplace mechanism for Paris locations with tight Geo-Ind bound
    noisy_lap_tight_sim={}
    #Obfuscating with different channels
    for nsim in range(N_sim_paris):
        simkey="sim {}".format(nsim) #Referring to the number of simulation of the noisy data
        noisyParisLapTight=obfuscate_paris(X_samp=paris_loc, C=CLap_paris_tight, Y=X_paris_wide,XLat=X_paris_lat,XLong=X_paris_long,YLat=X_paris_lat_wide,YLong=X_paris_long_wide)[1] #Freq of noisy Paris data under tight Laplace mechanism
        noisy_lap_tight_sim[simkey]=noisyParisLapTight
    #Noisy_lap_Oya_paris.append(noisy_lap_oya_sim)
    Noisy_lap_tight_paris.append(noisy_lap_tight_sim)
    

#Saving Paris locations sanitized with tight Laplace
paris_lap = open('Noisy Paris tight better Laplace.csv', 'w', encoding='utf-8')
for i in Noisy_lap_tight_paris:
    paris_lap.write(str(i)+'\n')
paris_lap.close()
      
    
##SF
N_sim_sf=3
X=np.array(X_sf)
Noisy_lap_tight_sf=[] #SF data with tight Laplace channels
#Noisy_lap_Oya_sf=[] #SF data with loose Laplace channels
#Call format: 
# DBA=Noisy_BA_sf[i]; DBA["sim j"]= j'th simulation of ith privacy level
for i in range(len(RDSlope)):
    print ("Slope iteration: ",i)
    #e_L=eps_Oya[i] #Loose epsilon 
    e_T=eps_tight_sf[i] #Empirical tight epsilon for SF
    #Different channels
    #CLap_sf_Oya=Laplace(eps=e_L, X=X_sf, Y=X_sf) #Laplace mechanism for SF locations with loose Geo-Ind bound
    CLap_sf_tight=LaplaceBetter(eps=e_T, X=X_sf, Y=X_sf_wide) #Laplace mechanism for SF locations with tight Geo-Ind bound
    noisy_lap_tight_sim={}
    #Obfuscating with different channels
    for nsim in range(N_sim_sf):
        simkey="sim {}".format(nsim) #Referring to the number of simulation of the noisy data
        noisySFLapTight=obfuscate_sf(X_samp=sf_loc, C=CLap_sf_tight, Y=X_sf_wide,XLat=X_sf_lat,XLong=X_sf_long,YLat=X_sf_lat_wide,YLong=X_sf_long_wide)[1] #Freq of noisy SF data under tight Laplace mechanism
        noisy_lap_tight_sim[simkey]=noisySFLapTight
    #Noisy_lap_Oya_sf.append(noisy_lap_oya_sim)
    Noisy_lap_tight_sf.append(noisy_lap_tight_sim)
    

#Saving SFlocations sanitized with tight better Laplace
sf_lap = open('Noisy SF tight better Laplace.csv', 'w', encoding='utf-8')
for i in Noisy_lap_tight_sf:
    sf_lap.write(str(i)+'\n')
sf_lap.close()




#ESTIMATION of original PDF!!!---!!!

##Running IBU on Noisy data to compute statistical utility

#Blahut-Arimoto mechanism
##Paris
N_sim_paris=5
Estimated_Paris_BA=[] #BA
for i in range(len(RDSlope)):
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    CBA_paris=BlahutArimotoParis(C0_paris,p0_paris,B=beta,R_BA=10,Num_X=N_X_paris,Num_Y=N_X_paris) #BA channel with slope-parameter beta
    #Extracting all the simulations for the present level of privacy (beta)
    noise_BA=Noisy_BA_paris[i]
    #noise_tight_lap=Noisy_lap_tight_paris[i]
    #noise_oya_lap=Noisy_lap_Oya_paris[i]
    est_BA={}
    #est_Oya_lap={}
    #est_tight_lap={}
    for n in range(N_sim_paris):
        simkey=simkey="sim {}".format(n)
        NBA=noise_BA[simkey]
        est_BA[simkey]=IBU(p0=p0_paris_unif, q=NBA, C=CBA_paris, R_IBU=10, X=X_paris, Y=X_paris)[1] #Estimation under BA mechanism
    #Storing estimations of simulations in the present privacy level
    Estimated_Paris_BA.append(est_BA) #BA
    
###Saving estimated distributions
#Paris locations sanitized with BA   
par_BA = open('Estimated Paris BA.csv', 'w', encoding='utf-8')
for i in Estimated_Paris_BA:
    par_BA.write(str(i)+'\n')
par_BA.close()

##SF
N_sim_sf=3
Estimated_SF_BA=[] #BA
#Estimated_SF_tight_lap=[] #Tight Laplace
#Estimated_SF_Oya_lap=[] #Loose (Oya) Laplace
for i in range(len(RDSlope)):
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    CBA_sf=BlahutArimotoSF(C0_sf,p0_sf,B=beta,R_BA=10,Num_X=N_X_sf,Num_Y=N_X_sf) #BA channel with slope-parameter beta
    #Extracting all the simulations for the present level of privacy (beta)
    noise_BA=Noisy_BA_sf[i]
    est_BA={}
    for n in range(N_sim_sf):
        simkey=simkey="sim {}".format(n)
        #Extracting the individual simulations of the noisy locations
        NBA=noise_BA[simkey]
        est_BA[simkey]=IBU(p0=p0_sf_unif, q=NBA, C=CBA_sf, R_IBU=10, X=X_sf, Y=X_sf)[1] #Estimation under BA mechanism
    #Storing estimations of simulations in the present privacy level
    Estimated_SF_BA.append(est_BA) #BA
    
    
###Saving estimated distributions
#SF locations sanitized with BA   
sf_BA = open('Estimated SF BA.csv', 'w', encoding='utf-8')
for i in Estimated_SF_BA:
    sf_BA.write(str(i)+'\n')
sf_BA.close()




##Running IBU on Noisy data to compute statistical utility
#Laplace

##Paris
N_sim_paris=5
Estimated_Paris_tight_lap=[] #Tight Laplace
for i in range(len(RDSlope)):
    e_T=eps_tight_paris[i] #Empirical tight epsilon for Paris
    CLap_paris_tight=LaplaceBetter(eps=e_T, X=X_paris, Y=X_paris_wide) #Laplace mechanism for Paris locations with tight Geo-Ind bound
    #Extracting all the simulations for the present level of privacy (beta)
    noise_tight_lap=Noisy_lap_tight_paris[i]
    est_tight_lap={}
    for n in range(N_sim_paris):
        simkey=simkey="sim {}".format(n)
        #Extracting the individual simulations of the noisy locations
        NTL=noise_tight_lap[simkey]
        est_tight_lap[simkey]=IBU(p0=p0_paris_unif, q=NTL, C=CLap_paris_tight, R_IBU=10, X=X_paris, Y=X_paris_wide)[1] #Estimation under tight Lap
    #Storing estimations of simulations in the present privacy level
    Estimated_Paris_tight_lap.append(est_tight_lap) #Tight Laplace


#Paris locations sanitized with tight better Laplace   
par_lap = open('Estimated Paris tight better Laplace.csv', 'w', encoding='utf-8')
for i in Estimated_Paris_tight_lap:
    par_lap.write(str(i)+'\n')
par_lap.close()

    

##SF
N_sim_sf=3
Estimated_SF_tight_lap=[] #Tight Laplace
for i in range(len(RDSlope)):
    #Different channels
    e_T=eps_tight_sf[i] #Empirical tight epsilon for SF
    CLap_sf_tight=LaplaceBetter(eps=e_T, X=X_sf, Y=X_sf_wide) #Laplace mechanism for sF locations with tight Geo-Ind bound
    #Extracting all the simulations for the present level of privacy (beta)
    noise_tight_lap=Noisy_lap_tight_sf[i]
    est_tight_lap={}
    for n in range(N_sim_sf):
        simkey=simkey="sim {}".format(n)
        #Extracting the individual simulations of the noisy locations
        NTL=noise_tight_lap[simkey]
        est_tight_lap[simkey]=IBU(p0=p0_sf_unif, q=NTL, C=CLap_sf_tight, R_IBU=10, X=X_sf, Y=X_sf_wide)[1] #Estimation under tight Lap
    #Storing estimations of simulations in the present privacy level
    Estimated_SF_tight_lap.append(est_tight_lap) #Tight Laplace
    

#SF locations sanitized with tight Laplace   
sf_lap = open('Estimated SF tight better Laplace.csv', 'w', encoding='utf-8')
for i in Estimated_SF_tight_lap:
    sf_lap.write(str(i)+'\n')
sf_lap.close()

    


######################################### 
    
##Comparing the real and the estimated distributions

##Blahut-Arimoto mechanism

#Paris
N_sim_paris=5
BA_paris_KW=[] #KW Distance for estimated dist. under BA 
for i in range(len(RDSlope)):
    print ("Beta iteration", i)
    #Extracting est. dist. of present privacy level
    est_BA=Estimated_Paris_BA[i] 
    KW_BA={}
    #Comparing the simulations of est. dist. under the privacy level
    for n in range(N_sim_paris):
        simkey="sim {}".format(n)
        #Extracting the individual simulations
        eBA=est_BA[simkey]
        #Comaparing BA estimates
        KW_BA[simkey]=KWdist(X=np.array(X_paris), Y=np.array(X_paris), a=p0_paris, b=eBA) #KW Distance
    #Storing the computed statistical utility
    BA_paris_KW.append(KW_BA) #KW Distance for estimated dist. under BA 

    
    
#SF
N_sim_sf=3
BA_sf_KW=[] #KW Distance for estimated dist. under BA 
for i in range(len(RDSlope)):
    print ("Beta iteration", i)
    #Extracting est. dist. of present privacy level
    est_BA=Estimated_SF_BA[i] 
    KW_BA={}
    #Comparing the simulations of est. dist. under the privacy level
    for n in range(N_sim_sf):
        simkey="sim {}".format(n)
        #Extracting the individual simulations
        eBA=est_BA[simkey]
        #Comaparing BA estimates
        KW_BA[simkey]=KWdist(X=np.array(X_sf), Y=np.array(X_sf), a=p0_sf, b=eBA) #KW Distance
    #Storing the computed statistical utility
    BA_sf_KW.append(KW_BA) #KW Distance for estimated dist. under BA 

##Saving statistical utilities
##Paris BA
#KW   
parKW = open('Statistical utility Paris with BA KW.csv', 'w', encoding='utf-8')
for i in BA_paris_KW:
    for n in range(N_sim_paris):
        simkey="sim {}".format(n)
        #parKW.write(simkey+'\n')
        parKW.write(str(i[simkey])+' ')
    parKW.write('\n')
parKW.close()


##SF BA
#KW
sfKW = open('Statistical utility SF with BA KW.csv', 'w', encoding='utf-8')
for i in BA_sf_KW:
    for n in range(N_sim_sf):
        simkey="sim {}".format(n)
        #parKW.write(simkey+'\n')
        sfKW.write(str(i[simkey])+' ')
    sfKW.write('\n')
sfKW.close()    
    
  
    

     
####Statistical utility of only tight better Laplace 
#Paris
TLap_paris_KW=[] #KW Distance for estimated dist. under tight Lap
for i in range(len(RDSlope)):
    #Extracting est. dist. of present privacy level
    est_tight_lap=Estimated_Paris_tight_lap[i]
    KW_TLap={}
    #Comparing the simulations of est. dist. under the privacy level
    for n in range(N_sim_paris):
        simkey="sim {}".format(n)
        #Extracting the individual simulations
        eTLap=est_tight_lap[simkey]
        #Comaparing tight Lap estimates
        KW_TLap[simkey]=KWdist(X=np.array(X_paris), Y=np.array(X_paris_wide), a=p0_paris, b=eTLap) #KW Distance
    #Storing the computed statistical utility
    TLap_paris_KW.append(KW_TLap) #KW Distance for estimated dist. under tight Lap
    
#SF
TLap_sf_KW=[] #KW Distance for estimated dist. under tight Lap
for i in range(len(RDSlope)):
    print ("Beta iteration", i)
    #Extracting est. dist. of present privacy level
    est_tight_lap=Estimated_SF_tight_lap[i]
    KW_TLap={}
    #Comparing the simulations of est. dist. under the privacy level
    for n in range(N_sim_sf):
        simkey="sim {}".format(n)
        #Extracting the individual simulations
        eTLap=est_tight_lap[simkey]
        #Comaparing tight Lap estimates
        KW_TLap[simkey]=KWdist(X=np.array(X_sf), Y=np.array(X_sf_wide), a=p0_sf, b=eTLap) #KW Distance
        KL_TLap[simkey]=KLDiv(a=p0_sf, b=eTLap) #KL Divergence
    #Storing the computed statistical utility
    TLap_sf_KW.append(KW_TLap) #KW Distance for estimated dist. under tight Lap
        
##Saving statistical utilities
##Paris Laplace
#KW   
parKW = open('Statistical utility Paris with Laplace KW.csv', 'w', encoding='utf-8')
for i in TLap_paris_KW:
    for n in range(N_sim_paris):
        simkey="sim {}".format(n)
        #parKW.write(simkey+'\n')
        parKW.write(str(i[simkey])+' ')
    parKW.write('\n')
parKW.close()
