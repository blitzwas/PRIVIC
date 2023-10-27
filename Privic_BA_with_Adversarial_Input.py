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

#Paris dataset
gowalla_paris_all = pd.read_csv (r'/Users/sayanbiswas/Desktop/Work/Projects/BAandIBU/Latest/Gowalla_Paris_grid.csv') #Importing Gowalla Paris (grid aggregated) dataset
gowalla_paris_values=gowalla_paris_all.values #Converting gowalla dataset to an array
N_paris=len(gowalla_paris_values) #Number of data points in Paris dataset

paris_loc=np.rint(gowalla_paris_values) #Rounding off latitudes and longitudes
paris_latitudes=np.rint(gowalla_paris_values[:,0]) #Latitudes rounded to nearest integer78
paris_longitudes=np.rint(gowalla_paris_values[:,1]) #Longitudes rounded to nearest integer

#Range of Paris dataset
paris_lat_max=np.max(paris_latitudes) #Maximum latitude
paris_lat_min=np.min(paris_latitudes) #Minimum latitude
paris_long_max=np.max(paris_longitudes) #Maximum longitude
paris_long_min=np.min(paris_longitudes) #Minimum longitude
paris_lat_range=np.array([paris_lat_min,paris_lat_max+1]) #Range of latitude
paris_long_range=np.array([paris_long_min,paris_long_max+1]) #Range of longitude

#San Francisco dataset
gowalla_sf_all = pd.read_csv (r'/Users/sayanbiswas/Desktop/Work/Projects/BAandIBU/Latest/Gowalla_SF_grid.csv') #Importing Gowalla SF (grid aggregated) dataset
gowalla_sf_values=gowalla_sf_all.values #Converting gowalla dataset to an array
N_sf=len(gowalla_sf_values) #Number of data points in SF dataset

sf_loc=np.rint(gowalla_sf_values) #Rounding off latitudes and longitudes
sf_latitudes=np.rint(gowalla_sf_values[:,0]) #Latitudes rounded to nearest integer78
sf_longitudes=np.rint(gowalla_sf_values[:,1]) #Longitudes rounded to nearest integer

#Range of SF dataset
sf_lat_max=np.max(sf_latitudes) #Maximum latitude
sf_lat_min=np.min(sf_latitudes) #Minimum latitude
sf_long_max=np.max(sf_longitudes) #Maximum longitude
sf_long_min=np.min(sf_longitudes) #Minimum longitude
sf_lat_range=np.array([sf_lat_min,sf_lat_max+1]) #Range of latitude
sf_long_range=np.array([sf_long_min,sf_long_max+1]) #Range of longitude


#Setup for Paris experiment

X_paris_lat=np.array(range(int(paris_lat_min),int(paris_lat_max+1)))    #Space of latitudes of original data in Paris dataset
X_paris_long=np.array(range(int(paris_long_min),int(paris_long_max+1))) #Space of longitudes of original data in Paris dataset
Y_paris_lat=np.array(range(int(paris_lat_min),int(paris_lat_max+1)))    #Space of latitudes of sanitized data in Paris dataset
Y_paris_long=np.array(range(int(paris_long_min),int(paris_long_max+1))) #Space of longitudes of sanitized data in Paris dataset
N_X_paris_lat=len(X_paris_lat) #Total length of the space of source latitudes of Paris dataset
N_X_paris_long=len(X_paris_long) #Total length of the space of source longitudes of Paris dataset
from itertools import product
X_paris=list(product(X_paris_lat, X_paris_long)) #all possible inpput pairs of (lat,long) of Paris datset
Y_paris=list(product(Y_paris_lat, Y_paris_long)) #all possible output pairs of (lat,long) of Paris dataset

N_X_paris=len(X_paris) #Total length of the space of Paris data

#Entire domain for Paris locations
X_paris_possible=len(X_paris_lat)*len(X_paris_long) #Total possible input locations of Paris dataset
Y_paris_possible=len(Y_paris_lat)*len(Y_paris_long) #Total possible output locations of Paris dataset

#Setup for SF experiment

X_sf_lat=np.array(range(int(sf_lat_min),int(sf_lat_max+1)))    #Space of latitudes of original data in SF dataset
X_sf_long=np.array(range(int(sf_long_min),int(sf_long_max+1))) #Space of longitudes of original data in SF dataset
Y_sf_lat=np.array(range(int(sf_lat_min),int(sf_lat_max+1)))    #Space of latitudes of sanitized data in SF dataset
Y_sf_long=np.array(range(int(sf_long_min),int(sf_long_max+1))) #Space of longitudes of sanitized data in SF dataset
N_X_sf_lat=len(X_sf_lat) #Total length of the space of source latitudes of SF dataset
N_X_sf_long=len(X_sf_long) #Total length of the space of source longitudes of SF dataset
from itertools import product
X_sf=list(product(X_sf_lat, X_sf_long)) #all possible inpput pairs of (lat,long) of SF datset
Y_sf=list(product(Y_sf_lat, Y_sf_long)) #all possible output pairs of (lat,long) of SF dataset


N_X_sf=len(X_sf) #Total length of the space of SF data

#Entire domain for SF locations
X_sf_possible=len(X_sf_lat)*len(X_sf_long) #Total possible input locations of SF dataset
Y_sf_possible=len(Y_sf_lat)*len(Y_sf_long) #Total possible output locations of SF dataset

########################################################
###Expanding the co-domain (to implement Laplace)
#Paris
X_paris_lat_wide=np.array(range(int(paris_lat_min)-5,int(paris_lat_max+1)+5))    #Wider space of latitudes of original data in Paris dataset
X_paris_long_wide=np.array(range(int(paris_long_min)-5,int(paris_long_max+1)+5)) #Wider space of longitudes of original data in Paris dataset
X_paris_wide=list(product(X_paris_lat_wide, X_paris_long_wide)) #Wider space of all possible inpput pairs of (lat,long) of SF datset

#SF
X_sf_lat_wide=np.array(range(int(sf_lat_min)-5,int(sf_lat_max+1)+5))    #Wider space of latitudes of original data in SF dataset
X_sf_long_wide=np.array(range(int(sf_long_min)-5,int(sf_long_max+1)+5)) #Wider space of longitudes of original data in SF dataset
X_sf_wide=list(product(X_sf_lat_wide, X_sf_long_wide)) #Wider space of all possible inpput pairs of (lat,long) of SF datset

########################################################

#Assigning each (latitude,longitude) pair it's positional index
#Converting between 1D pos to 2D val 
#FOR PARIS
def PosToValParis(t,XLat=X_paris_lat,XLong=X_paris_long):
    n=len(XLong)
    #First giving positions of corresponding latitude, longitude
    pos_lat=int((t-(t%n))/n)
    pos_long=int(t%n)
    #Note: (pos_lat,pos_long) is NOT (lat,long)
    lat=int(XLat[pos_lat])
    long=int(XLong[pos_long])
    return lat, long

def ValToPosParis(a,b,XLong=X_paris_long):
    n=len(XLong)
    return (n*a+b)


#FOR SF
def PosToValSF(t,XLat=X_sf_lat,XLong=X_sf_long):
    n=len(XLong)
    #First giving positions of corresponding latitude, longitude
    pos_lat=int((t-(t%n))/n)
    pos_long=int(t%n)
    #Note: (pos_lat,pos_long) is NOT (lat,long)
    lat=int(XLat[pos_lat])
    long=int(XLong[pos_long])
    return lat, long

def ValToPosSF(a,b,XLong=X_sf_long):
    n=len(XLong)
    return (n*a+b)

#######################################################################
#Blahut-Arimoto Algorithm
def BlahutArimotoParis(C0,p0,B,R_BA,Num_X=N_X_paris,Num_Y=N_X_paris):
    ##Inputs:
        #C0=Starting Channel 
        #p0=PDF of source data
        #B=rate-distortion parameter (high values of B implies low avg distortion=QoS and high MI=privacy)
        #R_BA=Number of iterations 
        #Num_X=length of source alphabet
        #Num_Y=length of image alphabet
    ##Output:
        #Returns a channel that computes the minimum MI for a given average distortion

    start_time = time.time()
    #Intermediate storage
    Channels=[] #Storing all the channels till convergence 
    Channels.append(C0) #The first element is the initial matrix

    exp_mat=np.zeros((Num_X,Num_Y)) #Exponential matrix pre-computed 
    for x in range(Num_X):
        x_val=np.array([PosToValParis(x)[0],PosToValParis(x)[1]]) #(lat,long) value at index x
        for y in range(Num_Y):
            y_val=np.array([PosToValParis(y)[0],PosToValParis(y)[1]]) #(lat,long) value at index y
            exp_mat[x,y]=np.exp(-B*(np.linalg.norm(x_val-y_val)))

    for i in range(R_BA): #starting the iteration of BA algorithm
        #print("BA iteration:", i)
        c=np.matmul(p0,Channels[i]) #Computing the output probabilities
        C_next=np.zeros((Num_X,Num_Y)) #Creating a dummy matrix at each step
        for x in range(Num_X): #Rows of the channel
            #print("BA Channel row: ",x)
            x_val=np.array([PosToValParis(x)[0],PosToValParis(x)[1]]) #(lat,long) value at index x
            for y in range(Num_Y): #Columns of the channel
                #print("BA Channel column: ",y)
                y_val=np.array([PosToValParis(y)[0],PosToValParis(y)[1]]) #(lat,long) value at index y
                C_next_num=(c[y])*(np.exp(-B*(np.linalg.norm(x_val-y_val)))) #Numerator of the next channel
                C_next_denom=np.dot(c,(exp_mat)[x,:]) #Denominator of the next channel
                C_next[x,y]=(C_next_num)/(C_next_denom) #Next channel
        Channels.append(C_next)
    print("Run time of BA for Paris: %s seconds ---" % (time.time() - start_time))
    return(Channels[R_BA])



#######################################################################

#Blahut-Arimoto Algorithm
def BlahutArimotoSF(C0,p0,B,R_BA,Num_X=N_X_sf,Num_Y=N_X_sf):
    ##Inputs:
        #C0=Starting Channel 
        #p0=PDF of source data
        #B=rate-distortion parameter (high values of B implies low avg distortion=QoS and high MI=privacy)
        #R_BA=Number of iterations 
        #Num_X=length of source alphabet
        #Num_Y=length of image alphabet
    ##Output:
        #Returns a channel that computes the minimum MI for a given average distortion

    start_time = time.time()
    #Intermediate storage
    Channels=[] #Storing all the channels till convergence 
    Channels.append(C0) #The first element is the initial matrix

    exp_mat=np.zeros((Num_X,Num_Y)) #Exponential matrix pre-computed 
    for x in range(Num_X):
        x_val=np.array([PosToValSF(x)[0],PosToValSF(x)[1]]) #(lat,long) value at index x
        for y in range(Num_Y):
            y_val=np.array([PosToValSF(y)[0],PosToValSF(y)[1]]) #(lat,long) value at index y
            exp_mat[x,y]=np.exp(-B*(np.linalg.norm(x_val-y_val)))

    for i in range(R_BA): #starting the iteration of BA algorithm
        #print("BA iteration:", i)
        c=np.matmul(p0,Channels[i]) #Computing the output probabilities
        C_next=np.zeros((Num_X,Num_Y)) #Creating a dummy matrix at each step
        for x in range(Num_X): #Rows of the channel
            #print("BA Channel row: ",x)
            x_val=np.array([PosToValSF(x)[0],PosToValSF(x)[1]]) #(lat,long) value at index x
            for y in range(Num_Y): #Columns of the channel
                #print("BA Channel column: ",y)
                y_val=np.array([PosToValSF(y)[0],PosToValSF(y)[1]]) #(lat,long) value at index y
                C_next_num=(c[y])*(np.exp(-B*(np.linalg.norm(x_val-y_val)))) #Numerator of the next channel
                C_next_denom=np.dot(c,(exp_mat)[x,:]) #Denominator of the next channel
                C_next[x,y]=(C_next_num)/(C_next_denom) #Next channel
        Channels.append(C_next)
    print("Run time of BA for SF: %s seconds ---" % (time.time() - start_time))
    return(Channels[R_BA])


###################################################################
#Adding noise to data
#PARIS
def obfuscate_paris(X_samp,C,Y,XLat=X_paris_lat,XLong=X_paris_long,YLat=X_paris_lat,YLong=X_paris_long): #Sanitization of data
    ##Inputs :
        #XLat
        #X_samp = sample of original data
        #C = privacy channel used
        #Y = Codomain of the mechanism/Space of noisy data
    Y_samp_paris=[]#Noisy samples
    for pos_x in range(len(X_samp)):
        x=X_samp[pos_x] #Getting the location data
        lat_x=int(x[0]) #Latitude component of x
        long_x=int(x[1]) #Longitude component of x
        #Finding the position of latitude of the data in the source alphabet for latitudes
        pos_lat_x=int(np.where(X_paris_lat==lat_x)[0][0]) 
        #Finding the position of longitude of the data in the source alphabet for longitudes
        pos_long_x=int(np.where(X_paris_long==long_x)[0][0]) 
        pos_x_1D=ValToPosParis(pos_lat_x,pos_long_x,XLong) #Linear index of the (lat,long) pair #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Adding noise
        prob_x=C[pos_x_1D,:] #x is sanitized wrt the PDF of x'th (in linear index) row of the channel 
        #Note Y is a m*n x 2 array, each row having a (lat,long) pair
        y_1D=int(random.choices(list(range(len(Y))),prob_x,k=1)[0]) #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Extracting latitude, longitude components from y_1D
        y_lat=PosToValParis(y_1D,YLat,YLong)[0]
        y_long=PosToValParis(y_1D,YLat,YLong)[1]
        Y_samp_paris.append(np.array([y_lat,y_long]))
    Y_samp_paris=np.array(Y_samp_paris)
    
    #Computing empirical probabilities of the noisy data
    q_freq_paris=np.array([]) #Empirical probability of noisy_samp
    N_Y_paris=len(Y)
    for pos_y in range(N_Y_paris):
        count_y=0 #Count of element y in noisy locations
        y=Y[pos_y]
        print("Obfuscating: Progress {a} out of {b}".format(a=pos_y, b=N_Y_paris))
        Y_truth=(Y_samp_paris==y)
        for nv in (Y_truth):
            count_y=count_y+(np.all(nv))
        #empirical probability of y in noisy_samp
        q_freq_paris=np.append(q_freq_paris,[count_y/(len(Y_samp_paris))])
    return Y_samp_paris, q_freq_paris #Returning the noisy data (in 2D form) and their empirical probabilities 

#SF
def obfuscate_sf(X_samp,C,Y,XLat=X_sf_lat,XLong=X_sf_long,YLat=X_sf_lat,YLong=X_sf_long): #Sanitization of data
    ##Inputs :
        #X_samp = sample of original data
        #C = privacy channel used
        #Y = Codomain of the mechanism/Space of noisy data
    Y_samp_sf=[] #Noisy samples
    for pos_x in range(len(X_samp)):
        x=X_samp[pos_x] #Getting the location data
        lat_x=int(x[0]) #Latitude component of x
        long_x=int(x[1]) #Longitude component of x
        #Finding the position of latitude of the data in the source alphabet for latitudes
        pos_lat_x=int(np.where(XLat==lat_x)[0][0]) 
        #Finding the position of longitude of the data in the source alphabet for longitudes
        pos_long_x=int(np.where(XLong==long_x)[0][0]) 
        pos_x_1D=ValToPosSF(pos_lat_x,pos_long_x,XLong) #Linear index of the (lat,long) pair #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Adding noise
        prob_x=C[pos_x_1D,:] #x is sanitized wrt the PDF of x'th (in linear index) row of the channel 
        #Note Y is a m*n x 2 array, each row having a (lat,long) pair
        y_1D=int(random.choices(list(range(len(Y))),prob_x,k=1)[0]) #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Extracting latitude, longitude components from y_1D
        y_lat=PosToValSF(y_1D,YLat,YLong)[0]
        y_long=PosToValSF(y_1D,YLat,YLong)[1]
        Y_samp_sf.append(np.array([y_lat,y_long]))
    Y_samp_sf=np.array(Y_samp_sf)
    
    #Computing empirical probabilities of the noisy data
    q_freq_sf=np.array([]) #Empirical probability of noisy_samp
    N_Y_sf=len(Y)
    for pos_y in range(N_Y_sf):
        y=Y[pos_y]
        count_y=0 #Count of element y in noisy locations
        print("Obfuscating: Progress {a} out of {b}".format(a=pos_y, b=N_Y_sf))
        Y_truth=(Y_samp_sf==y)
        for nv in (Y_truth):
            count_y=count_y+(np.all(nv))
        #empirical probability of y in noisy_samp
        q_freq_sf=np.append(q_freq_sf,[count_y/(len(Y_samp_sf))])
    return Y_samp_sf, q_freq_sf #Returning the noisy data (in 2D form) and their empirical probabilities 


##################################################################

#Iterative Bayesian Update
def IBU(p0,q,C,R_IBU,X,Y): #Estimating original distribution
    ##Inputs :
        #p0 = initial guess for the original PDF (full support)
        #q = the empirical PDF of the observed (noisy) data
        #C = the privacy mechanism used
        #R_IBU = number of iterationss' between consequtive iteration
        #X = domain of original data
        #Y = domain of noisy data
    ##Output :
        #Estimate of the original PDF
    start_time = time.time()
    Num_X=len(X)
    Num_Y=len(Y)
    PDFs=[p0] #List of estimated distributions
    for i in tqdm(range(R_IBU)):
        #print ("IBU iteration:", i)
        p=PDFs[i]
        p_next=np.zeros(Num_X)
        for x in range(len(X)): #computing the x'th component of the estimated PDF
            sum_ibu=0
            for y in range(len(Y)):
                num=q[y]*C[x,y]*p[x]
                denom=0
                for z in range(len(X)):
                    denom+=(p[z]*C[z,y])
                sum_ibu=(sum_ibu)+(num/denom)
            p_next[x]=sum_ibu
        PDFs.append(p_next)
    print("Run time of IBU: %s seconds ---" % (time.time() - start_time))
    return PDFs, PDFs[R_IBU]
    
    
####################################################
#CREATING A LINEAR PDF FOR 2D DATA for Paris
p0_paris=np.array([]) #Empirical probability of paris data
N_X_paris=len(X_paris)
for pos_x in tqdm(range(N_X_paris)):
    x=X_paris[pos_x]
    #print("Progress {a} out of {b}".format(a=pos_x, b=N_X_paris))
    freq_x=0 #empirical probability of x in Paris data
    for y in paris_loc:
        comp=(x==y)
        if np.all(comp): #Testing if both elements of x and y are the same
            freq_x+=1
    p0_paris=np.append(p0_paris,[freq_x])
p0_paris=p0_paris/(N_paris) #Normalising frequencies to give empirical probabilities

#CREATING A LINEAR PDF FOR 2D DATA for SF
p0_sf=np.array([]) #Empirical probability of paris data
N_X_sf=len(X_sf)
for pos_x in tqdm(range(N_X_sf)):
    x=X_sf[pos_x]
    #print("Progress {a} out of {b}".format(a=pos_x, b=N_X_sf))
    freq_x=0 #empirical probability of x in Paris data
    for y in sf_loc:
        comp=(x==y)
        if np.all(comp): #Testing if both elements of x and y are the same
            freq_x+=1
    p0_sf=np.append(p0_sf,[freq_x])
p0_sf=p0_sf/(N_sf) #Normalising frequencies to give empirical probabilities

##############################################

#Reshaping and initualizing uniform channels
C0_paris=np.reshape(np.array([1/N_X_paris]*(N_X_paris**2)),(N_X_paris,N_X_paris)) #Initial channel to start BA for Paris
p0_paris_unif=np.array([1/N_X_paris]*N_X_paris) #Initial guess of original distribution of Paris
C0_sf=np.reshape(np.array([1/N_X_sf]*(N_X_sf**2)),(N_X_sf,N_X_sf)) #Initial channel to start BA for SF
p0_sf_unif=np.array([1/N_X_sf]*N_X_sf) #Initial guess of original distribution of SF


#################################################################################
#Laplace mechanism
#Laplace
def Laplace(eps,X,Y):        #eps= privacy parameter
        #X= domain of original data
        #Y= domain of noisy data
    C=np.zeros((len(X),len(Y)))
    norm_const=float((eps**2)/(2*(np.pi))) #Normalising constant
    #Normalizer=[] #Normalizing constant for each value of X
    for i in range(len(X)):
        for j in range(len(Y)):
            C[i,j]=float((norm_const)*np.exp(-eps*np.linalg.norm(np.array(X[i])-np.array(Y[j]))))
    return (C)

#################################################################################
#Laplace mechanism
#Laplace better
def LaplaceBetter(eps,X,Y):        #eps= privacy parameter
        #X= domain of original data (lat,long)
        #Y= domain of noisy data (lat,long)
    C=np.zeros((len(X),len(Y)))
    #float((eps**2)/(2*(np.pi))) #Normalising constant
    Normalizer=[] #Normalizing constant for each value of X
    for i in range(len(X)):
        n=0
        for j in range(len(Y)):
            n+=float(np.exp(-eps*np.linalg.norm(np.array(X[i])-np.array(Y[j]))))
        Normalizer.append(1/n)
    for i in range(len(X)):
        norm_const=Normalizer[i]
        for j in range(len(Y)):
            C[i,j]=float((norm_const)*np.exp(-eps*np.linalg.norm(np.array(X[i])-np.array(Y[j]))))
    return (C)




#############################################################################
##Extracting different channels for locations of 0,high, and isolated probabilities
#Testing for different slope parameters
RDSlope=np.array([0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,
                  3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5])
eps_Oya=2*RDSlope
#Empirical tight epsilons for Paris 
eps_tight_paris=np.array([0.392818028068501,
0.7813925715019457,
1.1605111783813877,
1.522997813452586,
1.9201735805617113,
2.318468681063899,
2.7235710043448096,
3.131913318373869,
3.539017362258331,
3.946218013335979,
4.353319497725823,
4.7597883617851595,
5.1650376600730725,
5.568215644792336,
5.967803359717294,
6.3610873573101845,
6.741703436218012,
7.094334323347863,
7.398355371524807,
7.706459118837066,
8.109656883411978,
8.512103108219597,
8.914469010971294,
9.316911956393886,
9.719415985071155
])
#Empirical tight epsilons for SF
eps_tight_sf=np.array([0.3986617268975043,
0.7883813903959236,
1.1750263394594984,
1.5677798334718995,
1.9816354129563862,
2.3881286826627117,
2.7901762206493075,
3.18914410820109,
3.587243587425301,
3.986583121714903,
4.3873489652913085,
4.78887068465851,
5.1906269545833394,
5.59231245598343,
5.993791505639457,
6.39501755552611,
6.795978710824691,
7.196667510002639,
7.597048656480873,
7.997072913065153,
8.3968159232846,
8.796488394614757,
9.19621125393365,
9.595978444142562,
9.99579069328416
])
#Channels

###Paris
b=19 #Index of privacy parameter we are interested in
beta=RDSlope[b] #Slope parameter for BA
epsilon=eps_tight_paris[b] #Geo-Ind parameter for Laplace better
CLap_paris=LaplaceBetter(eps=epsilon, X=X_paris, Y=X_paris_wide) #Laplace mechanism for Paris with tight Geo-Ind bound
CBA_paris=BlahutArimotoParis(C0_paris,p0_paris,B=beta,R_BA=10,Num_X=N_X_paris,Num_Y=N_X_paris) #BA channel for Paris with slope-parameter beta

###SF
b=19 #Index of privacy parameter we are interested in
beta=RDSlope[b] #Slope parameter for BA
epsilon=eps_tight_sf[b] #Geo-Ind parameter for Laplace better
CLap_sf=LaplaceBetter(eps=epsilon, X=X_sf, Y=X_sf_wide) #Laplace mechanism for SF with tight Geo-Ind bound
CBA_sf=BlahutArimotoSF(C0_sf,p0_sf,B=beta,R_BA=10,Num_X=N_X_sf,Num_Y=N_X_sf) #BA channel for SF with slope-parameter beta
        
###########################################

###!!!!Finding behaviour of channels!!!!

#List of the privacy channels for different levels of geo-indistinguishability
##Channels via BA algorithm
Channels_BA_Paris={} #for Paris locations
Channels_BA_SF={} #for SF locations
#Channels via Laplace (tight, better) mechanism
Channels_Laplace_Paris={} #for Paris locations
Channels_Laplace_SF={} #for SF locations

for i in range(len(RDSlope)):
    print ("Beta iteration: {}".format(i))
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    eps_paris=eps_tight_paris[i] #Geo-Ind parameter for Paris
    eps_sf=eps_tight_sf[i]
    C_BA_paris=BlahutArimotoParis(C0_paris, p0_paris, B=beta, R_BA=10, Num_X=N_X_paris, Num_Y=N_X_paris) #BA channel for Paris
    C_BA_sf=BlahutArimotoSF(C0_sf, p0_sf, B=beta, R_BA=10, Num_X=N_X_sf, Num_Y=N_X_sf) #BA channel for SF
    C_Lap_paris=LaplaceBetter(eps=eps_paris, X=X_paris, Y=X_paris_wide) #Laplace channel for Paris
    C_Lap_sf=LaplaceBetter(eps=eps_sf, X=X_sf, Y=X_sf_wide) #Laplace channel for SF
    key="Beta {}".format(beta)
    Channels_BA_Paris[key]=C_BA_paris
    Channels_BA_SF[key]=C_BA_sf
    Channels_Laplace_Paris[key]=C_Lap_paris
    Channels_Laplace_SF[key]=C_Lap_sf





##Setting 1:
##Location of high probability in crowd
#Location points:
#loc1_par, loc1_sf: (latitude, longitude) for Paris and SF in setting 1
#!!
#!!PARIS
density1_BA_Paris=[] #List of obfucation PDFs of setting 1 location in Paris under BA of different privacy level
density1_Lap_Paris=[] #List of obfucation PDFs of setting 1 location in Paris under Laplace of different privacy level

l1_paris_index=52 #Crowded location index in Paris w.h.p #!!!!!DONE
l1_paris=(PosToValParis(l1_paris_index)) ##Crowded location index in Paris w.h.p


#!!SF
density1_BA_SF=[] #List of obfucation PDFs of setting 1 location in SF under BA of different privacy level
density1_Lap_SF=[] #List of obfucation PDFs of setting 1 location in SF under Laplace of different privacy level

l1_sf_index=52 #Crowded location index in SF w.h.p #!!!!!DONE
l1_sf=(PosToValSF(l1_sf_index)) ##Crowded location index in SF w.h.p

##Setting 2:
##Location of low probability in crowd
#Location points:
#loc2_par, loc2_sf: (latitude, longitude) for Paris and SF in setting 2
#!! 
#!!PARIS
density2_BA_Paris=[] #List of obfucation PDFs of setting 2 location in Paris under BA of different privacy level
density2_Lap_Paris=[] #List of obfucation PDFs of setting 2 location in Paris under Laplace of different privacy level

l2_paris_index=17 #Crowded location index in Paris w.l.p #!!!!!DONE
l2_paris=(PosToValParis(l2_paris_index)) ##Crowded location index in Paris w.l.p

#!!SF
density2_BA_SF=[] #List of obfucation PDFs of setting 2 location in SF under BA of different privacy level
density2_Lap_SF=[] #List of obfucation PDFs of setting 2 location in SF under Laplace of different privacy level

l2_sf_index=127 #Crowded location index in SF w.l.p #!!!!!DONE
l2_sf=(PosToValSF(l2_sf_index)) ##Crowded location index in SF w.l.p

##Setting 3:
##Location of high probability in isolation
#Location points:
#loc3_par, loc3_sf: (latitude, longitude) for Paris and SF in setting 3
#!!
#!!PARIS
density3_BA_Paris=[] #List of obfucation PDFs of setting 3 location in Paris under BA of different privacy level
density3_Lap_Paris=[] #List of obfucation PDFs of setting 3 location in Paris under Laplace of different privacy level

l3_paris_index=73 #Isolated location index in Paris w.h.p #!!!!!DONE
l3_paris=(PosToValParis(l3_paris_index)) ##Isolated location index in Paris w.h.p

#!!SF
density3_BA_SF=[] #List of obfucation PDFs of setting 3 location in SF under BA of different privacy level
density3_Lap_SF=[] #List of obfucation PDFs of setting 3 location in SF under Laplace of different privacy level

l3_sf_index=159 #Isolated location index in SF w.h.p #!!!!!DONE
l3_sf=(PosToValSF(l3_sf_index)) ##Isolated location index in SF w.h.p

##Setting 4:
##Location of low probability in isolation
#Location points:
#loc4_par, loc4_sf: (latitude, longitude) for Paris and SF in setting 4
#!!
#!!PARIS
density4_BA_Paris=[] #List of obfucation PDFs of setting 4 location in Paris under BA of different privacy level
density4_Lap_Paris=[] #List of obfucation PDFs of setting 4 location in Paris under Laplace of different privacy level


l4_paris_index=101 #Isolated location index in Paris w.l.p #!!!!!DONE
l4_paris=(PosToValParis(l4_paris_index)) ##Isolated location index in Paris w.l.p

#!!SF
density4_BA_SF=[] #List of obfucation PDFs of setting 4 location in SF under BA of different privacy level
density4_Lap_SF=[] #List of obfucation PDFs of setting 4 location in SF under Laplace of different privacy level

l4_sf_index=333 #Crowded location index in SF w.l.p #!!!!!DONE
l4_sf=(PosToValSF(l4_sf_index)) ##Crowded location index in SF w.l.p


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
#print(L_paris)
#Indices of X_sf_wide corresponding to X_sf
L_sf=[]
for i in range(len(X_sf_wide)):
    x1=np.array(X_sf_wide[i])
    for j in range(len((X_sf))):
        x2=np.array(X_sf[j])
        if ((x1==x2).all()):
            L_sf.append(i)
#print(L_sf)


for i in range(len(RDSlope)):
    print ("Beta iteration: {}".format(i))
    beta=RDSlope[i] #Slope parameter for Blahut-Arimoto
    eps_paris=eps_tight_paris[i] #Geo-Ind parameter for Paris
    eps_sf=eps_tight_sf[i]
    key="Beta {}".format(beta)
    #Calling the channels
    CBAPar=Channels_BA_Paris[key]
    CBASF=Channels_BA_SF[key]
    CLapPar=Channels_Laplace_Paris[key]
    CLapSF=Channels_Laplace_SF[key]
    ####
    ##Blahut-Arimoto obfuscation probabilities
    #Obfuscation location for Paris location in setting 1
    density1_BA_Paris.append(CBAPar[l1_paris_index]) 
    #Obfuscation location for SF location in setting 1
    density1_BA_SF.append(CBASF[l1_sf_index]) 
    #Obfuscation location for Paris location in setting 2
    density2_BA_Paris.append(CBAPar[l2_paris_index]) 
    #Obfuscation location for SF location in setting 2
    density2_BA_SF.append(CBASF[l2_sf_index]) 
    #Obfuscation location for Paris location in setting 3
    density3_BA_Paris.append(CBAPar[l3_paris_index]) 
    #Obfuscation location for SF location in setting 3
    density3_BA_SF.append(CBASF[l3_sf_index]) 
    #Obfuscation location for Paris location in setting 4
    density4_BA_Paris.append(CBAPar[l4_paris_index]) 
    #Obfuscation location for SF location in setting 4
    density4_BA_SF.append(CBASF[l4_sf_index]) 
    ####
    
    ##Laplace obfuscation probabilities
    #Obfuscation location for Paris location in setting 1
    density1_Lap_Paris.append(CLapPar[l1_paris_index][L_paris]) 
    #Obfuscation location for SF location in setting 1
    density1_Lap_SF.append(CLapSF[l1_sf_index][L_sf]) 
    #Obfuscation location for Paris location in setting 2
    density2_Lap_Paris.append(CLapPar[l2_paris_index][L_paris]) 
    #Obfuscation location for SF location in setting 2
    density2_Lap_SF.append(CLapSF[l2_sf_index][L_sf]) 
    #Obfuscation location for Paris location in setting 3
    density3_Lap_Paris.append(CLapPar[l3_paris_index][L_paris]) 
    #Obfuscation location for SF location in setting 3
    density3_Lap_SF.append(CLapSF[l3_sf_index][L_sf]) 
    #Obfuscation location for Paris location in setting 4
    density4_Lap_Paris.append(CLapPar[l4_paris_index][L_paris]) 
    #Obfuscation location for SF location in setting 4
    density4_Lap_SF.append(CLapSF[l4_sf_index][L_sf]) 


###########################################
priv=0
#plt.plot(density3_Lap_Paris[priv],color="red")
plt.plot(density3_BA_SF[priv],color="blue")
plt.plot(density4_BA_SF[priv],color="black")
plt.show()

priv=0
plt.plot(density3_Lap_SF[priv],color="red")
plt.plot(density3_BA_SF[priv],color="blue")
plt.show()
####


#Saving 2D obfuscation probabilities
##########~~~~~~~!!!
##BA!!!

##~~~~~~PARIS


#Beta=0.2

#Paris: Setting 1
b=0 #For beta value 0.2
ob1_BA_paris_beta_0_2=np.reshape(density1_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_BA_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_BA_paris_beta_0_2[i][0])+'\n')
w.close()

#Paris: Setting 2
b=0 #For beta value 0.2
ob2_BA_paris_beta_0_2=np.reshape(density2_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_BA_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_BA_paris_beta_0_2[i][0])+'\n')
w.close()

#Paris: Setting 3
b=0 #For beta value 0.2
ob3_BA_paris_beta_0_2=np.reshape(density3_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_BA_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_BA_paris_beta_0_2[i][0])+'\n')
w.close()

#Paris: Setting 4
b=0 #For beta value 0.2
ob4_BA_paris_beta_0_2=np.reshape(density4_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_BA_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_BA_paris_beta_0_2[i][0])+'\n')
w.close()
#~~~

#Beta=0.6

#Paris: Setting 1
b=2 #For beta value 0.6
ob1_BA_paris_beta_0_6=np.reshape(density1_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_BA_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_BA_paris_beta_0_6[i][0])+'\n')
w.close()

#Paris: Setting 2
b=2 #For beta value 0.6
ob2_BA_paris_beta_0_6=np.reshape(density2_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_BA_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_BA_paris_beta_0_6[i][0])+'\n')
w.close()

#Paris: Setting 3
b=2 #For beta value 0.2
ob3_BA_paris_beta_0_6=np.reshape(density3_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_BA_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_BA_paris_beta_0_6[i][0])+'\n')
w.close()

#Paris: Setting 4
b=2 #For beta value 0.6
ob4_BA_paris_beta_0_6=np.reshape(density4_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_BA_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_BA_paris_beta_0_6[i][0])+'\n')
w.close()

#~~~

#Beta=1

#Paris: Setting 1
b=4 #For beta value 1
ob1_BA_paris_beta_1_0=np.reshape(density1_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_BA_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_BA_paris_beta_1_0[i][0])+'\n')
w.close()

#Paris: Setting 2
b=4 #For beta value 1
ob2_BA_paris_beta_1_0=np.reshape(density2_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_BA_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_BA_paris_beta_1_0[i][0])+'\n')
w.close()

#Paris: Setting 3
b=4 #For beta value 1
ob3_BA_paris_beta_1_0=np.reshape(density3_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_BA_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_BA_paris_beta_1_0[i][0])+'\n')
w.close()

#Paris: Setting 4
b=4 #For beta value 1
ob4_BA_paris_beta_1_0=np.reshape(density4_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_BA_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_BA_paris_beta_1_0[i][0])+'\n')
w.close()
#~~~

#Beta=2

#Paris: Setting 1
b=9 #For beta value 2
ob1_BA_paris_beta_2_0=np.reshape(density1_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_BA_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_BA_paris_beta_2_0[i][0])+'\n')
w.close()

#Paris: Setting 2
b=9 #For beta value 2
ob2_BA_paris_beta_2_0=np.reshape(density2_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_BA_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_BA_paris_beta_2_0[i][0])+'\n')
w.close()

#Paris: Setting 3
b=9 #For beta value 2
ob3_BA_paris_beta_2_0=np.reshape(density3_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_BA_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_BA_paris_beta_2_0[i][0])+'\n')
w.close()

#Paris: Setting 4
b=9 #For beta value 2
ob4_BA_paris_beta_2_0=np.reshape(density4_BA_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_BA_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_BA_paris_beta_2_0[i][0])+'\n')
w.close()




##~~~~~~SAN FRANCISCO


#Beta=0.2

#SF: Setting 1
b=0 #For beta value 0.2
ob1_BA_sf_beta_0_2=np.reshape(density1_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_BA_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_BA_sf_beta_0_2[i][0])+'\n')
w.close()

#SF: Setting 2
b=0 #For beta value 0.2
ob2_BA_sf_beta_0_2=np.reshape(density2_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_BA_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_BA_sf_beta_0_2[i][0])+'\n')
w.close()

#SF: Setting 3
b=0 #For beta value 0.2
ob3_BA_sf_beta_0_2=np.reshape(density3_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_BA_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_BA_sf_beta_0_2[i][0])+'\n')
w.close()

#SF: Setting 4
b=0 #For beta value 0.2
ob4_BA_sf_beta_0_2=np.reshape(density4_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_BA_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_BA_sf_beta_0_2[i][0])+'\n')
w.close()
#~~~

#Beta=0.6

#SF: Setting 1
b=2 #For beta value 0.6
ob1_BA_sf_beta_0_6=np.reshape(density1_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_BA_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_BA_sf_beta_0_6[i][0])+'\n')
w.close()

#SF: Setting 2
b=2 #For beta value 0.6
ob2_BA_sf_beta_0_6=np.reshape(density2_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_BA_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_BA_sf_beta_0_6[i][0])+'\n')
w.close()

#SF: Setting 3
b=2 #For beta value 0.2
ob3_BA_sf_beta_0_6=np.reshape(density3_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_BA_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_BA_sf_beta_0_6[i][0])+'\n')
w.close()

#SF: Setting 4
b=2 #For beta value 0.6
ob4_BA_sf_beta_0_6=np.reshape(density4_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_BA_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_BA_sf_beta_0_6[i][0])+'\n')
w.close()

#~~~

#Beta=1

#SF: Setting 1
b=4 #For beta value 1
ob1_BA_sf_beta_1_0=np.reshape(density1_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_BA_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_BA_sf_beta_1_0[i][0])+'\n')
w.close()

#SF: Setting 2
b=4 #For beta value 1
ob2_BA_sf_beta_1_0=np.reshape(density2_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_BA_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_BA_sf_beta_1_0[i][0])+'\n')
w.close()

#SF: Setting 3
b=4 #For beta value 1
ob3_BA_sf_beta_1_0=np.reshape(density3_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_BA_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_BA_sf_beta_1_0[i][0])+'\n')
w.close()

#SF: Setting 4
b=4 #For beta value 1
ob4_BA_sf_beta_1_0=np.reshape(density4_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF BA Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_BA_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_BA_sf_beta_1_0[i][0])+'\n')
w.close()
#~~~

#Beta=2

#SF: Setting 1
b=9 #For beta value 2
ob1_BA_sf_beta_2_0=np.reshape(density1_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_BA_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_BA_sf_beta_2_0[i][0])+'\n')
w.close()

#SF: Setting 2
b=9 #For beta value 2
ob2_BA_sf_beta_2_0=np.reshape(density2_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_BA_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_BA_sf_beta_2_0[i][0])+'\n')
w.close()

#SF: Setting 3
b=9 #For beta value 2
ob3_BA_sf_beta_2_0=np.reshape(density3_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_BA_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_BA_sf_beta_2_0[i][0])+'\n')
w.close()

#SF: Setting 4
b=9 #For beta value 2
ob4_BA_sf_beta_2_0=np.reshape(density4_BA_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF BA Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_BA_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_BA_sf_beta_2_0[i][0])+'\n')
w.close()


##########~~~~~~~!!!

##Laplace!!!

##~~~~~~PARIS


#Beta=0.2

#Paris: Setting 1
b=0 #For beta value 0.2
ob1_Lap_paris_beta_0_2=np.reshape(density1_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_Lap_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_Lap_paris_beta_0_2[i][0])+'\n')
w.close()

#Paris: Setting 2
b=0 #For beta value 0.2
ob2_Lap_paris_beta_0_2=np.reshape(density2_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_Lap_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_Lap_paris_beta_0_2[i][0])+'\n')
w.close()

#Paris: Setting 3
b=0 #For beta value 0.2
ob3_Lap_paris_beta_0_2=np.reshape(density3_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_Lap_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_Lap_paris_beta_0_2[i][0])+'\n')
w.close()

#Paris: Setting 4
b=0 #For beta value 0.2
ob4_Lap_paris_beta_0_2=np.reshape(density4_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_Lap_paris_beta_0_2[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_Lap_paris_beta_0_2[i][0])+'\n')
w.close()
#~~~

#Beta=0.6

#Paris: Setting 1
b=2 #For beta value 0.6
ob1_Lap_paris_beta_0_6=np.reshape(density1_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_Lap_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_Lap_paris_beta_0_6[i][0])+'\n')
w.close()

#Paris: Setting 2
b=2 #For beta value 0.6
ob2_Lap_paris_beta_0_6=np.reshape(density2_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_Lap_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_Lap_paris_beta_0_6[i][0])+'\n')
w.close()

#Paris: Setting 3
b=2 #For beta value 0.2
ob3_Lap_paris_beta_0_6=np.reshape(density3_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_Lap_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_Lap_paris_beta_0_6[i][0])+'\n')
w.close()

#Paris: Setting 4
b=2 #For beta value 0.6
ob4_Lap_paris_beta_0_6=np.reshape(density4_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_Lap_paris_beta_0_6[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_Lap_paris_beta_0_6[i][0])+'\n')
w.close()

#~~~

#Beta=1

#Paris: Setting 1
b=4 #For beta value 1
ob1_Lap_paris_beta_1_0=np.reshape(density1_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_Lap_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_Lap_paris_beta_1_0[i][0])+'\n')
w.close()

#Paris: Setting 2
b=4 #For beta value 1
ob2_Lap_paris_beta_1_0=np.reshape(density2_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_Lap_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_Lap_paris_beta_1_0[i][0])+'\n')
w.close()

#Paris: Setting 3
b=4 #For beta value 1
ob3_Lap_paris_beta_1_0=np.reshape(density3_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_Lap_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_Lap_paris_beta_1_0[i][0])+'\n')
w.close()

#Paris: Setting 4
b=4 #For beta value 1
ob4_Lap_paris_beta_1_0=np.reshape(density4_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_Lap_paris_beta_1_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_Lap_paris_beta_1_0[i][0])+'\n')
w.close()
#~~~

#Beta=2

#Paris: Setting 1
b=9 #For beta value 2
ob1_Lap_paris_beta_2_0=np.reshape(density1_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 1 Paris Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob1_Lap_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob1_Lap_paris_beta_2_0[i][0])+'\n')
w.close()

#Paris: Setting 2
b=9 #For beta value 2
ob2_Lap_paris_beta_2_0=np.reshape(density2_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 2 Paris Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob2_Lap_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob2_Lap_paris_beta_2_0[i][0])+'\n')
w.close()

#Paris: Setting 3
b=9 #For beta value 2
ob3_Lap_paris_beta_2_0=np.reshape(density3_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 3 Paris Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob3_Lap_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob3_Lap_paris_beta_2_0[i][0])+'\n')
w.close()

#Paris: Setting 4
b=9 #For beta value 2
ob4_Lap_paris_beta_2_0=np.reshape(density4_Lap_Paris[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability Setting 4 Paris Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
         w.write(str(ob4_Lap_paris_beta_2_0[i][N_X_paris_long-j])+' ')
    w.write(str(ob4_Lap_paris_beta_2_0[i][0])+'\n')
w.close()




##~~~~~~SAN FRANCISCO


#Beta=0.2

#SF: Setting 1
b=0 #For beta value 0.2
ob1_Lap_sf_beta_0_2=np.reshape(density1_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_Lap_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_Lap_sf_beta_0_2[i][0])+'\n')
w.close()

#SF: Setting 2
b=0 #For beta value 0.2
ob2_Lap_sf_beta_0_2=np.reshape(density2_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_Lap_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_Lap_sf_beta_0_2[i][0])+'\n')
w.close()

#SF: Setting 3
b=0 #For beta value 0.2
ob3_Lap_sf_beta_0_2=np.reshape(density3_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_Lap_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_Lap_sf_beta_0_2[i][0])+'\n')
w.close()

#SF: Setting 4
b=0 #For beta value 0.2
ob4_Lap_sf_beta_0_2=np.reshape(density4_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_Lap_sf_beta_0_2[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_Lap_sf_beta_0_2[i][0])+'\n')
w.close()
#~~~

#Beta=0.6

#SF: Setting 1
b=2 #For beta value 0.6
ob1_Lap_sf_beta_0_6=np.reshape(density1_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_Lap_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_Lap_sf_beta_0_6[i][0])+'\n')
w.close()

#SF: Setting 2
b=2 #For beta value 0.6
ob2_Lap_sf_beta_0_6=np.reshape(density2_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_Lap_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_Lap_sf_beta_0_6[i][0])+'\n')
w.close()

#SF: Setting 3
b=2 #For beta value 0.2
ob3_Lap_sf_beta_0_6=np.reshape(density3_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_Lap_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_Lap_sf_beta_0_6[i][0])+'\n')
w.close()

#SF: Setting 4
b=2 #For beta value 0.6
ob4_Lap_sf_beta_0_6=np.reshape(density4_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_Lap_sf_beta_0_6[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_Lap_sf_beta_0_6[i][0])+'\n')
w.close()

#~~~

#Beta=1

#SF: Setting 1
b=4 #For beta value 1
ob1_Lap_sf_beta_1_0=np.reshape(density1_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_Lap_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_Lap_sf_beta_1_0[i][0])+'\n')
w.close()

#SF: Setting 2
b=4 #For beta value 1
ob2_Lap_sf_beta_1_0=np.reshape(density2_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_Lap_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_Lap_sf_beta_1_0[i][0])+'\n')
w.close()

#SF: Setting 3
b=4 #For beta value 1
ob3_Lap_sf_beta_1_0=np.reshape(density3_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_Lap_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_Lap_sf_beta_1_0[i][0])+'\n')
w.close()

#SF: Setting 4
b=4 #For beta value 1
ob4_Lap_sf_beta_1_0=np.reshape(density4_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF Laplace Beta 1_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_Lap_sf_beta_1_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_Lap_sf_beta_1_0[i][0])+'\n')
w.close()
#~~~

#Beta=2

#SF: Setting 1
b=9 #For beta value 2
ob1_Lap_sf_beta_2_0=np.reshape(density1_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 1 SF Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob1_Lap_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob1_Lap_sf_beta_2_0[i][0])+'\n')
w.close()

#SF: Setting 2
b=9 #For beta value 2
ob2_Lap_sf_beta_2_0=np.reshape(density2_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 2 SF Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob2_Lap_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob2_Lap_sf_beta_2_0[i][0])+'\n')
w.close()

#SF: Setting 3
b=9 #For beta value 2
ob3_Lap_sf_beta_2_0=np.reshape(density3_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 3 SF Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob3_Lap_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob3_Lap_sf_beta_2_0[i][0])+'\n')
w.close()

#SF: Setting 4
b=9 #For beta value 2
ob4_Lap_sf_beta_2_0=np.reshape(density4_Lap_SF[b], (N_X_sf_lat,N_X_sf_long))
w = open('Obfuscation Probability Setting 4 SF Laplace Beta 2_0.csv', 'w', encoding='utf-8')
for i in range(N_X_sf_lat):
    for j in range(1,N_X_sf_long):
         w.write(str(ob4_Lap_sf_beta_2_0[i][N_X_sf_long-j])+' ')
    w.write(str(ob4_Lap_sf_beta_2_0[i][0])+'\n')
w.close()




#####################################################################
#####################################################################
##Artificial island Paris

'''p0_paris_island=np.array([0.00069458225838,0.02689025600318,0.02133359793610,0.01111331613415,0.00218297281207,0.00992260369121,0.01488390553681,0.02153205000992,0.02302044056360,0.00654891843620,0.00535820599325,0.00208374677515,0.00059535622147,0.00089303433221,0.00039690414765,
0.00079380829530,0.00277832903354,0.02034133756698,0.01944830323477,0.01121254217107,0.01160944631871,0.01160944631871,0.01627307005358,0.01647152212741,0.02232585830522,0.01180789839254,0.00674737051002,0.00178606866442,0.00049613018456,0.00049613018456,
0.00506052788252,0.00674737051002,0.00476284977178,0.00109148640603,0.00575511014090,0.01081563802342,0.01448700138916,0.01468545346299,0.01299861083548,0.01885294701330,0.00377058940266,0.00535820599325,0.00416749355031,0.00297678110736,0.00416749355031,
0.00704504862076,0.00119071244295,0.00039690414765,0.00138916451677,0.00337368525501,0.00248065092280,0.00486207580869,0.05288747767414,0.01706687834888,0.02609644770788,0.01280015876166,0.00446517166104,0.00585433617781,0.00079380829530,0.00059535622147,
0.00168684262751,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00714427465767,0.01369319309387,0.01131176820798,0.01617384401667,0.01121254217107,0.00853343917444,0.00694582258385,0.00158761659059,0.00257987695971,
0.00188529470133,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00426671958722,0.03651518158365,0.02331811867434,0.02252431037904,0.00734272673149,0.00188529470133,0.00932724746974,0.01220480254019,0.00228219884898,
0.00039690414765,0.00000000000000,0.00000000000000,0.08424291000000,0.00000000000000,0.00000000000000,0.00238142488589,0.00714427465767,0.00754117880532,0.00625124032546,0.00496130184560,0.00019845207382,0.00049613018456,0.00406826751340,0.00069458225838,
0.00109148640603,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00347291129192,0.00000000000000,0.00019845207382,0.00347291129192,0.00337368525501,0.00188529470133,0.00257987695971,0.00932724746974,0.00099226036912,
0.00565588410399,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00178606866442,0.00029767811074,0.00059535622147,0.00218297281207,0.00039690414765,0.00000000000000,0.00000000000000,0.00069458225838,0.00496130184560,
0.00138916451677,0.00059535622147,0.00039690414765,0.00009922603691,0.00079380829530,0.00049613018456,0.00128993847986,0.00248065092280,0.00000000000000,0.00019845207382,0.00178606866442,0.00089303433221,0.00079380829530,0.00386981543957,0.00168684262751,
0.00089303433221,0.00000000000000,0.00059535622147,0.00009922603691,0.00000000000000,0.00039690414765,0.00019845207382,0.00019845207382,0.00128993847986,0.00138916451677,0.00515975391943,0.00000000000000,0.00049613018456,0.01905139908712,0.03105774955348])
'''
p0_paris_island=np.array([0.00069458225838,0.02689025600318,0.02133359793610,0.01111331613415,0.00218297281207,0.00992260369121,0.01488390553681,0.02153205000992,0.02302044056360,0.00654891843620,0.00535820599325,0.00208374677515,0.00059535622147,0.00089303433221,0.00039690414765,
0.00079380829530,0.00277832903354,0.02034133756698,0.01944830323477,0.01121254217107,0.01160944631871,0.024458028,0.029121652,0.029320104,0.03517444,0.02465648,0.00674737051002,0.00178606866442,0.00049613018456,0.00049613018456,
0.00506052788252,0.00674737051002,0.00476284977178,0.01109148640603,0.00575511014090,0.01081563802342,0.01448700138916,0.01468545346299,0.01299861083548,0.01885294701330,0.00377058940266,0.00535820599325,0.00416749355031,0.00297678110736,0.00416749355031,
0.00704504862076,0.00119071244295,0.00039690414765,0.00138916451677,0.00337368525501,0.00248065092280,0.00486207580869,0.05288747767414,0.01706687834888,0.02609644770788,0.01280015876166,0.00446517166104,0.00585433617781,0.00079380829530,0.00059535622147,
0.00168684262751,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00714427465767,0.01369319309387,0.01131176820798,0.01617384401667,0.01121254217107,0.00853343917444,0.00694582258385,0.00158761659059,0.00257987695971,
0.00188529470133,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00426671958722,0.03651518158365,0.02331811867434,0.02252431037904,0.00734272673149,0.00188529470133,0.00932724746974,0.01220480254019,0.00228219884898,
0.00039690414765,0.00000000000000,0.00000000000000,0.01,0.00000000000000,0.00000000000000,0.00238142488589,0.00714427465767,0.00754117880532,0.00625124032546,0.00496130184560,0.00019845207382,0.00049613018456,0.00406826751340,0.00069458225838,
0.00109148640603,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00347291129192,0.00000000000000,0.00019845207382,0.00347291129192,0.00337368525501,0.00188529470133,0.00257987695971,0.00932724746974,0.00099226036912,
0.00565588410399,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00178606866442,0.00029767811074,0.00059535622147,0.00218297281207,0.00039690414765,0.00000000000000,0.00000000000000,0.00069458225838,0.00496130184560,
0.00138916451677,0.00059535622147,0.00039690414765,0.00009922603691,0.00079380829530,0.00049613018456,0.00128993847986,0.00248065092280,0.00000000000000,0.00019845207382,0.00178606866442,0.00089303433221,0.00079380829530,0.00386981543957,0.00168684262751,
0.00089303433221,0.00000000000000,0.00059535622147,0.00009922603691,0.00000000000000,0.00039690414765,0.00019845207382,0.00019845207382,0.00128993847986,0.00138916451677,0.00515975391943,0.00000000000000,0.00049613018456,0.01905139908712,0.03105774955348])

#########!!!!Finding behaviour of channels for artificial island paris
##Only interested in setting 4 and setting 1

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
    
###########################################

#Investigation
#for t in test:
#    #print(t[6])
#    plt.plot(t[6])
#plt.show()

#for t in test2:
#    plt.plot(t)
#plt.show()


####

#Saving 2D obfuscation probabilities
##########~~~~~~~!!!


##~~~~~~Blahut-Arimoto


#Beta=0.2

#Setting 1
b=0 #For beta value 0.2
ob1_BA_paris_beta_0_2=np.reshape(density1_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_BA_paris_beta_0_2[i][j])+' ')
    w.write(str(ob1_BA_paris_beta_0_2[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=0 #For beta value 0.2
ob4_BA_paris_beta_0_2=np.reshape(density4_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) BA Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_BA_paris_beta_0_2[i][j])+' ')
    w.write(str(ob4_BA_paris_beta_0_2[i][N_X_paris_long-1])+'\n')
w.close()

#Beta=0.6

#Setting 1
b=2 #For beta value 0.6
ob1_BA_paris_beta_0_6=np.reshape(density1_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_BA_paris_beta_0_6[i][j])+' ')
    w.write(str(ob1_BA_paris_beta_0_6[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=2 #For beta value 0.6
ob4_BA_paris_beta_0_6=np.reshape(density4_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) BA Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_BA_paris_beta_0_6[i][j])+' ')
    w.write(str(ob4_BA_paris_beta_0_6[i][N_X_paris_long-1])+'\n')
w.close()

#Beta=0.8

#Setting 1
b=3 #For beta value 0.8
ob1_BA_paris_beta_0_8=np.reshape(density1_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) BA Beta 0_8.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(N_X_paris_long-1):
        w.write(str(ob1_BA_paris_beta_0_8[i][j])+' ')
    w.write(str(ob1_BA_paris_beta_0_8[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=3 #For beta value 0.8
ob4_BA_paris_beta_0_8=np.reshape(density4_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) BA Beta 0_8.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_BA_paris_beta_0_8[i][j])+' ')
    w.write(str(ob4_BA_paris_beta_0_8[i][N_X_paris_long-1])+'\n')
w.close()


#Beta=1

#Setting 1
b=4 #For beta value 1
ob1_BA_paris_beta_1=np.reshape(density1_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) BA Beta 1.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_BA_paris_beta_1[i][j])+' ')
    w.write(str(ob1_BA_paris_beta_1[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=4 #For beta value 1
ob4_BA_paris_beta_1=np.reshape(density4_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) BA Beta 1.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_BA_paris_beta_1[i][j])+' ')
    w.write(str(ob4_BA_paris_beta_1[i][N_X_paris_long-1])+'\n')
w.close()

#Beta=2

#Setting 1
b=9 #For beta value 2
ob1_BA_paris_beta_2=np.reshape(density1_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) BA Beta 2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_BA_paris_beta_2[i][j])+' ')
    w.write(str(ob1_BA_paris_beta_2[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=9 #For beta value 2
ob4_BA_paris_beta_2=np.reshape(density4_BA_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) BA Beta 2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_BA_paris_beta_2[i][j])+' ')
    w.write(str(ob4_BA_paris_beta_2[i][N_X_paris_long-1])+'\n')
w.close()


##~~~~~~Laplace


#Beta=0.2

#Setting 1
b=0 #For beta value 0.2
ob1_Lap_paris_beta_0_2=np.reshape(density1_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_Lap_paris_beta_0_2[i][j])+' ')
    w.write(str(ob1_Lap_paris_beta_0_2[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=0 #For beta value 0.2
ob4_Lap_paris_beta_0_2=np.reshape(density4_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) Laplace Beta 0_2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_Lap_paris_beta_0_2[i][j])+' ')
    w.write(str(ob4_Lap_paris_beta_0_2[i][N_X_paris_long-1])+'\n')
w.close()

#Beta=0.6

#Setting 1
b=2 #For beta value 0.6
ob1_Lap_paris_beta_0_6=np.reshape(density1_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_Lap_paris_beta_0_6[i][j])+' ')
    w.write(str(ob1_Lap_paris_beta_0_6[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=2 #For beta value 0.6
ob4_Lap_paris_beta_0_6=np.reshape(density4_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) Laplace Beta 0_6.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_Lap_paris_beta_0_6[i][j])+' ')
    w.write(str(ob4_Lap_paris_beta_0_6[i][N_X_paris_long-1])+'\n')
w.close()

#Beta=0.8

#Setting 1
b=3 #For beta value 0.8
ob1_Lap_paris_beta_0_8=np.reshape(density1_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) Laplace Beta 0_8.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(N_X_paris_long-1):
        w.write(str(ob1_Lap_paris_beta_0_8[i][j])+' ')
    w.write(str(ob1_Lap_paris_beta_0_8[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=3 #For beta value 0.8
ob4_Lap_paris_beta_0_8=np.reshape(density4_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) Laplace Beta 0_8.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_Lap_paris_beta_0_8[i][j])+' ')
    w.write(str(ob4_Lap_paris_beta_0_8[i][N_X_paris_long-1])+'\n')
w.close()

#Beta=1

#Setting 1
b=4 #For beta value 1
ob1_Lap_paris_beta_1=np.reshape(density1_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) Laplace Beta 1.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_Lap_paris_beta_1[i][j])+' ')
    w.write(str(ob1_Lap_paris_beta_1[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=4 #For beta value 1
ob4_Lap_paris_beta_1=np.reshape(density4_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) Laplace Beta 1.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_Lap_paris_beta_1[i][j])+' ')
    w.write(str(ob4_Lap_paris_beta_1[i][N_X_paris_long-1])+'\n')
w.close()

#Beta=2

#Setting 1
b=9 #For beta value 2
ob1_Lap_paris_beta_2=np.reshape(density1_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability strong Paris (artificial) Laplace Beta 2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob1_Lap_paris_beta_2[i][j])+' ')
    w.write(str(ob1_Lap_paris_beta_2[i][N_X_paris_long-1])+'\n')
w.close()


#Setting 4
b=9 #For beta value 2
ob4_Lap_paris_beta_2=np.reshape(density4_Lap_Paris_island[b], (N_X_paris_lat,N_X_paris_long))
w = open('Obfuscation Probability vulnerable Paris (artificial) Laplace Beta 2.csv', 'w', encoding='utf-8')
for i in range(N_X_paris_lat):
    for j in range(1,N_X_paris_long):
        w.write(str(ob4_Lap_paris_beta_2[i][j])+' ')
    w.write(str(ob4_Lap_paris_beta_2[i][N_X_paris_long-1])+'\n')
w.close()

######Saving heatmaps of the library locations 
##Creating 2D distribution of the library locations
####################################################


#Library dataset
lib_all = pd.read_csv(r'library_grid.csv') #Importing Paris libraries dataset
lib_values=lib_all.values #Converting library dataset to an array
N_lib=len(lib_values) #Number of data points in library dataset

lib_loc=np.rint(lib_values) #Rounding off latitudes and longitudes
lib_latitudes=np.rint(lib_values[:,0]) #Latitudes rounded to nearest integer
lib_longitudes=np.rint(lib_values[:,1]) #Longitudes rounded to nearest integer

#Range of library dataset
lib_lat_max=np.max(lib_latitudes) #Maximum latitude
lib_lat_min=np.min(lib_latitudes) #Minimum latitude
lib_long_max=np.max(lib_longitudes) #Maximum longitude
lib_long_min=np.min(lib_longitudes) #Minimum longitude
lib_lat_range=np.array([lib_lat_min,lib_lat_max+1]) #Range of latitude
lib_long_range=np.array([lib_long_min,lib_long_max+1]) #Range of longitude

#Setup for experiments on library dataset

X_lib_lat=np.array(range(int(lib_lat_min),int(lib_lat_max+1)))    #Space of latitudes of original data in library dataset
X_lib_long=np.array(range(int(lib_long_min),int(lib_long_max+1))) #Space of longitudes of original data in library dataset
Y_lib_lat=np.array(range(int(lib_lat_min),int(lib_lat_max+1)))    #Space of latitudes of sanitized data in library dataset
Y_lib_long=np.array(range(int(lib_long_min),int(lib_long_max+1))) #Space of longitudes of sanitized data in library dataset
N_X_lib_lat=len(X_lib_lat) #Total length of the space of source latitudes of library dataset
N_X_lib_long=len(X_lib_long) #Total length of the space of source longitudes of library dataset
from itertools import product
X_lib=list(product(X_lib_lat, X_lib_long)) #all possible inpput pairs of (lat,long) of library datset
Y_lib=list(product(Y_lib_lat, Y_lib_long)) #all possible output pairs of (lat,long) of library dataset

N_X_lib=len(X_lib) #Total length of the space of library data


#CREATING A LINEAR PDF FOR 2D DATA for library data
p0_lib=np.array([]) #Empirical probability of library data
N_X_lib=len(X_lib)
for pos_x in range(N_X_lib):
    x=X_lib[pos_x]
    print("Progress {a} out of {b}".format(a=pos_x, b=N_X_lib))
    freq_x=0 #empirical probability of x in library data
    for y in lib_loc:
        comp=(x==y)
        if np.all(comp): #Testing if both elements of x and y are the same
            freq_x+=1
    p0_lib=np.append(p0_lib,[freq_x])
p0_lib=p0_lib/(N_lib) #Normalising frequencies to give empirical probabilities




#Saving Original Distribution (2D)
p0_lib_2D=np.reshape(p0_lib, (N_X_lib_lat,N_X_lib_long))
w = open('Original Dist_Paris Libraries.csv', 'w', encoding='utf-8')
for i in range(N_X_lib_lat):
    for j in range(1,N_X_lib_long):
         w.write(str(p0_lib_2D[i][N_X_lib_long-j])+' ')
    w.write(str(p0_lib_2D[i][0])+'\n')
w.close()


#########################################################
#########################################################
#########################################################
#POIs sampled from artificial island paris dataset

p0_paris_pois=np.array([0.00569458225838,0.02689025600318,0.02133359793610,0.01111331613415,0.00218297281207,0.00992260369121,0.01488390553681,0.02153205000992,0.02302044056360,0.00654891843620,0.00535820599325,0.00208374677515,0.00059535622147,0.00089303433221,0.00039690414765,
0.00579380829530,0.00277832903354,0.02034133756698,0.01944830323477,0.01121254217107,0.01160944631871,0.024458028,0.029121652,0.029320104,0.03517444,0.02465648,0.00674737051002,0.00178606866442,0.00049613018456,0.00049613018456,
0.00506052788252,0.00674737051002,0.00476284977178,0.01109148640603,0.00575511014090,0.01081563802342,0.01448700138916,0.01468545346299,0.01299861083548,0.01885294701330,0.00377058940266,0.00535820599325,0.00416749355031,0.00297678110736,0.00416749355031,
0.00704504862076,0.00119071244295,0.00039690414765,0.00138916451677,0.00337368525501,0.00248065092280,0.00486207580869,0.05288747767414,0.01706687834888,0.02609644770788,0.01280015876166,0.00446517166104,0.00585433617781,0.00079380829530,0.00059535622147,
0.00168684262751,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00714427465767,0.01369319309387,0.01131176820798,0.01617384401667,0.01121254217107,0.00853343917444,0.00694582258385,0.00158761659059,0.00257987695971,
0.00188529470133,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00426671958722,0.03651518158365,0.02331811867434,0.02252431037904,0.00734272673149,0.00188529470133,0.00932724746974,0.01220480254019,0.00228219884898,
0.00039690414765,0.00000000000000,0.00000000000000,0.0,0.00000000000000,0.00000000000000,0.00238142488589,0.00714427465767,0.00754117880532,0.00625124032546,0.00496130184560,0.00019845207382,0.00049613018456,0.00406826751340,0.00069458225838,
0.00109148640603,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00347291129192,0.00000000000000,0.00019845207382,0.00347291129192,0.00337368525501,0.00188529470133,0.00257987695971,0.00932724746974,0.00099226036912,
0.00565588410399,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00000000000000,0.00178606866442,0.00029767811074,0.00059535622147,0.00218297281207,0.00039690414765,0.00000000000000,0.00000000000000,0.00069458225838,0.00496130184560,
0.00138916451677,0.00059535622147,0.00039690414765,0.00009922603691,0.00079380829530,0.00049613018456,0.00128993847986,0.00248065092280,0.00000000000000,0.00019845207382,0.00178606866442,0.00089303433221,0.00079380829530,0.00386981543957,0.00168684262751,
0.00089303433221,0.00000000000000,0.00059535622147,0.00009922603691,0.00000000000000,0.00039690414765,0.00019845207382,0.00019845207382,0.00128993847986,0.00138916451677,0.00515975391943,0.00000000000000,0.00049613018456,0.01905139908712,0.03105774955348])


N_pois=4316 #Number of POIs we wish to sample
#Consistent with the real number of bars in Paris (c.f. World Cities Culture Forum)
#http://www.worldcitiescultureforum.com/data/number-of-bars
pois_paris_island=random.choices(X_paris,p0_paris_pois,k=N_pois)
#Asssumption: POIs follow the distribution of the crowd (c.f. Hotelling’s Model of Spatial Competition and Nash equilibrium )
if (vul_loc in pois_paris_island):
    pois_paris_island.remove(vul_loc)
print(pois_paris_island)

#####ELASTIC GEO-INDISTINGUISHABILITY
###For reference:
##Vulnerable location in p0_paris_island_2D=vul_loc = (6,3)
##Strong location in p0_paris_island_2D=vul_loc = (3,7)

p0_paris_island_2D=np.reshape(p0_paris_island,(N_X_paris_lat,N_X_paris_long))

#Mass of a point
def ball_mass(x,max_dist):
    ##Inputs:
        #x=location of interest (array([int,int]))
        #max_dist = distance of threshold area (float)
    ##Outputs:
        #ball_x = locations in the ball around x (array)
        #mass_all_x = individual masses of locations in ball around x (array)
        #mass_x = total mass around x (float)
    x_lat=int(x[0]) #x_lat = latitude of location x (int)
    x_long=int(x[1]) #x_long = longitude of location x (int)
    ball_x=[] #Ball around x
    mass_all_x=[] #Mass around x
    for i in range(N_X_paris_lat):
        diff_lat=(x_lat-i)
        for j in range(N_X_paris_long):
            diff_long=(x_long-j)
            d_x=np.array([diff_lat,diff_long])
            if (np.linalg.norm(d_x)<=max_dist):
                ball_x.append(np.array([i,j]))
                mass_all_x.append(p0_paris_island_2D[i,j])
            mass_x=np.sum(mass_all_x)
    return ball_x, mass_x, mass_all_x
 

'''max_dist=3
ball,mass,mass_all=ball_mass(vul_loc,max_dist) 
ball_mass(vul_loc,max_dist)[1] 

ball
mass   
mass_all
'''
#Elastic Metric
def d_elastic(x,x1,dmax):
    ##Inputs
        #x = starting location of interest ((lat,long))
        #x1 = ending location ((lat,long))
        #dmax = max threshold of indistinguishability (float)
    ##Output 
        #dist_elastic = elastic distance between x1 and x2 (float)
    diff_lat=(x[0]-x1[0])
    diff_long=(x[1]-x1[1])
    diff_loc=np.array([diff_lat,diff_long])
    dist=np.linalg.norm(diff_loc) #ground distance between x and x1
    if (dist>=dmax):
        dist_elastic=dist*(ball_mass(x,dmax)[1])
    else:
        dist_elastic=dist*(ball_mass(x,dist)[1])
    return dist_elastic
        
    
#d_elastic(vul_loc,str_loc,100)    
#np.linalg.norm(diff_loc)

###Individual QoS -- nearest POI
def nearest_poi(x,P):
    ##Inputs:
        #x = location of interest ((lat,long))
        #P = set of POIs ([(lat,long),(lat,long),...(lat,long)])
    ##Output
        #poi_x = nearest POI to x
    d=100000
    for p in P:
        if (np.linalg.norm(np.array(x)-np.array(p))<d):
            d=np.linalg.norm(np.array(x)-np.array(p))
            poi_x=p
    return poi_x, d
            
###POI nearest to the vulnerable point

'''poi_vul=nearest_poi(vul_loc,pois_paris_island)
poi_vul'''
n_sim=10 #Number of simulations for obfuscating a point

#For different noise level
noisy_vul_BA=[] #Vulnerable location obfuscated with BA
noisy_vul_Lap=[] #Vulnerable location obfuscated with Laplace
for b in range(len(eps_tight_paris)): #Signifying beta=0.2 or epsilon=0.4
    noisy_vul_BA.append(random.choices(X_paris,density4_BA_Paris_island[b],k=n_sim))
    noisy_vul_Lap.append(random.choices(X_paris,density4_Lap_Paris_island[b],k=n_sim))

#Distance to the nearest POI
qos_dist_BA=[]
qos_dist_Lap=[]
for b in tqdm(range(len(eps_tight_paris))):
    baqos=[] #distance of the noisy locations under BA with privacy level given by b
    for vba in noisy_vul_BA[b]:
        bapoi=nearest_poi(vba,pois_paris_island)[0] #POI nearest to the reported noisy location
        baqos.append(np.linalg.norm(np.array(vul_loc)-np.array(bapoi)))
    qos_dist_BA.append(baqos)
    lapqos=[] #distance of the noisy locations under Laplace with privacy level given by b
    for vlap in noisy_vul_Lap[b]:
        lappoi=nearest_poi(vlap,pois_paris_island)[0] #POI nearest to the reported noisy location
        lapqos.append(np.linalg.norm(np.array(vul_loc)-np.array(lappoi)))
    qos_dist_Lap.append(lapqos)
  
#Saving QoS -- distance to POIs
wBA = open('Distance from POIs_BA.csv', 'w', encoding='utf-8')
wLap = open('Distance from POIs_Lap.csv', 'w', encoding='utf-8')
for b in range(len(eps_tight_paris)):
    wBA.write(str(qos_dist_BA[b])+'\n')
    wLap.write(str(qos_dist_Lap[b])+'\n')
wBA.close()
wLap.close()   