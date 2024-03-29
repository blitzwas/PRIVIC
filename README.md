# PRIVIC
## Language and recommended version: 
Python 3 or higher
## Code for experiments performed in the paper "PRIVIC: A privacy-preserving method for incremental collection of location data":
1. FilteringLocations.py --> Filtering out locations in Paris and San Francisco from the Gowalla dataset (File name: "gowalla_checkin.txt" downloaded from https://snap.stanford.edu/data/loc-gowalla.html). The dataset is very large (376.4 MB).
2. Gowalla_Paris_grid.csv --> 10,260 Gowalla check-ins from a central part of Paris bounded by latitudes (48.8286, 48.8798) and longitudes (2.2855, 2.3909) covering an area of 8Km×6Km discretized with a 16x12 grid.
3. Gowalla_SF_grid.csv --> 123,025 Gowalla check-ins from a northern part of San Francisco bounded by latitudes (37.7228, 37.7946) and longitudes (-122.5153, -122.3789) covering an area of 12Kmx8Km discretized with a 24x17 grid.
4. Paris_Grid.py and SF_Grid.py --> Discretizing the locations in Paris and San Francisco (obtained from the Gowalla dataset).
5. privic.py --> Blahut-Arimoto algorithm (BA), iterative Bayesian update (IBU), and PRIVIC.
6. Elastic.py --> Visualizing the elastic distinguishability property of BA and its ability to protect the isolated locations vs LAP
7. BA_vs_LAP_stat_utility.py --> Comparing the statistical utility of the noisy locations privatized with BA and LAP
8. Privic_BA_with_Adversarial_Input.py --> Experiments considering different levels of adversarial users reporting their locations falsely to compromise the privacy of (non-adversarial) users located in geo-spatially isolated points on the map.


# Required packages
## To be imported:
1. from pyemd import emd
2. import math
3. import numpy as np
4. import matplotlib.pyplot as plt
5. import random
6. from scipy.stats import binom
7. from scipy.stats import poisson
8. from scipy.stats import wasserstein_distance
9. from tqdm import tqdm
10. import time
11. import pandas as pd

# Downloading the dataset
The Gowalla check-ins dataset should be downloaded from https://snap.stanford.edu/data/loc-gowalla.html and prepared as follows:
1. Download the .gz file named "loc-gowalla_totalCheckins.txt.gz" from https://snap.stanford.edu/data/loc-gowalla.html. 
2. Gunzip the file and rename the gunzipped .txt file as "gowalla_checkin.txt".

The codes provided mostly generate the .csv files for the output data of the different experiments. All the visualisations provided in the paper were done using an external (paid) software called GraphPad PRISM using those output datasets. If one wishes to reproduce the plots, one may use any visualisation software or Python packages that help plot the output data from the .csv files produced by the codes. 

# Functions and variables:
"privic_functions.py" defines several functions and introduces many variables that are used by most of the other .py files. It is, therefore, recommended to import those or copy-paste the code from "privic_functions.py" at the beginning of the files "BA_vs_LAP_stat_utility.py", "Elastic.py", "privic_main.py", and "Privic_BA_with_Adversarial_Input.py". We highly recommend following the instructions in the README.md to run the codes following the desired sequence to avoid errors related to "function/variable not defined".


# Instructions to run the codes
1. Run FilteringLocations.py to filter out the Gowalla check-in locations in Paris and San Francisco from the entire Gowalla dataset. This code creates the csv files "Gowalla_Paris.csv" and "Gowalla_SF.csv".
2. "Gowalla_Paris_(No Head).csv" and "Gowalla_SF_(No Head).csv", respectively are created by essentially deleting the first row (column titles) from  "Gowalla_Paris.csv" and "Gowalla_SF.csv", respectively.
3. Run Paris_Grid.py and SF_Grid.py to discretize the locations and store the distributions for the datasets considered in the experiments, as described in Section 3.1 (for Paris) and Section 3.2 (for SF). Paris_Grid.py creates the csv files "Gowalla_Paris_grid.csv" and "Gowalla_Paris_grid_freq.csv" to store the discretized locations and the frequency of the locations in each grid for the Gowalla check-ins in Paris. SF_Grid.py creates the csv files "Gowalla_SF_grid.csv" and "Gowalla_SF_grid_freq.csv" to store the discretized locations and the frequency of the locations in each grid for the Gowalla check-ins in Paris. These are used to plot the original locations and their distributions (Fig. 6a and 6b in the paper). We used an external software GraphPad PRISM for creating the visualisations.
4. Run privic_functions.py to define the fundamental functions and variables called by the rest of the experiments.
5. Run BA_vs_LAP_stat_utility.py to create the csv files "Estimated Paris tight better Laplace.csv", "Estimated Paris BA.csv", "Estimated SF tight better Laplace.csv", and "Estimated SF BA.csv", representing the MLE estimated by IBU on the locations from Paris and SF obfuscated by LAP and BA, respectively. The last part of the file contains the code to compute the EMD between the true and the estimated distributions for the two mechanisms for the two sets of locations, producing the dataset for statistical utility. The visualisation of the statistical utility (Fig. 3a and 3b) was done using an external software GraphPad PRISM. Note that BA_vs_LAP_stat_utility.py has extensive use of some of the functions and variables defined in privic_functions.py.
6. Run Elastic.py to create the datasets for the obfuscation probabilities for the strong (index 52) and vulnerable (index 93) locations on the map under BA and LAP. Fig. 2 used a selected number of values of the privacy parameter (from the list RDSlope). In particular, the values 0.2, 0.6, 0.8, and 1.0 were used to illustrate the elastic mechanism property of BA as opposed to LAP (Fig. 2 in the paper). The visualisation of the obfuscation distributions was done using an external software GraphPad PRISM. Note that Elastic.py has extensive use of some of the functions and variables defined in privic_functions.py.
7. Run privic_main.py to run the main algorithm PRIVIC (Alg. 1 in the paper). The lists "KW_paris" and "KW_sf" store the EMD between the true and the estimated distribution of the Paris and SF datasets, respectively, which, in turn, is used to plot Fig. 8 and 9, respectively. Note that the value of the variable "beta" determines the privacy parameter illustrated in Fig. 8 and 9. (e.g., Fig. 8a and 9a use beta=0.5 and Fig. 8b and 9b use beta=1 both for the Paris and SF datasets. The visualisations were done using an external software GraphPad PRISM. Note that privic_main.py has extensive use of some of the functions and variables defined in privic_functions.py.
8. Run Privic_BA_with_Adversarial_Input.ipynb creates the datasets of the obfuscation probabilities of the vulnerable location for 9 different levels of adversarial levels (tuned by the variable "p", representing the index of the different levels of adversarial users maliciously dumping the weight on the vulnerable location on the map, contained in the list "p0_paris_island_adversarial") and 2 different levels of privacy (tuned by the variable "b", representing the index of the value of "beta" in BA, contained in the list RDSlope_short). Note that the geo-ind parameter epsilon = 0.8 and 1.2 are achieved by assigning b= 1 and 2, respectively. p=0,1,2,3,4,5,6,7,8 represent 0%, 5%, 10%, 15%, 20%, 24%, 29%, 34%, and 38% adversaries, respectively. This code creates the csv files "ObfusProb vul Paris BA Beta <x> Adv <y>.csv" representing the obfuscation probabilities of the vulnerable location in the Paris dataset with Beta=<x> under the adversarial level <y>. The visualisations were done using an external software GraphPad PRISM. Note that Privic_BA_with_Adversarial_Input.ipynb has extensive use of some of the functions and variables defined in privic_functions.py.
