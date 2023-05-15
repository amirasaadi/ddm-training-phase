# Generating behvioral data and return Reaction time and Choice of perceptual decision
# Writen by Jamal Esmaily Sadrabadi 
##
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from scipy.optimize import curve_fit
import sys
import matplotlib
import pickle


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

####### Params #####3
#np.random.seed(1)
#rand.seed(1)


NumTr = 3000
dt = 1  # ms
Time = 1000*5/dt
Bound = 30
SP_Var = 5
drif_Var = .2

timelap = np.linspace(0, Time, int(Time/dt))
Coherence = np.array([0, 3.2, 6.4, 12.8, 25.6, 51.2])/100
SP = np.random.normal(0, SP_Var, NumTr*len(Coherence))
K = 0.5 + np.random.normal(0, drif_Var, NumTr*len(Coherence))
AllTr = np.repeat(Coherence, NumTr)
rand.shuffle(AllTr)
V = np.vstack((SP*np.ones(NumTr*len(Coherence)), np.zeros([NumTr*len(Coherence), len(timelap)-1]).T)).T

### Behave ###
def GetBehave(V):
    RT = np.zeros(len(AllTr))
    Choice = np.zeros(len(AllTr))
    Indx_in = V > Bound
    Indx_out = V < -Bound
    for i in range(len(AllTr)):
        tmpindx1 = np.where(Indx_in[i, :])[0]
        tmpindx2 = np.where(Indx_out[i, :])[0]
        if len(tmpindx1) == 0:
            tmpindx1 = np.array([len(timelap)])
        if len(tmpindx2) == 0:
            tmpindx2 = np.array([len(timelap)])
        if tmpindx1[0] <= tmpindx2[0]:
            Choice[i] = 1
            RT[i] = tmpindx1[0]
        else:
            Choice[i] = 0
            RT[i] = tmpindx2[0]

    RT = RT*dt



    return RT,Choice


#### Main Loop ####
for t in range(len(timelap)-1):
    dX = np.random.normal(K*AllTr, np.ones([1, len(AllTr)]))
    V[:, t+1] = V[:, t] + dX*dt



########## Behave Data #########
RT, Choice = GetBehave(V)




plt.show()
