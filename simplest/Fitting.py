# Fit Bound and Drift of perceptual decision
# Written by Jamal Esmaily Sadrabadi 

##
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from scipy.optimize import curve_fit
import sys
import matplotlib
import pickle

from scipy.optimize import minimize
from noisyopt import minimizeCompass
from skopt import gp_minimize
from scipy import stats



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

with open('BehaveData.pkl', 'rb') as f:
    Choice, RT, AllTr = pickle.load(f)


##
np.random.seed(1)
rand.seed(1)


NumQ = 7
RTMaxRange = 5000
CutSize = RTMaxRange/NumQ
CorrIndex = np.where(Choice == 1)[0]
ErrIndex = np.where(Choice == 0)[0]
CorrRT = RT[CorrIndex]
ErrRT = RT[ErrIndex]
Behave_ProbMat = np.zeros([NumQ, 2])
for i in range(NumQ):
    tBin = np.arange(i * CutSize, (i + 1) * CutSize)
    NumCorrRT_tmp = np.where(np.logical_and(CorrRT >= i * CutSize, CorrRT <= (i + 1) * CutSize))[0].shape[0]
    NumErrRT_tmp = np.where(np.logical_and(ErrRT >= i * CutSize, ErrRT <= (i + 1) * CutSize))[0].shape[0]

    Behave_ProbMat[i, 0] = NumCorrRT_tmp/RT.shape[0]
    Behave_ProbMat[i, 1] = NumErrRT_tmp/RT.shape[0]

####### Params #####3
NumTr = len(AllTr)
dt = 1  # ms
Time = 1000 * 5 / dt
Bound0 = 6
K0 = .08
SP = 0
Vars0 = [Bound0, K0]



def DDMModel(Vars):
    Bound = Vars[0]
    K = Vars[1]


    timelap = np.linspace(0, Time, int(Time/dt))
    Coherence = np.unique(AllTr)
    V = SP*np.ones((NumTr, len(timelap)))
    a = 1

    #### Main Loop ####
    for t in range(len(timelap)-1):
        dX = np.random.normal(K*AllTr, np.ones([1, len(AllTr)]))
        V[:, t+1] = V[:, t] + dX*dt


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

        RT = RT * dt

        return RT, Choice



    ########## Behave Data #########
    RT, Choice = GetBehave(V)
    NumQ = 7
    RTMaxRange = 5000
    CutSize = RTMaxRange/NumQ
    CorrIndex = np.where(Choice == 1)[0]
    ErrIndex = np.where(Choice == 0)[0]
    CorrRT = RT[CorrIndex]
    ErrRT = RT[ErrIndex]
    Model_ProbMat = np.zeros([NumQ, 2])
    for qi in range(NumQ):
        NumCorrRT_tmp = np.where(np.logical_and(CorrRT >= qi * CutSize, CorrRT <= (qi + 1) * CutSize))[0].shape[0]
        NumErrRT_tmp = np.where(np.logical_and(ErrRT >= qi * CutSize, ErrRT <= (qi + 1) * CutSize))[0].shape[0]

        Model_ProbMat[qi, 0] = NumCorrRT_tmp/RT.shape[0]
        Model_ProbMat[qi, 1] = NumErrRT_tmp/RT.shape[0]

    CostVal = np.nansum(((Behave_ProbMat-Model_ProbMat)**2)/Model_ProbMat)
    CostVal = stats.chi2.cdf(CostVal, 1)


    return CostVal


res = gp_minimize(DDMModel,            # the function to minimize
                  [(5, 50), (.05, .8)],      # the bounds on each dimension of x
                  x0=Vars0,            # the starting point
                  n_calls=15,         # the number of evaluations of f including at x0
                  n_random_starts=3,  # the number of random initial points
                  random_state=777,
                  verbose=1)


Params = res.x

print(res)


