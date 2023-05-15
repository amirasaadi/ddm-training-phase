# Param recovery of Bound, Drift, and Non decision time
# Modified version of Jamal Esmaily Sadrabadi 
##
import numpy as np
import random as rand
from skopt import gp_minimize
from scipy import stats
import pandas as pd

# seed_number = 1
# np.random.seed(seed_number)
# rand.seed(seed_number)

nsub = 30
allRes = []
allParam = []

for iter in range(nsub):

    NumTr = 3000
    dt = 1  # ms
    Time = 1000 * 5 / dt
    Bound = int(np.random.uniform(10, 30))
    ndt = int(np.random.uniform(150, 450))
    SP_Var = 0
    drif_Var = 0
    drift_rate = np.random.uniform(.1, .8)
    timelap = np.linspace(0, Time, int(Time / dt))
    Coherence = np.array([0, 3.2, 6.4, 12.8, 25.6, 51.2]) / 100
    SP = np.random.normal(0, SP_Var, NumTr * len(Coherence))
    K = drift_rate + np.random.normal(0, drif_Var, NumTr * len(Coherence))
    AllTr = np.repeat(Coherence, NumTr)
    rand.shuffle(AllTr)
    V = np.vstack((SP * np.ones(NumTr * len(Coherence)), np.zeros([NumTr * len(Coherence), len(timelap) - 1]).T)).T
    allParam.append([Bound, drift_rate, ndt])

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

        RT = RT * dt

        return RT, Choice


    #### Main Loop ####
    for t in range(len(timelap) - 1):
        dX = np.random.normal(K * AllTr, np.ones([1, len(AllTr)]))
        V[:, t + 1] = V[:, t] + dX * dt

    ########## Behave Data #########
    RT, Choice = GetBehave(V)
    RT = RT + ndt

    NumQ = 7
    RTMaxRange = 5000
    CutSize = RTMaxRange / NumQ
    CorrIndex = np.where(Choice == 1)[0]
    ErrIndex = np.where(Choice == 0)[0]
    CorrRT = RT[CorrIndex]
    ErrRT = RT[ErrIndex]
    Behave_ProbMat = np.zeros([NumQ, 2])
    for i in range(NumQ):
        tBin = np.arange(i * CutSize, (i + 1) * CutSize)
        NumCorrRT_tmp = np.where(np.logical_and(CorrRT >= i * CutSize, CorrRT <= (i + 1) * CutSize))[0].shape[0]
        NumErrRT_tmp = np.where(np.logical_and(ErrRT >= i * CutSize, ErrRT <= (i + 1) * CutSize))[0].shape[0]

        Behave_ProbMat[i, 0] = NumCorrRT_tmp / RT.shape[0]
        Behave_ProbMat[i, 1] = NumErrRT_tmp / RT.shape[0]

    ####### Params #####3
    NumTr = len(AllTr)
    dt = 1  # ms
    Time = 1000 * 5 / dt
    Bound0 = 6
    K0 = .08
    SP = 0
    ndt0 = 150
    Vars0 = [Bound0, K0, ndt0]


    def DDMModel(Vars):
        Bound = Vars[0]
        K = Vars[1]
        ndt_in = Vars[2]

        timelap = np.linspace(0, Time, int(Time / dt))
        Coherence = np.unique(AllTr)
        V = SP * np.ones((NumTr, len(timelap)))
        a = 1

        #### Main Loop ####
        for t in range(len(timelap) - 1):
            dX = np.random.normal(K * AllTr, np.ones([1, len(AllTr)]))
            V[:, t + 1] = V[:, t] + dX * dt

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
        RT = RT + ndt_in
        NumQ = 7
        RTMaxRange = 5000
        CutSize = RTMaxRange / NumQ
        CorrIndex = np.where(Choice == 1)[0]
        ErrIndex = np.where(Choice == 0)[0]
        CorrRT = RT[CorrIndex]
        ErrRT = RT[ErrIndex]
        Model_ProbMat = np.zeros([NumQ, 2])
        for qi in range(NumQ):
            NumCorrRT_tmp = np.where(np.logical_and(CorrRT >= qi * CutSize, CorrRT <= (qi + 1) * CutSize))[0].shape[0]
            NumErrRT_tmp = np.where(np.logical_and(ErrRT >= qi * CutSize, ErrRT <= (qi + 1) * CutSize))[0].shape[0]

            Model_ProbMat[qi, 0] = NumCorrRT_tmp / RT.shape[0]
            Model_ProbMat[qi, 1] = NumErrRT_tmp / RT.shape[0]

        CostVal = np.nansum(((Behave_ProbMat - Model_ProbMat) ** 2) / Model_ProbMat)
        CostVal = stats.chi2.cdf(CostVal, 1)

        return CostVal


    res = gp_minimize(DDMModel,  # the function to minimize
                      [(5, 50), (.05, .8), (150, 300)],  # the bounds on each dimension of x
                      x0=Vars0,  # the starting point
                      n_calls=15,  # the number of evaluations of f including at x0
                      n_random_starts=3,  # the number of random initial points
                      random_state=777,
                      verbose=1)

    allRes.append(res)

b_predicted = np.zeros((nsub,))
d_predicted = np.zeros((nsub,))
ndt_predicted = np.zeros((nsub,))
for i in range(nsub):
    b_predicted[i] = allRes[i].x[0]
    d_predicted[i] = allRes[i].x[1]
    ndt_predicted[i] = allRes[i].x[2]

araay_allparam = np.asarray(allParam)

b_p = pd.DataFrame({'b_p': b_predicted})
d_p = pd.DataFrame({'d_p': d_predicted})
ndt_p = pd.DataFrame({'ndt_p': ndt_predicted})

b_t = pd.DataFrame({'b_t': araay_allparam[:, 0]})
d_t = pd.DataFrame({'d_t': araay_allparam[:, 1]})
ndt_t = pd.DataFrame({'ndt_t': araay_allparam[:, 2]})

csv_data = pd.concat([b_p, d_p, ndt_p, b_t, d_t, ndt_t], axis=1)
csv_data.to_csv('ddm_parameter_recovery.csv', index=False)
