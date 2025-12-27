# import packages
import os
from cec17_functions import cec17_test_func
import numpy as np
from copy import deepcopy


PopSize = 200
DimSize = 100
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1
curIter = 0
MaxIter = int(MaxFEs / PopSize * 2)
phi = 0.15


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


# initialize the M randomly
def Initialization():
    global Pop, Velocity, FitPop
    Velocity = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def CSO():
    global Pop, Velocity, FitPop, phi
    sequence = list(range(PopSize))
    np.random.shuffle(sequence)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    Xmean = np.mean(Pop, axis=0)
    for i in range(int(PopSize / 2)):
        idx1 = sequence[2 * i]
        idx2 = sequence[2 * i + 1]
        if FitPop[idx1] < FitPop[idx2]:
            Off[idx1] = deepcopy(Pop[idx1])
            FitOff[idx1] = FitPop[idx1]
            Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (
                    Pop[idx1] - Pop[idx2]) + phi * (Xmean - Pop[idx2])
            Off[idx2] = Pop[idx2] + Velocity[idx2]
            Off[idx2] = Check(Off[idx2])
            FitOff[idx2] = fitness(Off[idx2])
        else:
            Off[idx2] = deepcopy(Pop[idx2])
            FitOff[idx2] = FitPop[idx2]
            Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (
                        Pop[idx2] - Pop[idx1]) + phi * (Xmean - Pop[idx1])
            Off[idx1] = Pop[idx1] + Velocity[idx1]
            Off[idx1] = Check(Off[idx1])
            FitOff[idx1] = fitness(Off[idx1])

    Pop = deepcopy(Off)
    FitPop = deepcopy(FitOff)


def RunCSO():
    global FitPop, curIter, TrialRuns, DimSize
    All_Trial_Best = []
    All_Trial_Percent = []  # per-iteration percentage of iterations completed
    All_Trial_Diversity = []  # per-iteration swarm diversity
    All_Trial_Stagnation = []  # per-iteration stagnation duration
    for i in range(TrialRuns):
        BestList = []
        PercentList = []
        DiversityList = []
        StagnationList = []
        curIter = 0
        np.random.seed(945 + 3 * i)
        Initialization()
        # Initialize best tracking and improvement iteration marker
        best_so_far = float(np.min(FitPop))
        lastImproveIter = 0
        BestList.append(best_so_far)
        while curIter < MaxIter:
            CSO()
            curIter += 1
            # Track current best and update improvement marker
            current_best = float(np.min(FitPop))
            if current_best < best_so_far:
                best_so_far = current_best
                lastImproveIter = curIter

            # Percentage of iteration completed
            percent_iter = curIter / MaxIter
            PercentList.append(percent_iter)

            # Swarm diversity: average Euclidean distance to the mean particle
            Xmean = np.mean(Pop, axis=0)
            diversity = float(np.mean(np.linalg.norm(Pop - Xmean, axis=1)))
            DiversityList.append(diversity)

            # Stagnant growth duration (per user's formula)
            notImproveIter = (curIter - lastImproveIter) / MaxIter
            StagnationList.append(notImproveIter)

            # Record best list for compatibility with existing outputs
            BestList.append(current_best)
        All_Trial_Best.append(BestList)
        All_Trial_Percent.append(PercentList)
        All_Trial_Diversity.append(DiversityList)
        All_Trial_Stagnation.append(StagnationList)
    np.savetxt("./CSO_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")
    # Save additional per-iteration metrics alongside best fitness logs
    np.savetxt("./CSO_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D_percent.csv", All_Trial_Percent, delimiter=",")
    np.savetxt("./CSO_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D_diversity.csv", All_Trial_Diversity, delimiter=",")
    np.savetxt("./CSO_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D_stagnation.csv", All_Trial_Stagnation, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim

    for i in range(1, 31):
        if i == 2:
            continue
        FuncNum = i
        RunCSO()


if __name__ == "__main__":
    if os.path.exists('./CSO_Data/CEC2017') == False:
        os.makedirs('./CSO_Data/CEC2017')
    Dims = [50, 100]
    for Dim in Dims:
        main(Dim)

