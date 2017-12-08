import sys
sys.path.append("../../../")
import NaiveBayesNets as nbn
import pandas as pd
import numpy as np
from scipy import stats
import os
print(os.getcwd())

df = pd.read_csv("./NaiveBayesNets/data/Pima.tr.csv")
#df.head(10)

npreg = df['npreg']
age = df['age']
npreg_age = list(zip(npreg.tolist(),age.tolist()))



def test_MutualInfo(useries, vseries):
    """
    https://en.wikipedia.org/wiki/Mutual_information#Definition
    """
    ukde = stats.gaussian_kde(useries)
    vkde = stats.gaussian_kde(vseries)
    jointkeys = list(zip(useries.tolist(), vseries.tolist()))
    jointvalues = np.vstack([useries, vseries])
    jointkde = stats.gaussian_kde(jointvalues)
    MutualInfo = 0
    for uval, vval in jointkeys:
        uprob = ukde(uval)
        vprob = vkde(vval)
        uvprob = jointkde((uval, vval))
        MutualInfo += uvprob * np.log(uvprob / (uprob * vprob))
    if MutualInfo < 0:
        print("Major Error with Mutual Information Calculation", file=sys.stderr)
    return MutualInfo

result_test_MutualInfo = test_MutualInfo(npreg, age)
print(result_test_MutualInfo)




## test individual components of Probs module
bivariate = nbn.kdebayes.Probs2.Bivariate(xseries=npreg, yseries=age)
val_xy = bivariate.Evaluate(u = 5, v = 24)
print(val_xy)


## test Mutual Information
kde_npreg = nbn.kdebayes.Probs2.Univariate(series = npreg)
kde_age = nbn.kdebayes.Probs2.Univariate(series = age)

val_x = kde_npreg[5]
print(val_x)
val_y = kde_age[24]
print(val_y)

print(val_xy * np.log(val_xy / (val_x * val_y)))



def test_CalcMutualInfo(uprobs, vprobs, jointprobs, jointkeys):
    """
    Calculate Mutual Information statistic
    uprobs: dictionary of probabilities
    vprobs: dictionary of probabilities
    jointprobs: dictionary of probabilities
    """
    MI = [] ## collect Mutual Information
    for uval, vval in jointkeys:
        uprob = uprobs[uval]
        vprob = vprobs[vval]
        probxy = jointprobs[(uval, vval)]
        I = probxy * np.log(probxy / (uprob * vprob))
        MI.append(I)
    MI = np.sum(MI)
    if MI < 0:
        print("Major Error with Mutual Information Calculation", file=sys.stderr)
    return MI

mi_result = test_CalcMutualInfo(kde_npreg, kde_age, bivariate, npreg_age)
mi_result




probs = nbn.kdebayes.Probs2.Probs(useries = npreg, vseries=age)
MI = probs.CalcMutualInfo()
print(MI)
print(MI == mi_result)




def test_MutualInfMatrix(df, class_col_name):
    import itertools as it
    g = df.groupby(by = class_col_name) ## group df by class
    colnames = df.columns.tolist()
    colnames.remove(class_col_name)
    ## process the following steps for each class
    ClassMats = {} ## dictionary to store MutualInfMatrix for each class
    for i, frame in g:
        colcombos = it.combinations(colnames, 2) ## will return tuples
        MutualInfo = []
        for u, v in colcombos:
            ulist = frame[u] #.tolist() ## Probs2 takes a series
            vlist = frame[v] #.tolist()
            probs = nbn.kdebayes.Probs2.Probs(ulist, vlist) ## calculates all probs
            MI = probs.CalcMutualInfo()
            MutualInfo.append((u, v, MI)) ## no longer storing probs to save memory
        MutualInfMatrix = pd.DataFrame(MutualInfo, columns = ['U', 'V', "MI"])
        MutualInfMatrix.sort_values(by = "MI", ascending=False, inplace=True)
        MutualInfMatrix.reset_index(inplace=True, drop=True)
        ClassMats[i] = MutualInfMatrix ## store results for current class
    return ClassMats


#test_MutualInfMatrix(df, class_col_name='type')





def validation_MutualInfo(x, y, bins = 15):
    """
    The G-test statistic is proportional to the Kullback–Leibler divergence
    of the theoretical distribution from the empirical distribution:
    https://en.wikipedia.org/wiki/G-test#Relation_to_Kullback–Leibler_divergence
    """
    from scipy.stats import chi2_contingency
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

valid_mi = validation_MutualInfo(x = npreg, y = age, bins = 15)
print(valid_mi)
