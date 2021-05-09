# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm
from math import sqrt, floor, ceil

'''
(portfolio) is a list of loan entries, with each entry
containing the following information

[0] int: loan id
[1] int: loan rating (0 is best rating, 7 is worst,
                      8 is default - new rating only)

#####

(threshold) is a rating migration threshold table,
structured as lists of threshold for rating A
migrating to rating B. That is, migration[3] is a
list containing the thresholds of rating migration
for a loan rated 3.

#####

(pdlist) is a list with probabilities of default (PD)
per rating. That is, pdlist[2] is the PD for a loan
rated 2.

#####

(futurevalue) is a table of values per future rating,
structured as a list of values per rating per loan.
That is, futurevalue[17] is a list containing the values
per future rating of the portfolio[17] loan entry, with
futurevalue[17][3] being the value at future rating 3.
'''

def importdataset():
    df = pd.read_csv(r"C:\Users\Arthur\Dropbox\Mestrado Nova IMS\Credit and Market Risk\sample ajusted2.csv", sep = ";")
    df["installment"] = df["installment"]*df["term"]
    df = df[["customer_ref","rating_model","installment","loan_amount",
            "term_cat","interest_payment_cat","income_cat","category"]]
    df.columns = ['total_amnt' if x=='installment' else x for x in df.columns]
    return df

def probmig():
    probmig = [[0.759, 0.2, 0.024, 0.006, 0.002, 0, 0, 0, 0.009],
               [0.206, 0.572, 0.166, 0.026, 0.007, 0.003, 0.001, 0, 0.019],
               [0.026, 0.214, 0.543, 0.137, 0.028, 0.012, 0.005, 0.001, 0.034],
               [0.004, 0.036, 0.235, 0.473, 0.13, 0.043, 0.018, 0.007, 0.054],
               [0, 0.005, 0.039, 0.231, 0.407, 0.151, 0.056, 0.015, 0.096],
               [0, 0.001, 0.007, 0.037, 0.225, 0.382, 0.165, 0.054, 0.129],
               [0, 0, 0.001, 0.004, 0.042, 0.174, 0.282, 0.109, 0.388],
               [0, 0, 0, 0, 0, 0.003, 0.01, 0.02, 0.967]]
    return probmig

def migthreshold(probmig):
    thres = [[0 for i in range(len(probmig))] for j in range(len(probmig[0])-1)]
    for i in range(len(thres)):
        aux = 0
        for j in range(len(thres[0])):
            aux += probmig[i][j]
            thres[i][j] = norm.ppf(aux)
    return thres

def discounting(rate):
    factor = (1+rate)**(1/12) - 1
    factor = 1/(1+factor)
    d36 = (1/36)*(1-factor**36)/(1-factor)
    d60 = (1/60)*(1-factor**60)/(1-factor)
    return [d36, d60]

def futureval(loanset, prob, rate):
    recovery = 0.560956
    pd = [item[-1] for item in prob]
    pd.append(1)
    discount = discounting(rate)
    future = [[0 for j in range(len(pd))] for i in range(len(loanset))]
    expected = 0
    for i in range(len(future)):
        total = loanset["total_amnt"][i]
        term = loanset["term_cat"][i]
        loan = loanset["loan_amount"][i]
        rating = loanset["rating_model"][i]
        for j in range(len(pd)):
            future[i][j] = (1-pd[j])*total*discount[term] + pd[j]*loan*recovery
            expected += future[i][j]*prob[rating][j]
    return future, expected

def numgenerator(dim, correl, samplesize):
    '''
    This function generates random variables from a
    multivariate normal distribution with mean 0 and
    covariant matrix elements equal to 1 along the main
    diagonal and correl elsewhere.
    It generates samplesize amount of scenarios, each with
    dim amount of random variables.

    Parameters
    ----------
    dim : int
        Amount of loans in the portfolio.
    correl : float
        Correlation factor for rating migration.
        The same value is applied for all correlations.
        Bounded between 0 and 1, both inclusive.
    samplesize : int
        Amount of simulated portfolios.

    Returns
    -------
    list
        Matrix containing generated random variables from
        a multivariate normal distribution.
        Structured as a list of simulated portfolios,
        with each entry as a list of random variables.

    '''
    mean = [ 0 for i in range(dim) ]
    cov = [ [correl for i in range(dim)] for i in range(dim) ]
    for i in range(dim):
        cov[i][i] = 1
    return np.random.multivariate_normal(mean, cov, samplesize)

def normalRating(value, threshold_rating):
    '''
    This function returns the new loan rating at time horizon
    for a loan.
    Ratings values are int, varying from 0 to 8, both inclusive;
    0 is the best rating, 7 is the worst rating, 8 is default.

    Parameters
    ----------
    value : float
        Random variable from a simulated loan entry.
        This variable is generated with numgenerator function.
    threshold_rating : list
        List of rating migration thresholds for the
        old rating.
        This is a sublist from the threshold variable.

    Returns
    -------
    int
        Loan rating for the selected loan at the time horizon.

    '''
    for i in range(len(threshold_rating)):
        if value <= threshold_rating[i]:
            return i
    return i+1

def newrating(sim, threshold, portfolio):
    '''
    This function returns the new loan ratings at time horizon
    for all simulated portfolios.
    Ratings values are int, varying from 0 to 8, both inclusive;
    0 is the best rating, 7 is the worst rating, 8 is default.

    Parameters
    ----------
    sim : list
        Matrix containing generated random variables from
        a multivariate normal distribution.
        Structured as a list of simulated portfolios,
        with each entry as a list of random variables.
        This variable is provided by the numgenerator function.
    threshold : list
        Table of rating migration thresholds.
        Structured as lists of thresholds for rating A
        migrating to rating B. That is, migration[3] is a
        list containing the thresholds of rating migration
        for a loan rated 3.
    portfolio : list
        List of loan entries, with each entry containing
        the following information:
        [0] int: loan id
        [1] int: loan rating (0 is best rating, 8 is worst)

    Returns
    -------
    sim : list
        Matrix containing the new ratings for the simulated
        portfolios.

    '''
    sim_new = [[0 for j in range(len(sim[0]))] for i in range(len(sim))]
    for loan in range(len(portfolio)):
        thresholdvalues = threshold[(portfolio["rating_model"][loan])]
        for simnumber in range(len(sim)):
            simvalue = sim[simnumber][loan]
            sim_new[simnumber][loan] = int(normalRating(simvalue, thresholdvalues))
    return sim_new

def simportfoliovalues(sim, futurevalue):
    '''
    This function calculates the values of each simulated portfolio.
    The values are "meaningless" on their own, and should be
    compared to the original portfolio value to get the full picture.
    
    Parameters
    ----------
    sim : list
        Matrix containing the new ratings for the simulated
        portfolios.
        Rating values are int, between 0 and 8; 0 is the
        best rating, 7 is the worst rating, and 8 is default.
        This variable is provided by the newrating function.
    futurevalue : list
        List of values per future rating, structured as a list
        of values per rating per loan.
        That is, futurevalue[17] is a list containing the values
        per future rating of the portfolio[17] loan entry, with
        futurevalue[17][3] being the value at future rating 3.

    Returns
    -------
    portfoliovalue : list
        List of the simulated portfolio values, unsorted.

    '''
    portfoliovalue = [0 for i in range(len(sim))]
    loanamnt = len(sim[0])
    for simnumber in range(len(sim)):
        value = 0
        for loan in range(loanamnt):
            rating = sim[simnumber][loan]
            if rating == 8:
                beta = np.random.beta(2.105801,1.648153)
                value += futurevalue[loan][rating]*(beta/0.560956)
            else:
                value += futurevalue[loan][rating]
        portfoliovalue[simnumber] = value
    return portfoliovalue

def simulation(portfolio, threshold, futurevalue, correlation, samplesize):
    '''
    This function encompasses the entire simulation process.
    The output is "meaningless" on its own, and should
    be compared to the original portfolio value.

    Parameters
    ----------
    portfolio : list
        List of loan entries, with each entry containing
        the following information:
        [0] int: loan id
        [1] int: loan rating (0 is best rating, 8 is worst)
    threshold : list
        Table of rating migration thresholds.
        Structured as lists of thresholds for rating A
        migrating to rating B. That is, migration[3] is a
        list containing the thresholds of rating migration
        for a loan rated 3.
    futurevalue : list
        List of values per future rating, structured as a list
        of values per rating per loan.
        That is, futurevalue[17] is a list containing the values
        per future rating of the portfolio[17] loan entry, with
        futurevalue[17][3] being the value at future rating 3.
    correlation : float
        Correlation factor for rating migration.
        The same value is applied for all correlations.
        Bounded between 0 and 1, both inclusive.
    samplesize : int
        Amount of simulated portfolios.

    Returns
    -------
    values : float
        List of the simulated portfolio values, sorted.

    '''
    dim = len(portfolio)
    x = numgenerator(dim, correlation, samplesize)
    y = newrating(x, threshold, portfolio)
    values = simportfoliovalues(y, futurevalue)
    values.sort()
    return values

def changeportfolio(loanset, alternate, old, new, proportion):
    # Take sample from loanset
    nsample = proportion*len(loanset)
    aux = [loanset[loanset["category"]==x] for x in old]
    aux = pd.concat(aux)
    sample = aux.sample(n=nsample)
    # Count cat values from sample
    count = sample["category"].value_counts()
    # Take samples from loanset complement based on counts
    if len(count.index) < len(new):
        oldaux = []
        newaux = []
        for i in range(len(old)):
            if old[i] in count:
                oldaux.append(old[i])
                newaux.append(new[i])
        old = oldaux[:]
        new = newaux[:]    
    aux2 = [alternate[alternate["category"]==new[i]].sample(n=count[old[i]])
            for i in range(len(new))]
    # Create dataframe from sample complement
    # Append with samples
    aux3 = loanset.drop(sample.index)
    aux4 = pd.concat(aux2)
    df = pd.concat([aux3,aux4])
    df.reset_index(drop=True, inplace=True)
    return df

def varcalc(simvalue, portfoliovalue, quantile):
    index = len(simvalue)*quantile - 1
    plusminus = 1.96*sqrt(len(simvalue)*quantile*(1-quantile))
    var = simvalue[floor(index)]/portfoliovalue - 1
    lower = simvalue[floor(index-plusminus)]/portfoliovalue - 1
    upper = simvalue[ceil(index+plusminus)]/portfoliovalue - 1
    return [lower, var, upper]

### Setup 1
fullset = importdataset()
migration = probmig()
threshold = migthreshold(migration)
changeprop = 0.1
corr = 0.2
minrating = 4
nsim = 10000
quantiles = [0.05, 0.01, 0.005, 0.001]
riskfree = 0.03


### Test 0: rating test [sanity check]
sample = fullset.sample(n=2000, random_state=1)
sample.reset_index(drop=True, inplace=True)
test = sample[sample["rating_model"] <= minrating]
test.reset_index(drop=True, inplace=True)

sample_fv, sample_value = futureval(sample, migration, riskfree)
test_fv, test_value = futureval(test, migration, riskfree)

sim1 = simulation(sample, threshold, sample_fv, corr, nsim)
sim2 = simulation(test, threshold, test_fv, corr, nsim)

varsample = [varcalc(sim1, sample_value, x) for x in quantiles]
vartest = [varcalc(sim2, test_value, x) for x in quantiles]


### Setup 2: reasonable loanset
loanset = fullset[fullset["rating_model"] <= minrating]
baseline = loanset.sample(n=1500, random_state=2)
alternative = loanset.drop(baseline.index)

baseline.reset_index(drop=True, inplace=True)
alternative.reset_index(drop=True, inplace=True)

base_fv, base_value = futureval(baseline, migration, riskfree)
basesim = simulation(baseline, threshold, base_fv, corr, nsim)
varbase = [varcalc(basesim, base_value, x) for x in quantiles]


### Test 1: income test
alt1 = changeportfolio(baseline, alternative, [0,1,2,3], [4,5,6,7], changeprop)
alt2 = changeportfolio(baseline, alternative, [4,5,6,7], [0,1,2,3], changeprop)

alt1_fv, alt1_value = futureval(alt1, migration, riskfree)
alt2_fv, alt2_value = futureval(alt2, migration, riskfree)

sim1 = simulation(alt1, threshold, alt1_fv, corr, nsim)
sim2 = simulation(alt2, threshold, alt2_fv, corr, nsim)

var1 = [varcalc(sim1, alt1_value, x) for x in quantiles]
var2 = [varcalc(sim2, alt2_value, x) for x in quantiles]


### Test 2: loan interest test
alt1 = changeportfolio(baseline, alternative, [0,2,4,6], [1,3,5,7], changeprop)
alt2 = changeportfolio(baseline, alternative, [1,3,5,7], [0,2,4,6], changeprop)

alt1_fv, alt1_value = futureval(alt1, migration, riskfree)
alt2_fv, alt2_value = futureval(alt2, migration, riskfree)

sim1 = simulation(alt1, threshold, alt1_fv, corr, nsim)
sim2 = simulation(alt2, threshold, alt2_fv, corr, nsim)

var1 = [varcalc(sim1, alt1_value, x) for x in quantiles]
var2 = [varcalc(sim2, alt2_value, x) for x in quantiles]


### Test 3: term test
alt1 = changeportfolio(baseline, alternative, [0,1,4,5], [2,3,6,7], changeprop)
alt2 = changeportfolio(baseline, alternative, [2,3,6,7], [0,1,4,5], changeprop)

alt1_fv, alt1_value = futureval(alt1, migration, riskfree)
alt2_fv, alt2_value = futureval(alt2, migration, riskfree)

sim1 = simulation(alt1, threshold, alt1_fv, corr, nsim)
sim2 = simulation(alt2, threshold, alt2_fv, corr, nsim)

var1 = [varcalc(sim1, alt1_value, x) for x in quantiles]
var2 = [varcalc(sim2, alt2_value, x) for x in quantiles]


### Test 4: risk free rate test (discount curve test)
risk1 = 0.09
risk2 = -0.03

base1_fv, base1_value = futureval(baseline, migration, risk1)
base2_fv, base2_value = futureval(baseline, migration, risk2)

sim1 = simulation(baseline, threshold, base1_fv, corr, nsim)
sim2 = simulation(baseline, threshold, base2_fv, corr, nsim)

var1 = [varcalc(sim1, base1_value, x) for x in quantiles]
var2 = [varcalc(sim2, base2_value, x) for x in quantiles]


### Test 5: correlation test
corr1 = 0.35
corr2 = 0.05

sim1 = simulation(baseline, threshold, base_fv, corr1, nsim)
sim2 = simulation(baseline, threshold, base_fv, corr2, nsim)

var1 = [varcalc(sim1, base_value, x) for x in quantiles]
var2 = [varcalc(sim2, base_value, x) for x in quantiles]