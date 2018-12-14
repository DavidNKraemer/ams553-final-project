# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:45:37 2018

@author: andre
"""
from numpy import random
from scipy import stats
from scipy import misc

p = 0.5
q = 1-p
n = 7
bigN = 1000

randNums = []

for i in range(bigN):
    randNums.append(random.binomial(n, p))

freq = []

for i in range(n+1):
    freq.append(randNums.count(i))

print(freq)

for i in range(len(freq)):
    freq[i] -= (misc.comb(n,i)*(p**i)*(q**(n-i)))*bigN
    freq[i] = freq[i]**2
    freq[i] /= (misc.comb(n,i)*(p**i)*(q**(n-i)))*bigN
    
print(sum(freq))
p_val = 1-stats.chi2.cdf(sum(freq), n)

print(p_val)
