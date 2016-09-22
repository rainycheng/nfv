#!/usr/bin/env python
#coding=utf8

import os,sys,time
import numpy as np
#from sklearn import hmm
startprob = np.array([0.6,0.3,0.1])
transmat = np.array([[0.7,0.2,0.1],[0.3,0.5,0.2],[0.3,0.3,0.4]])
means = np.array([[0.0,0.0],[3.0,-3.0],[5.0,10.0]])
covars = np.tile(np.identity(2),(3,1,1))
#model = hmm.GaussianHMM(3, "full", startprob, transmat)

from sklearn.cluster import KMeans
from sklearn import datasets

estor = {'k_means_3': KMeans(n_clusters=3)}
iris = datasets.load_iris()
X = iris.data
Y = iris.target
print X
for name, est in estor.items():
   est.fit(X)
   print est.labels_

