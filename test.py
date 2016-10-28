#!/usr/bin/env python
#coding=utf8

import os,sys,time,Queue
import numpy as np
from hmmlearn import hmm
startprob = np.array([0.6,0.3,0.1])
transmat = np.array([[0.7,0.2,0.1],[0.3,0.5,0.2],[0.3,0.3,0.4]])
means = np.array([[0.0,0.0],[3.0,-3.0],[5.0,10.0]])
covars = np.tile(np.identity(2),(3,1,1))
a = np.array([1])
b = np.array([2])
print a
print b
print np.hstack((a,b))
print startprob
print startprob.reshape(1,-1)
print startprob.reshape(-1,1)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")

featureX = np.loadtxt('features.txt')
obvX = np.loadtxt('labels.txt')

print obvX
print obvX.reshape(-1,1)

print featureX
print featureX[:,0:3]

model.fit(featureX[:,0:2])

from sklearn.cluster import KMeans
from sklearn import datasets

estor = {'k_means_3': KMeans(n_clusters=3)}
iris = datasets.load_iris()
X = iris.data
Y = iris.target
#print X
for name, est in estor.items():
   est.fit(X)
   print est.labels_

q = Queue.Queue()

q.put('a')
q.put('b')

print q.qsize()

x = []
for i in range(0,10):
   x.append(i)
   if (len(x)>5):
      del x[0]
   print x
