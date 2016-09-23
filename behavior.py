#! /usr/bin/env python2.7
#encoding:utf-8
from __future__ import print_function
import threading,Queue,time,os,sys
import libvirt
import numpy as np
from xml.etree import ElementTree
from numpy import array
# apt-get install python-sklearn
from sklearn.cluster import KMeans
# $pip install -U --user hmmlearn
#https://github.com/hmmlearn/hmmlearn
from hmmlearn import hmm


class NFVMonitor(threading.Thread):
   """performance monitoring class"""
   def __init__(self,t_name,domID,queue):
      self.queue = queue
      #self._running state controls the termination of this thread
      self._running = True
      
      #open features.txt file, this file is used to record VNF performance monitoring features
      self.features = open('/home/stack/features.txt','w')
      
      #connect to libvirt
      conn = libvirt.open('qemu:///system')
      if conn == None:
         print('Failed to open connection to qemu:///system', file=sys.stderr)
         exit(1)
      
      #domID represents the VM instance ID, this ID can be obtained using 'virsh list'
      self.dom = conn.lookupByID(domID)
      if self.dom == None:
         print('Failed to find the domain '+domName, file=sys.stderr)
         exit(1)
      
      #initialize this thread
      threading.Thread.__init__(self, name=t_name)
   
   #terminate this thread, close the opened 'features.txt' file in __init__
   def terminate(self):
      self._running = False
      self.features.close()
   
   #get CPU performance events
   def getCPUstats(self):
      return self.dom.getCPUStats(True)

   #get memory performance events
   def getMEMstats(self):
      return self.dom.memoryStats()

   #get block device performance events
   def getDISKstats(self):
      return self.dom.blockStats('vda')

   #get network performance events
   def getNETstats(self):
      tree = ElementTree.fromstring(self.dom.XMLDesc())
      iface = tree.find('devices/interface/target').get('dev')
      return self.dom.interfaceStats(iface)

   #record perfomrance monitoring events every 1s
   def startMonitor(self):
      while self._running:
         #collect CPU,memory,block,network performance stats 
         cpu_stats = self.getCPUstats()
         mem_stats = self.getMEMstats()
         rd_req, rd_bytes, wr_req, wr_bytes, err = self.getDISKstats()
         net_stats = self.getNETstats()
         
         #concatenate CPU stats
         stats_vector = str(cpu_stats[0]['cpu_time']) + ' ' + str(cpu_stats[0]['system_time']) \
         + ' ' + str(cpu_stats[0]['user_time'])
         #concatenate memory stats
         for name in mem_stats:
            stats_vector = stats_vector + ' ' + str(mem_stats[name])
         #concatenate block stats
         stats_vector = stats_vector + ' ' + str(rd_req) + ' ' + str(rd_bytes) + ' ' \
                       + str(wr_req) + ' ' + str(wr_bytes) + ' ' + str(err)
         #concatenate network stats
         for i in range(0,8):
	    stats_vector = stats_vector + ' ' + str(net_stats[i])
         stats_vector = stats_vector + '\n'
         
         #NFVmonitor put monitoring events into a shared queue
         self.queue.put(stats_vector)
         
         #write stats_vector into features.txt file, do not foget to flush into disk
         self.features.write(stats_vector)
         self.features.flush()
         time.sleep(1) 

   def run(self):
      self.startMonitor()


class NFVCluster(threading.Thread):
   """transform VNF observations into a sequence of cluster labels"""
   def __init__(self,t_name):
      self._running = True
      #estimators are used to save different KMeans algos (# of clusters). Example:
      #http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html
      self.estimators = {'k_means_10': KMeans(n_clusters=3)}
                        # 'k_means_20': KMeans(n_clusters=20),
                        # 'k_means_30': KMeans(n_clusters=30)}

      #load VNF performance monitoring features into the featuresX vector
      self.featuresX = np.loadtxt('features.txt')

      #items()[0] is a 2-tuple, self.est is used to execute kmeans (est.fit(X))
      self.estname, self.est = self.estimators.items()[0]
      
      threading.Thread.__init__(self, name=t_name)
   
   def terminate(self):
      self._running = False
   
   #using k-means to cluster VNF based on its features
   #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
   def startCluster(self):
      #compute k-means clustering
      self.est.fit(self.featuresX)

      #labels.txt is used to record the labels of observations
      labels = open('/home/stack/labels.txt','w')
      #clusters.txt is used to record the cluster center points
      cluster_centers = open('/home/stack/centers.txt','w')
      
      #using str() to write readable formats into files
      for lab in self.est.labels_:
         labels.write(str(lab)+'\n')
      for cent in self.est.cluster_centers_:
	 cluster_centers.write(str(cent)+"\n")
      
      #close the opened files
      labels.close()
      cluster_centers.close()

   #predict the cluster label of each sample observation 
   def predictCluster(self):
      #sample features are stored in 'samples.txt' file 
      samples = np.loadtxt('samples.txt')
      #predicted labels are stored in 'predicts.txt' file
      predicts = open('/home/stack/predicts.txt','w')
      
      #using the est.predict(X) method, str() readable formats 
      #Predict the closest cluster each sample in X belongs to.
      for pred in self.est.predict(samples):
            predicts.write(str(pred)+'\n')

   def run(self):
      self.startCluster()
      #startCluster() must run before predictCluster
      #the trained parameters are stored in self.est after running est.fit(X)
      self.predictCluster()


class NFVHMM(threading.Thread):
   """Hidden Markov Models"""
   def __init__(self, t_name, queue):
      self._running = True
      # the shared queue is used, monitor is the producer, HMM is the consumer
      # HMM get an observation from the queue at a time, 
      # Monitor put an observation into the queue every 1s
      self.queue = queue
      
      self.hmm = hmm.GaussianHMM(n_components=2, covariance_type="full")

      threading.Thread.__init__(self, name=t_name)
   
   def terminate(self):
      self._running = False
   
   #get VNF observation from the queue
   def getObservation(self):
      try:
         #queue.get([block[,timeout]]) method. https://docs.python.org/2/library/queue.html
         #block=1 means block if necessary until an item is available, timeout=3 means block
         #at most 3 sesconds and raises the Empty exception if no item was available within 3s
         Observation = self.queue.get(1,3)
      except Exception, e:
         print (e)
         self.terminate()

      print (Observation)
      #indicate that a formerly enqueued task is complete
      self.queue.task_done()
      time.sleep(1)

   def startHMM(self):
      while self._running:
         self.getObservation()

   def run(self):
      self.startHMM()   

class NFVThrottle(threading.Thread):
   """VM resource throttle"""
   def __init__(self, t_name):
      self._running = True
      threading.Thread.__init__(self, name=t_name)

   def terminate(self):
      self._running = False

   def startThrottle(self):
   
   def run(self):
      self.startThrottle()


if __name__ == "__main__":
   q = Queue.Queue()
   nfv_monitor = NFVMonitor('nfv_monitor',33,q)
   nfv_hmm = NFVHMM('nfv_hmm',q)
   nfv_monitor.start()
   nfv_hmm.start()
   time.sleep(10)
   nfv_monitor.terminate()
   nfv_hmm.terminate()
   
   nfv_cluster = NFVCluster('nfv_cluster')
   nfv_cluster.start()
   #nfv_monitor.join()
   nfv_cluster.join()
   nfv_hmm.join()
   #features.close()

