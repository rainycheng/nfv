#! /usr/bin/env python2.7
#encoding:utf-8
from __future__ import print_function
import threading,Queue,time,os,sys
import libvirt
import numpy as np
from xml.etree import ElementTree
from numpy import array
from sklearn.cluster import KMeans

class NFVMonitor(threading.Thread):
   """performance monitoring class"""
   def __init__(self,t_name,domID,queue):
      self.queue = queue
      self._running = True
      self.features = open('/home/stack/features.txt','w')
      conn = libvirt.open('qemu:///system')
      if conn == None:
         print('Failed to open connection to qemu:///system', file=sys.stderr)
         exit(1)
      self.dom = conn.lookupByID(domID)
      if self.dom == None:
         print('Failed to find the domain '+domName, file=sys.stderr)
         exit(1)

      threading.Thread.__init__(self, name=t_name)
   
   def terminate(self):
      self._running = False
      self.features.close()

   def getCPUstats(self):
      return self.dom.getCPUStats(True)

   def getMEMstats(self):
      return self.dom.memoryStats()

   def getDISKstats(self):
      return self.dom.blockStats('vda')

   def getNETstats(self):
      tree = ElementTree.fromstring(self.dom.XMLDesc())
      iface = tree.find('devices/interface/target').get('dev')
      return self.dom.interfaceStats(iface)

   def startMonitor(self):
      while self._running:
         cpu_stats = self.getCPUstats()
         mem_stats = self.getMEMstats()
         rd_req, rd_bytes, wr_req, wr_bytes, err = self.getDISKstats()
         net_stats = self.getNETstats()
         stats_vector = str(cpu_stats[0]['cpu_time']) + ' ' + str(cpu_stats[0]['system_time']) \
         + ' ' + str(cpu_stats[0]['user_time'])
         for name in mem_stats:
            stats_vector = stats_vector + ' ' + str(mem_stats[name])
         stats_vector = stats_vector + ' ' + str(rd_req) + ' ' + str(rd_bytes) + ' ' + str(wr_req) \
         + ' ' + str(wr_bytes) + ' ' + str(err)
         for i in range(0,8):
	    stats_vector = stats_vector + ' ' + str(net_stats[i])
         stats_vector = stats_vector + '\n'
         #NFVmonitor put monitoring events into a shared queue
         self.queue.put(stats_vector)
         self.features.write(stats_vector)
         self.features.flush()
#         print(stats_vector)
         time.sleep(1) 

   def run(self):
      self.startMonitor()


class NFVCluster(threading.Thread):
   """transform VNF observations into a sequence of cluster labels"""
   def __init__(self,t_name):
      self._running = True
      self.estimators = {'k_means_10': KMeans(n_clusters=3)}
                        # 'k_means_20': KMeans(n_clusters=20),
                        # 'k_means_30': KMeans(n_clusters=30)}
      self.featuresX = np.loadtxt('features.txt')
      self.estname, self.est = self.estimators.items()[0]
      threading.Thread.__init__(self, name=t_name)
   
   def terminate(self):
      self._running = False

   def startCluster(self):
      self.est.fit(self.featuresX)
      labels = open('/home/stack/labels.txt','w')
      cluster_centers = open('/home/stack/centers.txt','w')
      for lab in self.est.labels_:
         labels.write(str(lab)+'\n')
      for cent in self.est.cluster_centers_:
	 cluster_centers.write(str(cent)+"\n")
      labels.close()
      cluster_centers.close()

   def predictCluster(self): 
      samples = np.loadtxt('samples.txt')
      predicts = open('/home/stack/predicts.txt','w')
      for pred in self.est.predict(samples):
            predicts.write(str(pred)+'\n')

   def run(self):
      self.startCluster()
      self.predictCluster()


class NFVHMM(threading.Thread):
   """Hidden Markov Models"""
   def __init__(self, t_name, queue):
      self._running = True
      self.queue = queue
      threading.Thread.__init__(self, name=t_name)
   
   def terminate(self):
      self._running = False

   def getObservation(self):
      try:
         Observation = self.queue.get(1,3)
      except Exception, e:
         print (e)
         self.terminate()

      print (Observation)
      self.queue.task_done()
      time.sleep(1)

   def startHMM(self):
      while self._running:
         self.getObservation()

   def run(self):
      self.startHMM()   

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

