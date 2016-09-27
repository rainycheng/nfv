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
   def __init__(self,t_name,domID,vec_queue):
      self.queue = vec_queue
      #self._running state controls the termination of this thread
      self._running = True
      
      #open features.txt file, this file is used to record VNF performance monitoring features
      self.features = open('/home/stack/features.txt','a')
      
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
      #self.features.close()
   
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
      #save previous stats to calculate rate features per second
      cpu_prev = self.getCPUstats()
      mem_prev = self.getMEMstats()
      disk_prev = self.getDISKstats()
      net_prev = self.getNETstats()
      
      while self._running:
         time.sleep(1)
         #collect CPU,memory,block,network performance stats 
         cpu_stats = self.getCPUstats()
         mem_stats = self.getMEMstats()
         disk_stats = self.getDISKstats()
         net_stats = self.getNETstats()
         #concatenate VNF features into VEC vector, features are ordered according to report 3
         #memory features 
         VEC = str(mem_stats['actual']) + ' ' + str(mem_stats['actual'] - mem_stats['unused'])
         #CPU features
         VEC = VEC + ' ' + str(cpu_stats[0]['cpu_time'])
         VEC = VEC + ' ' + str(cpu_stats[0]['cpu_time']-cpu_prev[0]['cpu_time'])
         #disk features
         for i in range(0,4):
            VEC = VEC + ' ' + str(disk_stats[i]) + ' ' + str(disk_stats[i] - disk_prev[i])
         #net features
         for i in range(0,8):
            VEC = VEC + ' ' + str(net_stats[i]) + ' ' + str(net_stats[i] - net_prev[i])  
         VEC = VEC + '\n'
         
         #NFVmonitor put monitoring events into a shared queue
         self.queue.put(VEC)
         
         #write stats_vector into features.txt file, do not foget to flush into disk
         self.features.write(VEC)
         self.features.flush()
         #save previous stats to calculate rate features per second 
         cpu_prev = cpu_stats
         mem_prev = mem_stats
         disk_prev = disk_stats
         net_prev = net_stats 
      self.features.close()

   def run(self):
      self.startMonitor()


class NFVCluster(threading.Thread):
   """transform VNF observations into a sequence of cluster labels"""
   def __init__(self, t_name, vec_queue, obv_queue):
      self._running = True
      #vec_queue is used to store VNF features vectors 
      #obv_queue is used to store observation cluster labels
      self.vec_queue = vec_queue
      self.obv_queue = obv_queue
      #estimators are used to save different KMeans algos (# of clusters). Example:
      #http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html
      self.estimators = {'k_means_10': KMeans(n_clusters=10)}
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

#   def labelCluster(self):
#      #sample features are stored in 'samples.txt' file 
#      samples = np.loadtxt('features.txt')
#      #predicted labels are stored in 'predicts.txt' file
#      predicts = open('/home/stack/predicts.txt','w')
#      
#      #using the est.predict(X) method, str() readable formats 
#      #Predict the closest cluster each sample in X belongs to.
#      for pred in self.est.predict(samples):
#            predicts.write(str(pred)+'\n')
   def predictCluster(self):
      try:
         #get a feature vector sample from vec_queue, return type string
         sample_vector = self.vec_queue.get(1,3)
         self.vec_queue.task_done()
      except Exception, e:
         print (e)
         self.terminate()
      #split sample_vector into type float array
      sampleX = np.array([float(i) for i in sample_vector.split(' ')])
      #the sample has only one feature, use X.reshape(1,-1) to adjust dimension
      #predict the cluster label of sampleX
      sample_label = self.est.predict(sampleX.reshape(1,-1))
      #put sample_label into obv_queue
      self.obv_queue.put(sample_label) 

   def run(self):
      self.startCluster()
      #startCluster() must run before predictCluster
      #the trained parameters are stored in self.est after running est.fit(X)
      while self._running:
         self.predictCluster()
         time.sleep(1)

class NFVHMM(threading.Thread):
   """Hidden Markov Models"""
   def __init__(self, t_name, obv_queue):
      self._running = True
      # the shared queue is used, monitor is the producer, HMM is the consumer
      # HMM get an observation from the queue at a time, 
      # Cluster put an observation label into the obv_queue every 1s
      self.queue = obv_queue
      # the threshold used to determine abnormal VNF behavior
      self.threshold = -10
      #self.X represents the observation sequences, type list
      self.X = []

      #use GaussianHMM, n_components is the number of hidden states
      self.hmm = hmm.GaussianHMM(n_components=5, covariance_type="full")

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
         #indicate that a formerly enqueued task is complete
         self.queue.task_done()         
      except Exception, e:
         print (e)
         self.terminate()
      #self.X is a list, add the new Observation label into self.X list tail
      self.X.append(Observation[0])
      #keep the list length of self.X fixed to a given number
      #delete the old items in the head of self.X list 
      if (len(self.X) > 20):
         del self.X[0]
   
   def trainHMM(self):
      #the trained VNF feature stats are stored in 'features.txt'
      #the corresponding VNF cluster labels are stored in 'labels.txt'  
      obvX = np.loadtxt('labels.txt')
      #the obvX has only one feature, using X.reshape(-1,1) to format dimension
      #hmm.fit is used to train the GaussianHMM model
      self.hmm.fit(obvX.reshape(-1,1))

   def predictHMM(self):
      self.getObservation()
#      print (self.X)
      sampleX = np.array(self.X)
      #hmm.score returns the Log likelihood of sampleX under the model
      return self.hmm.score(sampleX.reshape(-1,1)) 

   def startHMM(self):
      self.trainHMM()
      while self._running:
         Q = self.predictHMM()
         #if the Log likelihood Q is below a given threshold, report 'Abnormal'
         if (Q < self.threshold):
	    print ("Abnormal NFV. Q = " + str(Q) + "\n")
         time.sleep(1)   

   def run(self):
      self.startHMM()   

class NFVThrottle(threading.Thread):
   """VM resource throttle"""
   def __init__(self, t_name):
      self._running = True
      threading.Thread.__init__(self, name=t_name)

   def terminate(self):
      self._running = False

   #def startThrottle(self):
   
   #def run(self):
    #  self.startThrottle()


if __name__ == "__main__":
   # the shared queue q is used by NFVMonitor and NFVCluster, each item in the q is the performance vector
   Q_vec = Queue.Queue()
   # the shared queue qc is used by NFVCluster and NFVHMM, each item in the qc is the observation
   Q_obv = Queue.Queue()
 
   nfv_monitor = NFVMonitor('nfv_monitor', 33, Q_vec)
   nfv_monitor.start()
   time.sleep(5)
   nfv_cluster = NFVCluster('nfv_cluster', Q_vec, Q_obv)
   nfv_cluster.start()
   time.sleep(10)

   nfv_hmm = NFVHMM('nfv_hmm', Q_obv)
   nfv_hmm.start()
   time.sleep(50)
   
   nfv_monitor.terminate()
   nfv_cluster.terminate()
   nfv_hmm.terminate()

   #nfv_cluster = NFVCluster('nfv_cluster', Q_vec, Q_obv)
   #nfv_cluster.start()
   nfv_monitor.join()
   nfv_cluster.join()
   nfv_hmm.join()
   #nfv_monitor.join()
