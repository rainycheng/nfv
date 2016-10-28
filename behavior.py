#! /usr/bin/env python2.7
#encoding:utf-8
from __future__ import print_function
import threading,Queue,time,os,sys,subprocess
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
    def getNETstats(self, iface):
        #tree = ElementTree.fromstring(self.dom.XMLDesc())
        #iface = tree.find('devices/interface/target').get('dev')
        return self.dom.interfaceStats(iface)

    #record perfomrance monitoring events every 1s
    def startMonitor(self):
        #save previous stats to calculate rate features per second
        cpu_prev = self.getCPUstats()
        mem_prev = self.getMEMstats()
        disk_prev = self.getDISKstats()
        net_prev = []
        tree = ElementTree.fromstring(self.dom.XMLDesc())
        for iface in tree.findall('devices/interface/target'):
            net_prev.append(self.getNETstats(iface.get('dev')))
        
        while self._running:
            time.sleep(1)
            #collect CPU,memory,block,network performance stats 
            cpu_stats = self.getCPUstats()
            mem_stats = self.getMEMstats()
            disk_stats = self.getDISKstats()
#            net_stats = self.getNETstats()
            #concatenate VNF features into VEC vector, features are ordered according to report 3
            #memory features 
            VEC = str(mem_stats['major_fault'] - mem_prev['major_fault'])
            VEC = VEC + ' ' + str(mem_stats['minor_fault'] - mem_prev['minor_fault'])
            VEC = VEC + ' ' + str(mem_stats['rss'] - mem_prev['rss'])
            #CPU features
            VEC = VEC + ' ' + str(cpu_stats[0]['system_time']-cpu_prev[0]['system_time'])
            VEC = VEC + ' ' + str(cpu_stats[0]['user_time']-cpu_prev[0]['user_time'])
            #disk features
#            for i in range(0,4):
#                VEC = VEC + ' ' + str(disk_stats[i]) + ' ' + str(disk_stats[i] - disk_prev[i])
            #read bytes/s
            VEC = VEC + ' ' + str(disk_stats[1] - disk_prev[1])
            #write bytes/s
            VEC = VEC + ' ' + str(disk_stats[3] - disk_prev[3])

            #multiple net interfaces
            #tree = ElementTree.fromstring(self.dom.XMLDesc())
            count = 0
            net_stats = []
            for iface in tree.findall('devices/interface/target'):
                net_stats.append(self.getNETstats(iface.get('dev')))
                count = count + 1
            for j in range(0, count):
                #net features
                for i in range(0,8):
                    VEC = VEC + ' ' + str(net_stats[j][i] - net_prev[j][i])  
#                print ('read packets/s: ' + str(net_stats[j][0]-net_prev[j][0]))
#                print ('write packets/s: ' + str(net_stats[j][5]-net_prev[j][5]))

            VEC = VEC + '\n'
            #NFVmonitor put monitoring events into a shared queue
            self.queue.put(VEC)
#            print (VEC)        
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
        self.estimators = {'k_means_8': KMeans(n_clusters=8),
                           'k_means_10': KMeans(n_clusters=10),
                           'k_means_30': KMeans(n_clusters=30)}

        #load VNF performance monitoring features into the featuresX vector
        self.featuresX = np.loadtxt('features.txt')

        #items()[0] is a 2-tuple, self.est is used to execute kmeans (est.fit(X))
        self.estname1, self.est1 = self.estimators.items()[0]
        self.estname2, self.est2 = self.estimators.items()[1]
        
        threading.Thread.__init__(self, name=t_name)
    
    def terminate(self):
        self._running = False
    
    #using k-means to cluster VNF based on its features
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    def startCluster(self):
        #compute k-means clustering
        self.est1.fit(self.featuresX[:,0:7])
        self.est2.fit(self.featuresX[:,7:23])

        #labels.txt is used to record the labels of observations
        labels = open('/home/stack/labels.txt','w')
        #clusters.txt is used to record the cluster center points
        cluster_centers = open('/home/stack/centers.txt','w')
        
        #using str() to write readable formats into files
        i=0
        for lab in self.est1.labels_:
            lab_str = str(lab) + ' ' + str(self.est2.labels_[i]) +'\n'
            labels.write(lab_str)
            i = i+1

        for cent in self.est1.cluster_centers_:
            cluster_centers.write(str(cent)+"\n")
        for cent in self.est2.cluster_centers_:
            cluster_centers.write(str(cent)+"\n")
        
        #close the opened files
        labels.close()
        cluster_centers.close()

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
        sample_lab1 = self.est1.predict(sampleX[0:7].reshape(1,-1))
        sample_lab2 = self.est2.predict(sampleX[7:23].reshape(1,-1))
        #put sample_label into obv_queue
        self.obv_queue.put(np.hstack((sample_lab1,sample_lab2))) 

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
#        self.hmm = hmm.GaussianHMM(n_components=5, covariance_type="full")
        self.hmm = hmm.GaussianHMM(n_components=5)

        threading.Thread.__init__(self, name=t_name)
    
    def terminate(self):
        self._running = False
    
    #get VNF observation from the queue
    def getObservation(self):
        try:
            #queue.get([block[,timeout]]) method. https://docs.python.org/2/library/queue.html
            #block=1 means block if necessary until an item is available, timeout=3 means block
            #at most 3 sesconds and raises the Empty exception if no item was available within 3s
            Observation = []
            Observation.append(self.queue.get(1,3))
            #indicate that a formerly enqueued task is complete
            self.queue.task_done()         
        except Exception, e:
            print (e)
            self.terminate()
        #self.X is a list, add the new Observation label into self.X list tail
        self.X.append(Observation[0])
        #keep the list length of self.X fixed to a given number
        #delete the old items in the head of self.X list 
        if (len(self.X) > 10):
            del self.X[0]
#        print (self.X)
 
    def trainHMM(self):
        #the trained VNF feature stats are stored in 'features.txt'
        #the corresponding VNF cluster labels are stored in 'labels.txt'  
        obvX = np.loadtxt('labels.txt')
        #the obvX has only one feature, using X.reshape(-1,1) to format dimension
        #hmm.fit is used to train the GaussianHMM model
#        self.hmm.fit(obvX.reshape(-1,1))
        self.hmm.fit(obvX)
 
    def predictHMM(self):
        self.getObservation()
#        print (self.X)
        sampleX = np.array(self.X)
        #hmm.score returns the Log likelihood of sampleX under the model
#        return self.hmm.score(sampleX.reshape(-1,1)) 
        print (sampleX.reshape(-1,2))
        return self.hmm.score(sampleX.reshape(-1,2)) 

    def startHMM(self):
        self.trainHMM()
        while self._running:
            Q = self.predictHMM()
            #if the Log likelihood Q is below a given threshold, report 'Abnormal'
            if (Q < self.threshold):
                print ("Abnormal NFV. Q = " + str(Q) + "\n")
            else:
                print ("Likelihood Q = " + str(Q) + "\n")
            time.sleep(1)   

    def run(self):
        self.startHMM()   

class NFVThrottle(threading.Thread):
    """VM resource throttle"""
    def __init__(self, t_name, inst_name):
        self._running = True
        self.inst_name = inst_name
        threading.Thread.__init__(self, name=t_name)

    def terminate(self):
        self._running = False

    def throttleCPU(self, quota):
        cgdir = '/sys/fs/cgroup/cpu/machine/'
        COMMAND = 'echo ' + quota + ' > ' + cgdir + self.inst_name + '.libvirt-qemu/cpu.cfs_quota_us'      
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()

    def throttleMEM(self, quota):
        cgdir = '/sys/fs/cgroup/memory/machine/'
        COMMAND = 'echo ' + quota + ' > ' + cgdir + self.inst_name + '.libvirt-qemu/memory.limit_in_bytes'
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()

    def throttleDISK(self, quota, op):
        cgdir = '/sys/fs/cgroup/blkio/machine/'
        if (op == 'read_iops'):
            COMMAND = 'echo ' + '8:0 ' + quota + ' > ' + cgdir + self.inst_name \
                    + '.libvirt-qemu/blkio.throttle.read_iops_device'
        elif (op == 'write_iops'):
            COMMAND = 'echo ' + '8:0 ' + quota + ' > ' + cgdir + self.inst_name \
                    + '.libvirt-qemu/blkio.throttle.write_iops_device'
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()

    def throttleNET(self, quota, op):
        dev = 'tap0'
        dev1 = 'tap4a75cfa4-8c'
        rate = 'rate=1000'
        COMMAND = 'ovs-vsctl set interface ' + dev + 'ingress_policing_'+ rate
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()  

    def startThrottle(self):
        print ("OK!\n")  
 
    def run(self):
        self.startThrottle()

class NFVRegulator(threading.Thread):
    def __init__(self, t_name):
        self._running = True
        threading.Thread.__init__(self, name=t_name)
    
    def terminate(self):
        self._running = False
    
    def startRegulate(self):
        print ("OK!\n")

    def run(self):
        self.startRegulate()

if __name__ == "__main__":
   
    if len(sys.argv) == 1:
       print ("please give the domain id parameter!\n")
    # the shared queue q is used by NFVMonitor and NFVCluster, each item in the q is the performance vector
    Q_vec = Queue.Queue()
    # the shared queue qc is used by NFVCluster and NFVHMM, each item in the qc is the observation
    Q_obv = Queue.Queue()
 
    nfv_monitor = NFVMonitor('nfv_monitor', int(sys.argv[1]), Q_vec)
    nfv_monitor.start()
    time.sleep(15)
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
