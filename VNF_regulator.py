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
from sklearn import preprocessing
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
        self.features = open('/home/stack/features1.txt','a')
        
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
            time.sleep(5)
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
###            self.queue.put(VEC)
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
        global global_cluster1
        global global_cluster2
        self.estimators = {'k_means_8': KMeans(n_clusters=global_cluster1),
                           'k_means_10': KMeans(n_clusters=global_cluster2),
                           'k_means_30': KMeans(n_clusters=30)}

        #load VNF performance monitoring features into the featuresX vector
#        X = np.loadtxt('features5.txt')
#        file_stand = open('stand.txt','w') 
#        scaler = preprocessing.StandardScaler().fit(X)
#        self.featuresX = scaler.transform(X)
#        for i in self.featuresX:
#            for j in i:
#                file_stand(str(j)+' ')
#            file_stand.write('\n')
#        file_stand.close()

        #items()[0] is a 2-tuple, self.est is used to execute kmeans (est.fit(X))
        self.estname1, self.est1 = self.estimators.items()[0]
        self.estname2, self.est2 = self.estimators.items()[1]
        
        threading.Thread.__init__(self, name=t_name)
    
    def terminate(self):
        self._running = False

    # Preprocess the input VNF collected features
    # Use standardization method with zero mean and unit variance
    # da_input is the input data array of VNF features
    # f_output is the output file of the standardization result     
    def standardization(self, da_input, f_output):
        file_stand = open(f_output,'w')
        # The preprocessing module further provides a utility class StandardScaler that implements 
        # the Transformer API to compute the mean and standard deviation on a training set so as to 
        # be able to later reapply the same transformation on the testing set.   
        scaler = preprocessing.StandardScaler().fit(da_input)
        da_output = scaler.transform(da_input)
        
        # write the standardized features into file_stand.
        for i in da_output:
            for j in i:
                file_stand.write(str(j)+' ')
            file_stand.write('\n')
        file_stand.close()
        
        # return standardized features data array  
        return da_output
           
    #using k-means to cluster VNF based on its features
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    def trainCluster(self, f_raw, f_out_stand, f_out_label):
        #compute k-means clustering
        raw_features = np.loadtxt(f_raw)
        train_features = self.standardization(raw_features, f_out_stand)
        
        # the first 7 features represent CPU/MEM/BLOCK features
        # use these 7 features to train the cluster1 model
        self.est1.fit(train_features[:,0:7])
        # the last 16 features represent NETWORK features
        # use these 16 features to train the cluster2 model
        self.est2.fit(train_features[:,7:23])

        # labels.txt is used to record the labels of observations
        labels = open(f_out_label,'w')
        # clusters.txt is used to record the cluster center points
        cluster_centers = open('/home/stack/centers.txt','w')
        
        # write the cluster labels into the f_out_label file 
        i=0
        for lab in self.est1.labels_:
            #using str() to write readable formats into the file
            lab_str = str(lab) + ' ' + str(self.est2.labels_[i]) +'\n'
            labels.write(lab_str)
            i = i+1
        
        # write the cluster centroids into the centers.txt file
        for cent in self.est1.cluster_centers_:
            cluster_centers.write(str(cent)+"\n")
        for cent in self.est2.cluster_centers_:
            cluster_centers.write(str(cent)+"\n")
        
        #close the opened files
        labels.close()
        cluster_centers.close()

    # predictCluster is used to predict cluster labels online
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
        #the input data has only one sample, use X.reshape(1,-1) to adjust dimension
        #predict the cluster label of sampleX
        sample_lab1 = self.est1.predict(sampleX[0:7].reshape(1,-1))
        sample_lab2 = self.est2.predict(sampleX[7:23].reshape(1,-1))
        #put sample_label into obv_queue
        self.obv_queue.put(np.hstack((sample_lab1,sample_lab2))) 
    
    # offlinePredict is used to predict cluster labels offline
    # f_raw is the raw VNF features file
    # f_out_stand is the output file of standardized features
    # f_out_label is the output file of the predicted cluster labels
    def offlinePredict(self, f_raw, f_out_stand, f_out_label):
        # load the raw features
        raw_features = np.loadtxt(f_raw)
        # standardization of raw_features
        predict_features = self.standardization(raw_features, f_out_stand)
        file_label = open(f_out_label,'w')
        
        # predict the cluster label of each sample
        for off_vec in predict_features:
            # the input off_vec has only one sample, use reshape(1,-1) to adjust dimension 
            off_lab1 = self.est1.predict(np.array(off_vec)[0:7].reshape(1,-1))
            off_lab2 = self.est2.predict(np.array(off_vec)[7:23].reshape(1,-1))
            file_label.write(str(off_lab1[0])+' '+str(off_lab2[0])+'\n')            

    # run() is used in the multi-thread mode of online clustering for every second
    def run(self):
        self.trainCluster('features.txt','stand_fea.txt','train_label.txt')
        #trainCluster() must run before predictCluster
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
        global global_components
        self.hmm = hmm.GaussianHMM(n_components=global_components)

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
        global global_window
        if (len(self.X) > global_window):
            del self.X[0]
#        print (self.X)
 
    # train HMM model using f_input_labels
    def trainHMM(self, f_input_labels):
        # load the VNF clustered labels  
        obvX = np.loadtxt(f_input_labels)
        #the obvX has only one feature, using X.reshape(-1,1) to format dimension
        #hmm.fit is used to train the GaussianHMM model
#        self.hmm.fit(obvX.reshape(-1,1))
        self.hmm.fit(obvX)

    # use the trained HMM model to predict online VNF behavior log likelihood
    def predictHMM(self):
        self.getObservation()
#        print (self.X)
        sampleX = np.array(self.X)
        #hmm.score returns the Log likelihood of sampleX under the model
#        return self.hmm.score(sampleX.reshape(-1,1)) 
        print (sampleX.reshape(-1,2))
        return self.hmm.score(sampleX.reshape(-1,2)) 

    # use the trained HMM model to predict offline VNF behavior log likelihood
    # f_input_labels is the file containing VNF clustered labels
    # f_output_predict is the file containing the HMM log likelihood of predicted results 
    def offlinePredict(self, f_input_labels, f_output_predict):
        # obv_window is the slide window used to store cluster label observations 
        obv_window = []
        predict_result = open(f_output_predict,'w')
        obv_input = np.loadtxt(f_input_labels)
        
        # cluster labels are stored in obv_input
        for obv_X in obv_input:
            # the sliding window size is determined by global_window variable
            obv_window.append(obv_X)
            global global_window
            # the number of labels in the obv_window is fixed to the size of global_window
            if( len(obv_window) > global_window):
                del obv_window[0]
              
            # use the label sequence in the obv_window to calculate the log likelihood  
            obv_score = self.hmm.score(np.array(obv_window).reshape(-1,2))
            # store the log likelihood into the f_output_predict file.
            predict_result.write(str(obv_score)+'\n')

        print ('global_window:' + str(global_window)) 
        predict_result.close()

    # startHMM() includes the whole steps of training HMM model, 
    # predicting HMM log likelihood, and determining abnormal VNF behaviors.
    def startHMM(self):
        self.trainHMM()
        while self._running:
            Q = self.predictHMM()
            #if the Log likelihood Q is below a given threshold, report 'Abnormal'
            global global_threshold
            if (Q < global_threshold):
                print ("Abnormal NFV. Q = " + str(Q) + "\n")
            else:
                print ("Likelihood Q = " + str(Q) + "\n")
            time.sleep(1)   

    # run() is the default funtion of starting the thread 
    def run(self):
        self.startHMM()   

# NFVThrottle provides the mechanisms to throttle CPU/MEM/DISK/Network resources of NFV
class NFVThrottle(threading.Thread):
    """VM resource throttle"""
    # t_name is the thread name, inst_name is the instance name (through virsh list)
    def __init__(self, t_name, inst_name):
        self._running = True
        self.inst_name = inst_name
        threading.Thread.__init__(self, name=t_name)

    def terminate(self):
        self._running = False

    # throttle CPU quota
    def throttleCPU(self, quota):
        cgdir = '/sys/fs/cgroup/cpu/machine/'
        # cfs_quota_us is the hard limit CPU time slice allocated to this VM
        # https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/6/html/Resource_Management_Guide/sec-cpu.html 
        COMMAND = 'echo ' + str(quota) + ' > ' + cgdir + self.inst_name + '.libvirt-qemu/cpu.cfs_quota_us'      
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()

    # throttle memory resource
    def throttleMEM(self, quota):
        cgdir = '/sys/fs/cgroup/memory/machine/'
        # memory.limit_in_bytes is the maximum amount of user memory
        # https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/6/html/Resource_Management_Guide/sec-memory.html
        COMMAND = 'echo ' + str(quota) + ' > ' + cgdir + self.inst_name + '.libvirt-qemu/memory.limit_in_bytes'
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()

    # throttle disk bandwidth
    def throttleDISK(self, quota, op):
        cgdir = '/sys/fs/cgroup/blkio/machine/'
        if (op == 'read_bps'):
        # read_bps_device specifies the upper limit on the number of read operations a device can perform.
        # https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/6/html/Resource_Management_Guide/ch-Subsystems_and_Tunable_Parameters.html#sec-blkio    
            COMMAND = 'echo ' + '8:0 ' + str(quota) + ' > ' + cgdir + self.inst_name \
                    + '.libvirt-qemu/blkio.throttle.read_bps_device'
        elif (op == 'write_bps'):
        # write_bps_device specifies the upper limit on the number of write operations a device can perform.
            COMMAND = 'echo ' + '8:0 ' + str(quota) + ' > ' + cgdir + self.inst_name \
                    + '.libvirt-qemu/blkio.throttle.write_bps_device'
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()

    # throttle network interface bandwidth
    def throttleNET(self, in_quota, e_quota, dev):
        #dev = 'tap0'
        #dev1 = 'tap4a75cfa4-8c'
        #rate = 'rate=1000'
        # ingress_policing_rate limits this network interface sending bandwidth
        # http://openvswitch.org/support/dist-docs/ovs-vsctl.8.txt
        COMMAND = 'ovs-vsctl set interface ' + dev + 'ingress_policing_rate=' + str(in_quota)
        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
#        COMMAND = 'ovs-vsctl set interface ' + dev + 'egress_policing_rate=' + str(e_quota)  
#        p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#        p.wait()

    def startThrottle(self):
        print ("OK!\n")  
 
    def run(self):
        self.startThrottle()

# NFVRegulator regulates NFV resources usage when the HMM model reports abnormal VNF behaviors
class NFVRegulator(threading.Thread):
    def __init__(self, t_name, f_input):
        self._running = True
        threading.Thread.__init__(self, name=t_name)
        # features.txt is the performance monitoring events file
        self.f_input = f_input
        self.features = np.loadtxt(f_input)
        # shape[0] number of rows, shape[1] number of columns
        self.length = self.features.shape[0]
        # create NFVThrottle instance, 'instance-00000008' is the NFV VM name (virsh list)
        self.execute = NFVThrottle('nfv_throttle','instance-00000008')

    def terminate(self):
        self._running = False

    # calculate the percentage of resource throttling 
    def throttlePercent(self, da):
        # mean of the given data array (da)
        mean = np.mean(da)
        # standardard deviation of the given da
        std = np.std(da)
        # allow three std
        throttle = std / (mean + 3*std)
        return throttle

    # calculate CPU throttle threshold
    # return the CPU throttle quota
    def calcuCPUthresh(self):
        # calculate the last 100 monitoring samples
        # column 3,4 are cpu related features, CPU system and user usages 
        da_cpu = self.features[(self.length-100):self.length,3:5]
        da_filter = []
        for ca in da_cpu:
            # skip the sample if both features are zero
            if (not(ca[0]==0 and ca[1]==0)):
                da_filter.append(ca[0] + ca[1])
       
        throttle_percent = self.throttlePercent(da_filter)
        # two VCPU scenario, 100000 is the value of cpu.cfs_period_us
        cpu_quota = 2*100000*(1 - throttle_percent)
        
        return int(cpu_quota) 

    # calculate Memory throttle threshold
    # return Memory throttle quota
    def calcuMEMthresh(self):
        # calculate the last 100 monitoring samples
        # column 0,1,2 are memory related features, major faults, minor faults, residence size
        da_mem = self.features[(self.length-100):self.length,0:3]
        da_filter = []
        # only column 2, residence size is used to determine memory size limit
        for ma in da_mem[:,2]:
            # filter samples whose value is zero
            if (ma != 0):
                da_filter.append(ma)
        
        throttle_percent = 0
        if (len(da_filter) !=0 ):
            throttle_percent = self.throttlePercent(da_filter)
        # 4 GB memory scenario
        mem_quota = 4*1024*1024*1024*(1-throttle_percent)
        return int(mem_quota)
               
    # calculate Disk bandwidth throttle threshold
    # return Disk read/write bandwidth throttle quotas
    def calcuDISKthresh(self):
        # column 5,6 are disk read/write bandwidth usage features
        da_disk = self.features[(self.length-100):self.length,5:7]
        da_rfilter = []
        da_wfilter = []
        for ra in da_disk[:,0]:
            if (ra != 0):
                da_rfilter.append(ra)
        for wa in da_disk[:,1]:
            if (wa != 0):
                da_wfilter.append(wa)
        
        throttle_rpercent = 0
        throttle_wpercent = 0
        if (len(da_rfilter) != 0):
            throttle_rpercent = self.throttlePercent(da_rfilter)
        if (len(da_wfilter) != 0):
            throttle_wpercent = self.throttlePercent(da_wfilter)
        # the maximum read/write bandwidth is 100MB, can be set to a larger value
        read_quota = 104857600*(1-throttle_rpercent)
        write_quota = 104857600*(1-throttle_wpercent)

        return int(read_quota), int(write_quota)        

    # calculate network bandwidth throttle threshold
    # return network bandwidth throttle quotas
    def calcuNETthresh(self):
        # column [7,23) are network features 
        da_net = self.features[(self.length-100):self.length,7:23]
        # two network interfaces in this case
        # dev0, dev1 info in the xml file, /opt/stack/data/nova/instances
        da0_infilter = []
        da0_efilter = []
        da1_infilter = []
        da1_efilter = []
        # dev0 read bytes
        for a0_in in da_net[:,0]:
            if (a0_in != 0):
                da0_infilter.append(a0_in)
        # dev0 write bytes
        for a0_e in da_net[:,4]:
            if (a0_e != 0):
                da0_efilter.append(a0_e)
        # dev1 read bytes
        for a1_in in da_net[:,8]:
            if (a1_in != 0):
                da1_infilter.append(a1_in)
        # dev1 write bytes
        for a1_e in da_net[:,12]:
            if (a1_e != 0):
                da1_efilter.append(a1_e)
        # in: ingress, e: egress
        throttle0_in = 0
        throttle0_e = 0
        throttle1_in = 0
        throttle1_e = 0
        if (len(da0_infilter) != 0):
            throttle0_in = self.throttlePercent(da0_infilter)
        if (len(da0_efilter) != 0):
            throttle0_e = self.throttlePercent(da0_efilter)
        if (len(da1_infilter) != 0):
            throttle1_in = self.throttlePercent(da1_infilter)
        if (len(da1_efilter) != 0):
            throttle1_in = self.throttlePercent(da1_efilter)
        dev0 = 'tapa3502e00-82'
        dev1 = 'tapc56fd03b-31'
        # maximum bandwidth 10Gb, default Kbps
        quota0_in = 10240000 * (1-throttle0_in)  
        quota0_e = 10240000 * (1-throttle0_e)  
        quota1_in = 10240000 * (1-throttle1_in)  
        quota1_e = 10240000 * (1-throttle1_e)
        
        return int(quota0_in), int(quota0_e), int(quota1_in), int(quota1_e)  


    def startRegulate(self):
        # skip regulation if there are too few history samples
        while self._running:
            time.sleep(2)
            # features.txt is the performance monitoring events file
            self.features = np.loadtxt(self.f_input)
            # shape[0] number of rows, shape[1] number of columns
            self.length = self.features.shape[0]

            cpu_thresh = self.calcuCPUthresh()
            mem_thresh = self.calcuMEMthresh()
            disk_rthresh,disk_wthresh = self.calcuDISKthresh()
            net_in0,net_e0,net_in1,net_e1 = self.calcuNETthresh()
            # dev0 name is from xml file
            dev0 = 'tapa3502e00-82'
            # if0 is the corresponding interface name from ovs-ctl show command
            if0 = 'qvoa3502e00-82'
            dev1 = 'tapc56fd03b-31'
            if1 = 'qvoc56fd03b-31'
            self.execute.throttleCPU(cpu_thresh)
            #self.execute.throttleMEM(mem_thresh)
            #self.execute.throttleDISK(disk_rthresh, 'read_bps')
            #self.execute.throttleDISK(disk_wthresh, 'write_bps')        
            self.execute.throttleNET(net_in0, net_e0, if0)
            self.execute.throttleNET(net_in1, net_e1, if1)
            print (cpu_thresh, mem_thresh, disk_rthresh, disk_wthresh, net_in0, net_e0, net_in1, net_e1)
            #print (cpu_thresh)

    def run(self):
        # skip regulation if there are too few history samples
        if (self.length > 100):
            self.startRegulate()

if __name__ == "__main__":

    #the parameters are defined as global variables for easy tuning 
    #global_cluster1 is defined as the number of clusters to cluster CPU,MEM,BLOCK features
    global global_cluster1
    #global_cluster2 is defined as the number of clusters to cluster NETWORK features
    global global_cluster2
    #global_components is defined as the the number of hidden states in the HMM model
    global global_components
    #global_window is defined as the size of the sliding window used to store observations
    global global_window 
    #global_threshold is defined as the threshold to determine abnormal VNF behavior
    global global_threshold 
    #global_wthreshold is defined as the size of the sliding window used to store latest features to determine resource throttle threshold
    global global_wthrottle

    #the current tuned parameters, still needs further tuning work 
    global_cluster1 = 20
    global_cluster2 = 20
    global_components = 30
    global_window = 20
    global_threshold = -110
    global_wthrottle = 100
    # Input the instance ID to Monitor VNF instance
    if len(sys.argv) == 1:
       print ("please give the domain id parameter!\n")
    # the shared queue q is used by NFVMonitor and NFVCluster, each item in the q is the performance vector
    Q_vec = Queue.Queue()
    # the shared queue qc is used by NFVCluster and NFVHMM, each item in the qc is the observation
    Q_obv = Queue.Queue()

    nfv_regulator = NFVRegulator('nfv_regulator','new_reg_c2.txt')
    nfv_regulator.startRegulate()

    time.sleep(600)

    nfv_regulator.terminate()

    nfv_regulator.execute.throttleCPU(-1)
    # NFVMonitor is used to collect VNF instance performance featuress 
#    nfv_monitor = NFVMonitor('nfv_monitor', int(sys.argv[1]), Q_vec)
#    nfv_monitor.start()
#    time.sleep(600)

    # NFVCluster is used to cluster performance features samples into several clusters 
#    nfv_cluster = NFVCluster('nfv_cluster', Q_vec, Q_obv)
    # trainCluster function accept three parameters
    # 'train_features.txt' is the input file containing the collected VNF performance features
    # 'out_stand.txt' is the output file containing the preprocessed standardization features of 'train_features.txt'
    # 'train_label.txt' is the output file containing the clustered labels of each feature samples
#    nfv_cluster.trainCluster('train_features.txt','out_stand.txt','train_label.txt')
    # 'predict_features.txt' is the input file containing the collected VNF performance features being predicted
    # 'out_stand1.txt' is the output file containing the preprocessed standardization features of 'predict_features.txt'
    # 'predict_label.txt' is the output file containing the predicted cluster labels of each feature samples
#    nfv_cluster.offlinePredict('predict_features.txt','out_stand1.txt','predict_label.txt')
#    nfv_cluster.start()
#    time.sleep(10)
#

    # NFVHMM is used to predict abnormal behavior of VNF instances
#    nfv_hmm = NFVHMM('nfv_hmm', Q_obv)
#    nfv_hmm.trainHMM('train_label.txt')
#    nfv_hmm.offlinePredict('train_label.txt','hmm_result_man6.txt')
#    nfv_hmm.offlinePredict('predict_label.txt','hmm_result_man7.txt')
#    nfv_hmm.start()
#    time.sleep(50)
    
#    nfv_monitor.terminate()
#    nfv_cluster.terminate()
#    nfv_hmm.terminate()

    #nfv_cluster = NFVCluster('nfv_cluster', Q_vec, Q_obv)
    #nfv_cluster.start()
#    nfv_monitor.join()
#    nfv_cluster.join()
#    nfv_hmm.join()
