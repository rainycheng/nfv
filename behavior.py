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
      self.__running = True

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
      while True and self.__running:
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
         features.write(stats_vector)
         features.flush()
         print(stats_vector)
         time.sleep(2) 

   def run(self):
      self.startMonitor()

class NFVCluster(threading.Thread):
   """transform VNF observations into a sequence of cluster labels"""
   def __init__(self,t_name):
      self.__running = True
      threading.Thread.__init__(self, name=t_name)
   
   def terminate(self):
      self.__running = False

   def startCluster():
      features = np.loadtxt('features.txt')
      print (features)
      labels = open('/home/stack/labels.txt','w')
      labels.close()
   
   def run(self):
      self.startCluster()

features = open('/home/stack/features.txt','w')
q = Queue.Queue()
nfv_monitor = NFVMonitor('nfv_monitor',33,q)
nfv_monitor.start()
time.sleep(20)
nfv_monitor.terminate()
nfv_monitor.join()
features.close()

