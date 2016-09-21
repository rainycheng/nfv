#! /usr/bin/env python2.7
#encoding:utf-8
from __future__ import print_function
import threading,Queue,time,os,sys
import libvirt
from xml.etree import ElementTree

class Monitor(threading.Thread):
   """performance monitoring class"""
   def __init__(self,t_name,domID,queue):
      self.queue = queue

      conn = libvirt.open('qemu:///system')
      if conn == None:
         print('Failed to open connection to qemu:///system', file=sys.stderr)
         exit(1)
      self.dom = conn.lookupByID(domID)
      if self.dom == None:
         print('Failed to find the domain '+domName, file=sys.stderr)
         exit(1)

      threading.Thread.__init__(self, name=t_name)

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
      while True:
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



features = open('/home/stack/features.txt','a')
q = Queue.Queue()
nfv_monitor = Monitor('nfv_monitor',33,q)
nfv_monitor.start()
nfv_monitor.join()
