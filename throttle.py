#!/usr/bin/python
#from __future__ import print_function
import sys
import libvirt
import subprocess

#CPU throttle
cgdir = '/sys/fs/cgroup/cpu/machine/'
inst_name = 'instance-00000009'
quota = 80000
COMMAND = 'echo ' + quota + ' > ' + cgdir + inst_name + '.libvirt-qemu/cpu.cfs_quota_us'
p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#for line in p.stdout.readlines():
#	print line,
retval = p.wait()

#mem throttle
cgdir = '/sys/fs/cgroup/memory/machine/'
inst_name = 'instance-00000009'
quota = '1G'
COMMAND = 'echo ' + quota + ' > ' + cgdir + inst_name + '.libvirt-qemu/memory.limit_in_bytes'
p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
retval = p.wait()

#blkio throttle
cgdir = '/sys/fs/cgroup/blkio/machine/'
inst_name = 'instance-00000009'
quota = '8:0 ' + '1000'
COMMAND = 'echo ' + quota + ' > ' + cgdir + inst_name + '.libvirt-qemu/blkio.throttle.read_iops_device'
p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
retval = p.wait()

quota = '8:0 ' + '1000'
COMMAND = 'echo ' + quota + ' > ' + cgdir + inst_name + '.libvirt-qemu/blkio.throttle.write_iops_device'
p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
retval = p.wait()

#network throttle
dev = 'tap0'
rate = 'rate=1000'
#rate = 'burst=100'
COMMAND = 'ovs-vsctl set interface ' + dev + 'ingress_policing_'+ rate 
p = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
retval = p.wait()



