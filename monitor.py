#!/usr/bin/env python
from __future__ import print_function
import sys
import libvirt
from xml.etree import ElementTree

domName = 'instance-00000001'

conn = libvirt.open('qemu:///system')
if conn == None:
    print('Failed to open connection to qemu:///system', file=sys.stderr)
    exit(1)

#print ("argv:", sys.argv[1])
#input the domain ID

dom = conn.lookupByID(int(sys.argv[1]))
if dom == None:
    print('Failed to find the domain '+domName, file=sys.stderr)
    exit(1)

#get vCPU performance stats
cpu_stats = dom.getCPUStats(False)
for (i, cpu) in enumerate(cpu_stats):
    print('CPU '+str(i)+' Time: '+str(cpu['cpu_time']/1000000000.))
#    print('CPU '+str(i)+' Time_sys: '+str(cpu['system_time']/1000000000.))
#    print('CPU '+str(i)+' Time_user: '+str(cpu['user_time']/1000000000.))
cpu_aggregate = dom.getCPUStats(True)
print('cpu_time:    '+str(cpu_aggregate[0]['cpu_time']))
print('system_time: '+str(cpu_aggregate[0]['system_time']))
print('user_time:   '+str(cpu_aggregate[0]['user_time']))

#get memory statistics
mem_stats = dom.memoryStats()
print('Memory used:')
for name in mem_stats:
    print(' '+str(mem_stats[name])+'('+name+')')

#get disk I/O stats
block_dir = '/dev/disk/by-path/ip-10.149.59.163:3260-iscsi-iqn.2010-10.org.openstack:volume-8ab63a87-0044-4e5f-aae9-6ae2c57491c5-lun-1'
rd_req, rd_bytes, wr_req, wr_bytes, err = \
dom.blockStats('vda')
print('Read requests issued:  '+str(rd_req))
print('Bytes read:            '+str(rd_bytes))
print('Write requests issued: '+str(wr_req))
print('Bytes written:         '+str(wr_bytes))
print('Number of errors:      '+str(err))
 

#get network performance stats
tree = ElementTree.fromstring(dom.XMLDesc())
iface = tree.find('devices/interface/target').get('dev')
stats = dom.interfaceStats(iface)
print('read bytes:    '+str(stats[0]))
print('read packets:  '+str(stats[1]))
print('read errors:   '+str(stats[2]))
print('read drops:    '+str(stats[3]))
print('write bytes:   '+str(stats[4]))
print('write packets: '+str(stats[5]))
print('write errors:  '+str(stats[6]))
print('write drops:   '+str(stats[7]))

conn.close()
exit(0)
