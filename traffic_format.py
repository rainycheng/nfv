import sys,os

lsresult = os.popen('ls trafficData/').read()
datafiles = lsresult.split('\n')
i=1
for fname in datafiles:
    command1 = 'tcpprep --auto=bridge --pcap=trafficData/'+fname+' --cachefile='+'cacheFile/input'+str(i)+'.cache'
    command2 = 'tcprewrite --infile=trafficData/'+fname+' --outfile=outFile/itest'+str(i)+'.pcap'+' --cachefile=cacheFile/input'+str(i)+'.cache'+' --enet-dmac=fa:16:3e:12:50:ee --enet-smac=fa:16:3e:de:68:32 --endpoints=192.168.2.14:192.168.1.5 --skipbroadcast'
    print command1
    print command2
    i = i+1
    os.system(command1)
    os.system(command2)


