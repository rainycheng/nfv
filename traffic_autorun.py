import os,time

lsresult = os.popen('ls outputFile/').read()

outputfiles = lsresult.split('\n')

for fname in outputfiles:
    command1 = 'tcpreplay -i eth0 outputFile/' + fname
    os.system(command1)
    time.sleep(30)
